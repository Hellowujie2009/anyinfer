use super::onnx::{BFLOAT16, FLOAT, Graph};
use super::{Checkpoint, Tensor};
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use std::fs::{self, File};
use std::io;
use std::ops::Range;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub(super) fn parameter<'a>(
    ckpt: &'a Checkpoint,
    names: &[&str],
    shape: &[u64],
) -> Result<(&'a str, &'a Tensor)> {
    let mut matching = names
        .iter()
        .filter_map(|name| ckpt.tensors.get_key_value(*name));
    let (name, tensor) = matching
        .next()
        .with_context(|| format!("expected one of {names:?}"))?;
    ensure!(
        matching.next().is_none(),
        "ambiguous checkpoint parameters {names:?}"
    );
    ensure!(
        tensor.shape == shape,
        "unexpected shape {:?} for {name}, expected {shape:?}",
        tensor.shape
    );
    ensure!(
        matches!(tensor.dtype.as_str(), "BF16" | "F32"),
        "unsupported head weight dtype {} for {name}",
        tensor.dtype
    );
    Ok((name.as_str(), tensor))
}

fn write_parameter(
    graph: &mut Graph,
    directory: &Path,
    ckpt: &Checkpoint,
    names: &[&str],
    name: &str,
    full_shape: &[u64],
    rows: Range<u64>,
) -> Result<Value> {
    let (source, tensor) = parameter(ckpt, names, full_shape)?;
    ensure!(
        rows.start < rows.end && rows.end <= full_shape[0],
        "invalid row range"
    );
    let mut shape = full_shape.to_vec();
    shape[0] = rows.end - rows.start;
    let row_bytes = tensor.byte_len / full_shape[0];
    let size = row_bytes
        .checked_mul(shape[0])
        .context("shard size overflow")?;
    let start = row_bytes
        .checked_mul(rows.start)
        .context("shard offset overflow")?;
    let filename = format!("{name}.bin");
    let mut writer = File::create(directory.join(&filename))?;
    let n = io::copy(&mut tensor.read_range(start, size)?, &mut writer)?;
    ensure!(n == size, "{source} truncated while copying");
    let raw = format!("{name}_raw");
    let dtype = if tensor.dtype == "BF16" {
        BFLOAT16
    } else {
        FLOAT
    };
    graph.external(&raw, dtype, &shape, &filename);
    graph.node("Cast", &[&raw], name, Some(("to", FLOAT)));
    Ok(
        json!({"source":source,"file":filename,"dtype":tensor.dtype,"shape":shape,"bytes":size,"source_byte_offset":start}),
    )
}

fn emit(ckpt: &Checkpoint, dest: &Path, rank: u32, world_size: u32) -> Result<Value> {
    let h = ckpt.hc_mult;
    let d = ckpt.hidden_size;
    let flat = h
        .checked_mul(d)
        .context("hc_mult * hidden_size overflows")?;
    let vocab = ckpt.vocab_size;
    ensure!(
        vocab.is_multiple_of(u64::from(world_size)),
        "vocab_size must divide evenly among ranks"
    );
    let rows = vocab / u64::from(world_size);
    let start = rows * u64::from(rank);
    let mut g = Graph::default();
    g.input("residual", BFLOAT16, &[h, d]);
    g.output("logits", FLOAT, &[rows]);
    let mut parameters = Vec::new();
    for (name, keys, dims) in [
        (
            "gate_weight",
            ["hc_head_fn", "model.hc_head_fn"],
            vec![h, flat],
        ),
        (
            "gate_scale",
            ["hc_head_scale", "model.hc_head_scale"],
            vec![1],
        ),
        ("gate_bias", ["hc_head_base", "model.hc_head_base"], vec![h]),
        ("norm_weight", ["norm.weight", "model.norm.weight"], vec![d]),
    ] {
        parameters.push(write_parameter(
            &mut g,
            dest,
            ckpt,
            &keys,
            name,
            &dims,
            0..dims[0],
        )?);
    }
    parameters.push(write_parameter(
        &mut g,
        dest,
        ckpt,
        &[
            "head.weight",
            "model.head.weight",
            "lm_head.weight",
            "model.lm_head.weight",
        ],
        "vocab_weight",
        &[vocab, d],
        start..(start + rows),
    )?);
    g.scalar("norm_eps", ckpt.norm_eps);
    g.scalar("hc_eps", ckpt.hc_eps);
    g.integers("flattened_column", &[flat, 1]);
    g.integers("gate_column", &[h, 1]);
    g.integers("hidden_column", &[d, 1]);
    g.integers("vocab_vector", &[rows]);
    g.integers("stream_axis", &[0]);
    g.node("Cast", &["residual"], "x", Some(("to", FLOAT)));
    g.node("Reshape", &["x", "flattened_column"], "flat", None);
    g.node("Mul", &["x", "x"], "x_squared", None);
    g.node(
        "ReduceMean",
        &["x_squared"],
        "x_variance",
        Some(("keepdims", 0)),
    );
    g.node("Add", &["x_variance", "norm_eps"], "x_variance_eps", None);
    g.node("Sqrt", &["x_variance_eps"], "x_rms", None);
    g.node("Reciprocal", &["x_rms"], "x_inv_rms", None);
    g.node("MatMul", &["gate_weight", "flat"], "mixes", None);
    g.node("Mul", &["mixes", "x_inv_rms"], "mixes_norm", None);
    g.node("Mul", &["mixes_norm", "gate_scale"], "mixes_scaled", None);
    g.node(
        "Reshape",
        &["gate_bias", "gate_column"],
        "bias_column",
        None,
    );
    g.node("Add", &["mixes_scaled", "bias_column"], "gate_scores", None);
    g.node("Sigmoid", &["gate_scores"], "gate_prob", None);
    g.node("Add", &["gate_prob", "hc_eps"], "gate", None);
    g.node("Mul", &["x", "gate"], "weighted_streams", None);
    g.node(
        "ReduceSum",
        &["weighted_streams", "stream_axis"],
        "combined_fp32",
        Some(("keepdims", 0)),
    );
    g.node(
        "Cast",
        &["combined_fp32"],
        "combined_bf16",
        Some(("to", BFLOAT16)),
    );
    g.node("Cast", &["combined_bf16"], "combined", Some(("to", FLOAT)));
    g.node("Mul", &["combined", "combined"], "combined_squared", None);
    g.node(
        "ReduceMean",
        &["combined_squared"],
        "variance",
        Some(("keepdims", 0)),
    );
    g.node("Add", &["variance", "norm_eps"], "variance_eps", None);
    g.node("Sqrt", &["variance_eps"], "rms", None);
    g.node("Reciprocal", &["rms"], "inv_rms", None);
    g.node("Mul", &["combined", "inv_rms"], "normalized", None);
    g.node("Mul", &["normalized", "norm_weight"], "hidden_fp32", None);
    g.node(
        "Cast",
        &["hidden_fp32"],
        "hidden_bf16",
        Some(("to", BFLOAT16)),
    );
    g.node("Cast", &["hidden_bf16"], "hidden", Some(("to", FLOAT)));
    g.node(
        "Reshape",
        &["hidden", "hidden_column"],
        "normalized_column",
        None,
    );
    g.node(
        "MatMul",
        &["vocab_weight", "normalized_column"],
        "rank_logits",
        None,
    );
    g.node("Reshape", &["rank_logits", "vocab_vector"], "logits", None);
    fs::write(dest.join("head.onnx"), g.finish())?;
    let report = json!({"format_version":1,"model_type":ckpt.model_type,"rank":rank,"world_size":world_size,
        "vocab_start":start,"vocab_end":start+rows,"vocab_size":vocab,"hc_mult":h,"hidden_size":d,
        "graph":"head.onnx","input":{"name":"residual","dtype":"BF16","shape":[h,d]},
        "output":{"name":"logits","dtype":"F32","shape":[rows]},"parameters":parameters});
    fs::write(dest.join("head.json"), serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

pub fn prepare_head(root: &Path, output: &Path, rank: u32, world_size: u32) -> Result<Value> {
    ensure!(
        world_size > 0 && rank < world_size,
        "require world_size > 0 and 0 <= rank < world_size"
    );
    ensure!(
        !output.try_exists()?,
        "output already exists: {}",
        output.display()
    );
    let ckpt = Checkpoint::open(root)?;
    ensure!(
        ckpt.model_type == "deepseek_v4",
        "this output head implements deepseek_v4"
    );
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let tmp = parent.join(format!(".anyinfer-head-{}-{timestamp}", std::process::id()));
    fs::create_dir(&tmp)?;
    let result = emit(&ckpt, &tmp, rank, world_size).and_then(|report| {
        if output.try_exists()? {
            bail!("output already exists: {}", output.display());
        }
        fs::rename(&tmp, output)?;
        Ok(report)
    });
    if result.is_err() {
        let _ = fs::remove_dir_all(&tmp);
    }
    result
}
