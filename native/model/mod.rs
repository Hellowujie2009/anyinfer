//! Native, lazy reader for a single-host DeepSeek checkpoint.
mod head;
mod onnx;
mod oracle;
pub use oracle::golden_head;
#[cfg(test)]
mod tests;
pub use head::prepare_head;

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Take};
use std::path::{Component, Path, PathBuf};

#[derive(Deserialize)]
struct WeightIndex {
    weight_map: BTreeMap<String, String>,
}

#[derive(Deserialize)]
struct HeaderEntry {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

pub struct Tensor {
    pub dtype: String,
    pub shape: Vec<u64>,
    pub byte_len: u64,
    path: PathBuf,
    offset: u64,
}

impl Tensor {
    pub fn read_range(&self, start: u64, len: u64) -> Result<Take<File>> {
        ensure!(
            start <= self.byte_len && len <= self.byte_len - start,
            "tensor range exceeds payload"
        );
        let mut file = File::open(&self.path).with_context(|| self.path.display().to_string())?;
        file.seek(SeekFrom::Start(
            self.offset
                .checked_add(start)
                .context("tensor offset overflow")?,
        ))?;
        Ok(file.take(len))
    }
}

pub struct Checkpoint {
    pub model_type: String,
    pub hidden_size: u64,
    pub hc_mult: u64,
    pub vocab_size: u64,
    pub norm_eps: f32,
    pub hc_eps: f32,
    tensors: BTreeMap<String, Tensor>,
    shard_count: usize,
}

fn dtype_bits(dtype: &str) -> Result<u64> {
    Ok(match dtype {
        "F4" | "F4_E2M1" | "I4" | "U4" => 4,
        "F6_E2M3" | "F6_E3M2" => 6,
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" | "F8_E8M0" | "F8_E4M3FNUZ"
        | "F8_E5M2FNUZ" => 8,
        "U16" | "I16" | "F16" | "BF16" => 16,
        "U32" | "I32" | "F32" => 32,
        "U64" | "I64" | "F64" | "C64" => 64,
        _ => bail!("unrecognized safetensors dtype {dtype}"),
    })
}

fn read_shard(path: &Path) -> Result<BTreeMap<String, Tensor>> {
    let mut file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)?;
    let header_len = u64::from_le_bytes(prefix);
    ensure!(
        header_len > 0 && header_len <= 100_000_000,
        "invalid safetensors header length {header_len}"
    );
    let base = 8 + header_len;
    ensure!(base <= file_len, "truncated tensor header");
    let mut encoded = vec![0; header_len as usize];
    file.read_exact(&mut encoded)?;
    let mut header: BTreeMap<String, Value> = serde_json::from_slice(&encoded)?;
    header.remove("__metadata__");
    let mut ranges = Vec::new();
    let mut tensors = BTreeMap::new();
    for (name, entry) in header {
        let e: HeaderEntry = serde_json::from_value(entry).with_context(|| name.clone())?;
        let [start, end] = e.data_offsets;
        ensure!(
            start <= end && end <= file_len - base,
            "invalid data_offsets for {name}"
        );
        let elements = e
            .shape
            .iter()
            .try_fold(1u64, |n, d| n.checked_mul(*d))
            .context("shape overflows")?;
        let bits = elements
            .checked_mul(dtype_bits(&e.dtype)?)
            .context("tensor size overflows")?;
        ensure!(
            bits.is_multiple_of(8) && bits / 8 == end - start,
            "shape, dtype and length disagree for {name}"
        );
        ranges.push((start, end));
        tensors.insert(
            name,
            Tensor {
                dtype: e.dtype,
                shape: e.shape,
                byte_len: end - start,
                path: path.to_owned(),
                offset: base + start,
            },
        );
    }
    ranges.sort_unstable();
    let mut cursor = 0;
    for (start, end) in ranges {
        ensure!(
            start == cursor,
            "overlapping tensors or holes in payload at {cursor}"
        );
        cursor = end;
    }
    ensure!(cursor == file_len - base, "unreferenced trailing payload");
    Ok(tensors)
}

fn positive_integer(config: &Value, name: &str) -> Result<u64> {
    config[name]
        .as_u64()
        .filter(|n| *n > 0)
        .with_context(|| format!("invalid config field {name}"))
}

fn positive_float(config: &Value, name: &str) -> Result<f32> {
    let n = config[name]
        .as_f64()
        .with_context(|| format!("invalid config field {name}"))? as f32;
    ensure!(n.is_finite() && n > 0.0, "invalid config field {name}");
    Ok(n)
}

impl Checkpoint {
    pub fn open(root: &Path) -> Result<Self> {
        let config: Value = serde_json::from_reader(
            File::open(root.join("config.json")).context("open config.json")?,
        )?;
        let model_type = config["model_type"]
            .as_str()
            .context("missing model_type")?;
        let text_config = match model_type {
            "deepseek_v4" => &config,
            "deepseek_v41" if config["text_config"]["model_type"] == "deepseek_v41_text" => {
                &config["text_config"]
            }
            _ => bail!("unsupported model_type {model_type}"),
        };
        let mut ckpt = Self {
            model_type: model_type.to_owned(),
            hidden_size: positive_integer(text_config, "hidden_size")?,
            hc_mult: positive_integer(text_config, "hc_mult")?,
            vocab_size: positive_integer(text_config, "vocab_size")?,
            norm_eps: positive_float(text_config, "rms_norm_eps")?,
            hc_eps: positive_float(text_config, "hc_eps")?,
            tensors: BTreeMap::new(),
            shard_count: 0,
        };
        let index_file = root.join("model.safetensors.index.json");
        let index: Option<WeightIndex> = match File::open(&index_file) {
            Ok(file) => Some(serde_json::from_reader(file).context("invalid safetensors index")?),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.into()),
        };
        let shards: BTreeSet<&str> = match &index {
            Some(i) => i.weight_map.values().map(String::as_str).collect(),
            None => ["model.safetensors"].into(),
        };
        for filename in shards {
            let name = Path::new(filename);
            ensure!(
                matches!(
                    name.components().collect::<Vec<_>>().as_slice(),
                    [Component::Normal(_)]
                ) && !filename.contains(['/', '\\', ':'])
                    && filename.ends_with(".safetensors"),
                "invalid shard filename {filename}"
            );
            let shard =
                read_shard(&root.join(name)).with_context(|| format!("read shard {filename}"))?;
            for (key, tensor) in shard {
                if let Some(i) = &index {
                    ensure!(
                        i.weight_map.get(&key).map(String::as_str) == Some(filename),
                        "index does not assign {key} to {filename}"
                    );
                }
                ensure!(
                    ckpt.tensors.insert(key.clone(), tensor).is_none(),
                    "duplicate tensor {key}"
                );
            }
            ckpt.shard_count += 1;
        }
        if let Some(i) = index {
            ensure!(
                ckpt.tensors.len() == i.weight_map.len(),
                "index references missing tensors"
            );
        }
        ensure!(!ckpt.tensors.is_empty(), "empty checkpoint");
        Ok(ckpt)
    }

    pub fn tensor(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .with_context(|| format!("missing checkpoint tensor {name}"))
    }

    pub fn reader(&self, name: &str) -> Result<Take<File>> {
        let tensor = self.tensor(name)?;
        tensor.read_range(0, tensor.byte_len)
    }
}

pub fn inspect(root: &Path) -> Result<Value> {
    let ckpt = Checkpoint::open(root)?;
    let mut by_dtype = BTreeMap::<&str, (u64, u64)>::new();
    for t in ckpt.tensors.values() {
        let (count, bytes) = by_dtype.entry(&t.dtype).or_default();
        *count += 1;
        *bytes = bytes
            .checked_add(t.byte_len)
            .context("checkpoint size overflows")?;
    }
    let by_dtype: BTreeMap<_, _> = by_dtype
        .into_iter()
        .map(|(dtype, (count, bytes))| (dtype, json!({"tensors":count,"bytes":bytes})))
        .collect();
    let mut head_parameters = BTreeMap::new();
    for (name, t) in &ckpt.tensors {
        if name.contains("hc_head_") && !name.contains("mtp") {
            head_parameters.insert(name, json!({"shape":t.shape,"dtype":t.dtype}));
        }
    }
    let h = ckpt.hc_mult;
    let d = ckpt.hidden_size;
    let shapes: Vec<_> = [1u64,8,32,128,512].iter().map(|&t| json!({"tokens":t,"block_output":[t,d],"residual":[t,h,d],"post":[t,h],"comb":[t,h,h],"output":[t,h,d]})).collect();
    Ok(
        json!({"format_version":1,"model_type":ckpt.model_type,"shards":ckpt.shard_count,"tensor_count":ckpt.tensors.len(),
        "hidden_size":d,"hc_mult":h,"vocab_size":ckpt.vocab_size,"dtypes":by_dtype,"head_parameters":head_parameters,
        "hc_post_cases":shapes,"validation":"shard headers, index membership, dtype, shape and byte ranges"}),
    )
}
