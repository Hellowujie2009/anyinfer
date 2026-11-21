use anyhow::{Context, Result, ensure};
use clap::Args;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::time::Instant;

#[derive(Args)]
pub struct Options {
    /// Directory containing head.json and head.om. Repeat in rank order.
    #[arg(long, required = true)]
    rank_dir: Vec<PathBuf>,
    /// Visible device IDs in rank order.
    #[arg(long, value_delimiter = ',', required = true)]
    devices: Vec<u32>,
    /// Last-token residual, little-endian BF16 in [hc_mult, hidden_size] order.
    #[arg(long)]
    residual: PathBuf,
    /// Trusted full-vocabulary logits, little-endian FP32 in ascending token order.
    #[arg(long)]
    golden_logits: Option<PathBuf>,
    #[arg(long, default_value_t = 0.05)]
    atol: f64,
    #[arg(long, default_value_t = 0.01)]
    rtol: f64,
    #[arg(long, default_value_t = 3)]
    warmup: u32,
    #[arg(long, default_value_t = 10)]
    samples: u32,
    /// New JSON report file.
    #[arg(long)]
    output: PathBuf,
}

#[derive(Deserialize, Serialize)]
struct Head {
    format_version: u32,
    model_type: String,
    rank: u32,
    world_size: u32,
    vocab_start: u32,
    vocab_end: u32,
    vocab_size: u32,
    hidden_size: u32,
    hc_mult: u32,
    graph: String,
}

fn validate_heads(heads: &[Head], devices: &[u32]) -> Result<()> {
    let first = heads.first().context("missing head ranks")?;
    ensure!(heads.len() == devices.len(), "supply one device per rank");
    ensure!(
        devices.iter().copied().collect::<HashSet<_>>().len() == devices.len()
            && devices.iter().all(|&d| i32::try_from(d).is_ok()),
        "invalid or repeated device"
    );
    let world = u32::try_from(heads.len())?;
    ensure!(
        first.vocab_size > 0 && first.vocab_size.is_multiple_of(world),
        "invalid vocabulary partition"
    );
    let rows = first.vocab_size / world;
    for (rank, h) in heads.iter().enumerate() {
        let rank = rank as u32;
        ensure!(
            h.format_version == 1 && h.model_type == "deepseek_v4" && h.graph == "head.onnx",
            "unsupported head manifest for rank {rank}"
        );
        ensure!(
            h.rank == rank
                && h.world_size == world
                && h.vocab_size == first.vocab_size
                && h.vocab_start == rank * rows
                && h.vocab_end == (rank + 1) * rows
                && h.hidden_size > 0
                && h.hc_mult > 0
                && h.hidden_size == first.hidden_size
                && h.hc_mult == first.hc_mult,
            "inconsistent head partition for rank {rank}"
        );
    }
    Ok(())
}

struct Worker {
    child: Child,
    output: BufReader<ChildStdout>,
}
impl Worker {
    fn launch(bin: &Path, opt: &Options, h: &Head) -> Result<Self> {
        let r = h.rank as usize;
        let mut child = Command::new(bin)
            .args(
                [
                    opt.devices[r],
                    h.hc_mult,
                    h.hidden_size,
                    h.vocab_start,
                    h.vocab_end - h.vocab_start,
                    opt.warmup,
                    opt.samples,
                ]
                .map(|n| n.to_string()),
            )
            .arg(opt.rank_dir[r].join("head.om"))
            .arg(&opt.residual)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("start rank {r} with {}", bin.display()))?;
        let output = BufReader::new(child.stdout.take().context("missing worker output")?);
        Ok(Self { child, output })
    }
    fn marker(&mut self, prefix: &str) -> Result<String> {
        let mut line = String::new();
        loop {
            ensure!(
                self.output.read_line(&mut line)? > 0,
                "worker exited before {prefix}"
            );
            if let Some(message) = line.trim_end().strip_prefix("ANYINFER_HEAD_ERROR ") {
                anyhow::bail!("{message}");
            }
            if let Some(data) = line.trim_end().strip_prefix(prefix) {
                return Ok(data.to_string());
            }
            eprint!("{line}");
            line.clear();
        }
    }
    fn send(&mut self, command: &[u8]) -> Result<()> {
        self.child
            .stdin
            .as_mut()
            .context("closed worker input")?
            .write_all(command)?;
        Ok(())
    }
    fn finish(&mut self) -> Result<()> {
        drop(self.child.stdin.take());
        std::io::copy(&mut self.output, &mut std::io::sink())?;
        ensure!(self.child.wait()?.success(), "rank process failed");
        Ok(())
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        // Also release workers waiting at the ready barrier after another rank fails.
        drop(self.child.stdin.take());
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[derive(Deserialize, Serialize)]
struct RankResult {
    token_id: u32,
    logit: f32,
    soc: String,
    device_us: Vec<f64>,
    input_h2d_us: f64,
    output_d2h_us: f64,
    local_checked_argmax_us: f64,
}

fn winner(scores: impl IntoIterator<Item = (u32, f32)>) -> Result<(u32, f32)> {
    let mut best = None;
    for (token, logit) in scores {
        ensure!(logit.is_finite(), "non-finite logit for token {token}");
        if best.is_none_or(|(t, x)| logit > x || (logit == x && token < t)) {
            best = Some((token, logit));
        }
    }
    best.context("no logits")
}

#[derive(Default, Serialize)]
struct Comparison {
    elements: u64,
    failed_elements: u64,
    max_abs_error: f64,
    expected_token: u32,
    token_matches: bool,
}
fn compare(golden: &[u8], actual: &[u8], atol: f64, rtol: f64, c: &mut Comparison) -> Result<()> {
    ensure!(
        golden.len() == actual.len() && golden.len().is_multiple_of(4),
        "invalid logits size"
    );
    for (&g, &a) in golden
        .as_chunks::<4>()
        .0
        .iter()
        .zip(actual.as_chunks::<4>().0)
    {
        let expected = f32::from_le_bytes(g) as f64;
        let got = f32::from_le_bytes(a) as f64;
        ensure!(
            expected.is_finite() && got.is_finite(),
            "non-finite value in accuracy input"
        );
        let err = (expected - got).abs();
        c.max_abs_error = c.max_abs_error.max(err);
        c.failed_elements += u64::from(err > atol + rtol * expected.abs());
        c.elements += 1;
    }
    Ok(())
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    std::io::copy(&mut File::open(path)?, &mut hasher)?;
    Ok(hex::encode(hasher.finalize()))
}

fn evaluate(opt: &Options, bin: &Path) -> Result<Value> {
    ensure!(
        opt.atol.is_finite()
            && opt.atol >= 0.0
            && opt.rtol.is_finite()
            && opt.rtol >= 0.0
            && opt.samples > 0,
        "invalid tolerance or sample count"
    );
    let heads: Vec<Head> = opt
        .rank_dir
        .iter()
        .map(|dir| {
            let path = dir.join("head.json");
            ensure!(
                dir.join("head.om").is_file(),
                "missing {}",
                dir.join("head.om").display()
            );
            serde_json::from_reader(File::open(&path)?).with_context(|| path.display().to_string())
        })
        .collect::<Result<_>>()?;
    validate_heads(&heads, &opt.devices)?;
    let h = &heads[0];
    let expected_residual = u64::from(h.hidden_size)
        .checked_mul(u64::from(h.hc_mult))
        .and_then(|x| x.checked_mul(2))
        .context("residual size overflow")?;
    ensure!(
        fs::metadata(&opt.residual)?.len() == expected_residual,
        "residual has wrong byte length"
    );
    let residual_hash = hex::encode(Sha256::digest(fs::read(&opt.residual)?));
    let golden = opt
        .golden_logits
        .as_ref()
        .map(|path| -> Result<Vec<u8>> {
            ensure!(
                fs::metadata(path)?.len() == u64::from(h.vocab_size) * 4,
                "golden logits have wrong byte length"
            );
            Ok(fs::read(path)?)
        })
        .transpose()?;
    let digests: Vec<_> = opt
        .rank_dir
        .iter()
        .map(|dir| -> Result<_> {
            Ok((
                file_sha256(&dir.join("head.json"))?,
                file_sha256(&dir.join("head.om"))?,
            ))
        })
        .collect::<Result<_>>()?;
    let mut workers = heads
        .iter()
        .map(|h| Worker::launch(bin, opt, h))
        .collect::<Result<Vec<_>>>()?;
    for (rank, worker) in workers.iter_mut().enumerate() {
        worker
            .marker("ANYINFER_HEAD_READY")
            .with_context(|| format!("load/warm up rank {rank}"))?;
    }
    let started = Instant::now();
    for worker in &mut workers {
        worker.send(b"RUN\n")?;
    }
    let mut results = Vec::new();
    for (rank, worker) in workers.iter_mut().enumerate() {
        let data = worker
            .marker("ANYINFER_HEAD_RESULT ")
            .with_context(|| format!("execute rank {rank}"))?;
        let result: RankResult = serde_json::from_str(&data)?;
        let head = &heads[rank];
        ensure!(
            (head.vocab_start..head.vocab_end).contains(&result.token_id)
                && result.logit.is_finite()
                && !result.soc.is_empty()
                && [
                    result.input_h2d_us,
                    result.output_d2h_us,
                    result.local_checked_argmax_us
                ]
                .iter()
                .all(|&t| t.is_finite() && t >= 0.0)
                && result.device_us.len() == opt.samples as usize
                && result.device_us.iter().all(|t| t.is_finite() && *t > 0.0),
            "invalid result from rank {rank}"
        );
        results.push(result);
    }
    let (token, logit) = winner(results.iter().map(|r| (r.token_id, r.logit)))?;
    let wall_us = started.elapsed().as_micros();
    let mut comparison = None;
    if let Some(gold) = &golden {
        let (expected, _) = winner(
            gold.as_chunks::<4>()
                .0
                .iter()
                .enumerate()
                .map(|(i, &b)| (i as u32, f32::from_le_bytes(b))),
        )?;
        let mut c = Comparison {
            expected_token: expected,
            token_matches: expected == token,
            ..Default::default()
        };
        for (worker, head) in workers.iter_mut().zip(&heads) {
            let shard = &gold[head.vocab_start as usize * 4..head.vocab_end as usize * 4];
            let mut actual = vec![0; shard.len()];
            worker.send(b"LOGITS\n")?;
            worker.output.read_exact(&mut actual)?;
            compare(shard, &actual, opt.atol, opt.rtol, &mut c)?;
        }
        comparison = Some(c);
    }
    for worker in &mut workers {
        worker.finish()?;
    }
    let passed = comparison
        .as_ref()
        .is_none_or(|c| c.failed_elements == 0 && c.token_matches);
    let rank_reports: Vec<_> = heads.iter().zip(results).enumerate().map(|(i, (head, measured))| json!({
        "device": opt.devices[i], "directory": opt.rank_dir[i], "manifest":head, "measured":measured,
        "manifest_sha256":digests[i].0,"om_sha256":digests[i].1,
    })).collect();
    Ok(json!({
        "format_version":1,"stage":"deepseek_v4_last_token_head","status":if passed {"ok"} else {"failed_accuracy"},
        "residual_sha256":residual_hash,"golden_logits_sha256":golden.as_ref().map(|g|hex::encode(Sha256::digest(g))),
        "warmup":opt.warmup,"samples":opt.samples,"ranks":rank_reports,
        "token_id":token,"logit":logit,"host_timed_window_us":wall_us,
        "golden_comparison":comparison,"atol":opt.atol,"rtol":opt.rtol,
    }))
}

pub fn run(options: Options) -> Result<()> {
    if let Some(dir) = options
        .output
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
    {
        fs::create_dir_all(dir)?;
    }
    // Reserve a fresh report so concurrent runs cannot erase previous measurements.
    let mut output =
        File::create_new(&options.output).with_context(|| options.output.display().to_string())?;
    let result = std::env::current_exe()
        .context("resolve worker directory")
        .and_then(|exe| evaluate(&options, &exe.with_file_name("anyinfer-head-rank")));
    let report = match &result {
        Ok(value) => value.clone(),
        Err(error) => {
            json!({"format_version":1,"stage":"deepseek_v4_last_token_head","status":"error","error":format!("{error:#}")})
        }
    };
    serde_json::to_writer_pretty(&mut output, &report)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    result?;
    ensure!(
        report["status"] == "ok",
        "head precision check failed, see {}",
        options.output.display()
    );
    println!("{}", options.output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn shard(rank: u32) -> Head {
        Head {
            format_version: 1,
            model_type: "deepseek_v4".into(),
            rank,
            world_size: 2,
            vocab_start: rank * 4,
            vocab_end: (rank + 1) * 4,
            vocab_size: 8,
            hidden_size: 8,
            hc_mult: 4,
            graph: "head.onnx".into(),
        }
    }
    #[test]
    fn enforces_complete_rank_coverage_and_unique_devices() {
        assert!(validate_heads(&[shard(0), shard(1)], &[4, 6]).is_ok());
        assert!(validate_heads(&[shard(1), shard(0)], &[4, 6]).is_err());
        assert!(validate_heads(&[shard(0), shard(1)], &[4, 4]).is_err());
        let mut wrong = shard(1);
        wrong.vocab_start = 5;
        assert!(validate_heads(&[shard(0), wrong], &[4, 6]).is_err());
    }
    #[test]
    fn greedy_ties_and_accuracy_match_the_full_vocabulary() {
        assert_eq!(winner([(4, 1.0), (5, -1.0), (2, 1.0)]).unwrap(), (2, 1.0));
        assert!(winner([(0, f32::NAN)]).is_err());
        let mut c = Comparison::default();
        let gold: Vec<u8> = [1.0f32, 0.0, -2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let got: Vec<u8> = [1.01f32, 0.5, -2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        compare(&gold, &got, 0.0, 0.02, &mut c).unwrap();
        assert_eq!(
            (c.elements, c.failed_elements, c.max_abs_error),
            (3, 1, 0.5)
        );
    }

    #[cfg(unix)]
    #[test]
    fn two_child_processes_rendezvous_compare_logits_and_abort_on_rank_failure() -> Result<()> {
        use std::os::unix::fs::PermissionsExt;
        use std::time::{SystemTime, UNIX_EPOCH};
        let tmp = std::env::temp_dir().join(format!(
            "anyinfer-head-test-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        ));
        fs::create_dir(&tmp)?;
        let worker = tmp.join("ipc-test-worker");
        // Exercise the coordinator protocol without requiring the device runtime in host unit tests.
        fs::write(
            &worker,
            r#"#!/bin/sh
set -eu
test ! -e "$8.fail"
echo ANYINFER_HEAD_READY
read -r command
test "$command" = RUN
printf 'ANYINFER_HEAD_RESULT {"token_id":%s,"logit":3,"soc":"test","device_us":[1],"input_h2d_us":1,"output_d2h_us":1,"local_checked_argmax_us":1}\n' "$(($4 + 1))"
if read -r command; then
  test "$command" = LOGITS
  cat "$8.logits"
fi
"#,
        )?;
        fs::set_permissions(&worker, fs::Permissions::from_mode(0o700))?;
        let dirs = [tmp.join("0"), tmp.join("1")];
        let logits: Vec<u8> = [0.0f32, 3.0, 2.0, 1.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        for (rank, dir) in dirs.iter().enumerate() {
            fs::create_dir(dir)?;
            fs::write(
                dir.join("head.json"),
                serde_json::to_vec(&shard(rank as u32))?,
            )?;
            fs::write(dir.join("head.om"), b"test metadata")?;
            fs::write(dir.join("head.om.logits"), &logits)?;
        }
        let residual = tmp.join("residual");
        fs::write(&residual, [0; 64])?;
        let golden = tmp.join("golden");
        fs::write(&golden, logits.repeat(2))?;
        let mut opt = Options {
            rank_dir: dirs.to_vec(),
            devices: vec![0, 1],
            residual,
            golden_logits: Some(golden.clone()),
            atol: 0.0,
            rtol: 0.0,
            warmup: 0,
            samples: 1,
            output: tmp.join("report"),
        };
        let report = evaluate(&opt, &worker)?;
        assert_eq!(report["status"], "ok");
        assert_eq!(report["token_id"], 1);
        assert_eq!(report["golden_comparison"]["elements"], 8);
        fs::write(golden, [0; 32])?;
        assert_eq!(evaluate(&opt, &worker)?["status"], "failed_accuracy");
        opt.golden_logits = None;
        assert!(evaluate(&opt, &worker)?["golden_comparison"].is_null());
        fs::write(dirs[1].join("head.om.fail"), [])?;
        assert!(evaluate(&opt, &worker).is_err());
        fs::remove_dir_all(tmp)?;
        Ok(())
    }
}
