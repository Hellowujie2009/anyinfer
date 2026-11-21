use anyhow::{Context, Result, ensure};
use clap::{Args as ClapArgs, ValueEnum};
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Profile {
    Both,
    Pipe,
    Memory,
    Off,
}

#[derive(ClapArgs)]
pub struct Args {
    /// Pick individual IDs and inclusive ranges, e.g. 1,5-13,27.
    #[arg(long, default_value = "all")]
    range: String,
    #[arg(long, default_value_t = 0)]
    device: u32,
    /// Directory for this shard. It must be new to keep measurements from different runs separate.
    #[arg(long)]
    output: PathBuf,
    /// Kernel executable compiled for the actual chip on this host.
    #[arg(long)]
    worker: PathBuf,
    #[arg(long, value_enum, default_value = "both")]
    profile: Profile,
    #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u32).range(1..))]
    warmup: u32,
    #[arg(long, default_value_t = 7, value_parser = clap::value_parser!(u32).range(1..))]
    samples: u32,
    #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u32).range(1..))]
    iterations: u32,
    /// Override the hidden-dimension tile; omit for the kernel's shape-dependent default.
    #[arg(long, value_parser = parse_tile)]
    tile: Option<u32>,
}

fn parse_tile(raw: &str) -> std::result::Result<u32, String> {
    match raw.parse::<u32>() {
        Ok(n @ (128 | 256 | 512 | 1024)) => Ok(n),
        _ => Err("tile must be 128, 256, 512 or 1024".into()),
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct Case {
    id: usize,
    model: &'static str,
    operator: &'static str,
    tokens: u32,
    hidden: u32,
}

// Keep the ordering stable across releases. Non-power-of-two shapes exercise the
// distribution of work over cores on either side of common batching boundaries.
const TOKEN_COUNTS: [u32; 24] = [
    1, 2, 4, 7, 8, 15, 16, 31, 32, 48, 63, 64, 95, 96, 127, 128, 192, 255, 256, 384, 511, 512,
    1024, 2048,
];

pub fn select(range: &str) -> Result<Vec<Case>> {
    let mut cases = Vec::new();
    for (model, hidden) in [("V4-Flash", 4096), ("V4-Pro", 7168), ("V4.1-Flash", 5120)] {
        for tokens in TOKEN_COUNTS {
            cases.push(Case {
                id: cases.len() + 1,
                model,
                operator: "hc_post",
                tokens,
                hidden,
            });
        }
    }
    if range == "all" {
        return Ok(cases);
    }
    let mut wanted = BTreeSet::new();
    for part in range.split(',').map(str::trim) {
        let (lo, hi) = part.split_once('-').unwrap_or((part, part));
        let lo: usize = lo.parse().context("invalid first case ID")?;
        let hi: usize = hi.parse().context("invalid final case ID")?;
        ensure!(
            lo > 0 && lo <= hi && hi <= cases.len(),
            "unknown or reversed case range {part}"
        );
        wanted.extend(lo..=hi);
    }
    Ok(cases
        .into_iter()
        .filter(|c| wanted.contains(&c.id))
        .collect())
}

fn save(path: &Path, data: &Value) -> Result<()> {
    let temp = path.with_extension("part");
    fs::write(&temp, serde_json::to_vec_pretty(data)?)?;
    fs::rename(temp, path)?;
    Ok(())
}

fn unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn sample_devices(path: &Path, stop: Receiver<()>) -> Result<(u64, u64)> {
    let mut log = File::create(path)?;
    let mut successful = 0;
    let mut total = 0;
    loop {
        let timestamp = unix_ms();
        let reading = Command::new("timeout")
            .args(["10s", "npu-smi", "info"])
            .output();
        total += 1;
        if reading
            .as_ref()
            .is_ok_and(|r| r.status.success() && !r.stdout.is_empty())
        {
            successful += 1;
        }
        let row = match reading {
            Ok(r) => json!({"unix_ms":timestamp,"exit":r.status.code(),
                "stdout":String::from_utf8_lossy(&r.stdout),"stderr":String::from_utf8_lossy(&r.stderr)}),
            Err(e) => json!({"unix_ms":timestamp,"error":e.to_string()}),
        };
        writeln!(log, "{row}")?;
        log.flush()?;
        match stop.recv_timeout(Duration::from_secs(2)) {
            Err(mpsc::RecvTimeoutError::Timeout) => (),
            _ => return Ok((successful, total)),
        }
    }
}

fn execute(cmd: &mut Command, folder: &Path) -> Result<()> {
    let out = File::create(folder.join("stdout.log"))?;
    let err = File::create(folder.join("stderr.log"))?;
    let status = cmd
        .stdout(Stdio::from(out))
        .stderr(Stdio::from(err))
        .status()
        .with_context(|| format!("launch {:?}", cmd.get_program()))?;
    ensure!(
        status.success(),
        "{:?} exited {status}, see {}",
        cmd.get_program(),
        folder.display()
    );
    Ok(())
}

fn configure(cmd: &mut Command, args: &Args, case: &Case, output: &Path, replay: bool) {
    cmd.args([
        "--device",
        &args.device.to_string(),
        "--tokens",
        &case.tokens.to_string(),
        "--hidden",
        &case.hidden.to_string(),
        "--warmup",
        &if replay { 1 } else { args.warmup }.to_string(),
        "--samples",
        &if replay { 1 } else { args.samples }.to_string(),
        "--iterations",
        &if replay { 1 } else { args.iterations }.to_string(),
        "--output",
    ])
    .arg(output);
    if let Some(tile) = args.tile {
        cmd.arg("--tile").arg(tile.to_string());
    }
}

fn checked_result(path: &Path, case: &Case, device: u32) -> Result<Value> {
    let r: Value = serde_json::from_reader(File::open(path).context("kernel produced no result")?)?;
    let elements = u64::from(case.tokens) * u64::from(case.hidden) * 4;
    ensure!(
        r["status"] == "passed"
            && r["operator"] == "hc_post"
            && r["device"] == device
            && r["tokens"] == case.tokens
            && r["hidden"] == case.hidden
            && r["compared_elements"] == elements
            && r["mismatches"] == 0
            && r["median_us"].as_f64().is_some_and(|v| v > 0.0),
        "kernel shape, element check or timing failed, inspect {}",
        path.display()
    );
    Ok(r)
}

fn collect_task_rows(dir: &Path, rows: &mut Vec<Value>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            collect_task_rows(&entry.path(), rows)?;
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
        if !name.starts_with("op_summary") || !name.ends_with(".csv") {
            continue;
        }
        let mut csv = csv::ReaderBuilder::new()
            .flexible(true)
            .from_path(entry.path())?;
        let headers = csv.headers()?.clone();
        let names: Vec<usize> = headers
            .iter()
            .enumerate()
            .filter_map(|(i, h)| {
                let h = h.trim().to_ascii_lowercase().replace([' ', '_'], "");
                matches!(h.as_str(), "opname" | "optype" | "kernelname" | "taskname").then_some(i)
            })
            .collect();
        for record in csv.records() {
            let record = record?;
            if names.iter().filter_map(|&i| record.get(i)).any(|field| {
                field
                    .to_ascii_lowercase()
                    .replace('_', "")
                    .contains("hcpost")
            }) {
                let columns: serde_json::Map<String, Value> = headers
                    .iter()
                    .zip(&record)
                    .map(|(k, v)| (k.trim().to_owned(), json!(v.trim())))
                    .collect();
                rows.push(Value::Object(columns));
            }
        }
    }
    Ok(())
}

fn numeric_counters(rows: &[Value], group: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for row in rows.iter().filter_map(Value::as_object) {
        for (name, value) in row {
            let key: String = name
                .chars()
                .filter(char::is_ascii_alphanumeric)
                .map(|c| c.to_ascii_lowercase())
                .collect();
            let is_counter = match group {
                "Memory" => [
                    "bandwidth",
                    "readbw",
                    "writebw",
                    "readbytes",
                    "writebytes",
                    "activebw",
                ]
                .iter()
                .any(|s| key.contains(s)),
                "PipeUtilization" => {
                    ["vec", "mte", "scalar", "mac", "cube", "fixp"]
                        .iter()
                        .any(|s| key.contains(s))
                        && ["ratio", "cycle", "time", "utilization"]
                            .iter()
                            .any(|s| key.contains(s))
                }
                _ => false,
            };
            if is_counter
                && value
                    .as_str()
                    .and_then(|s| s.trim_end_matches('%').trim().parse::<f64>().ok())
                    .is_some_and(f64::is_finite)
            {
                names.insert(name.clone());
            }
        }
    }
    names
}

fn replay(args: &Args, case: &Case, worker: &Path, root: &Path, group: &str) -> Result<Value> {
    let dir = root.join(group);
    fs::create_dir(&dir)?;
    let trace = dir.join("trace");
    fs::create_dir(&trace)?;
    let output = dir.join("check.json");
    let mut cmd = Command::new("msprof");
    cmd.arg(format!("--output={}", trace.display()))
        .args(["--ai-core=on", "--task-time=on"])
        .arg(format!("--aic-metrics={group}"))
        .arg(worker);
    configure(&mut cmd, args, case, &output, true);
    execute(&mut cmd, &dir)?;
    checked_result(&output, case, args.device)?;
    let mut rows = Vec::new();
    collect_task_rows(&trace, &mut rows)?;
    if rows.is_empty() {
        // Some CANN installations keep only the raw PROF directory on capture.
        let exports = dir.join("export");
        fs::create_dir(&exports)?;
        let mut cmd = Command::new("msprof");
        cmd.args(["--export=on", "--type=text"])
            .arg(format!("--output={}", trace.display()));
        execute(&mut cmd, &exports)?;
        collect_task_rows(&trace, &mut rows)?;
    }
    ensure!(
        !rows.is_empty(),
        "no hc_post rows in msprof op_summary CSV under {}",
        trace.display()
    );
    let path = dir.join("kernel-counters.json");
    let numeric = numeric_counters(&rows, group);
    save(
        &path,
        &json!({"counter_group":group,"tasks":rows,"numeric_counter_columns":numeric}),
    )?;
    ensure!(
        !numeric.is_empty(),
        "no numeric {group} counters on hc_post tasks in {}",
        path.display()
    );
    Ok(
        json!({"group":group,"matched_tasks":rows.len(),"numeric_counter_columns":numeric,
        "path":format!("{group}/kernel-counters.json")}),
    )
}

fn report_result(result: Result<Value>) -> Value {
    match result {
        Ok(value) => json!({"ok":true,"result":value}),
        Err(error) => json!({"ok":false,"error":format!("{error:#}")}),
    }
}

pub fn run(args: &Args) -> Result<()> {
    let cases = select(&args.range)?;
    fs::create_dir(&args.output).context("create new benchmark shard directory")?;
    let root = args.output.canonicalize()?;
    let groups: &[&str] = match args.profile {
        Profile::Both => &["PipeUtilization", "Memory"],
        Profile::Pipe => &["PipeUtilization"],
        Profile::Memory => &["Memory"],
        Profile::Off => &[],
    };
    let worker = args.worker.canonicalize().context("find kernel worker")?;
    let telemetry = root.join("npu-smi.jsonl");
    let (tx, rx) = mpsc::channel();
    let sampler = thread::spawn(move || sample_devices(&telemetry, rx));
    let mut results = Vec::new();
    let started = unix_ms();
    let feedback = |results: &[Value], finished: bool| {
        json!({
            "schema":1,"workload":"native_hc_post_microbenchmark","device":args.device,"selected":cases,
            "started_unix_ms":started,"last_update_unix_ms":unix_ms(),"finished":finished,"results":results,
            "profile":args.profile,"warmup":args.warmup,"samples":args.samples,"iterations":args.iterations,
            "tile_override":args.tile
        })
    };
    save(&root.join("feedback.json"), &feedback(&results, false))?;
    for case in &cases {
        let dir = root.join(format!("case-{:03}", case.id));
        fs::create_dir(&dir)?;
        let output = dir.join("measurement.json");
        let mut cmd = Command::new(&worker);
        configure(&mut cmd, args, case, &output, false);
        let measurement = report_result(
            execute(&mut cmd, &dir).and_then(|()| checked_result(&output, case, args.device)),
        );
        let mut counters = Vec::new();
        if measurement["ok"] == true {
            for group in groups {
                counters.push(json!({"group":group,"capture":report_result(replay(args, case, &worker, &dir, group))}));
            }
        }
        let ok = measurement["ok"] == true && counters.iter().all(|c| c["capture"]["ok"] == true);
        results.push(
            json!({"case":case,"directory":dir.file_name().unwrap().to_string_lossy(),
            "measurement":measurement,"profiles":counters,"ok":ok}),
        );
        save(&root.join("feedback.json"), &feedback(&results, false))?;
        eprintln!(
            "case {} {} (tokens={}, hidden={})",
            case.id,
            if ok { "passed" } else { "failed" },
            case.tokens,
            case.hidden
        );
    }
    drop(tx);
    let telemetry = match sampler.join() {
        Ok(Ok((success, total))) => {
            json!({"ok":success>0,"successful_polls":success,"polls":total})
        }
        Ok(Err(e)) => json!({"ok":false,"error":e.to_string()}),
        Err(_) => json!({"ok":false,"error":"telemetry thread stopped"}),
    };
    let mut completed = feedback(&results, true);
    completed["telemetry"] = telemetry;
    save(&root.join("feedback.json"), &completed)?;
    ensure!(
        completed["telemetry"]["ok"] == true,
        "NPU telemetry failed, see npu-smi.jsonl and feedback.json"
    );
    ensure!(
        results.iter().all(|c| c["ok"] == true),
        "kernel verification or requested counter collection failed"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_shape_ids_and_shards() {
        let c = select("1,24-26,26,72").unwrap();
        assert_eq!(
            c.iter().map(|c| c.id).collect::<Vec<_>>(),
            [1, 24, 25, 26, 72]
        );
        assert_eq!(
            (c[1].hidden, c[1].tokens, c[2].hidden, c[4].hidden),
            (4096, 2048, 7168, 5120)
        );
        for bad in ["2-1", "73", "0", ""] {
            assert!(select(bad).is_err());
        }
    }

    #[test]
    fn timing_alone_cannot_pass_counter_validation() {
        let rows = [
            json!({"Task Duration(us)":"3.2","aiv_vec_ratio":"N/A","aiv_mte2_time(us)":"0.07",
            "aiv_read_main_memory_bandwidth(GB/s)":"0","Memory":"on"}),
        ];
        assert_eq!(
            numeric_counters(&rows, "PipeUtilization")
                .into_iter()
                .collect::<Vec<_>>(),
            ["aiv_mte2_time(us)"]
        );
        assert_eq!(numeric_counters(&rows, "Memory").len(), 1);
        assert!(numeric_counters(&[json!({"aiv_vec_ratio":"NaN"})], "PipeUtilization").is_empty());
    }
}
