use anyhow::{Context, Result, bail, ensure};
use clap::Args;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::runtime::MODEL_ALIAS;

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum AicMetrics {
    Both,
    Pipe,
    Memory,
}

impl AicMetrics {
    fn groups(self) -> &'static [&'static str] {
        match self {
            Self::Both => &["PipeUtilization", "Memory"],
            Self::Pipe => &["PipeUtilization"],
            Self::Memory => &["Memory"],
        }
    }
}

#[derive(Args, Debug)]
pub struct BenchArgs {
    /// Load a workload suite from disk (default: bundled benches/cases.json).
    #[arg(long)]
    pub cases: Option<PathBuf>,
    /// Inclusive IDs or workload families, for example @survey, @prefix, 1-3,5 or all.
    #[arg(long, default_value = "all")]
    pub range: String,
    /// Parent of the unique result directory created for this invocation.
    #[arg(long, default_value = "results")]
    pub output: PathBuf,
    /// Wall-clock deadline for each individual streaming request.
    #[arg(long, default_value_t = 600, value_parser = clap::value_parser!(u64).range(1..))]
    pub request_timeout_secs: u64,
    /// Profile a full replay of each case (default: a short replay after the baseline).
    #[arg(long)]
    pub profile: bool,
    /// AI Core counter groups. Both runs independent pipeline and memory replays after timing.
    #[arg(long, value_enum, default_value = "both")]
    aic_metrics: AicMetrics,
    /// Check greedy output IDs against a trusted reference before timing the workloads.
    #[arg(long)]
    pub golden: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: u32,
    name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    input_tokens: Vec<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    text_prompts: Vec<String>,
    output_tokens: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    output_pattern: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    shared_prefix: Option<SharedPrefix>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    arrivals: Option<Arrivals>,
    concurrency: usize,
    requests: u32,
    warmup: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SharedPrefix {
    tokens: usize,
    groups: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Arrivals {
    burst: u32,
    interval_ms: u32,
}

impl Case {
    fn output_at(&self, index: usize) -> u32 {
        if self.output_pattern.is_empty() {
            self.output_tokens
        } else {
            self.output_pattern[index % self.output_pattern.len()]
        }
    }

    fn arrival_offset(&self, index: u32) -> Option<Duration> {
        self.arrivals
            .as_ref()
            .map(|a| Duration::from_millis(u64::from(index / a.burst) * u64::from(a.interval_ms)))
    }
}

fn load_suite(path: Option<&Path>) -> Result<(Vec<Case>, String)> {
    let raw = match path {
        Some(path) => fs::read(path).with_context(|| format!("read {}", path.display()))?,
        None => include_bytes!("../benches/cases.json").to_vec(),
    };
    let cases = serde_json::from_slice(&raw).context("invalid benchmark suite")?;
    Ok((cases, hex::encode(Sha256::digest(&raw))))
}

pub fn list_cases(path: Option<&Path>, range: &str) -> Result<()> {
    let (cases, _) = load_suite(path)?;
    let selected = cases_in_range(&cases, range)?;
    let mut out = std::io::stdout().lock();
    writeln!(
        out,
        "id\tname\tinput_tokens_or_text_prompts\toutput_tokens\tconcurrency\trequests\twarmup\tprefix_tokens/groups\tburst/interval_ms"
    )?;
    for case in &selected {
        let input = if case.input_tokens.is_empty() {
            format!("{} text prompts", case.text_prompts.len())
        } else {
            case.input_tokens
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        };
        writeln!(
            out,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            case.id,
            case.name.replace(['\t', '\r', '\n'], " "),
            input,
            if case.output_pattern.is_empty() {
                case.output_tokens.to_string()
            } else {
                case.output_pattern
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(",")
            },
            case.concurrency,
            case.requests,
            case.warmup,
            case.shared_prefix
                .as_ref()
                .map(|p| format!("{}/{}", p.tokens, p.groups))
                .unwrap_or_default(),
            case.arrivals
                .as_ref()
                .map(|a| format!("{}/{}", a.burst, a.interval_ms))
                .unwrap_or_default(),
        )?;
    }
    eprintln!("{} cases selected", selected.len());
    Ok(())
}

impl BenchArgs {
    pub fn validate(&self, max_model_len: usize) -> Result<()> {
        if let Some(path) = &self.golden {
            crate::accuracy::validate_golden(path, max_model_len)?;
        }
        let (cases, _) = load_suite(self.cases.as_deref())?;
        for case in cases_in_range(&cases, &self.range)? {
            let required = case
                .input_tokens
                .iter()
                .enumerate()
                .map(|(i, n)| *n as u64 + u64::from(case.output_at(i)))
                .max()
                .unwrap_or(1 + u64::from(case.output_tokens));
            ensure!(
                required <= max_model_len as u64,
                "case {} requires {required} tokens; --max-model-len is {max_model_len}",
                case.id
            );
        }
        Ok(())
    }
}

const SURVEY_CASES: &[&str] = &[
    "smoke",
    "prefill-throughput",
    "coding-prompts",
    "decode-steady-in128-out1024-c8",
    "mixed-interactive-c16",
    "prefix-unique-c8",
    "prefix-one-c8",
    "arrival-burst8-r32",
];

fn cases_in_range<'a>(cases: &'a [Case], range: &str) -> Result<Vec<&'a Case>> {
    let mut ids = BTreeSet::new();
    ensure!(!cases.is_empty(), "empty case suite");
    for c in cases {
        ensure!(
            c.id > 0 && ids.insert(c.id),
            "invalid or duplicate case id {}",
            c.id
        );
        ensure!(
            c.input_tokens.is_empty() != c.text_prompts.is_empty(),
            "case {} needs exactly one input source",
            c.id
        );
        ensure!(
            c.text_prompts.iter().all(|text| !text.trim().is_empty())
                && c.input_tokens.iter().all(|n| *n > 0 && *n <= 1_048_576)
                && c.output_tokens > 0
                && c.concurrency > 0
                && c.requests > 0,
            "invalid case {}",
            c.id
        );
        ensure!(
            c.output_pattern.is_empty()
                || (c.output_pattern.len() == c.input_tokens.len()
                    && !c.output_pattern.contains(&0)
                    && c.output_pattern.iter().max() == Some(&c.output_tokens)),
            "case {} output_pattern must pair with input_tokens and have maximum output_tokens",
            c.id
        );
        if let Some(prefix) = &c.shared_prefix {
            ensure!(
                !c.input_tokens.is_empty()
                    && prefix.groups > 0
                    && c.input_tokens.iter().all(|n| *n > prefix.tokens),
                "case {} shared_prefix requires token input, a positive group count and a private suffix",
                c.id
            );
        }
        if let Some(arrivals) = &c.arrivals {
            ensure!(
                arrivals.burst > 0 && arrivals.interval_ms > 0,
                "case {} invalid arrival schedule",
                c.id
            );
        }
    }
    if range == "all" {
        return Ok(cases.iter().collect());
    }
    let mut selected = BTreeSet::new();
    for part in range.split(',').map(str::trim) {
        if let Some(family) = part.strip_prefix('@') {
            ensure!(!family.is_empty(), "empty workload family");
            let mut found = false;
            for case in cases {
                let matches = if family == "survey" {
                    SURVEY_CASES.contains(&case.name.as_str())
                } else {
                    case.name == family
                        || case
                            .name
                            .strip_prefix(family)
                            .is_some_and(|s| s.starts_with('-'))
                };
                if matches {
                    found = true;
                    selected.insert(case.id);
                }
            }
            ensure!(found, "no cases match {part} in this suite");
            continue;
        }
        let (start, end) = part.split_once('-').unwrap_or((part, part));
        let start = start
            .trim()
            .parse::<u32>()
            .context("expected numeric case ID")?;
        let end = end
            .trim()
            .parse::<u32>()
            .context("expected numeric case ID")?;
        ensure!(start <= end && start > 0, "invalid range {part}");
        let matched = ids.range(start..=end).count() as u64;
        ensure!(
            matched == u64::from(end - start) + 1,
            "range {part} includes unknown case IDs"
        );
        selected.extend(ids.range(start..=end).copied());
    }
    Ok(cases.iter().filter(|c| selected.contains(&c.id)).collect())
}

// SplitMix64 provides reproducible benchmark inputs independent of host RNG/version.
fn prompt(seed: u64, length: usize) -> Vec<u32> {
    let mut state = seed;
    (0..length)
        .map(|_| {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            256 + ((z ^ (z >> 31)) % 28_000) as u32
        })
        .collect()
}

fn request_seed(case_id: u32, phase: &str, index: u32) -> u64 {
    ((u64::from(case_id) << 32) | u64::from(index))
        ^ match phase {
            "warmup" => 0x8000_0000_0000_0000,
            "operator_replay" => 0x4000_0000_0000_0000,
            "operator_memory" => 0x2000_0000_0000_0000,
            _ => 0,
        }
}

fn token_input(case: &Case, phase: &str, index: u32, length: usize) -> Vec<u32> {
    let seed = request_seed(case.id, phase, index);
    let Some(prefix) = case.shared_prefix.as_ref().filter(|p| p.tokens > 0) else {
        return prompt(seed, length);
    };
    // Namespace the reusable block separately from the suffix, scoped to case and phase.
    let mut hash = Sha256::new();
    hash.update(b"anyinfer-shared-prefix-v1");
    hash.update(case.id.to_be_bytes());
    hash.update(phase.as_bytes());
    hash.update((index % prefix.groups).to_be_bytes());
    let digest = hash.finalize();
    let mut tokens = prompt(
        u64::from_be_bytes(digest[..8].try_into().unwrap()),
        prefix.tokens,
    );
    tokens.extend(prompt(seed, length - prefix.tokens));
    tokens
}

fn text_prompt(seed: u64, template: &str) -> String {
    let nonce = hex::encode(Sha256::digest(seed.to_be_bytes()));
    format!("Task {}\n\n{template}", &nonce[..16])
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("write {}", path.display()))
}

fn line(writer: &mut BufWriter<File>, value: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

#[derive(Debug, Serialize)]
struct Arrival {
    elapsed_ms: f64,
    tokens: u32,
}

#[derive(Debug, Serialize)]
struct Observation {
    index: u32,
    phase: &'static str,
    // String to preserve all 64 bits when read by JavaScript clients.
    prompt_seed: String,
    expected_prompt_tokens: Option<usize>,
    requested_output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    shared_prefix_group: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    client_issue_lag_ms: Option<f64>,
    text_prompt_index: Option<usize>,
    prompt_text_sha256: Option<String>,
    start_unix_ms: u128,
    request_id: Option<String>,
    prompt_tokens: Option<u32>,
    generated_tokens: u32,
    arrivals: Vec<Arrival>,
    finish_reason: Option<String>,
    server_metrics: Option<Value>,
    output_text_sha256: Option<String>,
    latency_ms: f64,
    error: Option<String>,
}

#[derive(Deserialize)]
struct Choice {
    #[serde(default)]
    text: String,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Deserialize)]
struct Event {
    id: Option<String>,
    choices: Option<Vec<Choice>>,
    usage: Option<Usage>,
    metrics: Option<Value>,
    error: Option<Value>,
}

fn record_event(
    data: &[u8],
    obs: &mut Observation,
    hash: &mut Sha256,
    elapsed_ms: f64,
    final_usage: &mut bool,
) -> Result<()> {
    let event: Event = serde_json::from_slice(data).context("malformed streaming event")?;
    if let Some(error) = event.error {
        bail!("backend generation error: {error}");
    }
    if obs.request_id.is_none() {
        obs.request_id = event.id;
    }
    let choices = event.choices.context("missing choices")?;
    let usage = event
        .usage
        .context("stream omitted continuous usage statistics")?;
    ensure!(
        usage.completion_tokens >= obs.generated_tokens,
        "generated token counter moved backwards"
    );
    let delta = usage.completion_tokens - obs.generated_tokens;
    if choices.is_empty() {
        ensure!(delta == 0, "final usage differs from stream");
        *final_usage = true;
        obs.server_metrics = event.metrics;
    } else {
        ensure!(
            choices.len() == 1 && !*final_usage,
            "unexpected stream choices"
        );
        hash.update(choices[0].text.as_bytes());
        if let Some(reason) = &choices[0].finish_reason {
            obs.finish_reason = Some(reason.clone());
        }
        if delta > 0 {
            obs.arrivals.push(Arrival {
                elapsed_ms,
                tokens: delta,
            });
        }
    }
    obs.generated_tokens = usage.completion_tokens;
    obs.prompt_tokens = Some(usage.prompt_tokens);
    Ok(())
}

async fn consume(
    client: &Client,
    url: &str,
    body: Vec<u8>,
    start: Instant,
    obs: &mut Observation,
    expected_output: u32,
) -> Result<()> {
    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .body(body)
        .send()
        .await?;
    if !response.status().is_success() {
        let status = response.status();
        let detail = response.text().await.unwrap_or_default();
        bail!(
            "HTTP {status}: {}",
            detail.chars().take(2048).collect::<String>()
        );
    }
    let mut stream = response.bytes_stream();
    let mut pending = Vec::new();
    let mut hash = Sha256::new();
    let mut final_usage = false;
    while let Some(bytes) = stream.next().await {
        let bytes = bytes?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        pending.extend_from_slice(&bytes);
        let mut used = 0;
        while let Some(pos) = pending[used..].iter().position(|b| *b == b'\n') {
            let line = &pending[used..used + pos];
            used += pos + 1;
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            let Some(data) = line.strip_prefix(b"data:") else {
                continue;
            };
            let data = data.strip_prefix(b" ").unwrap_or(data);
            if data == b"[DONE]" {
                ensure!(final_usage, "missing final usage");
                ensure!(
                    obs.finish_reason.as_deref() == Some("length"),
                    "unexpected finish reason {:?}",
                    obs.finish_reason
                );
                ensure!(
                    obs.generated_tokens == expected_output,
                    "generated {} of {expected_output} requested tokens",
                    obs.generated_tokens
                );
                let actual = obs.prompt_tokens.context("missing prompt token count")? as usize;
                ensure!(
                    actual > 0
                        && obs
                            .expected_prompt_tokens
                            .is_none_or(|expected| expected == actual),
                    "reported input token count differs from workload"
                );
                obs.output_text_sha256 = Some(hex::encode(hash.finalize()));
                return Ok(());
            }
            record_event(data, obs, &mut hash, elapsed_ms, &mut final_usage)?;
        }
        pending.drain(..used);
        ensure!(
            pending.len() <= 16 * 1024 * 1024,
            "oversized incomplete SSE line"
        );
    }
    bail!("stream ended before [DONE]")
}

async fn request(
    client: &Client,
    url: &str,
    case: &Case,
    phase: &'static str,
    index: u32,
    deadline: Option<tokio::time::Instant>,
) -> Observation {
    let seed = request_seed(case.id, phase, index);
    let output = case.output_at(index as usize);
    #[derive(Serialize)]
    #[serde(untagged)]
    enum Input {
        Tokens(Vec<u32>),
        Text(String),
    }
    let (input, expected_prompt_tokens, text_prompt_index, prompt_text_sha256) =
        if case.text_prompts.is_empty() {
            let length = case.input_tokens[(index as usize) % case.input_tokens.len()];
            (
                Input::Tokens(token_input(case, phase, index, length)),
                Some(length),
                None,
                None,
            )
        } else {
            let i = (index as usize) % case.text_prompts.len();
            let text = text_prompt(seed, &case.text_prompts[i]);
            let hash = hex::encode(Sha256::digest(text.as_bytes()));
            (Input::Text(text), None, Some(i), Some(hash))
        };
    #[derive(Serialize)]
    struct Payload {
        model: &'static str,
        prompt: Input,
        max_tokens: u32,
        temperature: f32,
        ignore_eos: bool,
        stream: bool,
        stream_options: Value,
    }
    let body = serde_json::to_vec(&Payload {
        model: MODEL_ALIAS,
        prompt: input,
        max_tokens: output,
        temperature: 0.0,
        ignore_eos: true,
        stream: true,
        stream_options: json!({"include_usage": true, "continuous_usage_stats": true}),
    })
    .expect("JSON constructed from primitives");
    if let Some(at) = deadline {
        tokio::time::sleep_until(at).await;
    }
    let mut obs = Observation {
        index,
        phase,
        prompt_seed: seed.to_string(),
        expected_prompt_tokens,
        requested_output_tokens: output,
        shared_prefix_group: case
            .shared_prefix
            .as_ref()
            .filter(|p| p.tokens > 0)
            .map(|p| index % p.groups),
        client_issue_lag_ms: deadline.map(|at| at.elapsed().as_secs_f64() * 1000.0),
        text_prompt_index,
        prompt_text_sha256,
        start_unix_ms: now_ms(),
        request_id: None,
        prompt_tokens: None,
        generated_tokens: 0,
        arrivals: Vec::new(),
        finish_reason: None,
        server_metrics: None,
        output_text_sha256: None,
        latency_ms: 0.0,
        error: None,
    };
    let start = Instant::now();
    let result = consume(client, url, body, start, &mut obs, output).await;
    obs.latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    obs.error = result.err().map(|e| format!("{e:#}"));
    obs
}

#[derive(Default)]
struct Stats {
    failed: usize,
    input: u64,
    generated: u64,
    latency: Vec<f64>,
    ttft: Vec<f64>,
    chunk_gaps: Vec<f64>,
    tpot: Vec<f64>,
    issue_lag: Vec<f64>,
    scheduled_e2e: Vec<f64>,
    by_shape: Option<BTreeMap<(u32, u32), ShapeTimes>>,
}

#[derive(Default)]
struct ShapeTimes {
    latency: Vec<f64>,
    ttft: Vec<f64>,
    tpot: Vec<f64>,
}

fn quantiles(values: &mut [f64]) -> Value {
    values.sort_unstable_by(f64::total_cmp);
    if values.is_empty() {
        return Value::Null;
    }
    let percentile = |q: usize| values[(values.len() * q).div_ceil(100) - 1];
    json!({"n":values.len(),"p50":percentile(50),"p95":percentile(95),"p99":percentile(99),"max":values.last()})
}

impl Stats {
    fn observe(&mut self, obs: &Observation) {
        if obs.error.is_some() {
            self.failed += 1;
            return;
        }
        self.input += u64::from(obs.prompt_tokens.unwrap_or(0));
        self.generated += u64::from(obs.generated_tokens);
        self.latency.push(obs.latency_ms);
        if let Some(lag) = obs.client_issue_lag_ms {
            self.issue_lag.push(lag);
            self.scheduled_e2e.push(lag + obs.latency_ms);
        }
        let mut shape = self.by_shape.as_mut().map(|groups| {
            groups
                .entry((obs.prompt_tokens.unwrap_or(0), obs.requested_output_tokens))
                .or_default()
        });
        if let Some(s) = &mut shape {
            s.latency.push(obs.latency_ms);
        }
        if let (Some(first), Some(last)) = (obs.arrivals.first(), obs.arrivals.last()) {
            self.ttft.push(first.elapsed_ms);
            if let Some(s) = &mut shape {
                s.ttft.push(first.elapsed_ms);
            }
            let tokens_after_first = obs.generated_tokens - first.tokens;
            if tokens_after_first > 0 {
                let tpot = (last.elapsed_ms - first.elapsed_ms) / f64::from(tokens_after_first);
                self.tpot.push(tpot);
                if let Some(s) = &mut shape {
                    s.tpot.push(tpot);
                }
            }
        }
        self.chunk_gaps.extend(
            obs.arrivals
                .windows(2)
                .map(|p| p[1].elapsed_ms - p[0].elapsed_ms),
        );
    }

    fn summary(&mut self, seconds: f64) -> Value {
        let mut summary = json!({
            "successful_requests":self.latency.len(), "failed_requests":self.failed,
            "wall_seconds":seconds, "prompt_tokens":self.input, "output_tokens":self.generated,
            "successful_requests_per_s":self.latency.len() as f64 / seconds,
            "output_tokens_per_s":self.generated as f64 / seconds,
            "total_tokens_per_s":(self.generated + self.input) as f64 / seconds,
            "ttft_ms":quantiles(&mut self.ttft), "e2e_ms":quantiles(&mut self.latency),
            "tpot_ms":quantiles(&mut self.tpot), "inter_chunk_gap_ms":quantiles(&mut self.chunk_gaps)
        });
        if !self.issue_lag.is_empty() {
            summary["client_issue_lag_ms"] = quantiles(&mut self.issue_lag);
            summary["scheduled_e2e_ms"] = quantiles(&mut self.scheduled_e2e);
        }
        if let Some(groups) = &mut self.by_shape {
            summary["by_request_shape"] = json!(groups.iter_mut().map(|(&(input, output), times)| json!({
                "prompt_tokens":input,"requested_output_tokens":output,"ttft_ms":quantiles(&mut times.ttft),
                "e2e_ms":quantiles(&mut times.latency),"tpot_ms":quantiles(&mut times.tpot)
            })).collect::<Vec<_>>());
        }
        summary
    }
}

async fn phase(
    client: &Client,
    url: &str,
    case: &Case,
    name: &'static str,
    count: u32,
    log: &mut BufWriter<File>,
) -> Result<Stats> {
    let mut stats = Stats {
        by_shape: (name == "measure" && case.input_tokens.len() > 1).then(BTreeMap::new),
        ..Stats::default()
    };
    let case = Arc::new(case.clone());
    let url: Arc<str> = Arc::from(url);
    let mut requests = 0..count;
    let mut active = tokio::task::JoinSet::new();
    let epoch = tokio::time::Instant::now();
    let submit = |active: &mut tokio::task::JoinSet<_>, index| {
        let case = case.clone();
        let url = url.clone();
        let client = client.clone();
        let deadline = case.arrival_offset(index).map(|d| epoch + d);
        active.spawn(async move { request(&client, &url, &case, name, index, deadline).await });
    };
    for index in requests.by_ref().take(case.concurrency) {
        submit(&mut active, index);
    }
    while let Some(completed) = active.join_next().await {
        let obs = completed.context("benchmark request task failed")?;
        if let Some(index) = requests.next() {
            submit(&mut active, index);
        }
        stats.observe(&obs);
        line(log, &obs)?;
    }
    Ok(stats)
}

async fn metrics(client: &Client, url: &str) -> Result<String> {
    let body = client
        .get(url)
        .timeout(Duration::from_secs(5))
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    ensure!(
        body.lines().any(|l| l.starts_with("vllm:")),
        "missing vLLM Prometheus metrics"
    );
    Ok(body)
}

fn has_scheduler_gauge(body: &str) -> bool {
    body.lines().any(|line| {
        let Some(end) = line.find(['{', ' ', '\t']) else {
            return false;
        };
        if !matches!(
            &line[..end],
            "vllm:num_requests_running"
                | "vllm:num_requests_waiting"
                | "vllm:kv_cache_usage_perc"
                | "vllm:gpu_cache_usage_perc"
        ) {
            return false;
        }
        let fields = &line[end..];
        let fields = fields.split(" # ").next().unwrap_or(fields);
        let value = fields.rsplit_once('}').map_or(fields, |(_, tail)| tail);
        value
            .split_whitespace()
            .next()
            .and_then(|s| s.parse::<f64>().ok())
            .is_some_and(f64::is_finite)
    })
}

const NO_GAUGE: &str = "no numeric request queue or KV cache gauge in /metrics";

async fn snapshot(client: &Client, url: &str, dest: PathBuf) -> Result<Option<String>> {
    match metrics(client, url).await {
        Ok(body) => {
            let missing = !has_scheduler_gauge(&body);
            fs::write(dest, body)?;
            Ok(missing.then(|| NO_GAUGE.to_string()))
        }
        Err(e) => Ok(Some(format!("{e:#}"))),
    }
}

async fn toggle_profiler(client: &Client, origin: &str, action: &str) -> Option<String> {
    let url = format!("{origin}/{action}_profile");
    let result = async {
        // stop_profile synchronously exports the raw CANN data on every rank.
        let response = client
            .post(&url)
            .timeout(Duration::from_secs(1800))
            .send()
            .await?;
        let status = response.status();
        if !status.is_success() {
            bail!(
                "{status}: {}",
                response
                    .text()
                    .await?
                    .chars()
                    .take(1024)
                    .collect::<String>()
            );
        }
        Ok::<(), anyhow::Error>(())
    }
    .await;
    result.err().map(|e| format!("{action}_profile: {e:#}"))
}

async fn sample_gauges(
    client: &Client,
    metrics_url: &str,
    mut samples: BufWriter<File>,
    monitor_errors: &mut u32,
) -> Result<()> {
    let mut ticker = tokio::time::interval(Duration::from_secs(1));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        ticker.tick().await;
        let sample = match metrics(client, metrics_url).await {
            Ok(body) if !has_scheduler_gauge(&body) => {
                *monitor_errors += 1;
                json!({"unix_ms":now_ms(),"error":NO_GAUGE})
            }
            Ok(body) => {
                let gauges: Vec<_> = body
                    .lines()
                    .filter(|l| {
                        [
                            "vllm:num_requests_",
                            "vllm:kv_cache_usage_perc",
                            "vllm:gpu_cache_usage_perc",
                            "vllm:num_preemptions_total",
                            "vllm:prefix_cache_",
                            "vllm:prompt_tokens_total",
                            "vllm:generation_tokens_total",
                        ]
                        .iter()
                        .any(|prefix| l.starts_with(prefix))
                    })
                    .collect();
                json!({"unix_ms":now_ms(),"metrics":gauges})
            }
            Err(e) => {
                *monitor_errors += 1;
                json!({"unix_ms":now_ms(),"error":format!("{e:#}")})
            }
        };
        line(&mut samples, &sample)?;
    }
}

async fn measure(
    client: &Client,
    url: &str,
    metrics_url: &str,
    case: &Case,
    dir: &Path,
    log: &mut BufWriter<File>,
) -> Result<(Value, bool)> {
    let samples = BufWriter::new(File::create(dir.join("engine-gauges.jsonl"))?);
    let mut monitor_errors = 0;
    let start_unix_ms = now_ms();
    let start = Instant::now();
    let mut stats = tokio::select! {
        result = phase(client, url, case, "measure", case.requests, log) => result?,
        result = sample_gauges(client, metrics_url, samples, &mut monitor_errors) => { result?; bail!("gauge sampler ended unexpectedly"); }
    };
    let mut summary = stats.summary(start.elapsed().as_secs_f64());
    summary["start_unix_ms"] = json!(start_unix_ms);
    summary["end_unix_ms"] = json!(now_ms());
    summary["gauge_scrape_errors"] = json!(monitor_errors);
    Ok((summary, stats.failed == 0 && monitor_errors == 0))
}

fn replay_workload(case: &Case, full: bool) -> Case {
    let mut replay = case.clone();
    replay.warmup = 0;
    if !full {
        let mut wave = case
            .concurrency
            .max(case.input_tokens.len())
            .max(case.text_prompts.len());
        // Leave room for cache population/reuse and several arrival bursts in a short replay.
        if let Some(prefix) = &case.shared_prefix {
            wave = wave.max(8).max(prefix.groups as usize).saturating_mul(3);
        }
        if let Some(arrivals) = &case.arrivals {
            wave = wave.max((arrivals.burst as usize).saturating_mul(3));
        }
        replay.requests = case.requests.min(u32::try_from(wave).unwrap_or(u32::MAX));
        replay.output_tokens = replay.output_tokens.min(8);
        for output in &mut replay.output_pattern {
            *output = (*output).min(8);
        }
    }
    replay
}

fn check_rank_exports(traces: &Path, case_id: u32, count: usize) -> Result<()> {
    for rank in 0..count {
        let marker = traces.join(format!("rank-{rank}/rank.json"));
        let record: Value = serde_json::from_slice(
            &fs::read(&marker).with_context(|| format!("read {}", marker.display()))?,
        )?;
        ensure!(
            record["case_id"] == case_id
                && record["rank"] == rank
                && record["status"] == "exported",
            "rank {rank} did not export case {case_id} (status {}, reason {})",
            record["status"],
            record["error"]
        );
    }
    Ok(())
}

async fn extract_operators(dir: &Path, expected_ranks: usize) -> Result<()> {
    let report = dir.join("operators.json");
    let child = tokio::process::Command::new("python3")
        .args([
            "-c",
            include_str!("../tools/operator_report.py"),
            "--trace-dir",
        ])
        .arg(dir.join("traces"))
        .arg("--output")
        .arg(&report)
        .arg("--expected-ranks")
        .arg(expected_ranks.to_string())
        .kill_on_drop(true)
        .spawn()
        .context("start Ascend operator report extractor")?;
    let result = tokio::time::timeout(Duration::from_secs(1800), child.wait_with_output())
        .await
        .context("operator extraction timed out")??;
    ensure!(
        result.status.success(),
        "operator extraction failed ({}), see operators.json or runtime logs",
        result.status
    );
    let parsed: Value =
        serde_json::from_slice(&fs::read(&report).context("missing operators.json")?)?;
    ensure!(
        parsed["ok"] == true,
        "operator report failed validation, see operators.json"
    );
    Ok(())
}

async fn capture_operators(
    client: &Client,
    origin: &str,
    replay: &Case,
    dir: &Path,
    expected_ranks: usize,
    group: &str,
    full: bool,
) -> Result<Value> {
    let traces = dir.join("traces");
    fs::create_dir(&traces)?;
    let mut log = BufWriter::new(File::create(dir.join("profile-requests.jsonl"))?);
    let phase_name = if group == "Memory" {
        "operator_memory"
    } else {
        "operator_replay"
    };
    let mut profile = json!({
        "ok":false,"status":"starting","full":full,"case_id":replay.id,"expected_ranks":expected_ranks,
        "replay":replay,"raw_path":"traces","path":"operators.json","requests_path":"profile-requests.jsonl",
        "start_unix_ms":now_ms(),"record_shapes":true,"profiler_level":"Level1","aic_metrics":group,
        "request_phase":phase_name
    });
    write_json(&dir.join("operator-profile.json"), &profile)?;
    let start_error = toggle_profiler(client, origin, "start").await;
    let mut errors = Vec::new();
    if let Some(error) = start_error {
        errors.push(error);
    } else {
        profile["workload_start_unix_ms"] = json!(now_ms());
        let url = format!("{origin}/v1/completions");
        match phase(client, &url, replay, phase_name, replay.requests, &mut log).await {
            Ok(stats) => {
                profile["successful_requests"] = json!(stats.latency.len());
                profile["failed_requests"] = json!(stats.failed);
                if stats.failed > 0 {
                    errors.push(format!("{} replay requests failed", stats.failed));
                }
            }
            Err(e) => errors.push(format!("replay failed: {e:#}")),
        }
        profile["workload_end_unix_ms"] = json!(now_ms());
    }
    // Even a failed start may have enabled profiling on a subset of the ranks.
    let stop_error = toggle_profiler(client, origin, "stop").await;
    profile["stopped"] = json!(stop_error.is_none());
    if let Some(err) = stop_error {
        errors.push(err);
    }
    if let Err(e) = check_rank_exports(&traces, replay.id, expected_ranks) {
        errors.push(format!("{e:#}"));
    }
    if let Err(e) = extract_operators(dir, expected_ranks).await {
        errors.push(format!("{e:#}"));
    }
    profile["ok"] = json!(errors.is_empty());
    profile["status"] = json!(if errors.is_empty() {
        "complete"
    } else {
        "failed"
    });
    profile["errors"] = json!(errors);
    profile["end_unix_ms"] = json!(now_ms());
    write_json(&dir.join("operator-profile.json"), &profile)?;
    Ok(profile)
}

async fn operator_replay(
    client: &Client,
    origin: &str,
    case: &Case,
    dir: &Path,
    expected_ranks: usize,
    args: &BenchArgs,
) -> Result<Value> {
    let replay = replay_workload(case, args.profile);
    let marker = dir
        .parent()
        .context("missing benchmark directory")?
        .join("active-case.json");
    let groups = args.aic_metrics.groups();
    let mut captures = Vec::new();
    let mut errors = Vec::new();
    for (index, &group) in groups.iter().enumerate() {
        let secondary = index > 0;
        let pass_dir = if secondary {
            dir.join("pmu-memory")
        } else {
            dir.to_owned()
        };
        if secondary {
            fs::create_dir(&pass_dir)?;
        }
        write_json(
            &marker,
            &json!({"case_id":case.id,"aic_metrics":group,"secondary_memory":secondary}),
        )?;
        let mut pass = capture_operators(
            client,
            origin,
            &replay,
            &pass_dir,
            expected_ranks,
            group,
            args.profile,
        )
        .await?;
        if secondary {
            for key in ["path", "raw_path", "requests_path"] {
                pass[key] = json!(format!(
                    "pmu-memory/{}",
                    pass[key].as_str().context("missing capture path")?
                ));
            }
        }
        let stopped = pass["stopped"] == true;
        if let Some(messages) = pass["errors"].as_array() {
            for message in messages.iter().filter_map(Value::as_str) {
                errors.push(format!("{group}: {message}"));
            }
        }
        captures.push(pass);
        if !stopped {
            break;
        }
    }
    let mut combined = captures[0].clone();
    let ok = captures.len() == groups.len() && captures.iter().all(|p| p["ok"] == true);
    combined["ok"] = json!(ok);
    combined["status"] = json!(if ok { "complete" } else { "failed" });
    let last = captures.last().expect("at least one counter group");
    combined["stopped"] = last["stopped"].clone();
    combined["end_unix_ms"] = last["end_unix_ms"].clone();
    combined["requested_groups"] = json!(groups);
    combined["errors"] = json!(errors);
    combined["captures"] = json!(captures);
    write_json(&dir.join("operator-profile.json"), &combined)?;
    Ok(combined)
}

pub fn run_directory(parent: &Path) -> Result<PathBuf> {
    fs::create_dir_all(parent)?;
    let stamp = now_ms();
    for attempt in 0..1000 {
        let dir = parent.join(format!("run-{stamp}-{}-{attempt}", std::process::id()));
        match fs::create_dir(&dir) {
            Ok(()) => return Ok(dir),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.into()),
        }
    }
    bail!("could not allocate unique run directory")
}

pub async fn run(
    args: &BenchArgs,
    base_url: &str,
    dir: &Path,
    expected_ranks: usize,
) -> Result<()> {
    let (cases, suite_hash) = load_suite(args.cases.as_deref())?;
    let selected = cases_in_range(&cases, &args.range)?;
    eprintln!("benchmark results: {}", dir.display());
    let golden = if let Some(source) = &args.golden {
        let data = fs::read(source).context("read accuracy reference")?;
        let digest = hex::encode(Sha256::digest(&data));
        fs::write(dir.join("golden.jsonl"), data)?;
        Some(digest)
    } else {
        None
    };
    write_json(
        &dir.join("manifest.json"),
        &json!({
            "schema_version":1,"version":env!("CARGO_PKG_VERSION"),"start_unix_ms":now_ms(),
            "suite_sha256":suite_hash, "cases":selected,"range":args.range,"golden_sha256":golden,
            "request_timeout_secs":args.request_timeout_secs,"prompt_generator":"splitmix64-v1",
            "workload": if selected.iter().all(|c| c.arrivals.is_none()) { "closed-loop" } else { "see per-case arrivals" },
            "shared_prefix_generator":"case-phase-sha256-splitmix64-v1",
            "stream_usage":"continuous","profile":args.profile,"text_prompt_generator":"task-sha256-be64-v1",
            "operator_capture":"separate_replay","expected_ranks":expected_ranks,"aic_metrics":args.aic_metrics.groups()
        }),
    )?;
    if golden.is_some() {
        let checks =
            crate::accuracy::verifier(dir.join("golden.jsonl"), dir.join("accuracy.jsonl"));
        crate::accuracy::run(&checks, base_url)
            .await
            .context("accuracy check failed before timing")?;
    }
    let client = Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(args.request_timeout_secs))
        .build()?;
    let url = format!("{base_url}/completions");
    let origin = base_url
        .strip_suffix("/v1")
        .context("expected base URL ending in /v1")?;
    let metrics_url = format!("{origin}/metrics");
    let mut reports = Vec::new();
    let mut success = true;
    for case in selected {
        eprintln!("case {} ({})", case.id, case.name);
        let subdir = dir.join(format!("case-{}", case.id));
        fs::create_dir(&subdir)?;
        let mut log = BufWriter::new(File::create(subdir.join("requests.jsonl"))?);
        let warmup = phase(&client, &url, case, "warmup", case.warmup, &mut log).await?;
        let mut report = if warmup.failed > 0 {
            json!({"case":case,"ok":false,"warmup_failed":warmup.failed,"measurement":null})
        } else {
            let before_error = snapshot(&client, &metrics_url, subdir.join("before.prom")).await?;
            let (summary, measured_ok) =
                measure(&client, &url, &metrics_url, case, &subdir, &mut log).await?;
            let after_error = snapshot(&client, &metrics_url, subdir.join("after.prom")).await?;
            json!({"case":case,"ok":measured_ok && before_error.is_none() && after_error.is_none(),
                "warmup_failed":0,"measurement":summary,"before_metrics_error":before_error,"after_metrics_error":after_error})
        };
        let baseline_ok = report["ok"] == true;
        report["baseline_ok"] = json!(baseline_ok);
        report["ok"] = json!(false);
        report["operator_profile"] = json!({"status":"pending","ok":false});
        write_json(&subdir.join("summary.json"), &report)?;
        let profile = if warmup.failed == 0 {
            operator_replay(&client, origin, case, &subdir, expected_ranks, args).await?
        } else {
            json!({"status":"skipped","ok":false,"errors":["warmup failed"]})
        };
        let stop_failed = profile["stopped"] == false;
        report["ok"] = json!(baseline_ok && profile["ok"] == true);
        report["operator_profile"] = profile;
        write_json(&subdir.join("summary.json"), &report)?;
        success &= report["ok"] == true;
        reports.push(report);
        if stop_failed {
            eprintln!("stopping suite because profiler shutdown was not confirmed");
            break;
        }
    }
    write_json(
        &dir.join("summary.json"),
        &json!({"schema_version":1,"ok":success,"cases":reports}),
    )?;
    ensure!(
        success,
        "benchmark or metrics collection failed; see {}",
        dir.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn scheduler_sampling_requires_a_numeric_queue_or_cache_gauge() {
        assert!(has_scheduler_gauge(
            "vllm:num_requests_running{model=\"a b\"} 0"
        ));
        assert!(has_scheduler_gauge("vllm:kv_cache_usage_perc 0.25"));
        assert!(!has_scheduler_gauge("vllm:prompt_tokens_total 999"));
        assert!(!has_scheduler_gauge("vllm:num_requests_waiting NaN"));
        assert!(!has_scheduler_gauge(
            "# TYPE vllm:num_requests_running gauge"
        ));
    }

    #[test]
    fn range_validation_and_prompt_replay() {
        let cases: Vec<Case> =
            serde_json::from_slice(include_bytes!("../benches/cases.json")).unwrap();
        let ids: Vec<_> = cases_in_range(&cases, "2-4, 1, 4")
            .unwrap()
            .iter()
            .map(|c| c.id)
            .collect();
        assert_eq!(ids, [1, 2, 3, 4]);
        let selected: Vec<_> = cases_in_range(&cases, "100-200")
            .unwrap()
            .into_iter()
            .map(|case| case.id)
            .collect();
        assert_eq!(selected, (100..=200).collect::<Vec<_>>());
        let last_id = cases.iter().map(|c| c.id).max().unwrap();
        for range in [
            "2-1",
            "",
            "0-4294967295",
            "1,",
            &format!("3,{}", last_id + 1),
        ] {
            assert!(cases_in_range(&cases, range).is_err());
        }
        assert_eq!(prompt(77, 2), prompt(77, 8)[..2]);
        assert_ne!(prompt(77, 8), prompt(78, 8));
        assert_ne!(text_prompt(77, "Explain"), text_prompt(78, "Explain"));
    }

    #[test]
    fn bounded_replays_preserve_input_mix_and_verify_rank_ownership() {
        let cases: Vec<Case> =
            serde_json::from_slice(include_bytes!("../benches/cases.json")).unwrap();
        let mixed = cases
            .iter()
            .find(|case| case.name == "mixed-context")
            .unwrap();
        let replay = replay_workload(mixed, false);
        assert_eq!(replay.input_tokens, mixed.input_tokens);
        assert_eq!(replay.requests as usize, mixed.concurrency);
        assert_eq!(replay.output_tokens, 8);
        assert_eq!(
            replay_workload(mixed, true).output_tokens,
            mixed.output_tokens
        );
        let dir = run_directory(&std::env::temp_dir()).unwrap();
        fs::create_dir(dir.join("rank-0")).unwrap();
        let marker = dir.join("rank-0/rank.json");
        write_json(&marker, &json!({"rank":0,"case_id":7,"status":"exported"})).unwrap();
        assert!(check_rank_exports(&dir, 7, 1).is_ok());
        assert!(check_rank_exports(&dir, 8, 1).is_err());
        assert!(check_rank_exports(&dir, 7, 2).is_err());
        write_json(&marker, &json!({"rank":0,"case_id":7,"status":"starting"})).unwrap();
        assert!(check_rank_exports(&dir, 7, 1).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn survey_mixed_workloads_and_shared_prefix_are_replayable() {
        let (cases, _) = load_suite(None).unwrap();
        let ids = |range| {
            cases_in_range(&cases, range)
                .unwrap()
                .iter()
                .map(|c| c.id)
                .collect::<Vec<_>>()
        };
        assert_eq!(ids("@survey"), [1, 6, 9, 211, 220, 227, 228, 238]);
        assert_eq!(ids("@prefix,227-228"), (224..=232).collect::<Vec<_>>());
        assert!(cases_in_range(&cases, "@").is_err());
        assert!(cases_in_range(&cases, "@pref").is_err());
        let cache = cases_in_range(&cases, "229").unwrap()[0];
        let a = token_input(cache, "measure", 0, 4096);
        let b = token_input(cache, "measure", 8, 4096);
        assert_eq!(a[..3840], b[..3840]);
        assert_ne!(a[3840..], b[3840..]);
        assert_ne!(a[..3840], token_input(cache, "measure", 1, 4096)[..3840]);
        for phase in ["warmup", "operator_replay", "operator_memory"] {
            assert_ne!(a[..3840], token_input(cache, phase, 0, 4096)[..3840]);
        }
        assert_ne!(
            token_input(cache, "operator_replay", 0, 4096),
            token_input(cache, "operator_memory", 0, 4096)
        );
        let plain = cases_in_range(&cases, "227").unwrap()[0];
        assert_eq!(
            token_input(plain, "measure", 5, 4096),
            prompt((227u64 << 32) | 5, 4096)
        );
        let mix = cases_in_range(&cases, "223").unwrap()[0];
        assert_eq!(
            [mix.output_at(0), mix.output_at(1), mix.output_at(2)],
            [512, 1, 512]
        );
        assert_eq!(replay_workload(mix, false).output_pattern, [8, 1]);
        let pace = cases_in_range(&cases, "238").unwrap()[0];
        assert_eq!(pace.arrival_offset(7), Some(Duration::ZERO));
        assert_eq!(pace.arrival_offset(8), Some(Duration::from_millis(250)));
        let extended: Vec<Case> =
            serde_json::from_slice(include_bytes!("../benches/extended-context.json")).unwrap();
        assert_eq!(cases_in_range(&extended, "@long").unwrap().len(), 10);
        assert_eq!(
            extended[9].input_tokens[0] + extended[9].output_at(0) as usize,
            1048576
        );
    }

    // The fixture only exercises the benchmark transport, accounting, and failure path.
    async fn receive_stream(body: &'static str, text_case: bool) -> Observation {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/v1/completions", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut req = Vec::new();
            let mut buf = [0_u8; 512];
            loop {
                let n = socket.read(&mut buf).await.unwrap();
                assert!(n > 0);
                req.extend_from_slice(&buf[..n]);
                if let Some(i) = req.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
                    let headers = std::str::from_utf8(&req[..i]).unwrap().to_ascii_lowercase();
                    let length: usize = headers
                        .lines()
                        .find_map(|line| {
                            line.strip_prefix("content-length: ")
                                .map(|n| n.trim().parse().unwrap())
                        })
                        .unwrap();
                    if req.len() >= i + 4 + length {
                        let payload: Value =
                            serde_json::from_slice(&req[i + 4..i + 4 + length]).unwrap();
                        assert_eq!(payload["stream_options"]["continuous_usage_stats"], true);
                        if text_case {
                            assert_eq!(
                                payload["prompt"],
                                text_prompt(4 << 32, "Review this Rust loop")
                            );
                        } else {
                            assert_eq!(payload["prompt"].as_array().unwrap().len(), 4);
                        }
                        break;
                    }
                }
            }
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n",
                body.len()
            );
            socket.write_all(head.as_bytes()).await.unwrap();
            for chunk in body.as_bytes().chunks(59) {
                socket.write_all(chunk).await.unwrap();
                tokio::task::yield_now().await;
            }
        });
        let (input_tokens, text_prompts) = if text_case {
            (vec![], vec!["Review this Rust loop".into()])
        } else {
            (vec![4], vec![])
        };
        let case = Case {
            id: 4,
            name: "fixture".into(),
            input_tokens,
            text_prompts,
            output_tokens: 3,
            output_pattern: vec![],
            shared_prefix: None,
            arrivals: None,
            concurrency: 1,
            requests: 1,
            warmup: 0,
        };
        let client = Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap();
        let observation = request(&client, &url, &case, "measure", 0, None).await;
        server.await.unwrap();
        observation
    }

    #[tokio::test]
    async fn records_coalesced_tokens_and_empty_text() {
        let observation = receive_stream(concat!(
            "data: {\"id\":\"q1\",\"choices\":[{\"text\":\"\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2}}\r\n\r\n",
            "data: {\"choices\":[{\"text\":\"hi\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":3}}\n\n",
            "data: {\"choices\":[{\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":3}}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":3}}\n\n",
            "data: [DONE]\n\n"
        ), false).await;
        assert!(observation.error.is_none(), "{:?}", observation.error);
        assert_eq!(observation.generated_tokens, 3);
        assert_eq!(
            observation
                .arrivals
                .iter()
                .map(|a| a.tokens)
                .collect::<Vec<_>>(),
            [2, 1]
        );
        assert_eq!(
            observation.output_text_sha256.unwrap(),
            hex::encode(Sha256::digest(b"hi"))
        );
    }

    #[tokio::test]
    async fn text_case_uses_tokenizer_usage_and_replayable_prompt() {
        let observation = receive_stream(concat!(
            "data: {\"choices\":[{\"text\":\"fine\",\"finish_reason\":\"length\"}],\"usage\":{\"prompt_tokens\":17,\"completion_tokens\":3}}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":17,\"completion_tokens\":3}}\n\n",
            "data: [DONE]\n\n"
        ), true).await;
        assert!(observation.error.is_none(), "{:?}", observation.error);
        assert_eq!(observation.expected_prompt_tokens, None);
        assert_eq!(observation.prompt_tokens, Some(17));
        assert_eq!(observation.text_prompt_index, Some(0));
        let seed = observation.prompt_seed.parse::<u64>().unwrap();
        let text = text_prompt(seed, "Review this Rust loop");
        assert_eq!(
            observation.prompt_text_sha256,
            Some(hex::encode(Sha256::digest(text.as_bytes())))
        );
    }

    #[tokio::test]
    async fn partial_stream_does_not_count_as_success() {
        let observation = receive_stream("data: {\"choices\":[{\"text\":\"hi\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":1}}\n\n", false).await;
        assert!(observation.error.as_deref().unwrap().contains("[DONE]"));
        let mut stats = Stats::default();
        stats.observe(&observation);
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.generated, 0);
        assert!(stats.ttft.is_empty());
    }
}
