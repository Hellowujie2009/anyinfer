use anyhow::{Context, Result, bail, ensure};
use clap::Args;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::runtime::MODEL_ALIAS;

#[derive(Args, Debug)]
pub struct AccuracyArgs {
    /// JSONL corpus with reference greedy token IDs.
    #[arg(
        long,
        conflicts_with = "capture_prompts",
        required_unless_present = "capture_prompts"
    )]
    pub golden: Option<PathBuf>,
    /// JSONL prompts to run against a trusted reference and save as a golden corpus.
    #[arg(
        long,
        conflicts_with = "golden",
        required_unless_present = "golden",
        requires = "reference_name"
    )]
    pub capture_prompts: Option<PathBuf>,
    /// Provenance recorded in the captured corpus (checkpoint revision, precision, runtime).
    #[arg(long, requires = "capture_prompts")]
    pub reference_name: Option<String>,
    /// Served model ID on the reference endpoint.
    #[arg(long, requires = "capture_prompts")]
    pub reference_model: Option<String>,
    /// New JSONL file for comparison results or captured golden rows.
    #[arg(long)]
    pub output: PathBuf,
}

pub fn verifier(golden: PathBuf, output: PathBuf) -> AccuracyArgs {
    AccuracyArgs {
        golden: Some(golden),
        capture_prompts: None,
        reference_name: None,
        reference_model: None,
        output,
    }
}

pub fn validate_golden(path: &Path, max_model_len: usize) -> Result<()> {
    verifier(path.to_owned(), PathBuf::new()).load(max_model_len)?;
    Ok(())
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    prompt_token_ids: Vec<u32>,
    max_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    expected_token_ids: Option<Vec<u32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reference_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reference_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    suite_size: Option<usize>,
}

impl AccuracyArgs {
    fn load(&self, max_model_len: usize) -> Result<Vec<Case>> {
        let (path, capture) = match (&self.golden, &self.capture_prompts) {
            (Some(p), None) => {
                ensure!(
                    self.reference_name.is_none() && self.reference_model.is_none(),
                    "reference options require --capture-prompts"
                );
                (p, false)
            }
            (None, Some(p)) => {
                ensure!(
                    self.reference_name
                        .as_ref()
                        .is_some_and(|s| !s.trim().is_empty()),
                    "--capture-prompts requires a reference name"
                );
                ensure!(
                    self.reference_model
                        .as_ref()
                        .is_none_or(|s| !s.trim().is_empty()),
                    "empty reference model ID"
                );
                (p, true)
            }
            _ => bail!("select --golden or --capture-prompts"),
        };
        let mut cases = Vec::new();
        let mut ids = BTreeSet::new();
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        for (index, line) in BufReader::new(file).lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let case: Case = serde_json::from_str(&line)
                .with_context(|| format!("{} line {}", path.display(), index + 1))?;
            ensure!(
                !case.id.trim().is_empty() && ids.insert(case.id.clone()),
                "empty or duplicate accuracy case ID {:?}",
                case.id
            );
            ensure!(
                !case.prompt_token_ids.is_empty() && case.max_tokens > 0,
                "case {} has an empty input or output",
                case.id
            );
            ensure!(
                case.prompt_token_ids
                    .len()
                    .checked_add(case.max_tokens)
                    .is_some_and(|n| n <= max_model_len),
                "case {} exceeds the context length {max_model_len}",
                case.id
            );
            if capture {
                ensure!(
                    case.expected_token_ids.is_none()
                        && case.reference_name.is_none()
                        && case.reference_model.is_none()
                        && case.suite_size.is_none(),
                    "capture prompt {} already has reference fields",
                    case.id
                );
            } else {
                ensure!(
                    case.expected_token_ids
                        .as_ref()
                        .is_some_and(|ids| ids.len() == case.max_tokens),
                    "case {} requires exactly {} expected token IDs",
                    case.id,
                    case.max_tokens
                );
            }
            cases.push(case);
        }
        ensure!(!cases.is_empty(), "empty accuracy corpus");
        ensure!(
            cases
                .iter()
                .all(|c| c.suite_size.is_none_or(|n| n == cases.len())),
            "incomplete captured suite, found {} cases",
            cases.len()
        );
        Ok(cases)
    }

    pub fn validate(&self, max_model_len: usize) -> Result<()> {
        self.load(max_model_len)?;
        ensure!(
            !self.output.try_exists()?,
            "output already exists: {}",
            self.output.display()
        );
        Ok(())
    }
}

#[derive(Deserialize)]
struct Choice {
    index: usize,
    token_ids: Vec<u32>,
    prompt_token_ids: Vec<u32>,
    finish_reason: String,
}

#[derive(Deserialize)]
struct Completion {
    id: String,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Deserialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

fn extract(response: Completion, case: &Case, model: &str) -> Result<(String, Vec<u32>)> {
    ensure!(
        response.model == model,
        "endpoint returned model {:?} for {model:?}",
        response.model
    );
    ensure!(response.choices.len() == 1, "expected a single completion");
    let choice = response
        .choices
        .into_iter()
        .next()
        .context("missing choice")?;
    ensure!(
        choice.index == 0 && choice.prompt_token_ids == case.prompt_token_ids,
        "echoed prompt IDs differ from submitted IDs"
    );
    ensure!(
        choice.finish_reason == "length",
        "unexpected finish reason {:?}",
        choice.finish_reason
    );
    ensure!(
        choice.token_ids.len() == case.max_tokens
            && response.usage.completion_tokens == case.max_tokens,
        "truncated output or inconsistent usage, expected {} tokens, returned {} IDs and {} usage tokens",
        case.max_tokens,
        choice.token_ids.len(),
        response.usage.completion_tokens
    );
    ensure!(
        response.usage.prompt_tokens == case.prompt_token_ids.len()
            && response.usage.total_tokens
                == response.usage.prompt_tokens + response.usage.completion_tokens,
        "inconsistent prompt or total usage"
    );
    Ok((response.id, choice.token_ids))
}

async fn generate(
    client: &Client,
    url: &str,
    model: &str,
    case: &Case,
) -> Result<(String, Vec<u32>)> {
    let payload = json!({
        "model":model,"prompt":case.prompt_token_ids,"max_tokens":case.max_tokens,"n":1,
        "temperature":0,"top_p":1,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0,
        "ignore_eos":true,"return_token_ids":true,"add_special_tokens":false,"stream":false
    });
    let response = client.post(url).json(&payload).send().await?;
    let status = response.status();
    if !status.is_success() {
        bail!(
            "HTTP {status}: {}",
            response
                .text()
                .await?
                .chars()
                .take(2048)
                .collect::<String>()
        );
    }
    extract(
        response
            .json()
            .await
            .context("expected a completion with prompt_token_ids and token_ids")?,
        case,
        model,
    )
}

fn comparison(case: &Case, request_id: String, actual: Vec<u32>) -> Result<Value> {
    let expected = case
        .expected_token_ids
        .as_ref()
        .context("missing expected_token_ids")?;
    let mismatch = expected
        .iter()
        .zip(&actual)
        .position(|(want, got)| want != got);
    Ok(json!({
        "id":case.id,"request_id":request_id,"reference_name":case.reference_name,"reference_model":case.reference_model,
        "prompt_tokens":case.prompt_token_ids.len(),"ok":mismatch.is_none(),"first_mismatch_index":mismatch,
        "expected_token_ids":expected,"actual_token_ids":actual
    }))
}

fn output(path: &Path) -> Result<BufWriter<File>> {
    if let Some(dir) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(dir)?;
    }
    Ok(BufWriter::new(
        File::create_new(path).with_context(|| format!("create {}", path.display()))?,
    ))
}

fn write_row(writer: &mut BufWriter<File>, row: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *writer, row)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

pub async fn run(args: &AccuracyArgs, base_url: &str) -> Result<()> {
    let mut cases = args.load(usize::MAX)?;
    let total = cases.len();
    let client = Client::builder()
        .timeout(Duration::from_secs(600))
        .no_proxy()
        .build()?;
    let url = format!("{}/completions", base_url.trim_end_matches('/'));
    let model = args.reference_model.as_deref().unwrap_or(MODEL_ALIAS);
    let mut failures = 0;
    if args.golden.is_some() {
        // Hash the exact evaluated rows, including order and case identities.
        let corpus_hash = hex::encode(Sha256::digest(serde_json::to_vec(&cases)?));
        let mut writer = output(&args.output)?;
        for case in &cases {
            let mut row = match generate(&client, &url, model, case).await {
                Ok((id, actual)) => comparison(case, id, actual)?,
                Err(e) => json!({"id":case.id,"ok":false,"error":format!("{e:#}")}),
            };
            row["suite_size"] = json!(total);
            row["corpus_sha256"] = json!(corpus_hash);
            if row["ok"] != true {
                failures += 1;
            }
            write_row(&mut writer, &row)?;
        }
    } else {
        for case in &mut cases {
            match generate(&client, &url, model, case).await {
                Ok((_, actual)) => {
                    case.expected_token_ids = Some(actual);
                    case.reference_name.clone_from(&args.reference_name);
                    case.reference_model = Some(model.to_owned());
                    case.suite_size = Some(total);
                }
                Err(e) => {
                    failures += 1;
                    eprintln!("capture {}: {e:#}", case.id);
                }
            }
        }
        ensure!(
            failures == 0,
            "reference capture failed for {failures}/{total} cases"
        );
        let mut writer = output(&args.output)?;
        for case in &cases {
            write_row(&mut writer, case)?;
        }
    }
    ensure!(
        failures == 0,
        "accuracy failed for {failures}/{total} cases; see {}",
        args.output.display()
    );
    eprintln!("accuracy {total}/{total}: {}", args.output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case() -> Case {
        serde_json::from_value(json!({
            "id":"golden","prompt_token_ids":[3,5],"max_tokens":2,"expected_token_ids":[7,9]
        }))
        .unwrap()
    }

    fn response() -> Value {
        json!({"id":"cmpl-1","model":MODEL_ALIAS,"choices":[{"index":0,"prompt_token_ids":[3,5],
            "token_ids":[7,9],"finish_reason":"length"}],"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}})
    }

    #[test]
    fn strict_matches_and_reports_first_divergence() {
        let mut parsed = response();
        let c = case();
        let (id, ids) = extract(
            serde_json::from_value(parsed.clone()).unwrap(),
            &c,
            MODEL_ALIAS,
        )
        .unwrap();
        assert_eq!(comparison(&c, id, ids).unwrap()["ok"], true);
        parsed["choices"][0]["token_ids"][1] = json!(12);
        let (id, ids) = extract(serde_json::from_value(parsed).unwrap(), &c, MODEL_ALIAS).unwrap();
        let result = comparison(&c, id, ids).unwrap();
        assert_eq!(result["ok"], false);
        assert_eq!(result["first_mismatch_index"], 1);
        assert_eq!(result["actual_token_ids"][1], 12);
    }

    #[test]
    fn rejects_truncation_wrong_prompt_and_token_accounting() {
        for (field, value) in [
            ("/choices/0/token_ids", json!([7])),
            ("/choices/0/prompt_token_ids", json!([3, 6])),
            ("/choices/0/finish_reason", json!("stop")),
            ("/usage/completion_tokens", json!(1)),
            ("/usage/total_tokens", json!(5)),
            ("/model", json!("wrong")),
        ] {
            let mut reply = response();
            *reply.pointer_mut(field).unwrap() = value;
            assert!(
                extract(serde_json::from_value(reply).unwrap(), &case(), MODEL_ALIAS).is_err(),
                "{field}"
            );
        }
    }

    #[test]
    fn preflight_checks_lengths_duplicate_ids_and_suite_coverage() {
        let path = std::env::temp_dir().join(format!(
            "anyinfer-accuracy-{}-{}.jsonl",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut c = case();
        fs::write(&path, serde_json::to_vec(&c).unwrap()).unwrap();
        assert!(validate_golden(&path, 4).is_ok());
        assert!(validate_golden(&path, 3).is_err());
        c.suite_size = Some(2);
        let row = serde_json::to_string(&c).unwrap();
        fs::write(&path, &row).unwrap();
        assert!(validate_golden(&path, 4).is_err());
        fs::write(&path, format!("{row}\n{row}\n")).unwrap();
        assert!(validate_golden(&path, 4).is_err());
        c.suite_size = None;
        c.expected_token_ids = Some(vec![7]);
        fs::write(&path, serde_json::to_vec(&c).unwrap()).unwrap();
        assert!(validate_golden(&path, 4).is_err());
        fs::remove_file(path).unwrap();
    }
}
