use std::{path::PathBuf, time::Duration};

use anyhow::{Context, Result, bail, ensure};
use clap::Args;
use futures_util::{StreamExt, stream};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::{
    fs::{File, OpenOptions},
    io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader, BufWriter},
};

use crate::runtime::MODEL_ALIAS;

#[derive(Debug, Args)]
pub struct OfflineArgs {
    /// One JSON object per line. Use '-' for stdin.
    #[arg(long)]
    input: PathBuf,
    /// New JSON Lines file, or '-' for stdout. Results include the input line number.
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u32).range(1..=4096))]
    concurrency: u32,
    #[arg(long, default_value_t = 3600, value_parser = clap::value_parser!(u64).range(1..))]
    request_timeout_secs: u64,
}

impl OfflineArgs {
    pub fn validate(&self) -> Result<()> {
        if self.input.as_os_str() != "-" {
            std::fs::File::open(&self.input).with_context(|| format!("open {:?}", self.input))?;
            ensure!(
                self.input.is_file(),
                "expected a regular input file or '-' for stdin"
            );
        }
        if self.output.as_os_str() != "-" {
            ensure!(
                !self.output.try_exists()?,
                "output {:?} already exists",
                self.output
            );
            if let Some(parent) = self.output.parent().filter(|p| !p.as_os_str().is_empty()) {
                ensure!(
                    parent.is_dir(),
                    "output directory {:?} does not exist",
                    parent
                );
            }
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Job {
    #[serde(default)]
    id: Value,
    prompt: Option<String>,
    messages: Option<Vec<Value>>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
}

impl Job {
    fn request(&self) -> Result<(&'static str, Value)> {
        let max_tokens = self.max_tokens.unwrap_or(128);
        let temperature = self.temperature.unwrap_or(0.0);
        ensure!(max_tokens > 0, "max_tokens must be positive");
        ensure!(
            temperature.is_finite() && temperature >= 0.0,
            "invalid temperature"
        );
        let mut body = json!({
            "model": MODEL_ALIAS, "temperature": temperature, "max_tokens": max_tokens,
            "stream": false
        });
        match (&self.prompt, &self.messages) {
            (Some(text), None) if !text.is_empty() => {
                body["prompt"] = json!(text);
                Ok(("completions", body))
            }
            (None, Some(messages)) if !messages.is_empty() => {
                body["messages"] = json!(messages);
                Ok(("chat/completions", body))
            }
            _ => bail!("supply exactly one nonempty prompt or messages array"),
        }
    }
}

async fn complete(client: &reqwest::Client, url: &str, job: &Job) -> Result<Value> {
    let (endpoint, body) = job.request()?;
    let response = client
        .post(format!("{url}/{endpoint}"))
        .json(&body)
        .send()
        .await?;
    let status = response.status();
    if !status.is_success() {
        let detail: String = response.text().await?.chars().take(2000).collect();
        bail!("worker returned {status}: {detail}");
    }
    let response: Value = response.json().await.context("invalid completion JSON")?;
    let choice = response
        .pointer("/choices/0")
        .context("missing completion choice")?;
    ensure!(
        choice["finish_reason"].is_string(),
        "completion missing finish reason"
    );
    Ok(response)
}

async fn process(client: &reqwest::Client, url: &str, line: String, number: u64) -> Value {
    match serde_json::from_str::<Job>(&line) {
        Err(err) => json!({"line": number, "error": err.to_string()}),
        Ok(job) => match complete(client, url, &job).await {
            Ok(response) => json!({"line":number, "id":job.id, "response":response}),
            Err(err) => json!({"line":number, "id":job.id, "error":format!("{err:#}")}),
        },
    }
}

pub async fn run(args: &OfflineArgs, url: &str) -> Result<()> {
    let reader: Box<dyn AsyncRead + Unpin + Send> = if args.input.as_os_str() == "-" {
        Box::new(tokio::io::stdin())
    } else {
        Box::new(
            File::open(&args.input)
                .await
                .with_context(|| format!("open {:?}", args.input))?,
        )
    };
    let writer: Box<dyn AsyncWrite + Unpin + Send> = if args.output.as_os_str() == "-" {
        Box::new(tokio::io::stdout())
    } else {
        Box::new(
            OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&args.output)
                .await
                .with_context(|| format!("create new output {:?}", args.output))?,
        )
    };

    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(args.request_timeout_secs))
        .build()?;
    let lines = BufReader::new(reader).lines();
    let inputs = stream::try_unfold((lines, 1_u64), |(mut lines, mut number)| async move {
        while let Some(line) = lines.next_line().await? {
            let current = number;
            number += 1;
            if !line.trim().is_empty() {
                return Ok::<_, std::io::Error>(Some(((current, line), (lines, number))));
            }
        }
        Ok(None)
    });
    let client_ref = &client;
    let results = inputs
        .map(|input| async move {
            let (number, line) = input.context("read input JSON Lines")?;
            Ok::<_, anyhow::Error>(process(client_ref, url, line, number).await)
        })
        .buffer_unordered(args.concurrency as usize);
    tokio::pin!(results);
    let mut output = BufWriter::new(writer);
    let mut failed = 0_u64;
    while let Some(result) = results.next().await {
        let row = result?;
        failed += u64::from(row.get("error").is_some());
        let mut bytes = serde_json::to_vec(&row)?;
        bytes.push(b'\n');
        output.write_all(&bytes).await?;
        output.flush().await?;
    }
    ensure!(
        failed == 0,
        "{failed} offline requests failed; see output JSON Lines"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_routes_text_and_chat_and_rejects_ambiguous_input() {
        let prompt: Job = serde_json::from_value(json!({"prompt":"x"})).unwrap();
        let (path, body) = prompt.request().unwrap();
        assert_eq!(path, "completions");
        assert_eq!(body["temperature"], 0.0);
        let chat: Job =
            serde_json::from_value(json!({"messages":[{"role":"user","content":"x"}]})).unwrap();
        assert_eq!(chat.request().unwrap().0, "chat/completions");
        for row in [
            json!({}),
            json!({"prompt":"x", "messages":[]}),
            json!({"prompt":"x","max_tokens":0}),
        ] {
            assert!(
                serde_json::from_value::<Job>(row)
                    .unwrap()
                    .request()
                    .is_err()
            );
        }
    }
}
