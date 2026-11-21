mod accuracy;
mod bench;
mod offline;
mod runtime;

use std::net::SocketAddr;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use runtime::{Runtime, RuntimeConfig};

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Explicit vLLM reference service and benchmark for comparison with AnyInfer"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// List matching workloads without starting the inference engine.
    ListCases {
        /// Use a supplied workload suite in place of the bundled cases.
        #[arg(long, env = "ANYINFER_CASE_CATALOG")]
        cases: Option<std::path::PathBuf>,
        /// Workload families, inclusive ranges or individual IDs, for example @survey or 1,7,100-120.
        #[arg(long, default_value = "all")]
        range: String,
    },
    /// Serve the loaded model through the OpenAI-compatible HTTP API.
    Serve {
        #[command(flatten)]
        runtime: RuntimeConfig,
        #[arg(long, default_value = "127.0.0.1:8000")]
        listen: SocketAddr,
    },
    /// Generate completions for a JSON Lines file, keeping the model resident.
    Offline {
        #[command(flatten)]
        runtime: RuntimeConfig,
        #[command(flatten)]
        jobs: offline::OfflineArgs,
    },
    /// Load the model once, then run the selected benchmark workloads.
    Bench {
        #[command(flatten)]
        runtime: RuntimeConfig,
        #[command(flatten)]
        cases: bench::BenchArgs,
    },
    /// Verify greedy token IDs or capture reference outputs using a local checkpoint.
    Accuracy {
        #[command(flatten)]
        runtime: RuntimeConfig,
        #[command(flatten)]
        checks: accuracy::AccuracyArgs,
    },
    /// Capture golden outputs from an independently trusted, running reference service.
    Capture {
        /// Reference HTTP API including its /v1 prefix.
        #[arg(long)]
        reference_url: reqwest::Url,
        #[command(flatten)]
        checks: accuracy::AccuracyArgs,
    },
}

async fn termination_signal() -> Result<()> {
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        tokio::select! {
            r = tokio::signal::ctrl_c() => r?,
            _ = term.recv() => (),
        }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let command = Cli::parse().command;
    let config = match &command {
        Command::ListCases { cases, range } => return bench::list_cases(cases.as_deref(), range),
        Command::Serve { runtime, .. }
        | Command::Offline { runtime, .. }
        | Command::Bench { runtime, .. }
        | Command::Accuracy { runtime, .. } => runtime,
        Command::Capture {
            reference_url,
            checks,
        } => {
            anyhow::ensure!(
                reference_url.scheme() == "http",
                "reference URL must use http"
            );
            anyhow::ensure!(
                checks.capture_prompts.is_some(),
                "capture requires --capture-prompts"
            );
            checks.validate(usize::MAX)?;
            return accuracy::run(checks, reference_url.as_str().trim_end_matches('/')).await;
        }
    };
    let bench_run = if let Command::Bench { cases, .. } = &command {
        cases.validate(config.max_model_len as usize)?;
        Some(bench::run_directory(&cases.output)?)
    } else {
        None
    };
    if let Command::Offline { jobs, .. } = &command {
        jobs.validate()?;
    }
    if let Command::Accuracy { checks, .. } = &command {
        checks.validate(config.max_model_len as usize)?;
    }
    let mut engine = tokio::select! {
        result = async {
            match &command {
                Command::Serve { listen, .. } => Runtime::start_on(config, *listen).await,
                Command::Bench { .. } => Runtime::start_benchmark(config, bench_run.as_deref().expect("allocated benchmark directory")).await,
                _ => Runtime::start(config).await,
            }
        } => result?,
        signal = termination_signal() => {signal?; bail!("interrupted while loading model")},
    };

    let base_url = engine.base_url.clone();
    let serving = matches!(command, Command::Serve { .. });
    let job = async {
        match &command {
            Command::Serve { listen, .. } => {
                eprintln!(
                    "Serving {} on {listen} (local endpoint {base_url})",
                    runtime::MODEL_ALIAS
                );
                std::future::pending().await
            }
            Command::Offline { jobs, .. } => offline::run(jobs, &base_url).await,
            Command::Bench { cases, runtime } => {
                bench::run(
                    cases,
                    &base_url,
                    bench_run.as_deref().expect("allocated benchmark directory"),
                    runtime.devices.split(',').count(),
                )
                .await
            }
            Command::Accuracy { checks, .. } => accuracy::run(checks, &base_url).await,
            Command::Capture { .. } | Command::ListCases { .. } => {
                unreachable!("handled before worker startup")
            }
        }
    };
    let result = tokio::select! {
        result = job => result,
        status = engine.wait() => {
            status.and_then(|()| {
                anyhow::ensure!(serving, "worker exited before the job completed");
                Ok(())
            })
        },
        signal = termination_signal() => {
            signal.and_then(|()| {
                anyhow::ensure!(serving, "job interrupted");
                Ok(())
            })
        },
    };
    engine.shutdown().await;
    result
}
