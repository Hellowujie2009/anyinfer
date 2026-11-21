#[path = "../native/head_runner/mod.rs"]
mod head_runner;
mod host_observation;
mod opbench;
mod scheduler_bench;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    version,
    about = "Native DeepSeek V4 checkpoint tools and Ascend kernel benchmarks"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compute trusted head logits from a BF16 residual and checkpoint weights on the CPU.
    GoldenHead {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        residual: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Measure host-side continuous batching bookkeeping with an empty device dispatch.
    SchedulerBench,
    /// Evaluate the compiled vocabulary head shards on local Ascend devices.
    RunHead(head_runner::Options),
    /// Read the checkpoint index, validate tensor metadata and show mHC dimensions.
    Inspect {
        #[arg(long)]
        model: PathBuf,
    },
    /// Export the weighted V4 final head for one vocabulary shard, including the BF16 rounding boundaries.
    PrepareHead {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 0)]
        rank: u32,
        #[arg(long, default_value_t = 1)]
        world_size: u32,
    },
    /// List native kernel workloads and their stable IDs.
    ListCases {
        #[arg(long, default_value = "all")]
        range: String,
    },
    /// Run the native Ascend kernel suite on one selected device.
    Bench(opbench::Args),
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::GoldenHead {
            model,
            residual,
            output,
        } => {
            let report = anyinfer::native::model::golden_head(&model, &residual, &output)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::SchedulerBench => println!(
            "{}",
            serde_json::to_string_pretty(&host_observation::report()?)?
        ),
        Command::RunHead(options) => head_runner::run(options)?,
        Command::Inspect { model } => {
            let report = anyinfer::native::model::inspect(&model)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::PrepareHead {
            model,
            output,
            rank,
            world_size,
        } => {
            let report = anyinfer::native::model::prepare_head(&model, &output, rank, world_size)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::ListCases { range } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&opbench::select(&range)?)?
            );
        }
        Command::Bench(args) => opbench::run(&args)?,
    }
    Ok(())
}
