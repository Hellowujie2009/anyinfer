use std::collections::BTreeSet;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail, ensure};
use clap::{Args, ValueEnum};
use serde::Serialize;
use serde_json::{Value, json};
use tokio::process::{Child, Command};
use tokio::time::{sleep, timeout};

pub const MODEL_ALIAS: &str = "deepseek-v4";
const BOOTSTRAP: &str = include_str!("../runtime/bootstrap.py");

#[derive(Clone, Copy, Debug, ValueEnum, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Hardware {
    A2,
    A3,
    A5,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Frontend {
    Rust,
    Python,
}

#[derive(Args, Clone, Debug, Serialize)]
pub struct RuntimeConfig {
    /// Local V4/V4.1 checkpoint directory including weights and tokenizer.
    #[arg(long)]
    pub model: PathBuf,
    #[arg(long, value_enum, ignore_case = true)]
    pub hardware: Hardware,
    /// OpenAI HTTP, tokenizer and streaming implementation in the pinned runtime.
    #[arg(long, value_enum, env = "ANYINFER_FRONTEND", default_value = "rust")]
    pub frontend: Frontend,
    /// Comma-separated local Ascend runtime device IDs. Include both dies on dual-die systems.
    #[arg(long, default_value = "0,1,2,3,4,5,6,7")]
    pub devices: String,
    /// TP ranks per DP replica. The default follows the hardware profile.
    #[arg(long)]
    pub tensor_parallel_size: Option<usize>,
    /// Total prompt and generated tokens permitted per sequence.
    #[arg(long, default_value_t = 32768, value_parser = clap::value_parser!(u32).range(1..))]
    pub max_model_len: u32,
    #[arg(long, default_value_t = 8192, value_parser = clap::value_parser!(u32).range(1..))]
    pub max_num_batched_tokens: u32,
    #[arg(long, default_value_t = 64, value_parser = clap::value_parser!(u32).range(1..))]
    pub max_num_seqs: u32,
    #[arg(long, default_value_t = 0.9)]
    pub memory_utilization: f64,
    /// Include individual server-side timings in responses from the Python frontend.
    #[arg(long)]
    pub per_request_metrics: bool,
    /// Number of draft tokens to verify when using a checkpoint with embedded DSpark weights.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub dspark_tokens: Option<u32>,
    #[arg(long, default_value_t = 3600, value_parser = clap::value_parser!(u64).range(1..=86400))]
    pub startup_timeout_secs: u64,
}

struct Topology {
    devices: String,
    count: usize,
    tp: usize,
    dp: usize,
}

fn topology(hardware: Hardware, devices: &str, tp: Option<usize>) -> Result<Topology> {
    let ids: Vec<u32> = devices
        .split(',')
        .map(|s| {
            s.trim()
                .parse()
                .context("--devices must contain numeric device IDs")
        })
        .collect::<Result<_>>()?;
    let count = ids.len();
    ensure!(
        count == ids.iter().collect::<BTreeSet<_>>().len(),
        "--devices repeats a device ID"
    );
    let tp = tp.unwrap_or(match hardware {
        Hardware::A2 => count,
        Hardware::A3 => count.min(4),
        Hardware::A5 => 1,
    });
    ensure!(
        tp > 0 && count.is_multiple_of(tp),
        "--tensor-parallel-size must divide the number of devices"
    );
    Ok(Topology {
        devices: ids.iter().map(u32::to_string).collect::<Vec<_>>().join(","),
        count,
        tp,
        dp: count / tp,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Model {
    V4,
    V41,
}

fn model_family(config: &Value, max_len: u32) -> Result<Model> {
    let (family, architecture) = match config["model_type"].as_str() {
        Some("deepseek_v4") => (Model::V4, "DeepseekV4ForCausalLM"),
        Some("deepseek_v41" | "deepseek_v4.1") => (Model::V41, "DeepseekV41ForCausalLM"),
        other => bail!("expected a DeepSeek V4 checkpoint, found model_type {other:?}"),
    };
    ensure!(
        config["architectures"]
            .as_array()
            .is_some_and(|a| a.iter().any(|v| v == architecture)),
        "checkpoint must declare {architecture}"
    );
    let text = config.get("text_config").unwrap_or(config);
    if let Some(limit) = text["max_position_embeddings"].as_u64() {
        ensure!(
            u64::from(max_len) <= limit,
            "--max-model-len exceeds checkpoint limit {limit}"
        );
    }
    Ok(family)
}

impl RuntimeConfig {
    fn worker_command(
        &self,
        listen: SocketAddr,
        instance_alias: &str,
        bench_run: Option<&Path>,
    ) -> Result<Command> {
        ensure!(
            !self.per_request_metrics || self.frontend == Frontend::Python,
            "per-request server timing responses require --frontend python in vLLM 0.30.0"
        );
        ensure!(
            self.memory_utilization > 0.0 && self.memory_utilization <= 1.0,
            "--memory-utilization must be in (0, 1]"
        );
        ensure!(
            self.max_num_seqs <= self.max_num_batched_tokens,
            "--max-num-seqs exceeds the batched token budget"
        );
        let model_dir = std::fs::canonicalize(&self.model)
            .with_context(|| format!("model directory {}", self.model.display()))?;
        let config: Value = serde_json::from_slice(
            &std::fs::read(model_dir.join("config.json")).context("read model config.json")?,
        )?;
        let family = model_family(&config, self.max_model_len)?;
        let topology = topology(self.hardware, &self.devices, self.tensor_parallel_size)?;
        let hardware = self.hardware.to_possible_value().unwrap();
        let mut cmd = Command::new("python3");
        cmd.kill_on_drop(true)
            .stdin(Stdio::null())
            .args(["-c", BOOTSTRAP, hardware.get_name()])
            .arg(topology.count.to_string())
            .arg(&model_dir)
            .arg("--host")
            .arg(listen.ip().to_string())
            .arg("--port")
            .arg(listen.port().to_string())
            .args(["--served-model-name", MODEL_ALIAS, instance_alias])
            .arg("--tensor-parallel-size")
            .arg(topology.tp.to_string())
            .arg("--data-parallel-size")
            .arg(topology.dp.to_string())
            .arg("--data-parallel-size-local")
            .arg(topology.dp.to_string())
            .args([
                "--distributed-executor-backend",
                "mp",
                "--enable-expert-parallel",
                "--enable-chunked-prefill",
            ])
            .arg("--max-model-len")
            .arg(self.max_model_len.to_string())
            .arg("--max-num-batched-tokens")
            .arg(self.max_num_batched_tokens.to_string())
            .arg("--max-num-seqs")
            .arg(self.max_num_seqs.to_string())
            .arg("--gpu-memory-utilization")
            .arg(self.memory_utilization.to_string())
            .args([
                "--compilation-config",
                r#"{"cudagraph_mode":"FULL_DECODE_ONLY"}"#,
            ])
            .env("ASCEND_RT_VISIBLE_DEVICES", &topology.devices)
            .env("HF_HUB_OFFLINE", "1")
            .env(
                "VLLM_USE_RUST_FRONTEND",
                if self.frontend == Frontend::Rust {
                    "1"
                } else {
                    "0"
                },
            )
            .env_remove("VLLM_RUST_FRONTEND_PATH")
            .env("VLLM_NO_USAGE_STATS", "1")
            .env("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
            .env(
                "VLLM_ENGINE_READY_TIMEOUT_S",
                self.startup_timeout_secs.to_string(),
            )
            .env("OMP_PROC_BIND", "false")
            .env("OMP_NUM_THREADS", "10")
            .env("TASK_QUEUE_ENABLE", "1");
        #[cfg(unix)]
        cmd.process_group(0);
        let a5 = self.hardware == Hardware::A5;
        let tokenizer = if family == Model::V41 {
            "deepseek_v41"
        } else {
            "deepseek_v4"
        };
        let block = if self.hardware == Hardware::A2 || family == Model::V41 {
            "128"
        } else {
            "32"
        };
        cmd.args([
            "--tokenizer-mode",
            tokenizer,
            "--reasoning-parser",
            tokenizer,
            "--block-size",
            block,
        ])
        .env("VLLM_USE_V2_MODEL_RUNNER", if a5 { "1" } else { "0" })
        .env(
            "HCCL_OP_EXPANSION_MODE",
            if a5 { "CCU_SCHED" } else { "AIV" },
        )
        .env(
            "HCCL_BUFFSIZE",
            if a5 && family == Model::V41 {
                "1600"
            } else {
                "1024"
            },
        );
        if !a5 {
            cmd.args(["--quantization", "ascend"]);
        }
        // v0.30 colocated recipes. DSA-CP targets runner V1 with TP sharding.
        let mut extra = json!({"enable_cpu_binding":true});
        if !a5 {
            extra["ascend_compilation_config"] =
                json!({"enable_npugraph_ex":true,"enable_static_kernel":false});
        }
        if topology.tp > 1
            && (self.hardware == Hardware::A3
                || (self.hardware == Hardware::A2
                    && family == Model::V4
                    && self.dspark_tokens.is_none()))
        {
            extra["enable_dsa_cp"] = json!(true);
        }
        if self.hardware == Hardware::A3 && family == Model::V41 {
            extra["enable_fused_mc2"] = json!(1);
            extra["enable_flashcomm1"] = json!(true);
            extra["enable_shared_expert_dp"] = json!(true);
        } else {
            extra["multistream_overlap_shared_expert"] = json!(true);
        }
        if a5 && family == Model::V41 {
            extra["combine_quant_mode"] = json!(4);
            extra["multistream_engram_overlap"] = json!(true);
        }
        cmd.arg("--additional-config").arg(extra.to_string());
        if family == Model::V4 {
            cmd.args([
                "--attention_config.indexer_kv_dtype",
                if a5 { "fp8" } else { "int8" },
            ]);
            if self.hardware != Hardware::A3 {
                cmd.arg("--no-disable-hybrid-kv-cache-manager");
            }
            if a5 {
                cmd.args(["--kv-cache-dtype", "fp8"]);
            }
        } else {
            cmd.args([
                "--engram-config",
                r#"{"cpu_offload":true,"dp_shared_memory":true}"#,
            ]);
            if a5 {
                cmd.args(["--quantization", "deepseek_v4_fp8", "--trust-remote-code"]);
            } else {
                cmd.args(["--safetensors-load-strategy", "lazy", "--async-scheduling"]);
            }
        }
        if let Some(tokens) = self.dspark_tokens {
            let text = config.get("text_config").unwrap_or(&config);
            let block_size = text["dspark_block_size"]
                .as_u64()
                .filter(|n| *n > 0)
                .context("--dspark-tokens requires a checkpoint containing DSpark draft weights")?;
            ensure!(
                u64::from(tokens) >= block_size,
                "--dspark-tokens must be >= the draft block size ({block_size})"
            );
            cmd.arg("--speculative-config").arg(
                json!({
                    "method":"dspark", "num_speculative_tokens": tokens, "enforce_eager": !a5
                })
                .to_string(),
            );
        }
        if self.per_request_metrics {
            cmd.arg("--enable-per-request-metrics");
        }
        if let Some(path) = bench_run {
            let dir = std::fs::canonicalize(path)?;
            // Local workers are spawned in fresh Python interpreters. Install the
            // pinned profiling subclass in an importable, per-run module directory.
            let module = dir.join("worker");
            std::fs::create_dir(&module)?;
            std::fs::write(
                module.join("anyinfer_bench_worker.py"),
                include_str!("../runtime/bench_worker.py"),
            )?;
            let mut paths = vec![module];
            if let Some(existing) = std::env::var_os("PYTHONPATH") {
                paths.extend(std::env::split_paths(&existing));
            }
            cmd.env("PYTHONPATH", std::env::join_paths(paths)?)
                .env("ANYINFER_BENCH_RUN", &dir)
                .args(["--worker-cls", "anyinfer_bench_worker.BenchmarkWorker"]);
            if std::env::var_os("VLLM_RPC_TIMEOUT").is_none() {
                // Worker stop returns after CANN has exported all rank traces.
                cmd.env("VLLM_RPC_TIMEOUT", "1800000");
            }
            cmd.arg("--profiler-config").arg(
                json!({
                    "profiler": "torch", "torch_profiler_dir": dir, "ignore_frontend": true,
                    "torch_profiler_with_stack": false, "torch_profiler_record_shapes": true
                })
                .to_string(),
            );
        }
        eprintln!(
            "Starting {family:?} with {:?} frontend on {:?} devices {} with TP={} DP={} EP={} context={} batch_tokens={} max_seqs={}",
            self.frontend,
            self.hardware,
            topology.devices,
            topology.tp,
            topology.dp,
            topology.count,
            self.max_model_len,
            self.max_num_batched_tokens,
            self.max_num_seqs
        );
        Ok(cmd)
    }
}

pub struct Runtime {
    pub base_url: String,
    child: Child,
    #[cfg(unix)]
    group: Option<i32>,
}

impl Runtime {
    pub async fn start(cfg: &RuntimeConfig) -> Result<Self> {
        Self::start_on(cfg, (Ipv4Addr::LOCALHOST, 0).into()).await
    }

    pub async fn start_on(cfg: &RuntimeConfig, listen: SocketAddr) -> Result<Self> {
        Self::launch(cfg, listen, None).await
    }

    pub async fn start_benchmark(cfg: &RuntimeConfig, run_dir: &Path) -> Result<Self> {
        Self::launch(cfg, (Ipv4Addr::LOCALHOST, 0).into(), Some(run_dir)).await
    }

    async fn launch(
        cfg: &RuntimeConfig,
        listen: SocketAddr,
        bench_run: Option<&Path>,
    ) -> Result<Self> {
        let reservation = TcpListener::bind(listen).with_context(|| format!("bind {listen}"))?;
        let listen = reservation.local_addr()?;
        let mut probe = listen;
        if listen.ip().is_unspecified() {
            probe.set_ip(if listen.is_ipv4() {
                IpAddr::V4(Ipv4Addr::LOCALHOST)
            } else {
                IpAddr::V6(Ipv6Addr::LOCALHOST)
            });
        }
        let base_url = format!("http://{probe}/v1");
        // The worker must rebind the port. A per-launch alias prevents a race from
        // attributing a concurrent process's readiness (or metrics) to this run.
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let instance_alias = format!("anyinfer-{}-{stamp}", std::process::id());
        let mut command = cfg.worker_command(listen, &instance_alias, bench_run)?;
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(3))
            .build()?;
        drop(reservation);
        let child = command
            .spawn()
            .context("start Python from the vLLM-Ascend release image")?;
        let mut runtime = Self {
            base_url,
            #[cfg(unix)]
            group: child.id().map(|id| id as i32),
            child,
        };
        let ready = async {
            let url = format!("{}/models", runtime.base_url);
            loop {
                if let Some(status) = runtime.child.try_wait()? {
                    bail!(
                        "Ascend worker exited during model load ({status}), inspect its log above"
                    );
                }
                if let Ok(response) = client.get(&url).send().await
                    && response.status().is_success()
                    && let Ok(models) = response.json::<Value>().await
                    && let Some(card) = models["data"]
                        .as_array()
                        .and_then(|a| a.iter().find(|v| v["id"] == instance_alias))
                {
                    ensure!(
                        cfg.frontend != Frontend::Rust || card["owned_by"] == "vllm-frontend-rs",
                        "model loaded under a different frontend than requested: {}",
                        card["owned_by"]
                    );
                    return Ok(());
                }
                sleep(Duration::from_millis(500)).await;
            }
        };
        timeout(Duration::from_secs(cfg.startup_timeout_secs), ready)
            .await
            .context("timed out waiting for Ascend model load")??;
        Ok(runtime)
    }

    pub async fn wait(&mut self) -> Result<()> {
        let status = self.child.wait().await?;
        ensure!(status.success(), "Ascend worker exited ({status})");
        Ok(())
    }

    pub async fn shutdown(&mut self) {
        #[cfg(unix)]
        if let Some(group) = self.group {
            // A dedicated process group includes the local TP and DP worker ranks.
            unsafe { libc::kill(-group, libc::SIGTERM) };
            let _ = timeout(Duration::from_secs(10), self.child.wait()).await;
        }
        self.kill_group();
        let _ = self.child.wait().await;
    }

    fn kill_group(&mut self) {
        #[cfg(unix)]
        if let Some(group) = self.group.take() {
            unsafe { libc::kill(-group, libc::SIGKILL) };
        }
        let _ = self.child.start_kill();
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        self.kill_group();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_node_topologies() {
        let a2 = topology(Hardware::A2, "0,1,2,3,4,5,6,7", None).unwrap();
        let a3 = topology(Hardware::A3, "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15", None).unwrap();
        let a5 = topology(Hardware::A5, "0,1,2,3,4,5,6,7", None).unwrap();
        assert_eq!((a2.tp, a2.dp), (8, 1));
        assert_eq!((a3.tp, a3.dp), (4, 4));
        assert_eq!((a5.tp, a5.dp), (1, 8));
        for devices in ["", "1,1", "0,-1", "a,1"] {
            assert!(topology(Hardware::A2, devices, None).is_err());
        }
        assert!(topology(Hardware::A3, "0,1", Some(3)).is_err());
    }

    #[test]
    fn validates_architecture_and_context() {
        let config = json!({"model_type":"deepseek_v41","architectures":["DeepseekV41ForCausalLM"],
            "text_config":{"max_position_embeddings":1024}});
        assert_eq!(model_family(&config, 1024).unwrap(), Model::V41);
        assert!(model_family(&config, 1025).is_err());
        assert!(model_family(&json!({"model_type":"deepseek_v3"}), 32).is_err());
    }
}
