# AnyInfer

单机多卡 DeepSeek V4 推理与性能实验工具。部署包优先启用 Rust 服务前端，负责 HTTP、分词、流式响应，并通过本机 IPC 连接模型执行进程。AnyInfer 的 Rust 代码负责机型与并行拓扑、进程管理、离线并发生成、精度回归和性能采集。模型前向和连续批处理复用 vLLM-Ascend 0.30.0，调度与模型编排仍有 Python，矩阵计算、稀疏注意力、MoE 和集合通信走昇腾算子、图执行和 HCCL。客户端直连服务端口。

第一次交付请按[构建、分片压测与回传手册](docs/user-manual.md)操作。构建入口是 `build.sh`，测试入口是部署目录内的 `benchmark.sh`，每台测试机回传一个 `bench-*.tar.gz`。

## 算子源码与交付

算子源码随源码包和部署包交付。固定版本的 Ascend C 内核、host tiling、PyTorch 绑定、Triton 内核以及 DeepSeek V4 的模型调用点在 `operators/vllm-ascend/`。例如稀疏注意力从 `csrc/attention/sparse_flash_mla/` 查看，压缩算子从 `csrc/attention/compressor/` 查看。修改后重新运行构建脚本，生成对应机型的算子二进制与运行镜像。源码清单及上游版本用于把回传结果定位到具体实现。HCCL、驱动和 CANN 内置算子的实现由对应发行版提供。[算子开发说明](docs/operators.md)

## 第一轮实测

在 Linux 编译机上从项目根目录执行。默认按编译机 CPU 架构生成软件包，包含 A2、A3、A5 的运行镜像、Rust 可执行文件和基准用例。每台目标机准备匹配的驱动、Docker 和本地权重。按 CPU 架构构建的命令、镜像空间和挂载要求见[打包与部署说明](docs/package.md)。

```bash
bash build.sh
# 单独打出可编辑的算子源码、许可证、依赖和编译入口
bash build.sh --sources-only
# 一份包同时用于 ARM64、x86_64 主机，构建环境需支持运行两种架构的镜像
bash build.sh --platform all
```

完整的可搬运目录为 `dist/anyinfer/`，使用 `--out DIR` 时为 `DIR/anyinfer/`。可以自己将整个目录打包，或直接使用脚本生成的部署归档和校验文件。在目标机解压后的 `anyinfer/` 目录执行下面的命令。两台机器可以同时运行，同一轮的两份回传包都带有构建标识和用例清单。

```bash
# 先在第一台 A2 用代表性负载定位热点
./benchmark.sh --list --range @survey
./benchmark.sh --hardware a2 --model /models/DeepSeek-V4-Flash-0731-w8a8 --range @survey --results /data/runs

# 第二台同配置 A2 专门测稳态解码，用完整重放观察后段内核
./benchmark.sh --hardware a2 --model /models/DeepSeek-V4-Flash-0731-w8a8 --range @decode --profile --results /data/runs
```

常规目录有 242 个用例，另附 10 个需单独选择的超长上下文用例。可以按实验目的、单个编号或任意编号区间选择。每个分片只加载一次模型。**将每台机器打印的 `bench-*.tar.gz` 交给 AI 即可**，内含同机算力和带宽参照、请求记录、引擎指标、逐 rank 算子报告与原始 trace。[用例与选取方法](docs/workloads.md) · [如何判断算子瓶颈](docs/operator-performance.md)

## 选择权重和机器

| 目标 | 单机起点 |
| --- | --- |
| A2 | V4-Flash-0731 的昇腾 W8A8 权重。官方单机配置为 8 × 64 GB。 |
| A3 | V4-Flash-0731 的昇腾 W8A8 权重。官方单机配置为 8 × 128 GB。双 Die 机器按逻辑设备编号，启动器默认探测本机设备。 |
| A5 | 950DT 单机使用适配该镜像的原生混合精度 V4-Flash 权重。 |

V4.1-Flash 使用自己的 tokenizer 和 Engram 配置，官方单机示例覆盖 A3 和 A5。V4-Pro 走 V4 执行路径，其官方参考部署需要多台常规服务器。Pro 单机运行取决于权重格式、实际本机容量和 KV 预算，当前没有单机实测数据。选购或分配硬件时，按具体权重版本核对容量。[V4-Flash 部署配置](https://docs.vllm.ai/projects/ascend/en/v0.30.0/tutorials/models/DeepSeek-V4-Flash.html) · [V4.1-Flash 部署配置](https://docs.vllm.ai/projects/ascend/en/v0.30.0/tutorials/models/DeepSeek-V4.1-Flash.html) · [V4-Pro 部署配置](https://docs.vllm.ai/projects/ascend/en/v0.30.0/tutorials/models/DeepSeek-V4-Pro.html)

## 在线与离线推理

启动在线接口后，可使用流式 chat completions，也可以直接调用 `/v1/completions` 和 `/v1/models`。服务对外使用固定模型名 `deepseek-v4`。

```bash
./run.sh serve --hardware a2 --model /models/DeepSeek-V4-Flash-0731-w8a8 -- --listen 127.0.0.1:8000
```

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4","messages":[{"role":"user","content":"写一个 Rust 迭代器示例。"}],"max_tokens":256,"stream":true}'
```

离线输入为 JSON Lines。每行提供 `prompt` 或 `messages`，可指定 `id`、`max_tokens`、`temperature`。

```jsonl
{"id":"test-1","prompt":"fn fibonacci(n: usize) -> usize {","max_tokens":128}
{"id":"test-2","messages":[{"role":"user","content":"解释 KV 缓存的作用。"}],"max_tokens":128}
```

```bash
./run.sh offline --hardware a2 --model /models/DeepSeek-V4-Flash-0731-w8a8 \
  --job-dir /data/jobs --results /data/runs -- \
  --input /job/input.jsonl --output /results/answers.jsonl --concurrency 16
```

结果随完成顺序逐行落盘，保留原始行号与 `id`。单行出错会记录错误并继续处理其他行，整批最终返回非零。输出文件采用新建模式。

## 用数据指导下一次修改

### 建立精度基线

随包提供预分词输入。第一次在已认可的参考模型上用贪心解码生成期望 token ID，记录具体权重版本、量化方案和运行时。参考结果随模型版本独立保存。也可以用源码构建的 `anyinfer capture --help` 连接已有的可信 HTTP 服务，它需要支持返回 token ID。

```bash
# 在已通过质量验收的参考部署上执行；这里的名称请替换为实际版本
./run.sh accuracy --hardware a2 --model /models/DeepSeek-V4-Flash-0731-w8a8 --results /data/reference -- \
  --capture-prompts /app/benches/accuracy-prompts.jsonl --reference-name flash0731-w8a8-approved \
  --output /results/golden.jsonl

# 将刚生成的 golden.jsonl 保存为 /data/reference/approved.jsonl，并分发给测试机
./run.sh --hardware a2 --model /models/DeepSeek-V4-Flash-0731-w8a8 --range 1-3 \
  --golden /data/reference/approved.jsonl --results /data/runs
```

传入 `--golden` 时，每个分片先在本次已加载的模型上逐 token 对比，全部通过后才开始计时。出现差异会保存首个偏离位置、预期 ID 和实际 ID。未提供基线的首轮可以用于摸底，汇总中的 `token_regression` 会标记为 `unchecked`。这组短样例覆盖生成回归，业务精度验收可将自己的预分词样本和独立参考输出加入同一语料。比较量化策略时也应使用覆盖目标任务的精度评测。

### 对照同机基准

固定形状用例直接发送可复现的 token ID，以服务端报告的 token 数核对输入与输出。实际文本用例经过 tokenizer，可用于观察编码任务下的吞吐与排队。各用例先预热，再按指定并发保持持续在途请求。用例的长度、并发、请求数及预热数见[用例定义](benches/cases.json)。

| 问题 | 首先查看 | 可验证的改动 |
| --- | --- | --- |
| 长输入的首 token 为什么慢 | TTFT、queue/prefill 时间、运行与等待队列、NPU 利用率 | 单独改变 `--max-num-batched-tokens`，重跑用例 3、6、7 |
| 高并发解码吞吐为何停滞 | 输出 tokens/s、TPOT、E2E P99、KV 占用、抢占计数 | 改变 `--max-num-seqs` 或 TP/DP 分配，再跑用例 4、5 |
| 时间花在哪些算子或通信上 | 查看 `case-N/operators.json`、各 rank 的时间线及通信记录 | 按输入 shape 和 dtype 找到热点，对照源码修改 tiling、访存或融合，再测相同用例 |

例如，以下命令在现有基线上改变单次迭代的 token 预算。每轮保持模型、权重、卡数和用例不变，再比较吞吐与尾延迟。带 profiler 的结果会标记采样模式。

```bash
./run.sh --hardware a2 --model /models/DeepSeek-V4-Flash-0731-w8a8 --range 3,6-7 \
  --results /data/experiment -- --max-num-batched-tokens 4096
```

服务前端可以在相同权重、拓扑、精度基线和用例下用 `-- --frontend rust` 与 `-- --frontend python` 做 A/B。Python 前端支持加 `--per-request-metrics` 诊断单个请求的服务端耗时，两种前端都保留客户端观测和引擎指标。若需要修改模型执行代码，一键构建流程支持后端源码覆盖和自定义镜像，详见[部署说明](docs/package.md)。

每个用例先完成未开启算子采样的端到端计时，再用同一模型独立重放采集算子时间。默认重放限制请求数，最多生成 8 个 token，实际负载写入 `operator_profile.replay`。研究长解码稳态时，可用 `--profile` 重放完整用例。各算子的调用次数、设备时间总和、平均、P50/P95、最大值、分 rank 时间及可用的硬件计数写入 `operators.json`。报告保留导出的 shape、dtype 和计数器单位，并记录每个计数器的有效样本数。跨流、跨卡的设备时间可以重叠，端到端收益请查看第一次计时得到的吞吐和延迟。[昇腾 profiler 字段说明](https://www.hiascend.com/document/detail/en/mindstudio/700/TITools/Profiling/atlasprofiling_16_1149.html)

默认 `--aic-metrics both` 会独立重放两次，依次采集流水线和访存计数器。两次重放采用相同负载配方和不同的输入种子，分别保留时间线。采集前在选中的每张 NPU 上实测 BF16 矩阵乘、可用的 W8A8 矩阵乘和大块拷贝作为参照。若指定访存计数却缺少数值，报告保留原始数据并标记采集不完整。也可只重放一个计数器组，例如以下命令。[torch-npu 采样配置](https://github.com/Ascend/pytorch/blob/v2.10.0-26.2.0/torch_npu/profiler/experimental_config.py)

```bash
./run.sh --hardware a2 --model /models/DeepSeek-V4-Flash-0731-w8a8 --range 5 \
  --golden /data/reference/approved.jsonl --results /data/memory -- --aic-metrics memory
```

查看 `source_candidates_by_exact_symbol` 能找到与 trace 同名的本地算子注册或 JIT 定义。相同名称也可能出现在 CANN 库中，进一步归因时需核对机型的模型分发分支、构建清单及原始 trace。`same_stream_neighbors` 列出同一流上相邻任务及间隔，可用来筛选值得检查的融合位置。融合前需验证数据依赖、精度和跨流同步。

```bash
python3 tools/report.py /data/unpacked-runs > runs.json
python3 tools/report.py --compare-cases /data/baseline/bench/run-AAA/case-5 /data/new/bench/run-BBB/case-5 > delta.json
```

对照工具检查用例、重放负载和 rank 数，并输出新增、消失或变慢的算子及原始端到端指标。对照前核对两侧的机型、权重和构建参数。客户端以收到首个含生成 token 的 SSE 事件计算 TTFT。同一事件可含多个 token，TPOT 按首个事件之后的 token 和耗时计算。默认用例的 P99 受样本量限制，正式比较尾延迟时可用 `--cases` 指定更高请求数。

## 本地开发

```bash
cargo test --locked
cargo build --release --locked
cargo run --locked -- bench --help
```

运行时命令和拓扑在 `src/runtime.rs`，请求生成和指标计算在 `src/bench.rs`，离线推理在 `src/offline.rs`。本地 CPU 测试覆盖参数、协议与统计逻辑。模型输出质量、设备兼容性和吞吐需在相应昇腾机器上用真实权重验证。
