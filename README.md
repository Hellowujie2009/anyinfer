# AnyInfer

AnyInfer 的目标是用 Rust 管理单机多卡推理，用自维护的 Ascend C 算子执行 DeepSeek V4。当前原生交付可以单独构建、测试 HcPost 融合核，可以从 checkpoint 提取词表输出头，并可测量 Rust 连续批处理调度器的主机开销。

| 路径 | 当前实现 |
| --- | --- |
| `native/acl_backend/` | 四路 mHC 残差混合，BF16 输入输出、FP32 累加。附全元素校验、设备计时及 D2D 带宽对照 |
| `src/native/scheduler.rs` | 请求入队、分块预填充、逐 token 解码调度、槽位回收、取消、设备错误传播 |
| `native/model/`、`native/head_runner/` | Rust 读取 safetensors，导出单 token 词表头，运行编译好的词表分片 |
| `benchmark.sh` | 按编号分片测试，采集算子计时、流水线、内存计数器与 NPU 遥测，输出可回传的压缩包 |

原生版本尚缺完整 Transformer 层前向、压缩 KV cache 和多卡注意力、专家计算与通信的串联，当前无法独立从文本生成下一段文本。完整模型的在线、离线及端到端 benchmark 暂由显式的 vLLM 对照版提供。

普通 Linux 主机上可以运行 `bash build.sh --host-only`，再运行 `dist/anyinfer-host/benchmark.sh`，比较两种原生 Rust token 输出队列的调度尾延迟。结果包包含逐轮和合并的延迟分位数、样本量以及调度线程的操作系统计数。

在装好 CANN 开发套件的 Linux 编译机上，以目标芯片的准确型号生成部署包。部署时需要匹配的 CPU 架构、CANN 运行环境和驱动。

```bash
bash build.sh --soc Ascend910B1 --out /data/anyinfer-release
cd /data/anyinfer-release/anyinfer
./benchmark.sh --list
./benchmark.sh --device 0 --range 1-12 --results /data/feedback
```

构建、分片和回传的完整步骤见[原生使用手册](docs/user-manual.md)。完整仓库中另有 `build-reference.sh`，用于生成 vLLM 对照包。
