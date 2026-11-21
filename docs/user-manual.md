# AnyInfer 原生算子构建与性能验证

当前可以独立测试 V4 的 HcPost 残差混合，另有词表输出头的导出和运行工具。整模型原生推理仍缺 attention、MoE、压缩缓存及跨卡通信。以下 benchmark 统计单个融合核及主机调度器，各卡可独立运行分片。

## 在普通 Linux 主机验证逐 token 调度

在源码根目录执行以下命令。有 Rust/Cargo 和系统链接器的 Linux 编译机就可以生成主机测试包。将包拷贝到 CPU 架构和 glibc 兼容的机器后也可以单独运行。

```bash
bash build.sh --host-only --out dist
dist/anyinfer-host/benchmark.sh --results "$PWD/results"
```

脚本生成 `bench-host-scheduler-*.tar.gz`。带设备核的部署包也接受 `./benchmark.sh --host-only`。两种方式都运行同一个 Rust 基准，主机测试不加载模型、Python 或 CANN。

`host-scheduler.json` 中的 `paired_decode.cases` 用 8 个活跃请求比较逐 token 发布到两种 Rust 输出队列的耗时。每个场景预热后跑 8 对，轮换两种实现的先后次序，每种实现采集 16,384 次稳态调度。消费者在同一线程每步读取，或每 64 步集中读取。看 `decode_tick` 的 P50、P99、P99.9、最大值和样本数，再对照 `submit` 与 `consumer_drain`。各对原始分位统计见 `runs`，P99.9 在样本不足 1,000 次时记为 null。`host` 记录 CPU 与亲和性，以及整个测试期间调度线程的上下文切换、缺页计数；这些计数不能将单次尖峰归因到特定事件。

预分配方式入队时按 `max_new_tokens × 4` 字节准备 token 存储，另有固定控制开销。多个请求按预算累加，慢消费者持有接收端时会延长内存占用时间。构建时的分配回归测试在预热后连续执行 4,096 个非终结解码步，每步发布并读取一个 token，检查同一线程上的 Rust 堆分配、扩容和释放次数均为零。测试计数器仅编入测试程序。端到端首 token 延迟和逐 token 延迟仍要在完整设备执行路径上测量。

## 1. 在编译机生成部署目录

使用与运行节点相同 CPU 架构和兼容 glibc 的 Linux 主机。准备 Rust/Cargo（本轮使用 Rust 1.100）、C/C++ 工具链、CMake、GNU coreutils、Bash、tar、gzip、flock，以及支持目标 SoC 的华为 CANN 开发套件。首次构建需要 Cargo 依赖缓存或网络。CANN 自身需要的工具按其安装要求配置。原生部署目录包含本项目的 Rust 和 Ascend C/C++ 源码。

```bash
sha256sum -c anyinfer-native-source-*.tar.gz.sha256
tar -xzf anyinfer-native-source-*.tar.gz
cd anyinfer
# 按实际安装路径加载编译环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh --soc Ascend910B1 --out /data/anyinfer-release
```

`--soc` 填目标卡的准确 CANN SoC 名称。多种芯片可重复传入此参数，前提是当前工具链支持每个目标。不同 CPU 架构或需要不同 CANN 版本时，分别构建。脚本会运行 Rust 测试，并编译、安装各 SoC 的设备核与主机代码。构建成功后会打印目录 `/data/anyinfer-release/anyinfer/` 和对应的可分发压缩包。

## 2. 在一台或多台测试机运行

测试机需要兼容的驱动、CANN 运行库和 `npu-smi`，默认计数器采集还需 `msprof`。可复制整个部署目录，也可使用构建好的压缩包。算子输入按形状可复现地生成，无需拷贝模型权重。下面三条运行命令可分别交给三台同类机器。

```bash
cd /data/anyinfer-release/anyinfer
./benchmark.sh --list
./benchmark.sh --device 0 --range 1-24  --results /data/feedback
./benchmark.sh --device 0 --range 25-48 --results /data/feedback
./benchmark.sh --device 0 --range 49-72 --results /data/feedback
```

支持逗号分隔的编号与闭区间，如 `--range 1,8,17-24`。省略范围会执行全部 72 个 HcPost 用例。三组隐藏维度分别为 4096、7168、5120，每组有从 1 到 2048 的 24 个 token 数，覆盖单 token、批处理边界附近及大预填充块。编号和实际形状见包内 `build/cases.json`。隐藏维度来自官方配置。[V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json) · [V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json) · [V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json)

一个分片使用一个逻辑设备。多 SoC 包用 `--soc` 选择对应核。同一主机上可以用不同设备并行执行分片，共用 CPU 和功耗预算时应记录并行运行条件。默认每个形状先预热，再采集 7 组，每组连续执行 8 次。随后分别重放采集 `PipeUtilization` 和 `Memory`。可用 `--profile pipe|memory|off` 选择单组或仅测耗时。各 CANN 版本、芯片和权限支持的 PMU 项不同，未获取到匹配的任务记录时脚本保留诊断并返回非零。[华为 msprof 采集参数](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0011.html)

## 3. 回传数据并选择下一轮改动

可直接在相同负载上改变隐藏维的分块长度。例如在同一张空闲卡上依次执行以下两条命令，然后比较两份结果中的设备耗时、实际 block 数、每核 UB 占用和两组计数器。省略 `--tile` 时由核按 token 数选择默认值。

```bash
./benchmark.sh --device 0 --range 1,13,24 --tile 128 --results /data/feedback
./benchmark.sh --device 0 --range 1,13,24 --tile 512 --results /data/feedback
```

每个分片输出 `bench-native-*.tar.gz` 及其 SHA-256 校验文件。运行中会持续更新结果目录，常规结束或测量失败时会打包。意外断电时可以从原始目录取回已落盘的数据。

| 文件 | 可判断的问题 |
| --- | --- |
| `status.json`、`benchmark/feedback.json` | 哪些形状完成、校验是否通过、哪些计数器采集成功。失败的运行还有 stderr |
| `benchmark/case-*/measurement.json` | 各组设备 Event 耗时、主机墙钟耗时、全量数值误差、逻辑字节数与有效 GB/s、同设备 D2D 参考带宽 |
| `benchmark/case-*/{PipeUtilization,Memory}/kernel-counters.json` | HcPost 逐任务原始计数器及单位。同目录 `trace/` 保留厂商输出 |
| `benchmark/npu-smi.jsonl` | 周期采集的整机设备信息、采样时刻、命令状态，可对照温度及其他负载 |
| `host-scheduler.json` | 空设备提交下三组请求形状的主机入队/调度开销及 token 统计 |
| `build/`、`command.txt` | 源码与二进制哈希、编译信息、形状表和运行选项 |

将结果压缩包连同轮次、机器型号一起交回编码环境。基线和候选版使用相同的 SoC、形状、采样设置和空闲条件。先看全元素精度和各组耗时分布，再结合计数器判断优先改善访存、核间分工还是向量计算。逻辑 GB/s 按输入输出最小流量计算。缓存命中、重复读取和 copy 引擎都影响它与 D2D 对照的比例，这个比值无法单独证明 HBM 或 Vector Core 已经饱和。端到端 TTFT、TPOT 和整机吞吐需要完整模型前向参与。

## 4. Checkpoint 与词表输出头

`bin/anyinfer inspect --model /models/DeepSeek-V4-Flash` 读取 checkpoint 元数据。`prepare-head` 按指定 rank 导出末层和所需权重，`golden-head` 从同一输入计算 CPU 参考 logits，`run-head` 执行已为本机编译的输出头分片。输入是到达最后一层的残差激活。[权重及精度参考](native-model.md) · [多卡输出头运行手册](../native/head_runner/README.md)
