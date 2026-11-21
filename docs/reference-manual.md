# vLLM 对照版构建、分片压测与回传手册

一轮实验涉及三份文件。编译机接收 `anyinfer-source-*.tar.gz` 完整工程源码包，测试机接收构建输出的 `anyinfer/` 整个部署目录，每台测试机向编码环境回传一个 `bench-*.tar.gz` 结果包。测试机可以离线运行。每台机器独立加载一份模型，并用本机的昇腾卡执行分给它的用例。

## 一、在编译机生成部署目录

准备一台 Linux 编译机，安装 Bash、Python 3.10+、GNU coreutils、tar、gzip 和支持 BuildKit 的 Docker Engine。首次构建需要能获取项目锁定的 Rust 构建镜像、对应昇腾镜像和 Cargo 依赖。内网编译机可预置镜像并使用 `--local-images`，同时准备 Cargo 缓存或可访问的内部镜像源。CANN 编译工具由对应的构建镜像提供。为镜像缓存、展开目录和压缩包预留磁盘空间。

将本轮**完整工程源码包和同名校验文件**放入一个空目录，然后执行下列命令。示例为 ARM64 的 A2 机群，`/data/anyinfer-release` 可换成自己的输出位置。

```bash
sha256sum -c anyinfer-source-*.tar.gz.sha256
tar -xzf anyinfer-source-*.tar.gz
cd anyinfer
bash build-reference.sh --profiles a2 --platform linux/arm64 --out /data/anyinfer-release
```

**成功后可直接分发的目录是 `/data/anyinfer-release/anyinfer/`**，其中有可执行的 `benchmark.sh`、引擎、镜像、用例、分析工具、可编辑的算子源码和本手册。构建脚本会测试并编译 Rust 程序、从随包源码编译指定机型的算子，然后校验产物。请以脚本成功退出作为本轮发布完成的依据。在同一输出路径再次发布时，上一轮部署目录保存在同级的 `previous-anyinfer-*` 下。编译失败时检查该次构建日志，已有部署目录可能仍是上一轮版本。

| 部署目标 | 构建参数 |
| --- | --- |
| 与编译机 CPU 架构相同，覆盖 A2、A3、A5 | `bash build-reference.sh --out /data/anyinfer-release` |
| 仅 A3，x86_64 测试机 | `--profiles a3 --platform linux/amd64` |
| A2 和 A5，测试机有两种 CPU 架构 | `--profiles a2,a5 --platform all`，编译机的 Docker 须能运行两种架构的容器 |

构建器会在 `/data/anyinfer-release/` 下生成部署压缩包和 SHA-256，与展开的 `anyinfer/` 并列。也可以按你的工作流，自己把上面的**整个目录**打包。在编译机执行以下命令。

```bash
mkdir -p /data/transfer
tar -C /data/anyinfer-release -czf /data/transfer/anyinfer-round01.tar.gz anyinfer
(cd /data/transfer && sha256sum anyinfer-round01.tar.gz > anyinfer-round01.tar.gz.sha256)
```

将这个压缩包和校验文件复制到各测试机。构建输出中的 `anyinfer-operators-*.tar.gz` 是独立算子源码交付件，适合单独审阅或重建算子。

## 二、在多台测试机并行运行

每台机准备与包相符的 CPU 架构、昇腾机型和驱动、可运行容器的 Docker、可用的 `npu-smi`，以及容纳所选权重和 KV 缓存的本机多卡。将对应 DeepSeek V4 权重、配置和 tokenizer 放在本地目录。示例使用 A2 的 V4-Flash W8A8 权重。权重需独立分发。运行机从包中加载镜像，不需要 Rust、CANN 编译器或源码构建步骤。

在每台机器的空目录校验、解压并查看本次分片。

```bash
sha256sum -c anyinfer-round01.tar.gz.sha256
tar -xzf anyinfer-round01.tar.gz
cd anyinfer
./benchmark.sh --list --range @survey
```

列表命令使用随包可执行文件，无需加载模型。首次性能摸底用 `--range @survey`，它选取 8 个代表用例。需要深挖时按 `@prefill`、`@decode`、`@mixed`、`@prefix`、`@arrival` 等负载族筛选，也可以按任意编号分片，例如 `--range 100-200` 包含两端。常规目录共 242 条，省略范围会运行整个目录。另附 64K 到 1M 上下文的 10 条用例，应按权重配置和设备容量选用。[用例覆盖、负载定义和超长上下文命令](workloads.md)

下面三条分别在三台 A2 上运行，每台使用**同一轮构建和同一份权重**。在每台机的部署目录先设置变量，再执行分配给该机的那一行。

```bash
MODEL=/models/DeepSeek-V4-Flash-0731-w8a8
RESULTS=/data/anyinfer-results/round01

# 机器 1，定位长输入瓶颈
./benchmark.sh --hardware a2 --model "$MODEL" --range @prefill --results "$RESULTS"
# 机器 2，观测持续解码
./benchmark.sh --hardware a2 --model "$MODEL" --range @decode --profile --results "$RESULTS"
# 机器 3，观测缓存和突发流量
./benchmark.sh --hardware a2 --model "$MODEL" --range @prefix,@arrival --results "$RESULTS"
```

| 要做的实验 | 在 benchmark 命令中加的参数 |
| --- | --- |
| 指定本机参与的设备 | `--devices 0,1,2,3,4,5,6,7`，省略时自动探测 |
| 验证生成精度再计时 | `--golden /data/reference/approved.jsonl`，各分片使用同一份已认可的参考输出 |
| 分析长解码稳态 | `--profile`，将每个用例的算子采样从短重放改为完整负载重放 |
| 缩短采样或单独诊断一组计数器 | `-- --aic-metrics memory` 或 `-- --aic-metrics pipe`。默认 `both` 分两次重放取数，引擎参数放在最后一个 `--` 后面 |
| 跳过本轮硬件参照测量 | `--skip-calibration`。本轮失去本机实测分母，仍保留端到端计时和算子原始指标 |
| 运行自己的用例 | 加 `--cases /data/cases.json --range 100-200`。自定义文件须包含所选的每个 ID，可先用 `./benchmark.sh --list --cases /data/cases.json --range 100-200` 核对 |

每台机只加载一次模型。默认优先使用镜像内的 Rust 服务前端。若镜像缺少对应可执行文件，启动器会提示并选用 Python 服务前端，实际选择见运行日志。每个用例先预热并按设定的并发持续发送请求，这段给出端到端成绩。随后同一模型独立重放负载，逐 rank 采集设备 trace。默认重放保留输入形状、限制请求数，最多解码 8 个 token。长分片和完整重放会产生更多 trace，请把 `--results` 指向有足够空间的目录。

### 建立可复用的精度参考

在已经通过质量验收的参考部署上运行下列命令。终端会打印本次独立结果目录，将其中的 `golden.jsonl` 保存为 `/data/reference/approved.jsonl` 并分发。后续每轮 benchmark 都加上 `--golden /data/reference/approved.jsonl`，验证通过后才进行计时。首次摸底可以先跳过此项，精度状态会记录为 `unchecked`。

```bash
./run.sh accuracy --hardware a2 --model "$MODEL" --results /data/reference \
  -- --capture-prompts /app/benches/accuracy-prompts.jsonl \
  --reference-name flash-w8a8-approved --output /results/golden.jsonl
```

## 三、将结果交给 AI，开始下一轮

每个分片结束时，终端显示 `Feedback archive:` 和完整路径。**把各机的 `bench-<机型>-<UTC时间>-<随机串>.tar.gz` 直接回传即可。** 同级的 `.tar.gz.sha256` 可用于校验传输。包内的 `feedback.json` 可以先看结果和热点，原始记录可进一步用于归因。同一台机重跑也会生成独立名称。发生启动、采样或推理错误时，已建立结果目录的运行会保留已有数据，并以非零状态退出。若进程被强制终止而来不及压缩，可打包对应的原始结果目录。

| 回传包中的内容 | 下一轮可据此判断 |
| --- | --- |
| 用例清单、命令、权重元数据、构建清单、镜像与算子源码指纹 | 测的哪一版，哪些运行可以对照 |
| `hardware-calibration.json`、`operator-smoke.json` 中的 `microbenchmark` | 每卡实测的大矩阵算力、逻辑拷贝带宽，以及第一张卡上 HcPost 用当前模型隐藏维度跑 5 种 token 数的耗时和数值核对 |
| `telemetry/samples.log`、`polls.tsv`，各用例的 `engine-gauges.jsonl` 和计时前后的 `.prom` | 设备端每次查询后等待 2 秒再采样，记录查询是否成功。计时阶段按 1 秒周期轮询引擎，观察 NPU 利用率与内存、排队、KV 占用和抢占 |
| `bench/run-*/case-*/summary.json`、`requests.jsonl` | 吞吐、首 token 延迟、每输出 token 延迟、端到端尾延迟和失败请求 |
| 同目录的 `operators.json`、`pmu-memory/operators.json`、`operator-profile.json` 及各自的 `traces/rank-*/` | 各次重放分别在哪些 shape 和 rank 耗时，有效速率与本机参照的关系，流水线忙碌程度、设备任务重叠、通信等待及可检查的融合位置 |
| `operator-load.json`、`operator-smoke.json`、已指定参考时的 `accuracy.jsonl` | 实际加载的算子库、HcPost 数值检查和模型输出回归 |

首轮回传实际运行的各分片。后续回传候选版和基线版中对应的分片，并说明目标，例如“改善长上下文预填充，同时守住解码 P99 和精度”。判断一个核离上限多远，要先核对数据类型、物理设备、计数器单位和可确认的工作量。对稀疏注意力、MoE 或融合算子，可结合卡上的流水线、访存、通信等待及相邻任务判断瓶颈。[算子读数与适用边界](operator-performance.md)

需要自己查看时，可在编码机把分片结果解压到 `/data/feedback`，然后在工程目录执行以下命令。

```bash
python3 tools/report.py /data/feedback > round-summary.json
# BASE_CASE 与 NEW_CASE 各为两次实际运行中的同一 case 目录，例如 case-150
python3 tools/report.py --compare-cases "$BASE_CASE" "$NEW_CASE" > case-delta.json
```

下一轮修改引擎或 `operators/` 中的热点实现后，可在编码环境执行 `python tools/package_source.py --out dist`，输出包含新代码和用例的完整源码包及校验文件。将这两个文件送回编译机，从第一步重复。模型路径和算子源码的对应关系见[算子开发说明](operators.md)，机型、镜像和驱动挂载细节见[部署说明](package.md)。
