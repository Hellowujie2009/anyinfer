# 在昇腾节点运行一轮性能实验

## 准备编译环境

在可以访问镜像仓库和 crates.io 的 Linux 构建机上运行以下命令。需要 Python 3.10+、Bash、启用 BuildKit 的 Docker Engine、GNU coreutils、tar 和 gzip。构建机可以没有 NPU。默认生成与构建机 CPU 架构匹配的包，在匹配 A2、A3、A5 的 CANN 工具链中分别构建算子。如果构建机和部署机的 CPU 架构不同，构建机的 Docker 需要已配置对应的 binfmt/QEMU。包的 CPU 架构须与目标机相同。完整操作示例见[使用手册](reference-manual.md)。

```bash
bash build-reference.sh
# 一份包交付给混合 CPU 机群，根目录的 run.sh 自动识别当前主机架构
bash build-reference.sh --platform all
# 只给 A2 测试机出包
bash build-reference.sh --profiles a2 --platform linux/arm64 --out dist-a2
# 给 x86_64 主机上的 A3 卡出包
bash build-reference.sh --profiles a3 --platform linux/amd64 --out dist-x86
# 只导出可编辑的算子源代码和编译入口，不需要 Docker
bash build-reference.sh --sources-only --out dist
```

脚本用锁定的 Cargo 依赖运行 release 测试、编译各运行时可执行的 musl 二进制，再在每个选定的镜像中启动二进制检查 CLI。构建成功会在输出路径发布可直接分发的 `anyinfer/` 目录，以及同级的 `anyinfer-<CPU>-<时间>-<随机串>.tar.gz` 和校验文件。目录内有 `benchmark.sh`、二进制、模型运行时镜像、用例、结果分析工具和构建清单。请为 Docker 镜像、展开目录、历史版本和最终压缩包预留磁盘空间。下一次构建复用同一构建机上的 Cargo 缓存卷。

`--platform all` 生成 `anyinfer-all-*.tar.gz`，内含 ARM64 和 x86_64 两份二进制以及最多六个运行时镜像。构建机要能执行两种 CPU 架构的容器。选择单一架构可以减少磁盘、传输和编译开销。如本地镜像按架构独立打标，可用 `ANYINFER_IMAGE_A2_ARM64`、`ANYINFER_IMAGE_A2_AMD64` 等变量分别指定，构建器也支持 `ANYINFER_BUILDER_IMAGE_ARM64` 和 `ANYINFER_BUILDER_IMAGE_AMD64`。

## 修改算子、模型执行或调度代码

算子源代码位于 `operators/vllm-ascend/csrc/`，包含 Ascend C 内核、tiling、算子注册和 PyTorch C++ 桥接。模型调用代码和 Triton 实现位于同一源码树下的 `vllm_ascend/`。例如，修改 HcPost 可从 `csrc/moe/hc_post/op_kernel/` 开始。仓库和解压后的完整软件包都包含可编辑源码、Catlass 头文件、锁定的构建依赖、许可与同一个 `build-reference.sh`。[算子源码与索引](operators.md)

每轮构建另行输出 `anyinfer-operators-<源码散列>-<批次>.tar.gz` 及 SHA-256 文件。这个源码归档含编译上下文、算子目录索引和原始源码，可独立传回编码环境。如果只需在编译机重建算子镜像，在解压后的 `anyinfer-operators/` 中执行 `./build-image.sh a2 linux/arm64`（可选 `a3`、`a5`、`linux/amd64`）。若要把算子、Rust 引擎和用例一起部署，使用完整工程或完整软件包根目录的 `bash build-reference.sh`。

再次运行 `bash build-reference.sh --profiles a2` 时，构建器校验本轮源码散列，使用匹配设备的 CANN 与 PyTorch 环境编译新 wheel，生成 Torch 扩展、vendor 自定义算子包和设备内核对象，随后将这些文件安装至镜像的 `/opt/anyinfer/operators/python`。每种硬件使用独立的构建产物；算子编译或产物检查失败会停止出包。构建清单记录基础镜像、编译器版本、原始代码、wheel 与已安装文件的 SHA-256。修改源码会使对应的 Docker 构建缓存失效，源码没有变化的迭代可复用已编译层。

目标机启动时核对库文件哈希、导入路径、CANN 的 OPP 搜索顺序和 NPU 算子注册，并用第一张选定的物理设备执行 HcPost，对照 CPU 参考输出。检查失败会在加载模型之前退出。算子级冒烟结果与端到端 golden 测试分别保存在结果目录。前者验证桥接和内核调用，后者检查模型生成。

把改过的后端文件按原包的模块路径放入源码树的 `backend-overlay/vllm/` 或 `backend-overlay/vllm_ascend/`。例如，修改了上游 v1 调度器时，将对应版本的完整文件写入 `backend-overlay/vllm/v1/core/sched/scheduler.py`。运行同一个 `bash build-reference.sh` 命令即可。构建器用本地 Docker BuildKit 从每个选中的基础镜像生成派生镜像，定位实际 Python 包目录、检查 Python 语法并覆盖文件，随后将派生镜像导出到软件包。每次回传包含文件散列 `build/overlay.sha256`、原始基础镜像 ID `build/<机型>.base.id` 和最终运行镜像 ID `build/<机型>.id`。建议先提交用于对照的基线，再为同一用例生成包含后端修改的包。

overlay 可以调整 Python 前端或调度器。要修改 C++、Ascend C 或插件自己的算子实现，直接编辑 `operators/` 后重新构建；安装路径中的原生扩展和 vendor 包由本地编译产物生成。

默认镜像基于上游 0.30.0。A2 使用 `quay.io/ascend/vllm-ascend:v0.30.0`，A3 和 A5 分别使用带 `-a3` 和 `-a5` 后缀的标签。脚本把实际镜像 ID 和仓库摘要写入清单。使用内部仓库时，可以给 `ANYINFER_BUILDER_IMAGE` 和各 `ANYINFER_IMAGE_A2`、`ANYINFER_IMAGE_A3`、`ANYINFER_IMAGE_A5` 指定镜像。优化后端算子或调度代码时，可在对应官方镜像上构建含改动的派生镜像，再一键打包。例如，以下命令使用本地 A3 派生镜像。使用 `--local-images` 时，构建镜像也必须存在于本地。构建 Rust 依赖仍需要可用的 Cargo 缓存或镜像源。[官方昇腾镜像与模型部署说明](https://docs.vllm.ai/projects/ascend/en/v0.30.0/tutorials/models/DeepSeek-V4-Flash.html#docker-image-installation)

```bash
ANYINFER_IMAGE_A3=my-v4-kernel:exp17 bash build-reference.sh --profiles a3 --local-images
```

## 部署包与模型

把包和校验文件拷贝到目标 Linux 节点。目标机需要 Docker、对应 CANN 镜像支持的宿主驱动和固件，以及可执行的 `npu-smi`。先用 `npu-smi info` 确认设备可用，再把适配该机型、能装入本机总显存的完整权重和 tokenizer 放到本地。模型目录内的符号链接应能在容器的只读挂载中解析。Hugging Face snapshot 使用指向目录外 blobs 的链接时，可以用 `cp -aL` 展开到独立目录。用户态 CANN、vLLM 和插件随镜像分发。目标机可以断网启动。

```bash
# 运行时替换为刚才生成的实际文件名
artifact=anyinfer-arm64-20261120T000000Z-AbCd1234.tar.gz
sha256sum -c "$artifact.sha256"
release_dir=${artifact%.tar.gz}
mkdir "$release_dir"
tar -xzf "$artifact" -C "$release_dir"
cd "$release_dir/anyinfer"
```

启动脚本按设备透传 `/dev/davinciN`，挂载主机的驱动库、DCMI 和管理节点，并按官方单机示例启用 host 网络及独立的共享内存空间。A2、A3 沿用上游的特权容器设置。A5 使用 950 系列所需的 UMMU、UBURMA 节点。默认共享内存上限 512 GiB，由 `--shm-size` 控制，实际占用取决于模型和并行配置。每次运行仅导入匹配机型的镜像，对已导入的相同镜像 ID 直接复用。[官方启动参数](https://docs.vllm.ai/projects/ascend/en/v0.30.0/tutorials/models/DeepSeek-V4-Flash.html#docker-image-installation)

## 在多套机器上分片

先在已验证的基准实现上冻结一份参考输出。以下命令使用包内的用例提示词；也可以用 `--job-dir /data/jobs -- --capture-prompts /job/prompts.jsonl` 载入自己的提示词。为每个目标节点指定同一份参考文件时，启动器将它随结果保存，并将验证和性能测量送入相同的模型实例。应使用与部署精度、tokenizer、生成设置匹配的参考。

```bash
./run.sh accuracy --hardware a2 --model /data/models/DeepSeek-V4-Flash-0731-w8a8 \
  --results /data/references -- --capture-prompts /app/benches/accuracy-prompts.jsonl \
  --reference-name verified-a2 --output /results/golden.jsonl
# 单独做精度对照
./run.sh accuracy --hardware a2 --model /data/models/DeepSeek-V4-Flash-0731-w8a8 \
  --job-dir /data/verified -- --golden /job/golden.jsonl --output /results/accuracy.jsonl
# 将上一步终端给出的结果目录内的 golden.jsonl 作为所有分片的基准
./run.sh --hardware a2 --model /data/models/DeepSeek-V4-Flash-0731-w8a8 \
  --range 1 --golden /data/verified/golden.jsonl --results /data/runs
```

每台机器各启动一个推理引擎，载入完整模型并独立执行本机用例。默认使用本机所有 `/dev/davinciN` 设备。需要指定设备时，传入 `--devices 0,1,2,3,4,5,6,7`。双 Die 的 A3 按逻辑设备 ID 计数。省略 `--range` 会运行全部用例。

```bash
# 环境 1
./run.sh --model /data/models/DeepSeek-V4-Flash-0731-w8a8 --hardware a2 --range 1-3 --results /data/runs
# 环境 2
./run.sh --model /data/models/DeepSeek-V4-Flash-0731-w8a8 --hardware a2 --range 4-9 --results /data/runs
# 为该用例做完整的额外 profiling 回放；默认每个用例已包含短回放
./run.sh --model /data/models/DeepSeek-V4-Flash-0731-w8a8 --hardware a2 --range 1 --profile --results /data/traces
```

通过 `--cases /path/to/cases.json` 可提供修改后的用例集，启动器会复制实际用例到本次输出。单机变更并行配置或引擎参数时，在 `--` 后传入 Rust 子命令支持的参数。性能对照实验应保持模型版本、硬件型号、卡数、用例、并发、预热和采样设置一致。带 profiler 的运行用于定位耗时，吞吐对比使用正常采样模式。[官方 PyTorch Profiler 说明](https://docs.vllm.ai/projects/ascend/en/v0.30.0/developer_guide/performance_and_debug/service_profiling_guide.html)

## 回传与分析

基准命令退出时，终端会显示原始目录和同级 `bench-<机型>-<UTC>-<随机串>.tar.gz` 回传包。失败也会打包已产生的数据。机器断电或进程被强制终止时，已写入的文件保存在原始目录内。汇总多台机器的结果时，解压到同一父目录，保留各分片的独立运行目录。

| 文件 | 可以回答的问题 |
| --- | --- |
| `feedback.json` | 分片整体是否成功、采样覆盖了哪些用例、主要热点和精度状态 |
| `status.json`、`stderr.log`、`stdout.log` | 启动是否成功、退出原因和 worker 完整日志 |
| `build/`、`image.txt`、`runtime-versions.json` | 源码、用例、Rust 编译器、依赖和实际后端版本是否一致 |
| `host.txt`、`command.txt`、`cases.json` | 使用哪台主机、驱动、哪些卡、模型、引擎参数和用例 |
| `checkpoint-metadata.sha256`、`weight-inventory.tsv` | 配置、索引和 tokenizer 是否相同，权重文件大小、修改时间是否变化 |
| `telemetry/topo.txt`、`telemetry/mapping.txt` | 驱动报告的设备拓扑、CPU 亲和性和芯片编号 |
| `telemetry/samples.log`、`telemetry/polls.tsv` | 各次设备查询起止、是否成功，以及设备报告的利用率、显存、功耗、温度和同一时刻的主机计数 |
| `bench/run-*/` | 每个用例的预热与测量、逐请求记录、吞吐与延迟统计和 Prometheus 指标 |
| `operator-load.json`、`operator-smoke.json` | 实际扩展、vendor、ACLNN 符号的加载路径，首次 NPU 执行与 CPU 参考的比较结果 |
| `build/<机型>.operators.json`、`build/operator-source.sha256`、`build/operator-index.json` | 本机实际部署的编译产物和源码散列；算子名称位于哪个可编辑源码目录 |
| `bench/run-*/case-*/operators.json`、`traces/rank-*/` | 每个用例的算子级统计，以及各卡在独立 profiling 回放中采集的原始 trace |

NPU 设备数据保留 `npu-smi info` 的原始输出，字段随设备和驱动变化。设备采样周期为单次命令执行时间加 2 秒，短用例请结合请求级记录与 trace。要核对每个权重的内容，可事先在模型根目录存放计算好的 `anyinfer-weights.sha256`，系统会将这份用户提供的校验清单一同带回。

构建清单中的 `<机型>.frontend.txt` 记录镜像内是否有可执行的 Rust 服务前端。进行 Python 前端对照、采集该前端的细粒度服务端请求统计时，可在命令末尾加 `-- --frontend python --per-request-metrics`，并在比较时保留所选前端和采样配置。

```bash
mkdir -p experiment
tar -xzf /path/to/bench-a2-FIRST.tar.gz -C experiment
tar -xzf /path/to/bench-a2-SECOND.tar.gz -C experiment
python3 tools/report.py experiment > experiment-summary.json
```

迭代时回传本次分片压缩包。在相同硬件和用例上重新构建、运行并对比时，同时提供上次的回传包和本次改动目标。模型装载失败时，`stderr.log`、`telemetry/initial.txt` 和镜像信息可以先用于定位版本或设备问题。

## 在线与离线

```bash
./run.sh serve --hardware a2 --model /data/models/DeepSeek-V4-Flash-0731-w8a8 -- --listen 0.0.0.0:8000
./run.sh offline --hardware a2 --model /data/models/DeepSeek-V4-Flash-0731-w8a8 \
  --job-dir /data/jobs --results /data/runs -- \
  --input /job/input.jsonl --output /results/answers.jsonl --concurrency 8
```

离线输入目录挂载至容器的 `/job`，本次独立输出目录挂载至 `/results`。在线服务占用指定的宿主机端口，使用回环地址可将监听范围限制在本机。
