# 从 DeepSeek V4 的一次前向找到可修改的昇腾算子

`operators/vllm-ascend/` 收录 vLLM-Ascend v0.30.0 的完整 `csrc/`、`cmake/`、Python 插件 `vllm_ascend/` 和构建入口。模型前向、PyTorch 注册层、host 端 tiling 和设备端 Ascend C 实现可以在同一工作区修改。源码来自上游提交 `c09d3e20b9c45c80b34ae3781440d90904a44023`。原始归档、Catlass gitlink、编译依赖的 SHA-256 和来源记录在 [SOURCE.json](../operators/SOURCE.json)。[上游固定版本](https://github.com/vllm-project/vllm-ascend/tree/c09d3e20b9c45c80b34ae3781440d90904a44023)

## 从 trace 定位源码

以下路径均相对于 `operators/vllm-ascend/`。同一模型可能因权重量化、SoC、prefill/decode 和并行配置选择不同的算子。先查该次运行的 `case-N/operators.json` 中的设备算子、shape 和 rank，再检查对应的前向分支。

| 前向操作 | 模型或插件的实际调用点 | 可编辑的设备算子目录 |
| --- | --- | --- |
| mHC 残差合并与扩展 | `vllm_ascend/models/deepseek_v4/model.py` 的 `hc_pre`、`hc_post` 调用 `_C_ascend.npu_hc_pre_v2`、`_C_ascend.npu_hc_post` | `csrc/moe/hc_pre/`、`csrc/moe/hc_post/` |
| 压缩历史 KV | `vllm_ascend/models/deepseek_v4/compressor.py` 调用 `_C_ascend.compressor` | `csrc/attention/compressor/`、`compressor_metadata/` |
| 压缩注意力的 top-k 索引 | `vllm_ascend/models/deepseek_v4/indexer.py` 调用 `_C_ascend.npu_quant_lightning_indexer_v2` | `csrc/attention/quant_lightning_indexer_v2/` 及相邻的 metadata 目录 |
| 稀疏注意力和 KV 尾部融合 | `vllm_ascend/attention/dsa_v1.py`、`dsa_v41.py` 及 `vllm_ascend/models/deepseek_v41/mixed_quant_attention.py` | `csrc/attention/sparse_flash_mla/`、`kv_quant_sparse_attn_sharedkv/`、`kv_compress_epilog_v2/` 等，按 SoC 选取 |
| MoE 专家路由及通信与计算融合 | `vllm_ascend/ops/fused_moe/moe_comm_method.py` 的融合分支调用 `_C_ascend.dispatch_ffn_combine` | A3 的 `csrc/mc2/dispatch_ffn_combine/` 及其量化变体。其他分支也可能调用 torch-npu 的 grouped matmul |

以 HcPost 为例，模型的 `hc_post` 传入 hidden states、residual、post、comb。PyTorch 的 PrivateUse1 后端在 `csrc/torch_binding.cpp` 注册 `npu_hc_post_npu`，该函数调用 `aclnnHcPost`。`csrc/moe/hc_post/op_host/` 描述输入、dtype、可用核心及切分策略。`op_kernel/hc_post.cpp` 和同目录头文件按硬件和数据类型选择设备实现。修改 tiling 时，同时检查 host 产生的参数和设备核读取的布局。[HcPost 设备入口](../operators/vllm-ascend/csrc/moe/hc_post/op_kernel/hc_post.cpp)

## 各硬件的编译范围

| 硬件 | 上游构建分支 | ACLNN 算子组数 |
| --- | --- | ---: |
| A2 | `ascend910b` | 39 |
| A3 | `ascend910_93` | 43 |
| A5 | `ascend950` | 27 |

每组可能包含设备实现、多个子算子或 host metadata。这个数量来自上游 `build_aclnn.sh` 的硬件选择列表。实际模型执行了哪些核、各核被调用多少次，见目标机器的 trace。Triton 源码另见 `vllm_ascend/ops/triton/`。A2/A3 还构建顶层 CMake 定义的 `vllm_ascend_kernels`。[逐组目录和支持矩阵](../operators/inventory.tsv) · [上游构建选择](../operators/vllm-ascend/csrc/build_aclnn.sh)

## 构建、数值验证、复测

```bash
# 构建机，示例目标为 A2、ARM64。变更工作区中的源码后再次执行同一命令。
bash build.sh --profiles a2 --platform linux/arm64 --out dist-a2

# 目标机，在解压后的软件包目录执行
./run.sh --hardware a2 --model /data/models/DeepSeek-V4-Flash-0731-w8a8 \
  --range 1 --profile --results /data/runs
```

每个硬件镜像单独编译。构建阶段在官方 CANN/PyTorch 环境中运行本地 `setup.py` 的扩展和 ACLNN 构建流程，将生成的 `vllm_ascend_C` 以及 `_cann_ops_custom/vendors/custom_transformer` 装入包内。标准构建所需的 Catlass 头文件、protobuf、Abseil、nlohmann/json 和 makeself 均随源码提供。首次准备基础镜像和 Rust 工具链的条件见[打包说明](package.md)。

目标机的 `operator-smoke.json` 记录 HcPost 的 BF16 数值检查。它在第一张可见 NPU 上执行扩展，用 CPU FP32 加权求和核对 50,688 个元素，同时记录实际扩展及 ACLNN 符号的动态库路径。这项冒烟检查覆盖一个设备核和一组确定性输入。模型级正确性应查看相同 checkpoint 的生成 token ID 基线；修改其他算子需要覆盖相应形状和数值路径。[检查程序](../operators/smoke_hc_post.py)

性能优化时，先保留一次基线运行，再只改一处热点源码，使用相同硬件、用例分片、并发和模型版本重建并复测。比较无 profiler 阶段的 TTFT、TPOT、吞吐以及尾延迟，将 `case-N/operators.json` 与相应 rank 的 trace 用于解释差异。多卡并行的核和通信可以重叠，设备时间之和不能换算为单请求耗时。[运行和回传说明](package.md)

## 依赖与源码校验

Rust 负责请求入口和基准工具，外部 vLLM 提供引擎调度及 continuous batching。随库的 vLLM-Ascend 插件负责 NPU 模型前向、量化路径和自定义算子分发。CANN 提供编译器、运行时及其内置算子，torch-npu 提供其自身的算子分发与实现，HCCL 提供卡间通信原语。这些系统组件由基础镜像和目标驱动提供。例如 `torch_npu.npu_grouped_matmul` 激活时，本仓可以调整调用、layout 和融合策略，其底层实现随 torch-npu/CANN 发布。

`upstream.sha256` 是首次导入时的逐文件基线，覆盖 vLLM-Ascend 的构建和运行源码以及 Catlass 的编译头文件和许可证。新 checkout 可运行下列命令核对原始版本；后续对源码的修改会显示为有意的差异。每次软件包构建另行记录当时实际输入的摘要。

```bash
(cd operators/vllm-ascend && sha256sum --quiet -c ../upstream.sha256)
(cd operators/build-deps && sha256sum --quiet -c SHA256SUMS)
```

保留了 vLLM-Ascend 仓库级 Apache-2.0 许可证、各文件的版权及许可头、Catlass 的 CANN Open Software License Agreement 2.0 及第三方清单，以及编译依赖的原始归档和许可证。Catlass 按上游 gitlink 引入完整 `include/`，示例和单元测试依赖的 googletest 在标准算子构建范围之外。上游 vLLM-Ascend 的教程、CI、独立测试套件由固定提交提供。[上游许可证](../operators/vllm-ascend/LICENSE) · [Catlass 许可证](../operators/vllm-ascend/csrc/third_party/catlass/LICENSE) · [编译依赖许可证](../operators/build-deps/licenses/)
