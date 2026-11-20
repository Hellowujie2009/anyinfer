# 如何判断一个昇腾算子还能优化多少

先在 `feedback.json` 找到影响目标用例的耗时算子和输入形状，再到该用例的 `operators.json` 查看各 rank 的 `rank_evidence`。设备任务时间总和适合排序。判断最终收益应使用同一用例正常计时阶段的 tokens/s、TTFT、TPOT 和尾延迟。

## 同卡实测参考

结果包根目录的 `hardware-calibration.json` 记录每张参与设备在模型加载前的独立微基准，包括输入形状、数值检查、NPU 事件耗时和多次重复的中位数。报告按 worker 的 `physical_device` 匹配物理设备，从同一设备、同类、数值校验成功且标记为 `reference_eligible` 的探针中选取最高的中位速率。

| 参考量 | 单位和计算方法 | 用途 |
| --- | --- | --- |
| `bf16_gemm` | TFLOP/s，普通 BF16 矩阵乘的有效运算量为 `2 × M × K × N`，一次乘加按两次操作计算 | 参照不同模型形状上的稠密 BF16 矩阵乘 |
| `stream_copy` | 十进制 GB/s，同卡不同缓冲区连续复制，计入读和写两份有效载荷 | 对照处理逻辑张量字节的速率 |
| `int8_quant_gemm` | TOP/s，使用预量化 INT8 输入和权重、带反量化输出的矩阵乘 | 支持该探针的设备可取得 W8A8 参考，保留完整形状和格式供权重路径匹配 |

`operator-smoke.json` 中 HcPost 的单核测试覆盖首张选定卡上的不同 token 数，使用实际权重配置的 hidden size 和混合流数。查看 `microbenchmark.shapes` 的逐形状结果和数值检查；同卡复制测试成功时，`feedback.json` 中该记录还给出有效带宽相对长缓冲区复制的比值。该测试反复调用同一组输入，缓存状态、输出分配和流上其他任务都可能与模型前向不同。

## 从一条模型 trace 读出硬件余量

* **计算吞吐与有效带宽。** 对单个可辨认的任务，使用该 rank 所有正耗时样本的总工作量除以它们的总耗时。`work_model` 记录每个调用的运算量或逻辑字节。稠密 BF16 MatMul/MatMulV2 需要两个二维矩阵、一个二维输出、ND 格式，以及能从三者维度唯一确定的 K。按所有输入各读一次、输出写一次估计逻辑字节。HcPost 需要 x、residual、post、comb 和输出的布局、数据类型、维度符合本地实现。Abs、Neg、Relu、Add、Mul 仅在 BF16 同形状且没有广播时使用同样的字节模型。
* **Roofline。** `rank_evidence.<rank>.empirical_roofline` 对满足上述条件的稠密矩阵乘给出 `min(实测 BF16 TFLOP/s，逻辑 FLOP/byte × 实测复制 GB/s ÷ 1000)`，以及算子的有效 TFLOP/s 与此估值的比值。`lower_reference_term` 说明在所设模型下较紧的一项。输入缓存命中可使有效带宽超过复制参考，矩阵形状、实现、频率及并发会改变结果。保留超过 1 的原始比值。物理 HBM 流量、硬件理论峰值和可兑现的端到端加速幅度均需额外证据。
* **流水线与访存证据。** `duration_weighted_pmu` 给出 CANN 导出的 pipeline ratios、主存字段按任务耗时加权的均值、有效样本数和覆盖时长。核对 MAC、Vector、MTE、Scalar、Fixpipe 的周期占比，以及 Memory 采样中的主存和缓存带宽。各流水线可重叠，Cube、Vector 和缓存层次应分别解读。完整硬件列和字段原有单位见 `hardware_counters`，各列样本算术平均与耗时加权统计分别保留。按核域导出的带宽不能直接当作整卡 HBM 带宽。可对照不同 tiling 或融合实现对同一形状的同名 PMU 列。[昇腾 profiler 字段定义](https://github.com/mindstudio-docs/master/blob/master/msprof/docs/zh/user_guide/profile_data_file_references.md)
* **小形状与多卡不均衡。** `launch_geometry` 对可明确匹配核域的任务列出 block 数、同卡实际查询的 Cube 或 Vector 核数及两者比值。它描述切分规模。结合 `mean_task_us`、调用次数和 PMU 看是否存在可增加并行度的切分。对比各 rank 时也对照请求量、形状和通信等待，累计时间差可能源于工作量不同。
* **时间线与融合。** 各 rank 的 `timelines` 按同一设备先求时间区间的并集，再给出跨任务重叠时长和最长无已列任务间隔。分母是最早有时间戳的任务开始到最后一个结束的时间，`timed_calls` 表明覆盖了多少条导出记录。设备上其他工作和导出缺失都可能出现在间隔中。`same_stream_neighbors` 提供同一流上的相邻算子和间隔，可据此检查融合位置的数据依赖与同步。HCCL 的等待、传输等另见主重放的 `communication`。应结合原始 trace 判断通信是否与计算重叠。

稀疏注意力、路由后的专家计算和复杂融合算子的有效工作量会受稀疏索引、专家 token 分布或实际执行分支影响。报告为无法确认的算子保留耗时、形状、原始 PMU 和时间线，`work_model` 留空。`reference_comparison_coverage.fraction_of_summed_task_time` 显示有同卡定量对照的形状覆盖了多少累计任务时间。对缺少定量模型的热点，下一轮可针对该形状增加独立探针或记录路由分布，复用同卡参考并重新核验精度。

对于这些复杂算子，也可以比较当前负载内的访存表现。每个 rank 的 `observed_main_memory_rates` 列出同一重放、同一核心类型、同一主存读或写带宽列（GB/s）下最高的分组均值及其来源形状。需要至少两组，每组有至少三次有效采样。各组 PMU 项的 `over_highest_observed_group` 以该均值为分母。例如某个稀疏内核的读带宽比值是 0.4，可检查它的索引访问模式与参考组的顺序访问是否不同。所有参照组都来自当前模型负载，它们与芯片能达到的最大值之间仍可能有差距。

## 选择下一轮实验

默认的两个 PMU 组各执行一次独立重放。主报告在 `case-N/operators.json`，额外的访存报告在 `case-N/pmu-memory/operators.json`。两次的输入形状和负载配置可对照，具体请求种子、任务起止和调度可以不同。在 `feedback.json` 用 `device_hotspots` 和 `memory_replay_hotspots` 分别查看同一形状。跨组比较以形状、rank、dtype 和调用数为单位，避免把两个时间窗当成一次采样。

例如，若某个高耗时形状的复制参考比值较高，且相邻算子重复读写相同张量，可试验融合，观察两者累计设备时间和正常推理吞吐。若小 batch 的矩阵乘远低于大矩阵参考且只切了少数 block，可试验小形状 tiling 或不同批量，复查首 token 延迟。若卡间通信等待或时间线空隙明显，应同时检查负载分配和提交时序。

需要单组诊断时，在 benchmark 参数末尾加 `-- --aic-metrics pipe` 或 `-- --aic-metrics memory`。当设备支持的访存计数有限，可用单组 pipe 重跑；双组运行缺失必需的访存样本时会保留计时与诊断文件。用 `--profile` 可为长解码完整重放。两轮包使用 `tools/report.py --compare-cases` 比较正常端到端结果及主重放，并核实两边测量状态、模型、TP/DP 布局、PMU 组与参考精度。
