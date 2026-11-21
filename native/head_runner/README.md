# 在昇腾上验证 DeepSeek V4 输出头

`run-head` 接收最后一个 Transformer 块输出的单个 token 的多流残差。每张本机卡执行一份带真实权重的输出头，得到连续词表分片的 logits。Rust 主进程合并各分片的最大值，相等时取较小的全局 token ID。这条命令用于单独验证输出头。

## 导出和编译

在已加载 CANN 环境的 Linux 构建机上，从同一份只读 DeepSeek V4 Flash 或 Pro checkpoint 导出全部 rank。下面以两张卡为例。`SOC` 填写目标机器的准确 SoC 型号。

```bash
set -euo pipefail
MODEL=/models/DeepSeek-V4-Flash
SOC=Ascend910B1
for rank in 0 1; do
  bin/anyinfer prepare-head --model "$MODEL" --rank "$rank" --world-size 2 --output "head-$rank"
  (cd "head-$rank" && atc --framework=5 --model=head.onnx \
    --input_format=ND --precision_mode_v2=origin --soc_version="$SOC" --output=head)
done
```

将每个 rank 的 `head.json` 与 `head.om` 放到运行环境的对应目录。ATC 生成的 OM 与目标芯片、CANN 能力有关。实际可编译的图精度组合应在目标工具链上验证。[华为 ATC ONNX 转换说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/devaids/atctool/atlasatc_16_0003.html)

## 输入与启动

从可信的模型运行中，采集同一 token 进入末端 mHC 输出头前的 `[hc_mult, hidden_size]` 连续 BF16 残差，以及该 token 在整个词表上的 FP32 logits。按小端原始字节写入两个文件，均不含数组头。golden logits 按 token ID 排列。每轮比较应固定原始 checkpoint、残差与 golden。

```bash
bin/anyinfer run-head \
  --rank-dir head-0 --rank-dir head-1 --devices 0,1 \
  --residual sample.residual.bf16 --golden-logits sample.logits.f32 \
  --atol 0.05 --rtol 0.01 --warmup 3 --samples 20 \
  --output results/head-sample-001.json
```

`--rank-dir` 按 rank 从 0 开始连续列出，设备 ID 与目录一一对应。单卡使用 `--world-size 1` 导出，并在启动时给出一个目录和设备 ID。每次运行选择新的报告文件名。省略 `--golden-logits` 时，报告包含执行结果和计时，`golden_comparison` 为 `null`。

## 读报告

- `ranks[].measured.device_us` 是每次单卡执行输出头 OM 的 ACL 事件时间，单位为微秒。进程先并发加载、上传输入和预热，全部就绪后才启动测量。设备事件记录同一条流上的图执行区间。[华为 ACL 事件计时文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_0081.html)
- `host_timed_window_us` 从主进程释放所有 rank 开始，到主进程得到全局 token 为止，包括全部样本、设备同步、每卡最后一轮 logits 的一次回拷以及主机归约。模型加载、预热和逐元素 golden 校验位于区间之外。每个 rank 在本地主机内存求分片最大值，常规模式通过进程管道只回传分片最大值与对应 ID。
- `input_h2d_us`、`output_d2h_us`、`local_checked_argmax_us` 是各 rank 的主机计时，分别对应一次残差上传、一次 logits 回拷，以及回拷后的有限值检查和 argmax。根据这些数值可以量化将归约移进设备端的可能收益。
- `golden_comparison` 记录比较元素数、超差个数、最大绝对误差、golden 的 token ID 与两者是否一致。每个 logit 满足 `abs(actual − golden) <= atol + rtol * abs(golden)` 且全局 token 相同才通过。超差会写报告并返回非零退出码。示例容差可按精度基线调整。
- `ranks[].manifest_sha256`、`ranks[].om_sha256`、`residual_sha256` 与 `golden_logits_sha256` 可用于核对跨轮次输入和二进制。更换 checkpoint 时重新导出所有 rank。
