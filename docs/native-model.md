# 权重检查与 V4 末层精度验证

以下命令使用交付目录中的 `bin/anyinfer`。权重目录需要提供官方配置 `config.json` 和 `model.safetensors.index.json` 引用的全部分片。单文件检查点可以直接提供 `model.safetensors`。准备末层时，各个 rank 需要使用同一份只读检查点。

```bash
bin/anyinfer inspect --model /models/DeepSeek-V4-Flash > checkpoint.json
```

`checkpoint.json` 记录分片数、各 dtype 的张量数和字节数、mHC 末层参数清单，以及 1、8、32、128、512 个 token 对应的 HcPost 输入输出形状。Rust 读取器会遍历每个分片头，核对索引归属、dtype、维度和文件边界。此项检查可以发现缺片、文件截断和索引错配。同长度的权重内容损坏需要利用发布方的校验和发现。检查支持顶层 `deepseek_v4` 和带 `deepseek_v41_text` 文本配置的 `deepseek_v41`。

## 从 V4-Flash 或 V4-Pro 权重导出末层

以下示例按词表行将末层平分给两张卡，每个 rank 的输出目录须使用新路径。设置 `SOC_VERSION` 为当前编译目标的 ATC SoC 标识。完成各 rank 的导出和 ATC 编译后，可将 rank 目录和程序一起复制到相同 SoC、兼容 CANN 版本的运行环境。

```bash
set -e
: "${SOC_VERSION:?请设置目标 SoC 的 ATC 标识}"
for rank in 0 1; do
  dir="heads/rank-$rank"
  bin/anyinfer prepare-head --model /models/DeepSeek-V4-Flash \
    --world-size 2 --rank "$rank" --output "$dir"
  (cd "$dir" && atc --framework=5 --model=head.onnx \
    --output=head --soc_version="$SOC_VERSION")
done
```

`head.onnx` 是带运算节点的标准 ONNX 文件。旁边的五个 `.bin` 文件包含真实参数，移动 ONNX 文件时需保留这些文件的相对位置。`head.json` 描述输入输出、rank 的词表区间、权重原名、源张量内的字节偏移以及导出长度。更改权重、SoC 或精度模式后应重新导出和编译。

图输入为一个序列最后一个 token 在所有 Transformer 层执行完毕后的残差，按 `[hc_mult, hidden_size]` 连续存放，类型为小端 BF16。图依次执行 mHC 门控、残差流求和、最终 RMSNorm 和本 rank 的词表投影，输出为连续词表行对应的 FP32 logits。图保留官方实现的两次 BF16 舍入。V4.1 的末层算子链和权重映射需要独立核对，目前仅开放配置与权重检查。该图仅覆盖末层，不能从 prompt 生成上述输入。[DeepSeek V4 官方模型实现](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L651)

## 生成同一残差的参考输出

将一条末层残差保存为小端 BF16 原始文件 `residual.bf16`，不带张量头。真实请求回放应使用同一权重、同一 token 的末层残差。以下命令逐行读取输出投影，使用独立的 Rust 标量实现计算全词表参考 logits，写入指定的新文件。

```bash
bin/anyinfer golden-head --model /models/DeepSeek-V4-Flash \
  --residual residual.bf16 --output golden.f32
```

`golden.f32` 按 token ID 升序保存小端 FP32 值，总字节数为 `4 * vocab_size`。输出 JSON 给出贪心 token ID 和 logit。对同一组权重和残差，可重复使用该文件作为 `run-head --golden-logits golden.f32` 的精度基准。CPU 参考用 FP64 累加长点积和均方归约，在算子边界舍入回 FP32，以降低长向量顺序累加的误差。设备矩阵乘法与归约可能使用不同的累加顺序，核查时同时查看逐元素容差和最终 token。CPU 开销应排除在 NPU 延迟统计之外。

完整的双卡运行命令、精度容差参数和设备计时口径见[多卡末层运行说明](../native/head_runner/README.md)。

## 已核对的 Flash 权重样本

一次本地 CPU 交叉验证使用官方 Flash 第 45 个分片中的完整门控参数、完整 RMSNorm 权重，以及输出投影的前四行。输入形状为 `[4,4096]`，按行展开后第 `i` 个 BF16 值为 `((i % 257) - 128) / 256`，索引从零开始。此例只检查词表 ID 0 到 3。[官方权重](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main)

ONNX 1.17 的完整模型检查通过。ONNX Runtime 1.17.1 的 CPU 后端与 Rust 精度参考均选出该区间内的 ID 2，四个 logits 最大绝对误差为 `2.3842e-7`。依词表顺序，Rust 参考结果为 `[-2.44799066, 2.38153577, 3.42176080, -0.28281155]`。使用完整检查点时可以复核全词表结果文件的前 16 个字节。昇腾上的 ATC 编译、精度和运行时间需分别在目标 SoC 上验证。
