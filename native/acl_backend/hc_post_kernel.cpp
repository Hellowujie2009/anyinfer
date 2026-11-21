#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t lanes = 4;
class Mix {
public:
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR residual, GM_ADDR post, GM_ADDR comb,
                              GM_ADDR y, uint32_t tokens, uint32_t hidden, uint32_t tile)
    {
        x_.SetGlobalBuffer((__gm__ bfloat16_t *)x);
        residual_.SetGlobalBuffer((__gm__ bfloat16_t *)residual);
        post_.SetGlobalBuffer((__gm__ float *)post);
        comb_.SetGlobalBuffer((__gm__ float *)comb);
        y_.SetGlobalBuffer((__gm__ bfloat16_t *)y);
        pipe_.InitBuffer(input_, 1, 5 * tile * sizeof(bfloat16_t));
        pipe_.InitBuffer(output_, 1, 4 * tile * sizeof(bfloat16_t));
        pipe_.InitBuffer(scratch_, 7 * tile * sizeof(float));
        const uint32_t columns = (hidden - 1) / tile + 1;
        const uint64_t work = static_cast<uint64_t>(tokens) * columns;
        for (uint64_t item = GetBlockIdx(); item < work; item += GetBlockNum()) {
            const uint32_t token = item / columns;
            const uint32_t offset = (item % columns) * tile;
            const uint32_t count = hidden - offset < tile ? hidden - offset : tile;
            const uint64_t base = static_cast<uint64_t>(token) * hidden;
            auto input = input_.AllocTensor<bfloat16_t>();
            DataCopy(input, x_[base + offset], count);
            for (uint32_t i = 0; i < lanes; ++i) {
                DataCopy(input[(i + 1) * tile], residual_[lanes * base + i * uint64_t(hidden) + offset], count);
            }
            input_.EnQue(input);
            input = input_.DeQue<bfloat16_t>();
            auto f = scratch_.Get<float>();
            for (uint32_t i = 0; i <= lanes; ++i) {
                Cast(f[i * tile], input[i * tile], RoundMode::CAST_NONE, count);
            }
            input_.FreeTensor(input);
            PipeBarrier<PIPE_V>();
            auto acc = f[5 * tile];
            auto tmp = f[6 * tile];
            auto output = output_.AllocTensor<bfloat16_t>();
            for (uint32_t j = 0; j < lanes; ++j) {
                // Form the residual reduction before adding the weighted block output.
                const uint64_t m = static_cast<uint64_t>(token) * 16 + j;
                Muls(acc, f[tile], comb_.GetValue(m), count);
                for (uint32_t i = 1; i < lanes; ++i) {
                    Muls(tmp, f[(i + 1) * tile], comb_.GetValue(m + i * lanes), count);
                    PipeBarrier<PIPE_V>();
                    Add(acc, acc, tmp, count);
                    PipeBarrier<PIPE_V>();
                }
                Muls(tmp, f, post_.GetValue(static_cast<uint64_t>(token) * lanes + j), count);
                PipeBarrier<PIPE_V>();
                Add(acc, acc, tmp, count);
                PipeBarrier<PIPE_V>();
                Cast(output[j * tile], acc, RoundMode::CAST_RINT, count);
                PipeBarrier<PIPE_V>();
            }
            output_.EnQue(output);
            output = output_.DeQue<bfloat16_t>();
            for (uint32_t j = 0; j < lanes; ++j) {
                DataCopy(y_[lanes * base + j * uint64_t(hidden) + offset], output[j * tile], count);
            }
            output_.FreeTensor(output);
        }
    }
private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> input_;
    TQue<QuePosition::VECOUT, 1> output_;
    TBuf<QuePosition::VECCALC> scratch_;
    GlobalTensor<bfloat16_t> x_, residual_, y_;
    GlobalTensor<float> post_, comb_;
};
}

extern "C" __global__ __aicore__ void anyinfer_hc_post_kernel(GM_ADDR x, GM_ADDR residual,
    GM_ADDR post, GM_ADDR comb, GM_ADDR y, uint32_t tokens, uint32_t hidden, uint32_t tile)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Mix op;
    op.Run(x, residual, post, comb, y, tokens, hidden, tile);
}
