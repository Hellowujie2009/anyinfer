#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

namespace reference {
inline uint16_t bf16(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
inline float fp32(uint16_t x) {
    const uint32_t bits = uint32_t(x) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
struct Inputs {
    std::vector<uint16_t> x, residual;
    std::vector<float> post, comb;
    Inputs(size_t tokens, size_t hidden)
        : x(tokens * hidden), residual(4 * x.size()), post(tokens * 4), comb(tokens * 16) {
        // Bounded dyadics keep the FP32 products and sums exact. Only the final BF16
        // rounding can lose precision. Deterministic asymmetric mixing detects transposes.
        uint32_t state = 0x6ac03ba1;
        auto next = [&] {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            return (int(state & 127) - 63) / 32.0f;
        };
        for (auto &v : x) v = bf16(next());
        for (auto &v : residual) v = bf16(next());
        for (auto &v : post) v = next();
        for (auto &v : comb) v = next();
    }
    uint16_t expected(size_t token, size_t out, size_t d, size_t hidden) const {
        double value = double(post[token * 4 + out]) * fp32(x[token * hidden + d]);
        for (size_t i = 0; i < 4; ++i)
            value += double(comb[token * 16 + i * 4 + out]) * fp32(residual[(token * 4 + i) * hidden + d]);
        return bf16(static_cast<float>(value));
    }
};
}
