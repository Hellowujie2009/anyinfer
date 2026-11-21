#pragma once
#include <cstdint>

inline uint32_t hc_tile(uint32_t tokens) { return tokens <= 4 ? 128 : 512; }
constexpr uint32_t hc_ub_bytes(uint32_t tile) { return (5 * 2 + 4 * 2 + 7 * 4) * tile; }
// A2/A3 have 192 KiB UB, with up to 256 bytes reserved, and 950 SIMD has 248 KiB.
// The input queue, output queue and scratch all start on 32-byte boundaries.
static_assert(hc_ub_bytes(1024) < 192 * 1024 - 256);
inline bool hc_valid_tile(uint32_t tile) {
    return tile == 128 || tile == 256 || tile == 512 || tile == 1024;
}
inline uint64_t hc_work(uint32_t tokens, uint32_t hidden, uint32_t tile) {
    return static_cast<uint64_t>(tokens) * (1 + (hidden - 1) / tile);
}
inline bool hc_valid_shape(uint32_t tokens, uint32_t hidden) {
    return tokens && hidden && hidden % 16 == 0 &&
           tokens <= UINT64_MAX / (18ull * hidden + 80);
}
