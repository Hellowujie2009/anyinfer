#include "reference.h"
#include "shape.h"
#include <iostream>

int main() {
    reference::Inputs inputs(1, 16);
    for (size_t i = 0; i < 4; ++i) {
        inputs.post[i] = i + 1;
        for (size_t j = 0; j < 4; ++j) inputs.comb[i * 4 + j] = 4 * i + j;
        inputs.residual[i * 16] = reference::bf16(i + 1);
    }
    inputs.x[0] = reference::bf16(2);
    const float expected[] = {82, 94, 106, 118};
    for (size_t j = 0; j < 4; ++j) {
        if (reference::fp32(inputs.expected(0, j, 0, 16)) != expected[j]) return 1;
    }
    if (reference::bf16(1.00390625f) != 0x3f80 || reference::bf16(1.01171875f) != 0x3f82
        || reference::bf16(-1.01171875f) != 0xbf82) return 2;
    if (!hc_valid_shape(2, 4096) || hc_valid_shape(2, 4095) || hc_valid_shape(0, 16)
        || hc_work(1, 7168, hc_tile(1)) != 56 || hc_work(16, 4224, hc_tile(16)) != 144) return 3;
    if (hc_work(16, 4224, 1024) != 80 || hc_work(1, 7168, 256) != 28
        || hc_valid_tile(0) || hc_valid_tile(1536) || !hc_valid_tile(1024) || hc_ub_bytes(1024) != 47104) return 4;
    std::cout << "HcPost reference and tile shape checks passed\n";
}
