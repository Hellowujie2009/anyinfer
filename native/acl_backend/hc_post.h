#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
/* A stream and five aligned device pointers on its device, contiguous ND row-major.
 * x [T,H] BF16, residual [T,4,H] BF16, post [T,4] FP32, comb [T,4,4] FP32,
 * y [T,4,H] BF16. Buffers must be disjoint and live until stream completion.
 * H is a multiple of 16. The return code is an aclError. */
int32_t anyinfer_hc_post_launch(void *stream, uint32_t blocks, uint32_t tokens,
                               uint32_t hidden, void *x, void *residual,
                               void *post, void *comb, void *y);
/* Explicit per-work-item tile for tuning. Supported widths: 128, 256, 512, 1024. */
int32_t anyinfer_hc_post_launch_tiled(void *stream, uint32_t blocks, uint32_t tokens,
                                     uint32_t hidden, uint32_t tile, void *x,
                                     void *residual, void *post, void *comb, void *y);
#ifdef __cplusplus
}
#endif
