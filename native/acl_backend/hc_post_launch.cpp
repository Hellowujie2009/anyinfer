#include "hc_post.h"
#include "shape.h"
#include "acl/acl.h"
#include "aclrtlaunch_anyinfer_hc_post_kernel.h"

extern "C" int32_t anyinfer_hc_post_launch(void *stream, uint32_t blocks, uint32_t tokens,
                                           uint32_t hidden, void *x, void *residual,
                                           void *post, void *comb, void *y) {
    return anyinfer_hc_post_launch_tiled(stream, blocks, tokens, hidden, hc_tile(tokens),
                                        x, residual, post, comb, y);
}

extern "C" int32_t anyinfer_hc_post_launch_tiled(void *stream, uint32_t blocks, uint32_t tokens,
                                                 uint32_t hidden, uint32_t tile, void *x,
                                                 void *residual, void *post, void *comb, void *y) {
    if (!stream || !blocks || !hc_valid_shape(tokens, hidden) || !hc_valid_tile(tile)
        || !x || !residual || !post || !comb || !y)
        return ACL_ERROR_INVALID_PARAM;
    return ACLRT_LAUNCH_KERNEL(anyinfer_hc_post_kernel)(
        blocks, stream, static_cast<uint8_t *>(x), static_cast<uint8_t *>(residual),
        static_cast<uint8_t *>(post), static_cast<uint8_t *>(comb), static_cast<uint8_t *>(y),
        tokens, hidden, tile);
}
