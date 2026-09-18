#ifndef HD_VISION_KERNELS_H
#define HD_VISION_KERNELS_H

/*
 * C ABI for the Qwen3-VL vision tower device kernels (vision_kernels.cu).
 * All pointers are device pointers; bf16 is uint16_t. See vision.h for the
 * host orchestration contract.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void hd_vision_patch(const void *x, const void *w, const void *bias,
                     void *y, int n, int in_dim, int out_dim);

void hd_vision_layernorm(const void *x, const void *w, const void *b,
                         void *y, int rows, int cols, float eps);

void hd_vision_gelu(const void *x, void *y, size_t n);

/* Exact GELU (erf) used by the vision patch mergers (nn.GELU() upstream),
 * distinct from the block MLP's gelu_pytorch_tanh. */
void hd_vision_gelu_exact(const void *x, void *y, size_t n);

void hd_vision_pos_interp(const void *pos_embed, const int *idx,
                          const float *wgt, void *y, int n, int hidden);

void hd_vision_rot(const float *inv_freq, const int *coords, void *y,
                   int n, int half);

void hd_vision_rot_cos_sin(const void *rot, float *cosd, float *sind,
                           int n, int half);

void hd_vision_spatial_merge(const void *in, void *out, int gh, int gw,
                             int hidden);

void hd_vision_qkv_split(const void *qkv, void *q, void *k, void *v,
                         int n, int heads, int d);

void hd_vision_attn_merge(const void *hm, void *sm, int heads, int n, int d);

void hd_vision_masked_scatter(const void *dst, const uint8_t *mask,
                              const void *src, void *out, int rows, int cols);

void hd_vision_deepstack_inject(const void *in, const uint8_t *mask,
                                const void *emb, void *out, int rows,
                                int cols);

#ifdef __cplusplus
}
#endif

#endif /* HD_VISION_KERNELS_H */