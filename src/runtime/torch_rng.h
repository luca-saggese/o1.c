/*
 * torch_rng.h — bit-exact port of PyTorch's CPU MT19937 RNG for float
 * normal/uniform draws.
 *
 * Semantics replicated from the frozen oracle venv (torch 2.12.1+cu130):
 *   - MT19937 engine: ATen/core/MT19937RNGEngine.h
 *   - uniform_real_distribution<float>: ATen/core/DistributionsHelper.h
 *     (24-bit mask, divisor 2^24)
 *   - normal_fill scalar path + serial normal_distribution<double> path:
 *     ATen/native/cpu/DistributionTemplates.h
 *
 * The dispatch rule matches torch's normal_kernel for float tensors:
 *   size >= 16 && contiguous  -> normal_fill path (uniform fill all, then
 *                                in-place Box-Muller blocks of 16; if
 *                                size % 16 != 0, the last 16 are redrawn)
 *   size <  16                -> serial normal_distribution<double> path
 *                                (random64, log1p(-u2), cached sin sample)
 *
 * Verified bit-exact against torch for n = 1, 8, 15, 16, 17, 31, 32, 1024.
 */
#ifndef HD_TORCH_RNG_H
#define HD_TORCH_RNG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hd_torch_rng {
    uint64_t seed_;
    int left_;
    int seeded_;
    uint32_t next_;
    uint32_t state[624];
} hd_torch_rng;

/* Seed the engine exactly like torch::manual_seed (state[0]=seed&0xffffffff,
 * standard init recurrence, left_=1, next_=0). */
void hd_torch_rng_seed(hd_torch_rng *rng, uint64_t seed);

/* Next raw tempered MT19937 u32. */
uint32_t hd_torch_rng_u32(hd_torch_rng *rng);

/* Uniform float in [0,1) matching torch's uniform_real_distribution<float>
 * (24-bit mantissa mask). */
float hd_torch_uniform_f32(hd_torch_rng *rng);

/* Fill out[0..n) with standard-normal floats, bit-exact with
 * torch.randn(..., generator=cpu_mt19937) for float tensors. */
void hd_torch_randn_f32(hd_torch_rng *rng, float *out, int64_t n);

#ifdef __cplusplus
}
#endif

#endif /* HD_TORCH_RNG_H */