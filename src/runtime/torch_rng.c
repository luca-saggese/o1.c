#include "torch_rng.h"

#include <math.h>
#include <string.h>

#define HD_MT_N 624
#define HD_MT_M 397
#define HD_MT_MATRIX_A 0x9908b0dfu
#define HD_MT_UPPER_MASK 0x80000000u
#define HD_MT_LOWER_MASK 0x7fffffffu

/* uniform_real_distribution<float>: 24-bit mantissa, divisor 2^24. */
#define HD_UNIFORM_MASK ((1u << 24) - 1u)

static uint32_t mt_twist(uint32_t u, uint32_t v) {
    return ((u & HD_MT_UPPER_MASK) | (v & HD_MT_LOWER_MASK)) >> 1 ^
           (v & 1u ? HD_MT_MATRIX_A : 0u);
}

static void mt_next_state(hd_torch_rng *rng) {
    uint32_t *p = rng->state;
    rng->left_ = HD_MT_N;
    rng->next_ = 0;
    int j;
    for (j = HD_MT_N - HD_MT_M + 1; --j; p++) {
        *p = p[HD_MT_M] ^ mt_twist(p[0], p[1]);
    }
    for (j = HD_MT_M; --j; p++) {
        *p = p[HD_MT_M - HD_MT_N] ^ mt_twist(p[0], p[1]);
    }
    *p = p[HD_MT_M - HD_MT_N] ^ mt_twist(p[0], rng->state[0]);
}

void hd_torch_rng_seed(hd_torch_rng *rng, uint64_t seed) {
    memset(rng, 0, sizeof(*rng));
    rng->seed_ = seed;
    rng->seeded_ = 1;
    rng->state[0] = (uint32_t)(seed & 0xffffffffu);
    for (int j = 1; j < HD_MT_N; j++) {
        rng->state[j] = 1812433253u * (rng->state[j - 1] ^ (rng->state[j - 1] >> 30)) + (uint32_t)j;
    }
    /* After init, torch's operator() triggers next_state() on first draw. */
    rng->left_ = 1;
    rng->next_ = 0;
}

uint32_t hd_torch_rng_u32(hd_torch_rng *rng) {
    if (--(rng->left_) == 0) {
        mt_next_state(rng);
    }
    uint32_t y = rng->state[rng->next_++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

float hd_torch_uniform_f32(hd_torch_rng *rng) {
    uint32_t v = hd_torch_rng_u32(rng);
    return (float)((v & HD_UNIFORM_MASK) * (1.0f / 16777216.0f));
}

/* normal_fill_16: in-place Box-Muller on 16 uniforms.
 * theta uses c10::pi<double> in double precision, cast to float. */
static void normal_fill_16(float *d) {
    for (int j = 0; j < 8; j++) {
        float u1 = 1.0f - d[j];
        float u2 = d[j + 8];
        float r = sqrtf(-2.0f * logf(u1));
        float th = (float)(6.283185307179586 * (double)u2);
        d[j] = r * cosf(th);
        d[j + 8] = r * sinf(th);
    }
}

void hd_torch_randn_f32(hd_torch_rng *rng, float *out, int64_t n) {
    if (n >= 16) {
        /* normal_fill scalar path: uniform fill all, then Box-Muller in
         * blocks of 16; redraw the tail block when n % 16 != 0. */
        for (int64_t i = 0; i < n; i++) {
            out[i] = hd_torch_uniform_f32(rng);
        }
        int64_t i;
        for (i = 0; i < n - 15; i += 16) {
            normal_fill_16(out + i);
        }
        if (n % 16 != 0) {
            float *d = out + (n - 16);
            for (int j = 0; j < 16; j++) {
                d[j] = hd_torch_uniform_f32(rng);
            }
            normal_fill_16(d);
        }
    } else {
        /* serial normal_distribution<double> path:
         * random64 = (u32 << 32) | u32, 53-bit uniform double,
         * r = sqrt(-2 log1p(-u2)), theta = 2 pi u1, cached sin sample. */
        int cached = 0;
        double cache = 0.0;
        for (int64_t i = 0; i < n; i++) {
            double ret;
            if (cached) {
                ret = cache;
                cached = 0;
            } else {
                uint64_t v1 = ((uint64_t)hd_torch_rng_u32(rng) << 32) | hd_torch_rng_u32(rng);
                uint64_t v2 = ((uint64_t)hd_torch_rng_u32(rng) << 32) | hd_torch_rng_u32(rng);
                double u1 = (double)(v1 & ((1ULL << 53) - 1)) * (1.0 / 9007199254740992.0);
                double u2 = (double)(v2 & ((1ULL << 53) - 1)) * (1.0 / 9007199254740992.0);
                double r = sqrt(-2.0 * log1p(-u2));
                double theta = 2.0 * 3.14159265358979323846 * u1;
                double sample = r * sin(theta);
                cache = sample;
                cached = 1;
                ret = r * cos(theta);
            }
            out[i] = (float)ret;
        }
    }
}
