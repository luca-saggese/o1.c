#ifndef HD_LORA_H
#define HD_LORA_H

/*
 * M3 LoRA support: merge-on-load.
 *
 * A musubi HiDream-O1 LoRA adapter (.safetensors) is parsed, validated, and
 * merged once into the resident BF16 base weights at model load. After the
 * merge the denoising forward is unchanged (zero LoRA overhead).
 *
 * Contract (pinned to musubi-tuner 4e7c7149):
 *   W' = W + multiplier * (alpha/rank) * (up @ down)
 *   down [rank, in_dim], up [out_dim, rank], FP32 compute, BF16 result.
 *
 * Phase-1 scope: T2I linear LoRA only. Conv/I2I/unknown targets fail
 * explicitly; the adapter is never partially applied.
 */

#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime.h>

#include "hidream.h"
#include "weights.h"

#ifdef __cplusplus
extern "C" {
#endif

/* One resolved linear target. */
typedef struct {
    char lora_prefix[256];   /* lora_unet_<module path with _> */
    char base_tensor[256];   /* model.<module path>.weight */
    uint32_t rank;
    uint32_t in_dim;
    uint32_t out_dim;
    float alpha;
    float multiplier;
    /* device pointers into the adapter staging (freed after merge). */
    void *down_dev;
    void *up_dev;
    /* pointer into the resident base arena. */
    void *base_dev;
} hd_lora_entry;

typedef struct {
    hd_lora_entry *entries;
    size_t count;
    char sha256[65];
    char base_profile[16];
} hd_lora_adapter;

/* One --lora path[:multiplier] spec. */
typedef struct {
    const char *path;
    float multiplier;
} hd_lora_spec;

typedef struct {
    const hd_lora_spec *items;
    size_t count;
} hd_lora_config;

/*
 * Parses and validates the adapter, then merges it into the resident base
 * weights in-place. `store` must be loaded and idle. Returns HD_OK only if
 * every target resolved and merged; otherwise the store is left untouched
 * (validation happens before any mutation).
 */
hd_status hd_lora_apply(const hd_lora_config *cfg, hd_weight_store *store,
                        int device_id);

/* GPU merge (implemented in lora_merge.cu). Mutates base weights in place. */
hd_status hd_lora_merge_gpu(hd_lora_entry *entries, size_t count,
                            cudaStream_t stream);

/* Last error string set by this module. */
const char *hd_lora_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* HD_LORA_H */