#ifndef HD_WEIGHTS_H
#define HD_WEIGHTS_H

/*
 * M1.1 weight ingestion: host-side inventory and validation of the frozen
 * HiDream safetensors shards, plus deterministic device placement.
 *
 * Host-side (V0) work needs no CUDA: it proves every expected tensor is
 * resolved, no unexpected tensor is silently ignored, and that representative
 * raw payload bytes match the oracle fingerprint.
 */

#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime.h>

#include "hidream.h"
#include "safetensors.h"
#include "gemm.h"

/* How many representative tensors to fingerprint by default. */
#define HD_WEIGHT_PROBE_MAX 32

typedef struct {
    char name[256];
    char shard[128];
    int64_t numel;
    int64_t nbytes;
    char sha256[65];      /* digest of the raw payload */
    char first_values[256]; /* human-readable preview of the first few values */
} hd_weight_probe;

typedef struct {
    int64_t n_expected;        /* tensors declared by the frozen manifest */
    int64_t n_found;           /* tensors resolved in the shard index */
    int64_t missing;
    int64_t unexpected;
    int64_t shape_mismatch;
    int64_t dtype_mismatch;
    int64_t numel_mismatch;
    int64_t total_bytes;
    int64_t total_numel;
    int64_t largest_numel;
    char largest_name[256];
    int n_probes;
    hd_weight_probe probes[HD_WEIGHT_PROBE_MAX];
} hd_weight_inventory;

/*
 * Cross-checks the shard index against the frozen tensor manifest.
 * Fails closed on any missing/unexpected tensor or metadata disagreement.
 */
hd_status hd_weights_inventory(const char *model_dir,
                               const hd_tensor_manifest *manifest,
                               hd_weight_inventory *out);

/*
 * Fingerprints a representative tensor set spanning the required roles:
 * patch projection, early attention Q/KV, middle MLP, late attention/output,
 * final norm/head. Missing role representatives are a hard error so the
 * coverage guarantee cannot silently regress.
 */
hd_status hd_weights_probe_representatives(const hd_st_index *idx,
                                           hd_weight_inventory *out);

/* Renders the first few raw values of a payload for human inspection. */
void hd_weight_format_preview(hd_dtype dtype, const void *data, int64_t numel,
                              char *out, size_t out_len);

/* ------------------------------------------------------------------ */
/* Device placement                                                    */
/* ------------------------------------------------------------------ */

typedef struct {
    void *dev_ptr;
    int64_t nbytes;
    char name[256];
} hd_device_alloc;

typedef struct {
    int device_id;
    char device_name[256];
    int compute_major;
    int compute_minor;
    int64_t device_mem_bytes;

    int64_t host_bytes_loaded;
    int64_t device_bytes_allocated;
    int64_t n_allocations;
    int64_t largest_numel;
    char largest_name[256];

    hd_device_alloc *allocs;
    int64_t n_allocs;

    /* M2 production GEMM runtime (persistent cuBLAS/cuBLASLt handles). */
    struct hd_gemm_runtime *gemm;

    /* M3 loader: single aligned CUDA weight arena. When arena_ptr is set,
     * every allocs[i].dev_ptr points inside it and hd_weight_store_free
     * releases the arena once instead of per-tensor cudaFree. */
    void *arena_ptr;
    int64_t arena_bytes;
    cudaStream_t upload_stream;
} hd_weight_store;

/* Queries the CUDA device identity without allocating model memory. */
hd_status hd_device_info(int device_id, hd_weight_store *out);

/*
 * Reads each tensor from disk and copies it into its own explicitly tracked
 * device buffer. BF16 weights stay BF16; no dtype conversion is performed.
 * Tensors are placed one at a time so peak host memory stays bounded.
 */
hd_status hd_weights_to_device(const char *model_dir, const hd_st_index *idx,
                               int device_id, hd_weight_store *out);

/*
 * M3: loads a materialized GGUF pack (tools/hidream_convert.py output) into
 * a single aligned CUDA arena using pinned staging + a dedicated nonblocking
 * upload stream. The pack is already BF16 in production order, so the
 * payload streams sequentially with no per-tensor cast or lookup.
 */
hd_status hd_weights_to_device_gguf(const char *gguf_path, int device_id,
                                    hd_weight_store *out);

void hd_weight_store_free(hd_weight_store *s);

/* Last error string set by this module. */
const char *hd_weights_last_error(void);

#endif /* HD_WEIGHTS_H */