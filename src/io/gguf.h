#ifndef HD_GGUF_H
#define HD_GGUF_H

/*
 * Minimal GGUF v3 reader for the o1.c materialized weight pack.
 *
 * Scope: parse the header (magic/version/tensor_count/KV count), the tensor
 * info table (name/shape/type/offset), and stream tensor payloads. No
 * ggml/llama.cpp dependency. The pack is produced by tools/hidream_convert.py
 * with general.alignment = 256 and BF16 tensors in production order.
 *
 * GGUF layout (v3, little-endian):
 *   [u32 magic "GGUF"][u32 version][u64 tensor_count][u64 kv_count]
 *   [kv pairs...][tensor infos...][pad to alignment][tensor data...]
 */

#include <stddef.h>
#include <stdint.h>

#include "hidream.h"

/* ggml_type values we accept. */
#define HD_GGML_TYPE_F32 0
#define HD_GGML_TYPE_F16 1
#define HD_GGML_TYPE_BF16 30

typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t dims[8];
    uint32_t type;        /* ggml_type */
    uint64_t offset;      /* relative to tensor_data */
    uint64_t nbytes;      /* payload bytes for this tensor */
} hd_gguf_tensor;

typedef struct {
    char *path;
    int64_t n_tensors;
    hd_gguf_tensor *tensors;  /* in file order (production order) */
    uint64_t alignment;
    uint64_t tensor_data_off; /* absolute file offset of tensor_data */
    uint64_t payload_bytes;
    char *arch;               /* general.architecture */
    char *name;               /* general.name */
    char *profile;            /* hidream.profile */
    char *variant;            /* hidream.variant */
    char *revision;           /* hidream.revision */
    char *dtype;              /* hidream.dtype */
    char *quantization;       /* hidream.quantization */
    int64_t num_layers;       /* hidream.num_layers */
    int64_t layout_version;   /* hidream.layout_version */
} hd_gguf_file;

/* Parses the GGUF header + tensor table. No payload I/O. */
hd_status hd_gguf_open(const char *path, hd_gguf_file *out);
void hd_gguf_close(hd_gguf_file *f);

/* Reads one tensor's payload into dst (must be >= nbytes). */
hd_status hd_gguf_read_tensor(const hd_gguf_file *f, const hd_gguf_tensor *t,
                              void *dst);

/* Last error string set by this module. */
const char *hd_gguf_last_error(void);

#endif /* HD_GGUF_H */