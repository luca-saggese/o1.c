#ifndef HD_SAFETENSORS_H
#define HD_SAFETENSORS_H

/*
 * Minimal safetensors reader for the frozen HiDream Dev/Base shards.
 *
 * Scope (M1.1): resolve the model.safetensors.index.json weight map, then read
 * tensor headers and raw tensor payloads directly out of the shard files.
 * No model math, no Python, no network.
 *
 * safetensors layout: [u64 little-endian header length][JSON header][payload].
 * The header JSON maps tensor name -> {dtype, shape, data_offsets}, with
 * data_offsets relative to the start of the payload region.
 */

#include <stddef.h>
#include <stdint.h>

#include "hidream.h"

typedef struct {
    char *name;
    char *shard;       /* basename of the owning shard file */
    hd_dtype dtype;
    int64_t *shape;
    int rank;
    int64_t numel;
    int64_t data_begin; /* absolute file offset of the payload */
    int64_t nbytes;
} hd_st_tensor;

typedef struct {
    char *dir;              /* model directory */
    char *index_path;
    int64_t n_tensors;
    hd_st_tensor *tensors;  /* sorted by name */
    int64_t total_bytes;
} hd_st_index;

/* Loads <dir>/model.safetensors.index.json and every shard header. */
hd_status hd_st_index_load(const char *dir, hd_st_index *out);
void hd_st_index_free(hd_st_index *idx);

const hd_st_tensor *hd_st_index_find(const hd_st_index *idx, const char *name);

/* Reads nbytes at absolute offset into a caller-provided buffer. */
hd_status hd_st_read(const char *dir, const char *shard, int64_t offset,
                     int64_t nbytes, void *dst);

/* Reads one tensor's raw payload into dst (must be at least nbytes). */
hd_status hd_st_read_tensor(const hd_st_index *idx, const hd_st_tensor *t,
                            void *dst);

/* Parses just the header of a single .safetensors file (used for discovery). */
hd_status hd_st_header_scan(const char *path, hd_st_index *out);

/* Last error string set by this module. */
const char *hd_st_last_error(void);

#endif /* HD_SAFETENSORS_H */