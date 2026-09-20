#include "weights.h"
#include "sha256.h"
#include "hd_cuda.h"
#include "o1_timing.h"
#include "gguf.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#include <cuda_runtime.h>

/* M3 loader tuning. */
#define HD_LOADER_STAGE_SLOTS 4
#define HD_LOADER_STAGE_BYTES (128u * 1024u * 1024u) /* 128 MiB per slot */
#define HD_LOADER_ALIGN 256u

static char g_w_error[512] = "";

static void w_err(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_w_error, sizeof(g_w_error), fmt, ap);
    va_end(ap);
}

const char *hd_weights_last_error(void) { return g_w_error; }

/* Role probes required by the M1.1 gate. Each entry is a glob-ish matcher over
 * tensor names; the first match in name order is fingerprinted. */
typedef struct {
    const char *role;
    const char *prefix;   /* exact name prefix to match */
    int required;
} hd_role_probe;

static const hd_role_probe g_roles[] = {
    {"patch_projection", "model.visual.patch_embed.proj.weight", 1},
    {"early_q_projection", "model.language_model.layers.0.self_attn.q_proj.weight", 1},
    {"early_kv_projection", "model.language_model.layers.0.self_attn.k_proj.weight", 1},
    {"early_v_projection", "model.language_model.layers.0.self_attn.v_proj.weight", 1},
    {"middle_mlp", "model.language_model.layers.18.mlp.gate_proj.weight", 1},
    {"middle_mlp_down", "model.language_model.layers.18.mlp.down_proj.weight", 1},
    {"late_attention_out", "model.language_model.layers.35.self_attn.o_proj.weight", 1},
    {"late_mlp_up", "model.language_model.layers.35.mlp.up_proj.weight", 1},
    {"final_norm", "model.language_model.norm.weight", 1},
    {"final_head", "lm_head.weight", 1},
    {"timestep_embed", "model.t_embedder1.mlp.0.weight", 1},
    {"input_embed", "model.x_embedder.proj1.weight", 1},
    {"final_projection", "model.final_layer2.linear.weight", 1},
    {"visual_block_early", "model.visual.blocks.0.attn.qkv.weight", 1},
    {"visual_block_late", "model.visual.blocks.26.mlp.linear_fc1.weight", 1},
    {"token_embed", "model.language_model.embed_tokens.weight", 1},
    {"deepstack_merger", "model.visual.deepstack_merger_list.0.linear_fc1.weight", 1},
    {NULL, NULL, 0},
};

void hd_weight_format_preview(hd_dtype dtype, const void *data, int64_t numel,
                              char *out, size_t out_len) {
    int n = numel < 4 ? (int)numel : 4;
    size_t used = 0;
    out[0] = '\0';
    for (int i = 0; i < n; i++) {
        char tmp[64];
        if (dtype == HD_DTYPE_F32) {
            const float *p = data;
            snprintf(tmp, sizeof(tmp), i ? " %.9g" : "%.9g", (double)p[i]);
        } else if (dtype == HD_DTYPE_BF16) {
            const uint16_t *p = data;
            uint32_t bits = (uint32_t)p[i] << 16;
            float f;
            memcpy(&f, &bits, 4);
            snprintf(tmp, sizeof(tmp), i ? " %.9g" : "%.9g", (double)f);
        } else if (dtype == HD_DTYPE_F16) {
            const uint16_t *p = data;
            uint32_t sign = (uint32_t)(p[i] >> 15) << 31;
            uint32_t exp = (p[i] >> 10) & 0x1f;
            uint32_t man = p[i] & 0x3ff;
            uint32_t bits;
            if (exp == 0) {
                if (man == 0) bits = sign;
                else {
                    exp = 127 - 15 + 1;
                    while (!(man & 0x400)) { man <<= 1; exp--; }
                    man &= 0x3ff;
                    bits = sign | (exp << 23) | (man << 13);
                }
            } else if (exp == 0x1f) {
                bits = sign | 0x7f800000u | (man << 13);
            } else {
                bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
            }
            float f;
            memcpy(&f, &bits, 4);
            snprintf(tmp, sizeof(tmp), i ? " %.9g" : "%.9g", (double)f);
        } else if (dtype == HD_DTYPE_I64) {
            snprintf(tmp, sizeof(tmp), i ? " %lld" : "%lld", (long long)((const int64_t *)data)[i]);
        } else if (dtype == HD_DTYPE_I32) {
            snprintf(tmp, sizeof(tmp), i ? " %d" : "%d", ((const int32_t *)data)[i]);
        } else {
            snprintf(tmp, sizeof(tmp), " ?");
        }
        size_t tl = strlen(tmp);
        if (used + tl + 1 >= out_len) break;
        memcpy(out + used, tmp, tl + 1);
        used += tl;
    }
}

static int meta_ptr_cmp(const void *a, const void *b) {
    const hd_tensor_meta *const *x = a;
    const hd_tensor_meta *const *y = b;
    return strcmp((*x)->name, (*y)->name);
}

hd_status hd_weights_inventory(const char *model_dir,
                               const hd_tensor_manifest *manifest,
                               hd_weight_inventory *out) {
    memset(out, 0, sizeof(*out));
    out->n_expected = manifest->n_tensors;

    hd_st_index idx;
    hd_status st = hd_st_index_load(model_dir, &idx);
    if (st != HD_OK) { w_err("%s", hd_st_last_error()); return st; }
    out->n_found = idx.n_tensors;
    out->total_bytes = idx.total_bytes;

    /* The versioned manifest preserves discovery order, so sort pointers to it
     * before merging with the name-sorted shard index. */
    const hd_tensor_meta **sorted = NULL;
    if (manifest->n_tensors > 0) {
        sorted = malloc((size_t)manifest->n_tensors * sizeof(*sorted));
        if (!sorted) {
            hd_st_index_free(&idx);
            w_err("oom manifest sort buffer");
            return HD_ERR_OOM;
        }
        for (int64_t k = 0; k < manifest->n_tensors; k++) sorted[k] = &manifest->tensors[k];
        qsort(sorted, (size_t)manifest->n_tensors, sizeof(*sorted), meta_ptr_cmp);
    }

    int64_t i = 0, j = 0;
    while (i < manifest->n_tensors && j < idx.n_tensors) {
        int c = strcmp(sorted[i]->name, idx.tensors[j].name);
        if (c == 0) {
            const hd_tensor_meta *e = sorted[i];
            const hd_st_tensor *a = &idx.tensors[j];
            if (e->rank != a->rank) {
                out->shape_mismatch++;
            } else {
                for (int d = 0; d < e->rank; d++) {
                    if (e->shape[d] != a->shape[d]) { out->shape_mismatch++; break; }
                }
            }
            if (e->dtype != a->dtype) out->dtype_mismatch++;
            if (e->numel != a->numel) out->numel_mismatch++;
            if (a->numel > out->largest_numel) {
                out->largest_numel = a->numel;
                snprintf(out->largest_name, sizeof(out->largest_name), "%s", a->name);
            }
            i++; j++;
        } else if (c < 0) {
            out->missing++;
            i++;
        } else {
            out->unexpected++;
            j++;
        }
    }
    if (i < manifest->n_tensors) out->missing += manifest->n_tensors - i;
    if (j < idx.n_tensors) out->unexpected += idx.n_tensors - j;

    free(sorted);

    /* Account for the full payload once the cross-check has consumed the
     * per-tensor metadata; totals come from the index so an early break cannot
     * under-report. */
    out->total_numel = 0;
    for (int64_t k = 0; k < idx.n_tensors; k++) out->total_numel += idx.tensors[k].numel;

    hd_st_index_free(&idx);

    if (out->missing || out->unexpected || out->shape_mismatch ||
        out->dtype_mismatch || out->numel_mismatch) {
        w_err("weight inventory mismatch: %lld missing, %lld unexpected, "
              "%lld shape, %lld dtype, %lld numel",
              (long long)out->missing, (long long)out->unexpected,
              (long long)out->shape_mismatch, (long long)out->dtype_mismatch,
              (long long)out->numel_mismatch);
        return HD_ERR_MISMATCH;
    }
    return HD_OK;
}

hd_status hd_weights_probe_representatives(const hd_st_index *idx,
                                           hd_weight_inventory *out) {
    int64_t scratch_cap = 0;
    uint8_t *scratch = NULL;
    hd_status st = HD_OK;

    for (const hd_role_probe *r = g_roles; r->prefix; r++) {
        const hd_st_tensor *t = hd_st_index_find(idx, r->prefix);
        if (!t) {
            if (r->required) {
                w_err("representative tensor for role '%s' not found: %s",
                      r->role, r->prefix);
                st = HD_ERR_MISSING;
                break;
            }
            continue;
        }
        if (out->n_probes >= HD_WEIGHT_PROBE_MAX) {
            w_err("too many weight probes (max %d)", HD_WEIGHT_PROBE_MAX);
            st = HD_ERR_MANIFEST;
            break;
        }
        if (t->nbytes > scratch_cap) {
            uint8_t *ns = realloc(scratch, (size_t)t->nbytes);
            if (!ns) { w_err("oom probe scratch"); st = HD_ERR_OOM; break; }
            scratch = ns;
            scratch_cap = t->nbytes;
        }
        st = hd_st_read_tensor(idx, t, scratch);
        if (st != HD_OK) { w_err("%s", hd_st_last_error()); break; }

        hd_weight_probe *p = &out->probes[out->n_probes];
        memset(p, 0, sizeof(*p));
        snprintf(p->name, sizeof(p->name), "%s", t->name);
        snprintf(p->shard, sizeof(p->shard), "%s", t->shard);
        p->numel = t->numel;
        p->nbytes = t->nbytes;
        hd_sha256_ctx ctx;
        hd_sha256_init(&ctx);
        hd_sha256_update(&ctx, scratch, (size_t)t->nbytes);
        uint8_t digest[32];
        hd_sha256_final(&ctx, digest);
        hd_sha256_hex(digest, p->sha256);
        hd_weight_format_preview(t->dtype, scratch, t->numel,
                                 p->first_values, sizeof(p->first_values));
        out->n_probes++;
    }

    free(scratch);
    return st;
}

/* ------------------------------------------------------------------ */
/* Device placement                                                    */
/* ------------------------------------------------------------------ */

hd_status hd_device_info(int device_id, hd_weight_store *out) {
    memset(out, 0, sizeof(*out));
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) {
        w_err("cudaGetDeviceCount: %s", cudaGetErrorString(e));
        return HD_ERR_IO;
    }
    if (device_id < 0 || device_id >= count) {
        w_err("device %d out of range (visible=%d)", device_id, count);
        return HD_ERR_PROFILE;
    }
    struct cudaDeviceProp prop;
    e = cudaGetDeviceProperties(&prop, device_id);
    if (e != cudaSuccess) {
        w_err("cudaGetDeviceProperties: %s", cudaGetErrorString(e));
        return HD_ERR_IO;
    }
    out->device_id = device_id;
    snprintf(out->device_name, sizeof(out->device_name), "%s", prop.name);
    out->compute_major = prop.major;
    out->compute_minor = prop.minor;
    out->device_mem_bytes = (int64_t)prop.totalGlobalMem;
    return HD_OK;
}

/* ------------------------------------------------------------------ */
/* M3 pipelined loader                                                 */
/* ------------------------------------------------------------------ */

typedef struct {
    int fd;
    char shard[128];
} hd_shard_fd;

typedef struct {
    const hd_st_tensor *t;
    int64_t orig_idx;
} hd_io_entry;

static int tensor_io_cmp(const void *a, const void *b) {
    const hd_io_entry *x = a;
    const hd_io_entry *y = b;
    int c = strcmp(x->t->shard, y->t->shard);
    if (c) return c;
    if (x->t->data_begin < y->t->data_begin) return -1;
    if (x->t->data_begin > y->t->data_begin) return 1;
    return 0;
}

static int64_t hd_dev_nbytes(const hd_st_tensor *t) {
    return (t->dtype == HD_DTYPE_F32) ? t->numel * (int64_t)sizeof(uint16_t)
                                      : t->nbytes;
}

static int64_t hd_align_up(int64_t v, int64_t a) {
    return (v + a - 1) & ~(a - 1);
}

hd_status hd_weights_to_device(const char *model_dir, const hd_st_index *idx,
                               int device_id, hd_weight_store *out) {
    O1_TIMING_BEGIN("MODEL_LOAD");
    hd_weight_store info;
    hd_status st = hd_device_info(device_id, &info);
    if (st != HD_OK) return st;

    cudaError_t e = cudaSetDevice(device_id);
    if (e != cudaSuccess) { w_err("cudaSetDevice: %s", cudaGetErrorString(e)); return HD_ERR_IO; }
    (void)model_dir; /* tensors carry their own shard paths in the index */

    info.allocs = calloc((size_t)idx->n_tensors, sizeof(hd_device_alloc));
    if (!info.allocs) { w_err("oom alloc table"); return HD_ERR_OOM; }

    /* ---- pass 1: placement plan (no I/O, no CUDA alloc) ---- */
    int64_t arena_bytes = 0;
    int64_t max_raw = 0, max_dev = 0;
    for (int64_t i = 0; i < idx->n_tensors; i++) {
        const hd_st_tensor *t = &idx->tensors[i];
        int64_t dev = hd_dev_nbytes(t);
        info.allocs[i].nbytes = dev;
        snprintf(info.allocs[i].name, sizeof(info.allocs[i].name), "%s", t->name);
        arena_bytes = hd_align_up(arena_bytes, HD_LOADER_ALIGN) + dev;
        if (t->nbytes > max_raw) max_raw = t->nbytes;
        if (dev > max_dev) max_dev = dev;
    }
    arena_bytes = hd_align_up(arena_bytes, HD_LOADER_ALIGN);

    /* ---- one aligned CUDA arena ---- */
    O1_TIMING_BEGIN("CUDA_ALLOC");
    e = cudaMalloc(&info.arena_ptr, (size_t)arena_bytes);
    O1_TIMING_END("CUDA_ALLOC");
    if (e != cudaSuccess) {
        w_err("cudaMalloc arena %lld bytes: %s", (long long)arena_bytes,
              cudaGetErrorString(e));
        st = HD_ERR_OOM;
        goto fail;
    }
    info.arena_bytes = arena_bytes;
    O1_TIMING_COUNTER_SET("cuda_malloc_calls", 1);

    /* ---- pinned staging + dedicated nonblocking upload stream ---- */
    uint8_t *raw_slot = NULL;
    uint8_t *dev_slot[2] = {NULL, NULL};
    cudaEvent_t slot_ev[2] = {NULL, NULL};
    e = cudaMallocHost(&raw_slot, (size_t)max_raw);
    if (e != cudaSuccess) { w_err("cudaMallocHost raw: %s", cudaGetErrorString(e)); st = HD_ERR_OOM; goto fail; }
    e = cudaMallocHost(&dev_slot[0], (size_t)max_dev);
    if (e != cudaSuccess) { w_err("cudaMallocHost dev0: %s", cudaGetErrorString(e)); st = HD_ERR_OOM; goto fail; }
    e = cudaMallocHost(&dev_slot[1], (size_t)max_dev);
    if (e != cudaSuccess) { w_err("cudaMallocHost dev1: %s", cudaGetErrorString(e)); st = HD_ERR_OOM; goto fail; }
    e = cudaStreamCreateWithFlags(&info.upload_stream, cudaStreamNonBlocking);
    if (e != cudaSuccess) { w_err("cudaStreamCreate: %s", cudaGetErrorString(e)); st = HD_ERR_OOM; goto fail; }
    cudaEventCreateWithFlags(&slot_ev[0], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&slot_ev[1], cudaEventDisableTiming);

    /* ---- open each shard once, keep the fd ---- */
    int n_shards = 0;
    hd_shard_fd shards[16];
    for (int64_t i = 0; i < idx->n_tensors; i++) {
        const char *s = idx->tensors[i].shard;
        int found = 0;
        for (int k = 0; k < n_shards; k++) {
            if (strcmp(shards[k].shard, s) == 0) { found = 1; break; }
        }
        if (found) continue;
        if (n_shards >= 16) { w_err("too many shards"); st = HD_ERR_MANIFEST; goto fail; }
        char path[2048];
        snprintf(path, sizeof(path), "%s/%s", idx->dir, s);
        int fd = open(path, O_RDONLY);
        if (fd < 0) { w_err("open %s: %s", path, strerror(errno)); st = HD_ERR_IO; goto fail; }
        snprintf(shards[n_shards].shard, sizeof(shards[n_shards].shard), "%s", s);
        shards[n_shards].fd = fd;
        n_shards++;
    }

    /* ---- pass 2: sequential pread + async H2D, double-buffered ---- */
    hd_io_entry *order = malloc((size_t)idx->n_tensors * sizeof(*order));
    if (!order) { w_err("oom order table"); st = HD_ERR_OOM; goto fail; }
    for (int64_t i = 0; i < idx->n_tensors; i++) {
        order[i].t = &idx->tensors[i];
        order[i].orig_idx = i;
    }
    qsort(order, (size_t)idx->n_tensors, sizeof(*order), tensor_io_cmp);

    O1_TIMING_BEGIN("FILE_READ");
    int64_t arena_off = 0;
    int cur = 0;
    for (int64_t i = 0; i < idx->n_tensors; i++) {
        const hd_st_tensor *t = order[i].t;
        int64_t oi = order[i].orig_idx;
        int fd = -1;
        for (int k = 0; k < n_shards; k++) {
            if (strcmp(shards[k].shard, t->shard) == 0) { fd = shards[k].fd; break; }
        }
        if (fd < 0) { w_err("no fd for %s", t->shard); st = HD_ERR_IO; goto fail; }

        /* Wait for this slot's previous upload before overwriting it. */
        if (i >= 2) cudaEventSynchronize(slot_ev[cur]);

        /* pread in a loop: a single syscall is capped at ~2 GiB. */
        size_t got = 0;
        while (got < (size_t)t->nbytes) {
            ssize_t rd = pread(fd, (uint8_t *)raw_slot + got,
                               (size_t)t->nbytes - got,
                               (off_t)t->data_begin + (off_t)got);
            if (rd <= 0) {
                w_err("short read %s (%zu/%lld)", t->shard, got,
                      (long long)t->nbytes);
                st = HD_ERR_IO;
                goto fail;
            }
            got += (size_t)rd;
        }
        O1_TIMING_COUNTER_ADD("file_read_calls", 1);
        O1_TIMING_COUNTER_ADD("file_read_bytes", (double)t->nbytes);

        uint8_t *src = raw_slot;
        int64_t dev = hd_dev_nbytes(t);
        if (t->dtype == HD_DTYPE_F32) {
            hd_f32_buf_to_bf16((const float *)raw_slot, dev_slot[cur], (size_t)t->numel);
            src = dev_slot[cur];
        } else {
            memcpy(dev_slot[cur], raw_slot, (size_t)t->nbytes);
            src = dev_slot[cur];
        }

        e = cudaMemcpyAsync((uint8_t *)info.arena_ptr + arena_off, src,
                            (size_t)dev, cudaMemcpyHostToDevice, info.upload_stream);
        if (e != cudaSuccess) {
            w_err("cudaMemcpyAsync %s: %s", t->name, cudaGetErrorString(e));
            st = HD_ERR_IO;
            goto fail;
        }
        cudaEventRecord(slot_ev[cur], info.upload_stream);
        O1_TIMING_COUNTER_ADD("h2d_calls", 1);
        O1_TIMING_COUNTER_ADD("h2d_bytes", (double)dev);

        info.allocs[oi].dev_ptr = (uint8_t *)info.arena_ptr + arena_off;
        arena_off = hd_align_up(arena_off, HD_LOADER_ALIGN) + dev;
        info.n_allocs++;
        info.device_bytes_allocated += t->nbytes;
        info.host_bytes_loaded += t->nbytes;
        cur ^= 1;
    }
    O1_TIMING_END("FILE_READ");

    /* largest tensor: iterate in name order (matches the legacy loader's
     * tie-break, where embed_tokens and lm_head share the same numel). */
    for (int64_t i = 0; i < idx->n_tensors; i++) {
        const hd_st_tensor *t = &idx->tensors[i];
        if (t->numel > info.largest_numel) {
            info.largest_numel = t->numel;
            snprintf(info.largest_name, sizeof(info.largest_name), "%s", t->name);
        }
    }

    free(order);
    for (int k = 0; k < n_shards; k++) close(shards[k].fd);
    cudaEventDestroy(slot_ev[0]);
    cudaEventDestroy(slot_ev[1]);
    cudaFreeHost(raw_slot);
    cudaFreeHost(dev_slot[0]);
    cudaFreeHost(dev_slot[1]);

    e = cudaStreamSynchronize(info.upload_stream);
    if (e != cudaSuccess) { w_err("upload stream sync: %s", cudaGetErrorString(e)); st = HD_ERR_IO; goto fail; }

    O1_TIMING_COUNTER_SET("tensor_count", (double)idx->n_tensors);
    O1_TIMING_COUNTER_SET("disk_gbps", info.host_bytes_loaded / 1e9 /
                          (o1_timing_region_seconds("FILE_READ") > 0
                               ? o1_timing_region_seconds("FILE_READ") : 1e-9));
    /* H2D is overlapped with FILE_READ on the upload stream; report the
     * effective rate over the same window. */
    O1_TIMING_COUNTER_SET("h2d_gbps", info.host_bytes_loaded / 1e9 /
                          (o1_timing_region_seconds("FILE_READ") > 0
                               ? o1_timing_region_seconds("FILE_READ") : 1e-9));

    /* M2: persistent cuBLASLt GEMM runtime (created once, destroyed in
     * hd_weight_store_free). 64 MiB persistent workspace for cuBLASLt
     * algorithms; no allocation happens in the forward hot path. */
    info.gemm = hd_gemm_runtime_init(device_id, 64u << 20);
    if (!info.gemm) {
        w_err("gemm runtime init failed: %s", hd_cuda_errbuf());
        st = HD_ERR_IO;
        goto fail;
    }

    O1_TIMING_END("MODEL_LOAD");
    *out = info;
    return HD_OK;

fail:
    if (info.arena_ptr) cudaFree(info.arena_ptr);
    if (info.upload_stream) cudaStreamDestroy(info.upload_stream);
    if (info.gemm) hd_gemm_runtime_destroy(info.gemm);
    free(info.allocs);
    return st;
}

/* ------------------------------------------------------------------ */
/* M3 GGUF pack loader                                                 */
/* ------------------------------------------------------------------ */

hd_status hd_weights_to_device_gguf(const char *gguf_path, int device_id,
                                    hd_weight_store *out) {
    O1_TIMING_BEGIN("MODEL_LOAD");
    hd_weight_store info;
    hd_status st = hd_device_info(device_id, &info);
    if (st != HD_OK) return st;

    cudaError_t e = cudaSetDevice(device_id);
    if (e != cudaSuccess) { w_err("cudaSetDevice: %s", cudaGetErrorString(e)); return HD_ERR_IO; }

    hd_gguf_file gf;
    st = hd_gguf_open(gguf_path, &gf);
    if (st != HD_OK) { w_err("gguf: %s", hd_gguf_last_error()); return st; }

    info.allocs = calloc((size_t)gf.n_tensors, sizeof(hd_device_alloc));
    if (!info.allocs) { w_err("oom alloc table"); hd_gguf_close(&gf); return HD_ERR_OOM; }

    /* ---- one aligned CUDA arena ---- */
    O1_TIMING_BEGIN("CUDA_ALLOC");
    e = cudaMalloc(&info.arena_ptr, (size_t)gf.payload_bytes);
    O1_TIMING_END("CUDA_ALLOC");
    if (e != cudaSuccess) {
        w_err("cudaMalloc arena %llu bytes: %s",
              (unsigned long long)gf.payload_bytes, cudaGetErrorString(e));
        hd_gguf_close(&gf);
        free(info.allocs);
        return HD_ERR_OOM;
    }
    info.arena_bytes = (int64_t)gf.payload_bytes;
    O1_TIMING_COUNTER_SET("cuda_malloc_calls", 1);

    /* ---- pinned staging + dedicated nonblocking upload stream ---- */
    uint8_t *stage[HD_LOADER_STAGE_SLOTS] = {0};
    cudaEvent_t ev[HD_LOADER_STAGE_SLOTS] = {0};
    for (int i = 0; i < HD_LOADER_STAGE_SLOTS; i++) {
        e = cudaMallocHost(&stage[i], HD_LOADER_STAGE_BYTES);
        if (e != cudaSuccess) {
            w_err("cudaMallocHost stage %d: %s", i, cudaGetErrorString(e));
            st = HD_ERR_OOM;
            goto fail;
        }
        cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming);
    }
    e = cudaStreamCreateWithFlags(&info.upload_stream, cudaStreamNonBlocking);
    if (e != cudaSuccess) { w_err("cudaStreamCreate: %s", cudaGetErrorString(e)); st = HD_ERR_OOM; goto fail; }

    /* ---- stream the payload sequentially: file -> pinned -> arena ---- */
    int fd = open(gguf_path, O_RDONLY);
    if (fd < 0) { w_err("open %s: %s", gguf_path, strerror(errno)); st = HD_ERR_IO; goto fail; }

    O1_TIMING_BEGIN("FILE_READ");
    uint64_t payload = gf.payload_bytes;
    uint64_t off = 0;
    int slot = 0;
    while (off < payload) {
        size_t n = (size_t)((payload - off) < HD_LOADER_STAGE_BYTES
                                ? (payload - off) : HD_LOADER_STAGE_BYTES);
        /* Wait for this slot's previous upload before overwriting it. */
        if (off >= (uint64_t)HD_LOADER_STAGE_BYTES * HD_LOADER_STAGE_SLOTS)
            cudaEventSynchronize(ev[slot]);

        size_t got = 0;
        while (got < n) {
            ssize_t rd = pread(fd, stage[slot] + got, n - got,
                               (off_t)(gf.tensor_data_off + off + got));
            if (rd <= 0) {
                w_err("short read gguf (%zu/%zu)", got, n);
                close(fd);
                st = HD_ERR_IO;
                goto fail;
            }
            got += (size_t)rd;
        }
        O1_TIMING_COUNTER_ADD("file_read_calls", 1);
        O1_TIMING_COUNTER_ADD("file_read_bytes", (double)n);

        e = cudaMemcpyAsync((uint8_t *)info.arena_ptr + off, stage[slot], n,
                            cudaMemcpyHostToDevice, info.upload_stream);
        if (e != cudaSuccess) {
            w_err("cudaMemcpyAsync: %s", cudaGetErrorString(e));
            close(fd);
            st = HD_ERR_IO;
            goto fail;
        }
        cudaEventRecord(ev[slot], info.upload_stream);
        O1_TIMING_COUNTER_ADD("h2d_calls", 1);
        O1_TIMING_COUNTER_ADD("h2d_bytes", (double)n);

        off += n;
        slot = (slot + 1) % HD_LOADER_STAGE_SLOTS;
    }
    O1_TIMING_END("FILE_READ");
    close(fd);

    e = cudaStreamSynchronize(info.upload_stream);
    if (e != cudaSuccess) { w_err("upload stream sync: %s", cudaGetErrorString(e)); st = HD_ERR_IO; goto fail; }

    /* ---- bind pointers: tensor i lives at arena + offset ---- */
    for (int64_t i = 0; i < gf.n_tensors; i++) {
        const hd_gguf_tensor *t = &gf.tensors[i];
        hd_device_alloc *a = &info.allocs[i];
        a->dev_ptr = (uint8_t *)info.arena_ptr + t->offset;
        a->nbytes = (int64_t)t->nbytes;
        snprintf(a->name, sizeof(a->name), "%s", t->name);
        info.n_allocs++;
        info.device_bytes_allocated += (int64_t)t->nbytes;
        info.host_bytes_loaded += (int64_t)t->nbytes;
        if (t->nbytes > (uint64_t)info.largest_numel) {
            info.largest_numel = (int64_t)t->nbytes;
            snprintf(info.largest_name, sizeof(info.largest_name), "%s", t->name);
        }
    }

    for (int i = 0; i < HD_LOADER_STAGE_SLOTS; i++) {
        cudaEventDestroy(ev[i]);
        cudaFreeHost(stage[i]);
    }

    O1_TIMING_COUNTER_SET("tensor_count", (double)gf.n_tensors);
    O1_TIMING_COUNTER_SET("disk_gbps", info.host_bytes_loaded / 1e9 /
                          (o1_timing_region_seconds("FILE_READ") > 0
                               ? o1_timing_region_seconds("FILE_READ") : 1e-9));
    O1_TIMING_COUNTER_SET("h2d_gbps", info.host_bytes_loaded / 1e9 /
                          (o1_timing_region_seconds("FILE_READ") > 0
                               ? o1_timing_region_seconds("FILE_READ") : 1e-9));

    info.gemm = hd_gemm_runtime_init(device_id, 64u << 20);
    if (!info.gemm) {
        w_err("gemm runtime init failed: %s", hd_cuda_errbuf());
        hd_gguf_close(&gf);
        st = HD_ERR_IO;
        goto fail;
    }

    hd_gguf_close(&gf);
    O1_TIMING_END("MODEL_LOAD");
    *out = info;
    return HD_OK;

fail:
    for (int i = 0; i < HD_LOADER_STAGE_SLOTS; i++) {
        if (ev[i]) cudaEventDestroy(ev[i]);
        if (stage[i]) cudaFreeHost(stage[i]);
    }
    if (info.arena_ptr) cudaFree(info.arena_ptr);
    if (info.upload_stream) cudaStreamDestroy(info.upload_stream);
    if (info.gemm) hd_gemm_runtime_destroy(info.gemm);
    hd_gguf_close(&gf);
    free(info.allocs);
    return st;
}

void hd_weight_store_free(hd_weight_store *s) {
    if (!s) return;
    if (s->gemm) hd_gemm_runtime_destroy(s->gemm);
    if (s->upload_stream) cudaStreamDestroy(s->upload_stream);
    if (s->arena_ptr) {
        cudaFree(s->arena_ptr);
    } else {
        for (int64_t i = 0; i < s->n_allocs; i++) {
            if (s->allocs[i].dev_ptr) cudaFree(s->allocs[i].dev_ptr);
        }
    }
    free(s->allocs);
    memset(s, 0, sizeof(*s));
}