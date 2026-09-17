#include "weights.h"
#include "sha256.h"
#include "hd_cuda.h"
#include "o1_timing.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

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

    uint8_t *host = NULL;
    int64_t host_cap = 0;

    for (int64_t i = 0; i < idx->n_tensors; i++) {
        const hd_st_tensor *t = &idx->tensors[i];

        if (t->nbytes > host_cap) {
            uint8_t *nh = realloc(host, (size_t)t->nbytes);
            if (!nh) { w_err("oom host staging buffer"); st = HD_ERR_OOM; goto fail; }
            host = nh;
            host_cap = t->nbytes;
        }

        st = hd_st_read_tensor(idx, t, host);
        if (st != HD_OK) { w_err("%s", hd_st_last_error()); goto fail; }

        /* The engine computes in BF16 (M1_NUMERICAL_CONTRACT: "BF16 forward;
         * weights loaded FP32"). Cast FP32 weights to BF16 on upload so the
         * primitive ABI (which consumes BF16 dev buffers) sees the same
         * values as the oracle's bf16-loaded weights. Non-FP32 tensors are
         * uploaded as-is. */
        int64_t dev_nbytes = t->nbytes;
        if (t->dtype == HD_DTYPE_F32) {
            dev_nbytes = t->numel * (int64_t)sizeof(uint16_t);
        }

        void *dev = NULL;
        e = cudaMalloc(&dev, (size_t)dev_nbytes);
        if (e != cudaSuccess) {
            w_err("cudaMalloc %lld bytes for %s: %s", (long long)dev_nbytes,
                  t->name, cudaGetErrorString(e));
            st = HD_ERR_OOM;
            goto fail;
        }
        if (t->dtype == HD_DTYPE_F32) {
            /* fp32 -> bf16 host-side cast, then upload */
            uint8_t *cast = malloc((size_t)dev_nbytes);
            if (!cast) {
                w_err("oom bf16 cast buffer for %s", t->name);
                cudaFree(dev);
                st = HD_ERR_OOM;
                goto fail;
            }
            hd_f32_buf_to_bf16((const float *)host, cast, (size_t)t->numel);
            e = cudaMemcpy(dev, cast, (size_t)dev_nbytes, cudaMemcpyHostToDevice);
            free(cast);
        } else {
            e = cudaMemcpy(dev, host, (size_t)t->nbytes, cudaMemcpyHostToDevice);
        }
        if (e != cudaSuccess) {
            w_err("cudaMemcpy %s: %s", t->name, cudaGetErrorString(e));
            cudaFree(dev);
            st = HD_ERR_IO;
            goto fail;
        }

        hd_device_alloc *a = &info.allocs[info.n_allocs];
        a->dev_ptr = dev;
        a->nbytes = dev_nbytes;
        snprintf(a->name, sizeof(a->name), "%s", t->name);
        info.n_allocs++;
        info.device_bytes_allocated += t->nbytes;
        info.host_bytes_loaded += t->nbytes;
        if (t->numel > info.largest_numel) {
            info.largest_numel = t->numel;
            snprintf(info.largest_name, sizeof(info.largest_name), "%s", t->name);
        }
    }

    free(host);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { w_err("cudaDeviceSynchronize: %s", cudaGetErrorString(e)); st = HD_ERR_IO; goto fail; }

    /* M2: persistent cuBLAS/cuBLASLt GEMM runtime (created once, destroyed
     * in hd_weight_store_free). 64 MiB workspace for cuBLASLt plans. */
    info.gemm = hd_gemm_runtime_init(device_id, 0);
    if (!info.gemm) {
        w_err("gemm runtime init failed: %s", hd_cuda_errbuf());
        st = HD_ERR_IO;
        goto fail;
    }

    O1_TIMING_END("MODEL_LOAD");
    *out = info;
    return HD_OK;

fail:
    free(host);
    if (info.gemm) hd_gemm_runtime_destroy(info.gemm);
    for (int64_t i = 0; i < info.n_allocs; i++) cudaFree(info.allocs[i].dev_ptr);
    free(info.allocs);
    return st;
}

void hd_weight_store_free(hd_weight_store *s) {
    if (!s) return;
    if (s->gemm) hd_gemm_runtime_destroy(s->gemm);
    for (int64_t i = 0; i < s->n_allocs; i++) {
        if (s->allocs[i].dev_ptr) cudaFree(s->allocs[i].dev_ptr);
    }
    free(s->allocs);
    memset(s, 0, sizeof(*s));
}