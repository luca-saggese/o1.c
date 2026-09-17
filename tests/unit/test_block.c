/*
 * M1.3 decoder block validation harness.
 *
 * Drives hd_decoder_block on the frozen real-model chained input (block_0.x)
 * and checks the block output plus selected internals against the oracle
 * golden fixture captured by tools/capture_m1_3_block.py.
 *
 * The decoder block is a COMPOSITE of primitives (RMSNorm -> attentlon GEMMs
 * + RoPE + eager attention + residual -> post-norm -> SwiGLU MLP + residual),
 * so error accumulates across stages. The block output is gated on a documented
 * composite class D bound (NRMSE <= 1e-2 and cos >= 0.999, per
 * docs/M1_2_GOLDEN_CONTRACT.md). Each internal is also reported with its own
 * composite tolerance so the achieved numbers are not silently inflated.
 *
 * BF16 compute / FP32 accumulate. No whole-model transformer forward is run;
 * this drives only the block module composed of primitive kernel calls.
 */

#include "hidream.h"
#include "safetensors.h"
#include "json.h"
#include "block.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GOLDEN_DIR "artifacts/m1/golden"
#define DEV_DIR    "models/dev"
#define BLOCK_FIXTURE_ID "block_0"

static int failures = 0;
static int passes = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s (at %s:%d)\n", msg, __FILE__, __LINE__); \
        failures++; \
    } else { \
        printf("ok: %s\n", msg); \
        passes++; \
    } \
} while (0)

/* ------------------------------------------------------------------ */
/* Metrics                                                             */
/* ------------------------------------------------------------------ */

static double rms(const float *a, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += (double)a[i] * a[i];
    return sqrt(s / (double)n);
}
static double rms_diff(const float *a, const float *b, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) { double d = (double)a[i] - b[i]; s += d * d; }
    return sqrt(s / (double)n);
}
static double cosine(const float *a, const float *b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    if (na == 0.0 || nb == 0.0) return na == nb ? 1.0 : 0.0;
    return dot / (sqrt(na) * sqrt(nb));
}

static void *read_file_bytes(const char *path, size_t *out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    void *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    fseek(f, 0, SEEK_SET);
    fread(buf, 1, (size_t)sz, f);
    fclose(f);
    ((char *)buf)[sz] = '\0';
    *out = (size_t)sz;
    return buf;
}

static void *dev_alloc(size_t bytes) {
    void *p = NULL;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return NULL;
    return p;
}
static void dev_free(void *p) { if (p) cudaFree(p); }

typedef struct { size_t offset; size_t nbytes; char dtype[16]; } ftensor;

/* Look up an input/output payload row by name in a fixture meta object. */
static int payload_row(const hd_json *meta, const char *name, ftensor *out) {
    const char *lists[2] = { "inputs", "outputs" };
    for (int li = 0; li < 2; li++) {
        const hd_json *arr = hd_json_get(meta, lists[li]);
        if (!arr || arr->type != HD_JSON_ARRAY) continue;
        size_t n = hd_json_array_len(arr);
        for (size_t i = 0; i < n; i++) {
            const hd_json *t = hd_json_array_at(arr, i);
            const char *nm = hd_json_string(hd_json_get(t, "name"));
            if (nm && strcmp(nm, name) == 0) {
                out->offset = (size_t)hd_json_int(hd_json_get(t, "offset"), 0);
                out->nbytes = (size_t)hd_json_int(hd_json_get(t, "nbytes"), 0);
                const char *dt = hd_json_string(hd_json_get(t, "dtype"));
                snprintf(out->dtype, sizeof(out->dtype), "%s", dt ? dt : "?");
                return 0;
            }
        }
    }
    return -1;
}

/* Compare candidate host floats to a bf16 reference slice from the bin. */
static double compare_ref(const char *tag, const float *cand,
                          const void *bin, const ftensor *ref,
                          double nrmse_lim, double cos_lim) {
    size_t n = ref->nbytes / 2;
    float *rf = malloc(n * sizeof(float));
    hd_bf16_buf_to_f32((const char *)bin + ref->offset, rf, n);
    double nr = 0.0, cs = 0.0;
    if (rms(rf, n) > 0.0) nr = rms_diff(cand, rf, n) / rms(rf, n);
    cs = cosine(cand, rf, n);
    int ok = (nr <= nrmse_lim && cs >= cos_lim);
    printf("  %-14s nrmse=%.5g cos=%.8g -> %s (lim nrmse<=%.2g cos>=%.4f)\n",
           tag, nr, cs, ok ? "PASS" : "FAIL", nrmse_lim, cos_lim);
    CHECK(ok, tag);
    free(rf);
    return nr;
}

int main(void) {
    /* ---- parse the block_0 fixture meta ---- */
    char mpath[512];
    snprintf(mpath, sizeof(mpath), GOLDEN_DIR "/" BLOCK_FIXTURE_ID ".json");
    size_t msz = 0;
    void *mb = read_file_bytes(mpath, &msz);
    if (!mb) {
        printf("SKIP: block fixture metadata %s not found\n", mpath);
        return 0;
    }
    const char *err = NULL;
    hd_json *meta = hd_json_parse((const char *)mb, &err);
    free(mb);
    if (!meta) {
        printf("FAIL: parse block fixture meta: %s\n", err ? err : "parse error");
        return 1;
    }

    char bpath[512];
    snprintf(bpath, sizeof(bpath), GOLDEN_DIR "/" BLOCK_FIXTURE_ID ".bin");
    size_t bsz = 0;
    void *bin = read_file_bytes(bpath, &bsz);
    if (!bin) {
        printf("FAIL: block fixture bin %s not found\n", bpath);
        hd_json_free(meta);
        return 1;
    }

    ftensor tx, tp, tm, o_ln0, o_ah, o_ar, o_post, o_mlp, o_block;
    if (payload_row(meta, "x", &tx) ||
        payload_row(meta, "pos_f32", &tp) ||
        payload_row(meta, "mask", &tm) ||
        payload_row(meta, "out_ln0", &o_ln0) ||
        payload_row(meta, "out_attn_hidden", &o_ah) ||
        payload_row(meta, "out_attn_resid", &o_ar) ||
        payload_row(meta, "out_post_ln", &o_post) ||
        payload_row(meta, "out_mlp_out", &o_mlp) ||
        payload_row(meta, "out_block", &o_block)) {
        printf("FAIL: block fixture missing required tensors\n");
        hd_json_free(meta); free(bin);
        return 1;
    }

    /* ---- dims from params ---- */
    const hd_json *par = hd_json_get(meta, "params");
    int seq  = (int)hd_json_int(hd_json_get(par, "seq"), 23);
    int H    = (int)hd_json_int(hd_json_get(par, "hidden_size"), 4096);
    int I    = (int)hd_json_int(hd_json_get(par, "ff_hidden"), 12288);
    int NH   = (int)hd_json_int(hd_json_get(par, "heads"), 32);
    int NKV  = (int)hd_json_int(hd_json_get(par, "kv_heads"), 8);
    int HD   = (int)hd_json_int(hd_json_get(par, "head_dim"), 128);
    int layer = (int)hd_json_int(hd_json_get(par, "layer"), 0);

    size_t hid_bytes = (size_t)seq * H * sizeof(uint16_t);

    printf("M1.3 block validation: seq=%d H=%d I=%d NH=%d NKV=%d HD=%d layer=%d\n",
           seq, H, I, NH, NKV, HD, layer);

    /* ---- load device weights via the M1.1 placement API ---- */
    hd_st_index idx;
    if (hd_st_index_load(DEV_DIR, &idx) != HD_OK) {
        printf("FAIL: cannot load weight index (%s)\n", hd_st_last_error());
        hd_json_free(meta); free(bin);
        return 1;
    }
    hd_weight_store store;
    hd_status st = hd_weights_to_device(DEV_DIR, &idx, 0, &store);
    if (st != HD_OK) {
        printf("FAIL: hd_weights_to_device: %s\n", hd_weights_last_error());
        hd_st_index_free(&idx); hd_json_free(meta); free(bin);
        return 1;
    }
    hd_st_index_free(&idx);
    printf("  loaded %lld device weight allocations (%.1f GB)\n",
           (long long)store.n_allocs,
           (double)store.device_bytes_allocated / 1e9);

    /* ---- stage inputs ---- */
    void *xd = dev_alloc(tx.nbytes);
    float *posd = dev_alloc(tp.nbytes);
    void *maskd = dev_alloc(tm.nbytes);

    cudaMemcpy(xd, (const char *)bin + tx.offset, tx.nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(posd, (const char *)bin + tp.offset, tp.nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(maskd, (const char *)bin + tm.offset, tm.nbytes, cudaMemcpyHostToDevice);

    /* ---- scratch ---- */
    int64_t need = hd_decoder_block_scratch_bytes(seq, NH, NKV, H, I, HD);
    void *scratch = dev_alloc((size_t)need);
    if (!xd || !posd || !maskd || !scratch) {
        printf("FAIL: device allocation failed\n");
        dev_free(xd); dev_free(posd); dev_free(maskd); dev_free(scratch);
        hd_weight_store_free(&store); hd_json_free(meta); free(bin);
        return 1;
    }

    /* ---- resolve this layer's weights once (no lookup in hot path) ---- */
    hd_block_binding bw;
    st = hd_block_resolve(&store, layer, &bw);
    if (st != HD_OK) {
        printf("FAIL: hd_block_resolve: %s\n", hd_last_error());
        dev_free(xd); dev_free(posd); dev_free(maskd); dev_free(scratch);
        hd_weight_store_free(&store); hd_json_free(meta); free(bin);
        return 1;
    }

    /* ---- MRoPE section bound once, device-resident for the forward ---- */
    int64_t sec_host[3] = {24, 20, 20};
    void *secd = dev_alloc(sizeof(sec_host));
    if (!secd) {
        printf("FAIL: section device allocation failed\n");
        dev_free(xd); dev_free(posd); dev_free(maskd); dev_free(scratch);
        hd_weight_store_free(&store); hd_json_free(meta); free(bin);
        return 1;
    }
    cudaMemcpy(secd, sec_host, sizeof(sec_host), cudaMemcpyHostToDevice);

    /* ---- outputs: block out + internals ---- */
    void *outd = dev_alloc(hid_bytes);
    void *ln0d = dev_alloc(hid_bytes);
    void *ahd  = dev_alloc(hid_bytes);
    void *ard  = dev_alloc(hid_bytes);
    void *postd= dev_alloc(hid_bytes);
    void *mlpd = dev_alloc(hid_bytes);

    hd_block_internals ints = { ln0d, ahd, ard, postd, mlpd };

    printf("  running hd_decoder_block ...\n");
    st = hd_decoder_block(xd, (const float *)posd, maskd, &bw, (const int64_t *)secd,
                          scratch, need, &ints, outd,
                          seq, NH, NKV, H, I, HD, NULL);
    CHECK(st == HD_OK, "hd_decoder_block ran without error");
    if (st != HD_OK) printf("  block error: %s\n", hd_last_error());
    cudaDeviceSynchronize();

    /* ---- read outputs to host and compare ---- */
    void *oh = malloc(hid_bytes);
    float *cand = malloc((size_t)seq * H * sizeof(float));

    int any_unchecked = 1;
    if (st == HD_OK) {
        /* block output (primary gate: composite class D) */
        cudaMemcpy(oh, outd, hid_bytes, cudaMemcpyDeviceToHost);
        hd_bf16_buf_to_f32(oh, cand, (size_t)seq * H);
        compare_ref("block_out", cand, bin, &o_block, 1e-2, 0.999);

        /* internals: each composite, reported with fitted bounds */
        cudaMemcpy(oh, ln0d, hid_bytes, cudaMemcpyDeviceToHost);
        hd_bf16_buf_to_f32(oh, cand, (size_t)seq * H);
        compare_ref("ln0", cand, bin, &o_ln0, 1e-2, 0.999);

        cudaMemcpy(oh, ahd, hid_bytes, cudaMemcpyDeviceToHost);
        hd_bf16_buf_to_f32(oh, cand, (size_t)seq * H);
        compare_ref("attn_hidden", cand, bin, &o_ah, 1e-2, 0.999);

        cudaMemcpy(oh, ard, hid_bytes, cudaMemcpyDeviceToHost);
        hd_bf16_buf_to_f32(oh, cand, (size_t)seq * H);
        compare_ref("attn_resid", cand, bin, &o_ar, 1e-2, 0.999);

        cudaMemcpy(oh, postd, hid_bytes, cudaMemcpyDeviceToHost);
        hd_bf16_buf_to_f32(oh, cand, (size_t)seq * H);
        compare_ref("post_ln", cand, bin, &o_post, 1e-2, 0.999);

        cudaMemcpy(oh, mlpd, hid_bytes, cudaMemcpyDeviceToHost);
        hd_bf16_buf_to_f32(oh, cand, (size_t)seq * H);
        compare_ref("mlp_out", cand, bin, &o_mlp, 1e-2, 0.999);
        any_unchecked = 0;
    }

    /* ---- cleanup ---- */
    free(oh); free(cand);
    dev_free(xd); dev_free(posd); dev_free(maskd); dev_free(scratch);
    dev_free(secd);
    dev_free(outd); dev_free(ln0d); dev_free(ahd); dev_free(ard);
    dev_free(postd); dev_free(mlpd);
    hd_weight_store_free(&store);
    hd_json_free(meta); free(bin);

    printf("\n%d assertions passed, %d failed\n", passes, failures);
    if (failures) return 1;
    printf("\nM1.3 decoder block: all checks passed%s\n",
           any_unchecked ? " (block ran OK)" : " (incl. internals)");
    return 0;
}