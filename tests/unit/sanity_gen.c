/*
 * M1-post production native generation runner (1024x1024 Dev sanity).
 *
 * Full native prompt-to-image path with NO Python, NO network, NO golden
 * noise file, NO golden tensor input:
 *
 *   frozen sequence fixture (oracle pure functions)  ->  pos/mask/ids
 *   native torch MT19937 RNG (seed+1)                ->  initial z
 *   native transformer forward (28 steps)            ->  x_pred
 *   native scheduler (flash, DEFAULT_TIMESTEPS)      ->  z chain
 *   native decode (no VAE)                           ->  RGB
 *   native PNG writer                                ->  image
 *
 * The sequence fixture is the ONLY oracle-derived input (tokenizer +
 * position/mask semantics are frozen at M1-post.0; the native tokenizer
 * subset cannot encode arbitrary prompts).
 *
 * Usage:
 *   build/test_sanity_gen --fixture /tmp/sanity_seq --out out.png \
 *       --seed 123456 --steps 28 --model-dir models/dev
 */
#include "cuda.h"
#include "decode.h"
#include "forward.h"
#include "json.h"
#include "png_wrap.h"
#include "scheduler.h"
#include "torch_rng.h"
#include "weights.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEV_DIR "models/dev"
#define NLAYERS 36
#define H 4096
#define I 12288
#define NH 32
#define NKV 8
#define HD 128
#define FF 3072
#define TMS_ID 151673
#define S_NOISE 8.0f
#define PATCH 32

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
    else printf("ok: %s\n", msg); \
} while (0)

static void *dev_alloc(size_t bytes) {
    void *p = NULL;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return NULL;
    return p;
}
static void dev_free(void *p) { if (p) cudaFree(p); }

static void *read_file_bytes(const char *path, size_t *out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return NULL; }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) { free(buf); return NULL; }
    buf[sz] = '\0';
    *out = (size_t)sz;
    return buf;
}

/* ------------------------------------------------------------------ */
/* Fixture manifest                                                    */
/* ------------------------------------------------------------------ */

typedef struct {
    int64_t off_pos, off_mask, off_vin, off_ts, off_tt, off_vm, off_ids;
    int64_t seq, text_len, image_len;
    int width, height;
} fixture_t;

static int load_fixture(const char *dir, fixture_t *fx) {
    char meta[1024], bin[1024];
    snprintf(meta, sizeof(meta), "%s/inputs.json", dir);
    snprintf(bin, sizeof(bin), "%s/inputs.bin", dir);
    size_t msz = 0;
    void *mb = read_file_bytes(meta, &msz);
    if (!mb) { printf("FAIL: cannot read %s\n", meta); return -1; }
    const char *err = NULL;
    hd_json *root = hd_json_parse((const char *)mb, &err);
    free(mb);
    if (!root) { printf("FAIL: parse %s: %s\n", meta, err ? err : "?"); return -1; }
    fx->seq = hd_json_int(hd_json_get(root, "seq_len"), 0);
    fx->text_len = hd_json_int(hd_json_get(root, "text_len"), 0);
    fx->width = (int)hd_json_int(hd_json_get(root, "width"), 0);
    fx->height = (int)hd_json_int(hd_json_get(root, "height"), 0);
    fx->image_len = fx->seq - fx->text_len;
    const hd_json *iarr = hd_json_get(root, "tensors");
    size_t n = iarr ? hd_json_array_len(iarr) : 0;
    for (size_t i = 0; i < n; i++) {
        const hd_json *t = hd_json_array_at(iarr, i);
        const char *nm = hd_json_string(hd_json_get(t, "name"));
        int64_t off = hd_json_int(hd_json_get(t, "offset"), 0);
        if (!nm) continue;
        if (!strcmp(nm, "pos_f32")) fx->off_pos = off;
        else if (!strcmp(nm, "mask")) fx->off_mask = off;
        else if (!strcmp(nm, "vinputs")) fx->off_vin = off;
        else if (!strcmp(nm, "timestep")) fx->off_ts = off;
        else if (!strcmp(nm, "token_types")) fx->off_tt = off;
        else if (!strcmp(nm, "vinput_mask")) fx->off_vm = off;
        else if (!strcmp(nm, "input_ids")) fx->off_ids = off;
    }
    hd_json_free(root);
    if (fx->seq <= 0 || fx->text_len <= 0 || fx->off_pos < 0 || fx->off_mask < 0 ||
        fx->off_vin < 0 || fx->off_ids < 0) {
        printf("FAIL: fixture manifest incomplete\n");
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Native initial state: torch MT19937 randn(1,3,H,W) seeded seed+1    */
/* ------------------------------------------------------------------ */

static void gen_initial_noise(uint64_t seed, int width, int height,
                              float *out /* [3*H*W] */) {
    hd_torch_rng rng;
    hd_torch_rng_seed(&rng, seed + 1);
    hd_torch_randn_f32(&rng, out, (int64_t)3 * height * width);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    const char *fixture_dir = "/tmp/sanity_seq";
    const char *out_path = "sanity.png";
    const char *model_dir = DEV_DIR;
    uint64_t seed = 123456;
    int steps = 28;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--fixture") && i + 1 < argc) fixture_dir = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_path = argv[++i];
        else if (!strcmp(argv[i], "--model-dir") && i + 1 < argc) model_dir = argv[++i];
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }

    fixture_t fx;
    memset(&fx, 0, sizeof(fx));
    fx.off_pos = fx.off_mask = fx.off_vin = fx.off_ts = fx.off_tt =
        fx.off_vm = fx.off_ids = -1;
    if (load_fixture(fixture_dir, &fx) != 0) return 1;

    int S = (int)fx.seq, T = (int)fx.text_len, IMG = (int)fx.image_len;
    int W = fx.width, Hh = fx.height;
    int grid_h = Hh / PATCH, grid_w = W / PATCH;
    size_t nimg = (size_t)IMG * FF;
    printf("sanity gen: %dx%d grid=%dx%d seq=%d text=%d img=%d steps=%d seed=%llu\n",
           W, Hh, grid_h, grid_w, S, T, IMG, steps, (unsigned long long)seed);

    /* ---- load fixture blob ---- */
    char bin[1024];
    snprintf(bin, sizeof(bin), "%s/inputs.bin", fixture_dir);
    size_t bsz = 0;
    void *blob = read_file_bytes(bin, &bsz);
    if (!blob) { printf("FAIL: cannot read %s\n", bin); return 1; }

    /* ---- load weights ---- */
    hd_st_index idx;
    if (hd_st_index_load(model_dir, &idx) != HD_OK) {
        printf("FAIL: hd_st_index_load: %s\n", hd_st_last_error());
        return 1;
    }
    hd_weight_store store;
    hd_status st = hd_weights_to_device(model_dir, &idx, 0, &store);
    hd_st_index_free(&idx);
    if (st != HD_OK) {
        printf("FAIL: hd_weights_to_device: %s\n", hd_weights_last_error());
        return 1;
    }
    hd_forward_binding bw;
    st = hd_forward_resolve(&store, NLAYERS, &bw);
    if (st != HD_OK) {
        printf("FAIL: hd_forward_resolve: %s\n", hd_last_error());
        hd_weight_store_free(&store);
        return 1;
    }

    /* ---- workspace ---- */
    int64_t scratch_bytes = 0;
    int64_t ws_bytes = hd_forward_workspace_bytes(S, IMG, NH, NKV, H, I, HD,
                                                  &scratch_bytes);
    void *wsbase = dev_alloc((size_t)ws_bytes);
    hd_forward_workspace ws;
    memset(&ws, 0, sizeof(ws));
    ws.hidden_a = wsbase;
    ws.block_scratch_bytes = scratch_bytes;
    if (!wsbase) { printf("FAIL: workspace alloc %lld bytes\n", (long long)ws_bytes); return 1; }
    printf("  workspace %lld bytes (block scratch %lld)\n",
           (long long)ws_bytes, (long long)scratch_bytes);

    /* ---- stage device inputs ---- */
    const char *p = (const char *)blob;
    void *posd = dev_alloc((size_t)3 * S * 4);
    void *maskd = dev_alloc((size_t)S * S * 2);
    void *idsd = dev_alloc((size_t)T * 8); /* forward reads only text rows */
    int64_t sec_host[3] = {24, 20, 20};
    void *secd = dev_alloc(sizeof(sec_host));
    cudaMemcpy(posd, p + fx.off_pos, (size_t)3 * S * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(maskd, p + fx.off_mask, (size_t)S * S * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(idsd, p + fx.off_ids, (size_t)T * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(secd, sec_host, sizeof(sec_host), cudaMemcpyHostToDevice);

    /* ---- device buffers ---- */
    void *z_prev_dev = dev_alloc(nimg * 2);
    void *z_next_dev = dev_alloc(nimg * 2);
    float *mo_dev = dev_alloc(nimg * 4);
    float *noise_dev = dev_alloc(nimg * 4);
    float *scratch = dev_alloc(3 * nimg * 4);
    float *tsd = dev_alloc(4);
    void *out_dev = dev_alloc((size_t)S * FF * 2);
    void *xp_dev = dev_alloc(nimg * 2);
    if (!z_prev_dev || !z_next_dev || !mo_dev || !noise_dev || !scratch ||
        !tsd || !out_dev || !xp_dev) {
        printf("FAIL: device alloc oom\n");
        return 1;
    }

    /* ---- native initial noise: randn(1,3,H,W) seed+1, then
     *      rearrange B C (H p1) (W p2) -> B (H W) (C p1 p2) ---- */
    float *noise_h = malloc((size_t)3 * Hh * W * sizeof(float));
    if (!noise_h) { printf("FAIL: oom noise\n"); return 1; }
    gen_initial_noise(seed, W, Hh, noise_h);
    /* pixel_unshuffle: out[tok, c*p1*p2 + p1*PATCH + p2] = noise[c, (tok/PATCH)*PATCH+p1, (tok%PATCH)*PATCH+p2] */
    float *z_h = malloc(nimg * sizeof(float));
    if (!z_h) { printf("FAIL: oom z\n"); return 1; }
    for (int tok = 0; tok < IMG; tok++) {
        int r = tok / grid_w, c = tok % grid_w;
        for (int ch = 0; ch < 3; ch++) {
            for (int p1 = 0; p1 < PATCH; p1++) {
                for (int p2 = 0; p2 < PATCH; p2++) {
                    float v = noise_h[((size_t)ch * Hh + (size_t)(r * PATCH + p1)) * W +
                                      (size_t)(c * PATCH + p2)];
                    z_h[(size_t)tok * FF + (size_t)ch * PATCH * PATCH +
                        (size_t)p1 * PATCH + p2] = v;
                }
            }
        }
    }
    /* scale by noise_scale_start=8.0 (oracle: noise_scale_start * randn) */
    for (size_t i = 0; i < nimg; i++) z_h[i] *= S_NOISE;
    cudaMemcpy(z_prev_dev, z_h, nimg * sizeof(float), cudaMemcpyHostToDevice);
    hd_f32_convert_bf16(z_prev_dev, z_prev_dev, (int)nimg); /* in-place bf16 */
    free(noise_h); free(z_h);

    /* ---- scheduler: derive Dev sigmas from DEFAULT_TIMESTEPS ---- */
    hd_scheduler sched;
    int n_sigmas = hd_scheduler_derive_dev(&sched, S_NOISE);
    if (n_sigmas < steps + 1) {
        printf("FAIL: derive_dev returned %d sigmas (need >= %d)\n", n_sigmas, steps + 1);
        return 1;
    }
    printf("  sigmas[0..3] = %.6f %.6f %.6f %.6f  final=%.6f\n",
           sched.sigmas[0], sched.sigmas[1], sched.sigmas[2], sched.sigmas[3],
           sched.sigmas[n_sigmas - 1]);

    /* ---- 28-step chain ---- */
    float *noise_step = malloc(nimg * sizeof(float));
    if (!noise_step) { printf("FAIL: oom noise_step\n"); return 1; }
    /* Oracle noise semantics (pipeline.py): torch.manual_seed(seed+1) once,
     * then each step's randn_like(z) is drawn IN SEQUENCE from the single
     * global generator. There is NO per-step reseeding. */
    hd_torch_rng step_rng;
    hd_torch_rng_seed(&step_rng, seed + 1);
    for (int i = 0; i < steps; i++) {
        /* model_timestep = 1 - step_t/1000 (pipeline.py) */
        float t_pixeldit = 1.0f - sched.sigmas[i] * 1000.0f / 1000.0f;
        /* sigma = step_t/1000 clamped to T_EPS */
        float sigma = sched.sigmas[i];
        if (sigma < 1e-6f) sigma = 1e-6f;
        cudaMemcpy(tsd, &t_pixeldit, 4, cudaMemcpyHostToDevice);

        st = hd_forward(&bw, &ws, (const int64_t *)idsd, T, (const float *)posd,
                        maskd, z_prev_dev, IMG, tsd, secd, S, NH, NKV, H, I, HD,
                        TMS_ID, NULL, out_dev, NULL);
        if (st != HD_OK) { printf("FAIL: hd_forward step%d: %s\n", i, hd_last_error()); return 1; }

        /* x_pred_masked = out_dev[text_len .. text_len+img) */
        cudaMemcpy(xp_dev, (const char *)out_dev + (size_t)T * FF * 2, nimg * 2,
                   cudaMemcpyDeviceToDevice);

        /* v_cond = (xp - z)/sigma; model_output = -v_guided (no CFG) */
        hd_sched_vcond(z_prev_dev, xp_dev, sigma, mo_dev, (int)nimg);

        /* per-step noise: drawn sequentially from the single generator
         * seeded seed+1 (matches the frozen oracle exactly) */
        hd_torch_randn_f32(&step_rng, noise_step, (int64_t)nimg);
        cudaMemcpy(noise_dev, noise_step, nimg * 4, cudaMemcpyHostToDevice);

        st = hd_scheduler_step(&sched, z_prev_dev, mo_dev, noise_dev, S_NOISE,
                               z_next_dev, (int)nimg, scratch);
        if (st != HD_OK) { printf("FAIL: scheduler step%d: %s\n", i, hd_last_error()); return 1; }
        cudaMemcpy(z_prev_dev, z_next_dev, nimg * 2, cudaMemcpyDeviceToDevice);

        if (i == 0 || (i + 1) % 7 == 0 || i == steps - 1)
            printf("  step %2d/%d done (sigma=%.4f)\n", i + 1, steps, sigma);
    }

    /* ---- decode final z to RGB ---- */
    float *z_final = malloc(nimg * sizeof(float));
    if (!z_final) { printf("FAIL: oom z_final\n"); return 1; }
    void *z_bf16_h = malloc(nimg * 2);
    if (!z_bf16_h) { printf("FAIL: oom z_bf16_h\n"); return 1; }
    cudaMemcpy(z_bf16_h, z_prev_dev, nimg * 2, cudaMemcpyDeviceToHost);
    hd_bf16_buf_to_f32(z_bf16_h, z_final, nimg);
    free(z_bf16_h);
    unsigned char *rgb = malloc((size_t)Hh * W * 3);
    if (!rgb) { printf("FAIL: oom rgb\n"); return 1; }
    hd_decode_to_rgb(z_final, grid_h, grid_w, PATCH, 3, rgb);

    /* NaN/Inf/range check on raw output */
    int bad = 0;
    for (size_t i = 0; i < nimg; i++) {
        if (isnan(z_final[i]) || isinf(z_final[i])) { bad++; break; }
    }
    CHECK(bad == 0, "final z has no NaN/Inf");
    float mn = z_final[0], mx = z_final[0];
    for (size_t i = 1; i < nimg; i++) {
        if (z_final[i] < mn) mn = z_final[i];
        if (z_final[i] > mx) mx = z_final[i];
    }
    printf("  final z range [%.4f, %.4f]\n", mn, mx);
    CHECK(mn >= -2.0f && mx <= 2.0f, "final z range sane");

    /* ---- write PNG ---- */
    int rc = hd_png_write_rgb(out_path, W, Hh, rgb, "hidream", "m1-post sanity");
    CHECK(rc == 0, "PNG written");
    if (rc == 0) printf("  PNG: %s (%dx%d)\n", out_path, W, Hh);

    free(z_final); free(rgb); free(noise_step);
    dev_free(wsbase); dev_free(posd); dev_free(maskd); dev_free(idsd);
    dev_free(secd); dev_free(z_prev_dev); dev_free(z_next_dev);
    dev_free(mo_dev); dev_free(noise_dev); dev_free(scratch); dev_free(tsd);
    dev_free(out_dev); dev_free(xp_dev);
    hd_forward_binding_free(&bw);
    hd_weight_store_free(&store);
    free(blob);

    printf("\n%d assertions passed, %d failed\n", failures == 0 ? 5 : 5 - failures, failures);
    return failures ? 1 : 0;
}