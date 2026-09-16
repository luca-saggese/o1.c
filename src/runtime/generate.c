/*
 * M1.8 production native generation engine (see generate.h).
 *
 * One engine for every mode. The unified request is lowered by the unified
 * sequence builder into the single transformer contract; the scheduler
 * abstraction drives the denoising chain; the unified output path decodes
 * to RGB. No Python, no network, no golden input for normal inference.
 *
 * Denoising chain (oracle pipeline.py generate_image parity):
 *   t_pixeldit = 1 - step_t/1000
 *   sigma      = step_t/1000 clamped to T_EPS
 *   x_pred     = forward(z, t_pixeldit)
 *   v_cond     = (x_pred - z) / sigma
 *   v_guided   = v_uncond + g*(v_cond - v_uncond)   (CFG when g > 0)
 *   model_output = -v_guided
 *   z_next     = sched.step(model_output, step_t, z, s_noise, noise_clip_std)
 *
 * Initial state: torch MT19937 randn(1,3,H,W) seeded seed+1, pixel_unshuffle
 * to [img, 3072], scaled by noise_scale_start. Per-step noise: randn_like(z)
 * drawn IN SEQUENCE from a single generator seeded seed+1 (oracle semantics:
 * torch.manual_seed(seed+1) once, no per-step reseed), clamped to
 * +/-noise_clip_std*noise.std() (oracle dynamic clamp), scaled by the
 * noise_scale_schedule (start -> end linear ramp).
 */

#include "generate.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda.h"
#include "decode.h"
#include "forward.h"
#include "scheduler.h"
#include "sequence.h"
#include "tokenizer.h"
#include "torch_rng.h"
#include "weights.h"

#define NLAYERS 36
#define H 4096
#define I 12288
#define NH 32
#define NKV 8
#define HD 128
#define FF 3072
#define TMS_ID 151673
#define BOI_ID 151669
#define PATCH 32
#define T_EPS 1e-6f

static void *dev_alloc(size_t bytes) {
    void *p = NULL;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return NULL;
    return p;
}
static void dev_free(void *p) { if (p) cudaFree(p); }

/* ------------------------------------------------------------------ */
/* Native initial state: torch MT19937 randn(1,3,H,W) seeded seed+1    */
/* ------------------------------------------------------------------ */

static void gen_initial_noise(uint64_t seed, int width, int height,
                              float *out /* [3*H*W] */) {
    hd_torch_rng rng;
    hd_torch_rng_seed(&rng, seed + 1);
    hd_torch_randn_f32(&rng, out, (int64_t)3 * height * width);
}

/* pixel_unshuffle: out[tok, c*p1*p2 + p1*PATCH + p2] =
 *   noise[c, (tok/grid_w)*PATCH+p1, (tok%grid_w)*PATCH+p2] */
static void pixel_unshuffle(const float *noise, int height, int width,
                            int grid_h, int grid_w, float *z_out) {
    int img = grid_h * grid_w;
    for (int tok = 0; tok < img; tok++) {
        int r = tok / grid_w, c = tok % grid_w;
        for (int ch = 0; ch < 3; ch++) {
            for (int p1 = 0; p1 < PATCH; p1++) {
                for (int p2 = 0; p2 < PATCH; p2++) {
                    float v = noise[((size_t)ch * height + (size_t)(r * PATCH + p1)) *
                                        width +
                                    (size_t)(c * PATCH + p2)];
                    z_out[(size_t)tok * FF + (size_t)ch * PATCH * PATCH +
                          (size_t)p1 * PATCH + p2] = v;
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Sequence construction (T2I)                                         */
/* ------------------------------------------------------------------ */

/*
 * Build the full template input_ids: chat template + boi + tms (oracle
 * build_t2i_text_sample). Returns malloc'd int64 array via *out_ids and
 * length via *out_len. Caller frees with free().
 */
static hd_status build_t2i_ids(const char *prompt, int64_t **out_ids,
                               int *out_len) {
    char *tmpl = NULL;
    hd_status st = hd_tokenizer_build_template(prompt, &tmpl);
    if (st != HD_OK) return st;
    size_t tlen = strlen(tmpl);
    /* template + boi + tms (TIMESTEP_TOKEN_NUM = 1) */
    char *full = malloc(tlen + 64);
    if (!full) { free(tmpl); return HD_ERR_OOM; }
    memcpy(full, tmpl, tlen);
    full[tlen] = '\0';
    strcat(full, "<|boi_token|><|tms_token|>");
    free(tmpl);

    int *ids = NULL;
    size_t n = 0;
    st = hd_tokenizer_encode(full, &ids, &n);
    free(full);
    if (st != HD_OK) return st;

    int64_t *ids64 = malloc(n * sizeof(int64_t));
    if (!ids64) { hd_tokenizer_free_ids(ids); return HD_ERR_OOM; }
    for (size_t i = 0; i < n; i++) ids64[i] = ids[i];
    hd_tokenizer_free_ids(ids);
    *out_ids = ids64;
    *out_len = (int)n;
    return HD_OK;
}

/* ------------------------------------------------------------------ */
/* Generation                                                          */
/* ------------------------------------------------------------------ */

hd_status hd_generate(const hd_generation_request *req, const char *model_dir,
                      int device_id, unsigned char **out_rgb,
                      int *out_w, int *out_h) {
    if (!req || !model_dir || !out_rgb || !out_w || !out_h) {
        hd_set_error("generate: null argument");
        return HD_ERR_MISSING;
    }
    *out_rgb = NULL;
    *out_w = *out_h = 0;

    if (req->mode != HD_MODE_T2I) {
        hd_set_error("generate: mode '%s' staged behind unified sequence "
                     "builder (M1-post.3); T2I is the production path",
                     hd_mode_name(req->mode));
        return HD_ERR_MISSING;
    }
    if (req->scheduler != HD_SCHED_FLASH) {
        hd_set_error("generate: scheduler '%s' staged behind scheduler "
                     "matrix (M1-post.2); flash is the Dev production path",
                     hd_scheduler_name(req->scheduler));
        return HD_ERR_MISSING;
    }

    int W = req->width, Hh = req->height;
    int grid_h = Hh / PATCH, grid_w = W / PATCH;
    if (grid_h <= 0 || grid_w <= 0 || Hh % PATCH || W % PATCH) {
        hd_set_error("generate: dimensions %dx%d not multiple of patch %d",
                     W, Hh, PATCH);
        return HD_ERR_MISSING;
    }
    int IMG = grid_h * grid_w;
    size_t nimg = (size_t)IMG * FF;

    /* ---- tokenize prompt -> sequence ---- */
    int64_t *ids = NULL;
    int text_len = 0;
    hd_status st = build_t2i_ids(req->prompt, &ids, &text_len);
    if (st != HD_OK) {
        hd_set_error("generate: tokenize: %s", hd_last_error());
        return st;
    }
    hd_sequence seq;
    st = hd_seq_t2i(ids, text_len, Hh, W, PATCH, 151655, 151656, 151652,
                    TMS_ID, 1, 1, 4096, &seq);
    free(ids);
    if (st != HD_OK) {
        hd_set_error("generate: sequence: %s", hd_last_error());
        return st;
    }
    int S = seq.S;
    if (seq.image_len != IMG) {
        hd_set_error("generate: sequence image_len %d != %d", seq.image_len, IMG);
        hd_sequence_free(&seq);
        return HD_ERR_MISMATCH;
    }

    /* ---- load weights ---- */
    hd_st_index idx;
    if (hd_st_index_load(model_dir, &idx) != HD_OK) {
        hd_set_error("generate: index: %s", hd_st_last_error());
        hd_sequence_free(&seq);
        return HD_ERR_IO;
    }
    hd_weight_store store;
    st = hd_weights_to_device(model_dir, &idx, device_id, &store);
    hd_st_index_free(&idx);
    if (st != HD_OK) {
        hd_set_error("generate: weights: %s", hd_weights_last_error());
        hd_sequence_free(&seq);
        return st;
    }
    hd_forward_binding bw;
    st = hd_forward_resolve(&store, NLAYERS, &bw);
    if (st != HD_OK) {
        hd_set_error("generate: resolve: %s", hd_last_error());
        hd_weight_store_free(&store);
        hd_sequence_free(&seq);
        return st;
    }

    /* ---- workspace ---- */
    int64_t scratch_bytes = 0;
    int64_t ws_bytes = hd_forward_workspace_bytes(S, IMG, NH, NKV, H, I, HD,
                                                  &scratch_bytes);
    void *wsbase = dev_alloc((size_t)ws_bytes);
    if (!wsbase) {
        hd_set_error("generate: workspace alloc %lld bytes", (long long)ws_bytes);
        hd_forward_binding_free(&bw);
        hd_weight_store_free(&store);
        hd_sequence_free(&seq);
        return HD_ERR_OOM;
    }
    hd_forward_workspace ws;
    memset(&ws, 0, sizeof(ws));
    ws.hidden_a = wsbase;
    ws.block_scratch_bytes = scratch_bytes;

    /* ---- stage device inputs ---- */
    void *posd = dev_alloc((size_t)3 * S * 4);
    void *maskd = dev_alloc((size_t)S * S * 2);
    void *idsd = dev_alloc((size_t)text_len * 8);
    int64_t sec_host[3] = {24, 20, 20};
    void *secd = dev_alloc(sizeof(sec_host));
    if (!posd || !maskd || !idsd || !secd) {
        hd_set_error("generate: input alloc oom");
        hd_forward_binding_free(&bw);
        hd_weight_store_free(&store);
        hd_sequence_free(&seq);
        dev_free(wsbase); dev_free(posd); dev_free(maskd); dev_free(idsd);
        dev_free(secd);
        return HD_ERR_OOM;
    }
    cudaMemcpy(posd, seq.pos_f32, (size_t)3 * S * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(maskd, seq.mask_bf16, (size_t)S * S * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(idsd, seq.input_ids, (size_t)text_len * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(secd, sec_host, sizeof(sec_host), cudaMemcpyHostToDevice);

    /* ---- device buffers ---- */
    void *z_prev_dev = NULL;
    void *z_next_dev = NULL;
    float *mo_dev = NULL;
    float *noise_dev = NULL;
    float *scratch = NULL;
    float *tsd = NULL;
    void *out_dev = NULL;
    void *xp_dev = NULL;
    float *z_final = NULL;
    unsigned char *rgb = NULL;
    z_prev_dev = dev_alloc(nimg * 2);
    z_next_dev = dev_alloc(nimg * 2);
    mo_dev = dev_alloc(nimg * 4);
    noise_dev = dev_alloc(nimg * 4);
    scratch = dev_alloc(3 * nimg * 4);
    tsd = dev_alloc(4);
    out_dev = dev_alloc((size_t)S * FF * 2);
    xp_dev = dev_alloc(nimg * 2);
    if (!z_prev_dev || !z_next_dev || !mo_dev || !noise_dev || !scratch ||
        !tsd || !out_dev || !xp_dev) {
        hd_set_error("generate: device alloc oom");
        goto fail;
    }

    /* ---- native initial noise ---- */
    float *noise_h = malloc((size_t)3 * Hh * W * sizeof(float));
    float *z_h = malloc(nimg * sizeof(float));
    if (!noise_h || !z_h) { hd_set_error("generate: oom noise"); goto fail; }
    gen_initial_noise(req->seed, W, Hh, noise_h);
    pixel_unshuffle(noise_h, Hh, W, grid_h, grid_w, z_h);
    for (size_t i = 0; i < nimg; i++) z_h[i] *= req->noise_scale_start;
    cudaMemcpy(z_prev_dev, z_h, nimg * sizeof(float), cudaMemcpyHostToDevice);
    hd_f32_convert_bf16(z_prev_dev, z_prev_dev, (int)nimg);
    free(noise_h); free(z_h);

    /* ---- scheduler ---- */
    hd_scheduler sched;
    int n_sigmas = hd_scheduler_derive_dev(&sched, req->noise_clip_std);
    if (n_sigmas < req->steps + 1) {
        hd_set_error("generate: derive_dev returned %d sigmas (need >= %d)",
                     n_sigmas, req->steps + 1);
        goto fail;
    }

    /* ---- denoising chain ---- */
    float *noise_step = malloc(nimg * sizeof(float));
    if (!noise_step) { hd_set_error("generate: oom noise_step"); goto fail; }
    /* oracle noise: single global generator seeded seed+1, drawn in sequence
     * per step (initial noise uses its own generator seed+1; then
     * torch.manual_seed(seed+1) reseeds the global generator and each step
     * draws randn_like(z) in sequence). */
    hd_torch_rng step_rng;
    hd_torch_rng_seed(&step_rng, req->seed + 1);
    for (int i = 0; i < req->steps; i++) {
        float t_pixeldit = 1.0f - sched.sigmas[i];
        float sigma = sched.sigmas[i];
        if (sigma < T_EPS) sigma = T_EPS;
        cudaMemcpy(tsd, &t_pixeldit, 4, cudaMemcpyHostToDevice);

        st = hd_forward(&bw, &ws, (const int64_t *)idsd, text_len,
                        (const float *)posd, maskd, z_prev_dev, IMG, tsd,
                        secd, S, NH, NKV, H, I, HD, TMS_ID, NULL, out_dev,
                        NULL);
        if (st != HD_OK) {
            hd_set_error("generate: forward step %d: %s", i, hd_last_error());
            free(noise_step);
            goto fail;
        }
        cudaMemcpy(xp_dev, (const char *)out_dev + (size_t)text_len * FF * 2,
                   nimg * 2, cudaMemcpyDeviceToDevice);

        /* v_cond = (xp - z)/sigma; model_output = -v_guided (no CFG) */
        hd_sched_vcond(z_prev_dev, xp_dev, sigma, mo_dev, (int)nimg);

        /* per-step noise: randn_like(z) drawn in sequence from the global
         * generator (seeded seed+1 once), clamped to
         * +/-noise_clip_std*noise.std() (oracle dynamic clamp), scaled by
         * noise_scale_schedule[i] */
        hd_torch_randn_f32(&step_rng, noise_step, (int64_t)nimg);
        float s_noise = req->noise_scale_start +
                        (req->noise_scale_end - req->noise_scale_start) *
                            (float)i / (float)(req->steps - 1);
        float clip = req->noise_clip_std;
        if (clip > 0.0f) {
            /* unbiased sample std (torch.std default) */
            double mean = 0.0, m2 = 0.0;
            for (size_t k = 0; k < nimg; k++) mean += noise_step[k];
            mean /= (double)nimg;
            for (size_t k = 0; k < nimg; k++) {
                double d = (double)noise_step[k] - mean;
                m2 += d * d;
            }
            double std = sqrt(m2 / (double)(nimg - 1));
            float clip_val = clip * (float)std;
            for (size_t k = 0; k < nimg; k++) {
                if (noise_step[k] > clip_val) noise_step[k] = clip_val;
                else if (noise_step[k] < -clip_val) noise_step[k] = -clip_val;
            }
        }
        cudaMemcpy(noise_dev, noise_step, nimg * 4, cudaMemcpyHostToDevice);

        st = hd_scheduler_step(&sched, z_prev_dev, mo_dev, noise_dev, s_noise,
                               z_next_dev, (int)nimg, scratch);
        if (st != HD_OK) {
            hd_set_error("generate: scheduler step %d: %s", i, hd_last_error());
            free(noise_step);
            goto fail;
        }
        cudaMemcpy(z_prev_dev, z_next_dev, nimg * 2, cudaMemcpyDeviceToDevice);

        if (req->progress_cb) req->progress_cb(i + 1, req->steps, req->progress_user);
    }
    free(noise_step);

    /* ---- decode final z to RGB ---- */
    void *z_bf16_h = malloc(nimg * 2);
    z_final = malloc(nimg * sizeof(float));
    rgb = malloc((size_t)Hh * W * 3);
    if (!z_final || !z_bf16_h || !rgb) {
        hd_set_error("generate: oom decode");
        free(z_final); free(z_bf16_h); free(rgb);
        z_final = NULL; rgb = NULL;
        goto fail;
    }
    cudaMemcpy(z_bf16_h, z_prev_dev, nimg * 2, cudaMemcpyDeviceToHost);
    hd_bf16_buf_to_f32(z_bf16_h, z_final, nimg);
    free(z_bf16_h);

    /* NaN/Inf/range check on raw output (fails closed) */
    int bad = 0;
    for (size_t i = 0; i < nimg; i++) {
        if (isnan(z_final[i]) || isinf(z_final[i])) { bad = 1; break; }
    }
    if (bad) {
        hd_set_error("generate: final latent contains NaN/Inf");
        free(z_final); free(rgb);
        goto fail;
    }
    float mn = z_final[0], mx = z_final[0];
    for (size_t i = 1; i < nimg; i++) {
        if (z_final[i] < mn) mn = z_final[i];
        if (z_final[i] > mx) mx = z_final[i];
    }
    if (mn < -2.0f || mx > 2.0f) {
        hd_set_error("generate: final latent range [%.4f, %.4f] out of bounds",
                     mn, mx);
        free(z_final); free(rgb);
        goto fail;
    }

    hd_decode_to_rgb(z_final, grid_h, grid_w, PATCH, 3, rgb);
    free(z_final);

    *out_rgb = rgb;
    *out_w = W;
    *out_h = Hh;

    dev_free(wsbase); dev_free(posd); dev_free(maskd); dev_free(idsd);
    dev_free(secd); dev_free(z_prev_dev); dev_free(z_next_dev);
    dev_free(mo_dev); dev_free(noise_dev); dev_free(scratch); dev_free(tsd);
    dev_free(out_dev); dev_free(xp_dev);
    hd_forward_binding_free(&bw);
    hd_weight_store_free(&store);
    hd_sequence_free(&seq);
    return HD_OK;

fail:
    dev_free(wsbase); dev_free(posd); dev_free(maskd); dev_free(idsd);
    dev_free(secd); dev_free(z_prev_dev); dev_free(z_next_dev);
    dev_free(mo_dev); dev_free(noise_dev); dev_free(scratch); dev_free(tsd);
    dev_free(out_dev); dev_free(xp_dev);
    if (z_final) free(z_final);
    if (rgb) free(rgb);
    hd_forward_binding_free(&bw);
    hd_weight_store_free(&store);
    hd_sequence_free(&seq);
    return HD_ERR_MISSING;
}