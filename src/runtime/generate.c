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

#include "hd_cuda.h"
#include "decode.h"
#include "forward.h"
#include "hd_image.h"
#include "layout.h"
#include "o1_timing.h"
#include "scheduler.h"
#include "sequence.h"
#include "tokenizer.h"
#include "torch_rng.h"
#include "weights.h"
#include "hd_lora.h"

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
#define CONDITION_IMAGE_SIZE 384

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
/* Reference preprocessing (oracle pipeline.py ref path)               */
/* ------------------------------------------------------------------ */

/*
 * Oracle reference preprocessing parity:
 *   resize_pilimage(pil, max_size, PATCH)   (area-preserving, patch-aligned)
 *   TENSOR_TRANSFORM: float32 [0,1] -> normalize [-1,1]
 *   einops rearrange "C (H p1) (W p2) -> (H W) (C p1 p2)"
 * Returns the resized grid dims via *grid_h/*grid_w (patches).
 */
static hd_status preprocess_ref(const hd_image *src, int max_size,
                                float *patches_out, int *grid_h, int *grid_w) {
    hd_image resized;
    hd_status st = hd_image_resize(src, max_size, PATCH, &resized);
    if (st != HD_OK) return st;
    if (resized.width % PATCH || resized.height % PATCH) {
        hd_image_free(&resized);
        hd_set_error("generate: ref dims %dx%d not patch-aligned",
                     resized.width, resized.height);
        return HD_ERR_MISMATCH;
    }
    int gh = resized.height / PATCH, gw = resized.width / PATCH;
    st = hd_image_to_patches(&resized, PATCH, patches_out);
    if (st != HD_OK) {
        hd_image_free(&resized);
        return st;
    }
    /* TENSOR_TRANSFORM normalize [-1,1]: (x - 0.5) / 0.5 */
    size_t n = (size_t)gh * gw * FF;
    for (size_t i = 0; i < n; i++) patches_out[i] = patches_out[i] * 2.0f - 1.0f;
    hd_image_free(&resized);
    *grid_h = gh;
    *grid_w = gw;
    return HD_OK;
}

/*
 * Oracle cond grid (pipeline.py): cond_img_size = 384 (K<=4), 288 (K<=8),
 * 192 (else); calculate_dimensions(cond_img_size, ref ratio); then
 * spatial_merge divides the grid (cond_h/cond_w in ref_geom are the
 * post-merge grid). Returns cond grid in patches.
 */
static void ref_cond_grid(int K, int ref_w, int ref_h, int spatial_merge,
                          int *cond_h, int *cond_w) {
    int cond_img_size = CONDITION_IMAGE_SIZE;
    if (K > 4 && K <= 8) cond_img_size = CONDITION_IMAGE_SIZE * 48 / 64;
    else if (K > 8) cond_img_size = CONDITION_IMAGE_SIZE / 2;
    int cw, ch;
    hd_image_calc_dims(cond_img_size, (float)ref_w, (float)ref_h, PATCH,
                       &cw, &ch);
    *cond_w = cw / PATCH / spatial_merge;
    *cond_h = ch / PATCH / spatial_merge;
    if (*cond_w < 1) *cond_w = 1;
    if (*cond_h < 1) *cond_h = 1;
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
    if (req->scheduler != HD_SCHED_FLASH && req->scheduler != HD_SCHED_DEFAULT) {
        hd_set_error("generate: scheduler '%s' unsupported (flash for dev, "
                     "default for base)",
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
    int use_cfg = (req->scheduler == HD_SCHED_DEFAULT) && (req->guidance_scale > 1.0f);

    /* ---- tokenize prompt -> conditional sequence ----
       nbranch=1 (conditional) for flash; nbranch=2 (conditional + unconditional
       " ") for the base/default CFG path. The unconditional branch mirrors the
       oracle `build_t2i_text_sample(" ")` (empty caption). */
    O1_TIMING_BEGIN("PROMPT_TOKENIZE");
    int nbranch = use_cfg ? 2 : 1;
    hd_sequence decks[2];
    for (int b = 0; b < 2; b++) memset(&decks[b], 0, sizeof(decks[b]));
    int seq_text_len[2], seq_S[2];
    const char *cap[2] = { req->prompt, " " };
    hd_status st;

    for (int b = 0; b < nbranch; b++) {
        int64_t *ids = NULL;
        int tlen = 0;
        st = build_t2i_ids(cap[b], &ids, &tlen);
        if (st != HD_OK) {
            hd_set_error("generate: tokenize: %s", hd_last_error());
            return st;
        }
        st = hd_seq_t2i(ids, tlen, Hh, W, PATCH, 151655, 151656, 151652,
                        TMS_ID, 1, 1, 4096, &decks[b]);
        free(ids);
        if (st != HD_OK) {
            hd_set_error("generate: sequence: %s", hd_last_error());
            for (int q = 0; q < b; q++) hd_sequence_free(&decks[q]);
            return st;
        }
        if (decks[b].image_len != IMG) {
            hd_set_error("generate: sequence image_len %d != %d",
                         decks[b].image_len, IMG);
            for (int q = 0; q <= b; q++) hd_sequence_free(&decks[q]);
            return HD_ERR_MISMATCH;
        }
        seq_text_len[b] = decks[b].text_len;
        seq_S[b] = decks[b].S;
    }
    O1_TIMING_END("PROMPT_TOKENIZE");

    /* canonical (conditional) deck drives workspace sizing & output decode */
    int S = seq_S[0];
    int text_len = seq_text_len[0];
    int seq_S_uncond = seq_S[1], seq_text_len_uncond = seq_text_len[1];
    int text_len_uncond = seq_text_len[1];

    /* ---- load weights ---- */
    O1_TIMING_BEGIN("INPUT_PREPARE");
    hd_weight_store store = {0};
    size_t mdlen = strlen(model_dir);
    int is_gguf = mdlen > 5 && strcmp(model_dir + mdlen - 5, ".gguf") == 0;
    if (is_gguf) {
        st = hd_weights_to_device_gguf(model_dir, device_id, &store);
        if (st != HD_OK) {
            hd_set_error("generate: weights: %s", hd_weights_last_error());
            hd_sequence_free(&decks[0]);
            return st;
        }
    } else {
        hd_st_index idx;
        if (hd_st_index_load(model_dir, &idx) != HD_OK) {
            hd_set_error("generate: index: %s", hd_st_last_error());
            hd_sequence_free(&decks[0]);
            return HD_ERR_IO;
        }
        st = hd_weights_to_device(model_dir, &idx, device_id, &store);
        hd_st_index_free(&idx);
        if (st != HD_OK) {
            hd_set_error("generate: weights: %s", hd_weights_last_error());
            hd_sequence_free(&decks[0]);
            return st;
        }
    }
    hd_forward_binding bw;
    st = hd_forward_resolve(&store, NLAYERS, &bw);
    if (st != HD_OK) {
        hd_set_error("generate: resolve: %s", hd_last_error());
        hd_weight_store_free(&store);
        hd_sequence_free(&decks[0]);
        return st;
    }

    /* ---- LoRA merge-on-load (optional) ----
     * Applies adapters to the resident base weights in place, then the
     * normal cuBLAS forward path runs unchanged. */
    if (req->lora) {
        O1_TIMING_BEGIN("LORA_APPLY");
        st = hd_lora_apply(req->lora, &store, device_id);
        O1_TIMING_END("LORA_APPLY");
        if (st != HD_OK) {
            hd_set_error("generate: lora: %s", hd_lora_last_error());
            hd_forward_binding_free(&bw);
            hd_weight_store_free(&store);
            hd_sequence_free(&decks[0]);
            return st;
        }
    }

    /* ---- workspace ----
       Two forward contexts: the canonical conditional deck and (with CFG)
       the unconditional deck. Both share one arena: the canonical deck takes
       the front, the unconditional deck is laid out after it. The canonical
       ws.sdpa is pointed at the conditional plan by the SDPA block below. */
    int64_t scratch_bytes = 0, scratch_bytes_un = 0;
    int64_t ws_bytes = hd_forward_workspace_bytes(S, IMG, NH, NKV, H, I, HD,
                                                  &scratch_bytes);
    int64_t ws_bytes_un = 0;
    if (use_cfg) {
        ws_bytes_un = hd_forward_workspace_bytes(seq_S_uncond, IMG, NH, NKV,
                                                 H, I, HD, &scratch_bytes_un);
    }
    int64_t ws_total = ws_bytes + ws_bytes_un;
    void *wsbase = dev_alloc((size_t)ws_total);
    if (!wsbase) {
        hd_set_error("generate: workspace alloc %lld bytes", (long long)ws_total);
        hd_forward_binding_free(&bw);
        hd_weight_store_free(&store);
        hd_sequence_free(&decks[0]);
        if (use_cfg) hd_sequence_free(&decks[1]);
        return HD_ERR_OOM;
    }
    hd_forward_workspace ws;
    memset(&ws, 0, sizeof(ws));
    ws.hidden_a = wsbase;
    ws.block_scratch_bytes = scratch_bytes;

    hd_forward_workspace ws_uncond;
    memset(&ws_uncond, 0, sizeof(ws_uncond));
    ws_uncond.hidden_a = (void *)((uint8_t *)wsbase + ws_bytes);
    ws_uncond.block_scratch_bytes = scratch_bytes_un;

    /* ---- cuDNN SDPA attention plan (M2 pre-baseline) ----
     * Built once for each distinct sequence length (the conditional S and,
     * when CFG is active, the shorter unconditional S). Reused for every
     * denoise step. On failure we fall back to the eager reference backend
     * (ws.sdpa stays NULL) rather than aborting generation. */
    {
        for (int b = 0; b < nbranch; b++) {
            hd_sdpa_plan *plan = NULL;
            int Ss = seq_S[b];
            float attn_scale = (float)(1.0 / sqrt((double)HD));
            int rc = hd_sdpa_create(&plan, 1, NH, NKV, Ss, Ss, HD, attn_scale);
            if (rc != 0) {
                hd_set_error("generate: sdpa plan disabled (%s); using eager reference",
                             hd_cuda_errbuf());
                if (plan) hd_sdpa_destroy(plan);
                plan = NULL;
            }
            if (b == 0) ws.sdpa = plan; else ws_uncond.sdpa = plan;
        }
    }

    /* ---- stage device inputs (one buffer set per branch) ---- */
    void *posd[2] = {NULL, NULL};
    void *maskd[2] = {NULL, NULL};
    void *idsd[2] = {NULL, NULL};
    int64_t sec_host[3] = {24, 20, 20};
    void *secd = dev_alloc(sizeof(sec_host));
    for (int b = 0; b < nbranch; b++) {
        int Ss = seq_S[b];
        posd[b] = dev_alloc((size_t)3 * Ss * 4);
        maskd[b] = dev_alloc((size_t)Ss * Ss * 2);
        idsd[b] = dev_alloc((size_t)seq_text_len[b] * 8);
        if (!posd[b] || !maskd[b] || !idsd[b] || !secd) {
            hd_set_error("generate: input alloc oom");
            hd_forward_binding_free(&bw);
            hd_weight_store_free(&store);
            for (int q = 0; q < nbranch; q++) hd_sequence_free(&decks[q]);
            dev_free(wsbase); dev_free(secd);
            dev_free(posd[0]); dev_free(maskd[0]); dev_free(idsd[0]);
            dev_free(posd[1]); dev_free(maskd[1]); dev_free(idsd[1]);
            return HD_ERR_OOM;
        }
        cudaMemcpy(posd[b], decks[b].pos_f32, (size_t)3 * Ss * 4,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(maskd[b], decks[b].mask_bf16, (size_t)Ss * Ss * 2,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(idsd[b], decks[b].input_ids, (size_t)seq_text_len[b] * 8,
                   cudaMemcpyHostToDevice);
    }
    cudaMemcpy(secd, sec_host, sizeof(sec_host), cudaMemcpyHostToDevice);

    /* ---- device buffers ---- */
    void *z_prev_dev = NULL;
    void *z_next_dev = NULL;
    float *z_f32_dev = NULL;
    float *mo_dev = NULL;      /* conditioned v-guide (fp32) */
    float *mo_uncond_dev = NULL;
    float *noise_dev = NULL;
    float *scratch = NULL;
    float *tsd = NULL;
    void *out_dev = NULL;
    void *xp_dev = NULL;
    void *out_uncond_dev = NULL;   /* unconditional branch output     */
    void *xp_uncond_dev = NULL;     /* unconditional branch x_pred     */
    float *scratch_uncond = NULL;   /* Euler scratch for uncond step   */
    float *z_cfg_dev = NULL;        /* CFG combine scratch (unused now) */
    /* UniPC persistent history */
    void *unipc_hist_dev = NULL;   /* [3 * nimg * 4] cur/prev1/prev2 fp32  */
    void *unipc_last_dev = NULL;   /* [nimg*4] last_sample fp32 (pre-pred) */
    float *z_final = NULL;
    unsigned char *rgb = NULL;
    z_prev_dev = dev_alloc(nimg * 2);
    z_next_dev = dev_alloc(nimg * 2);
    z_f32_dev = dev_alloc(nimg * 4);
    mo_dev = dev_alloc(nimg * 4);
    mo_uncond_dev = dev_alloc(nimg * 4);
    noise_dev = dev_alloc(nimg * 4);
    scratch = dev_alloc(3 * nimg * 4);
    tsd = dev_alloc(4);
    out_dev = dev_alloc((size_t)S * FF * 2);
    xp_dev = dev_alloc(nimg * 2);
    out_uncond_dev = dev_alloc((size_t)seq_S_uncond * FF * 2);
    xp_uncond_dev = dev_alloc(nimg * 2);
    scratch_uncond = dev_alloc(3 * nimg * 4);
    z_cfg_dev = dev_alloc(nimg * 4);
    if (req->scheduler == HD_SCHED_DEFAULT) {
        unipc_hist_dev = dev_alloc(3 * nimg * 4);   /* cur/prev1/prev2 */
        unipc_last_dev = dev_alloc(nimg * 4);
    }
    if (!z_prev_dev || !z_next_dev || !z_f32_dev || !mo_dev || !noise_dev ||
        !scratch || !tsd || !out_dev || !xp_dev || !mo_uncond_dev ||
        (use_cfg && (!out_uncond_dev || !xp_uncond_dev || !scratch_uncond ||
                     !z_cfg_dev)) ||
        (req->scheduler == HD_SCHED_DEFAULT &&
         (!unipc_hist_dev || !unipc_last_dev))) {
        hd_set_error("generate: device alloc oom");
        goto fail;
    }

    /* ---- native initial noise ---- */
    O1_TIMING_BEGIN("INITIAL_NOISE");
    float *noise_h = malloc((size_t)3 * Hh * W * sizeof(float));
    float *z_h = malloc(nimg * sizeof(float));
    if (!noise_h || !z_h) { hd_set_error("generate: oom noise"); goto fail; }
    gen_initial_noise(req->seed, W, Hh, noise_h);
    pixel_unshuffle(noise_h, Hh, W, grid_h, grid_w, z_h);
    for (size_t i = 0; i < nimg; i++) z_h[i] *= req->noise_scale_start;
    cudaMemcpy(z_f32_dev, z_h, nimg * sizeof(float), cudaMemcpyHostToDevice);
    hd_f32_convert_bf16(z_f32_dev, z_prev_dev, (int)nimg);
    free(noise_h); free(z_h);
    O1_TIMING_END("INITIAL_NOISE");

    /* ---- scheduler ---- */
    hd_scheduler sched;
    int n_sigmas;
    if (req->scheduler == HD_SCHED_DEFAULT)
        n_sigmas = hd_scheduler_derive_default(&sched, req->steps, req->shift,
                                               req->noise_clip_std);
    else
        n_sigmas = hd_scheduler_derive_dev(&sched, req->noise_clip_std);
    if (n_sigmas < req->steps + 1) {
        hd_set_error("generate: derive returned %d sigmas (need >= %d)",
                     n_sigmas, req->steps + 1);
        goto fail;
    }
    O1_TIMING_END("INPUT_PREPARE");

    /* ---- denoising chain ---- */
    float *noise_step = malloc(nimg * sizeof(float));
    if (!noise_step) { hd_set_error("generate: oom noise_step"); goto fail; }
    /* oracle noise: single global generator seeded seed+1, drawn in sequence
     * per flash step; the default (UniPC) path draws NO per-step noise. */
    hd_torch_rng step_rng;
    hd_torch_rng_seed(&step_rng, req->seed + 1);

    /* UniPC persistent history (base/default path): ring of converted
       model outputs (ORDER=2) + last_sample. We mirror the oracle's
       model_outputs ring as two device slots (prev and prev2) plus a current
       conv scratch. Scratch aliases the scheduler scratch buffers. */
    float *unipc_scratch = scratch;               /* [3*nimg] fp32          */
    float *unipc_cur = unipc_scratch;              /* conv pointwise output  */
    float *unipc_corr = unipc_scratch + nimg;      /* corrected sample       */
    float *unipc_prev = unipc_scratch + 2 * nimg;  /* predictor output       */
    float *unipc_slot_0 = (float *)unipc_hist_dev;            /* conv_t      */
    float *unipc_slot_1 = (float *)unipc_hist_dev + nimg;     /* conv_{t-1}  */
    float *unipc_slot_2 = (float *)unipc_hist_dev + 2 * nimg; /* conv_{t-2}  */
    int unipc_lower = 0;   /* warmup counter */
    int unipc_this = 0;    /* last predictor order (0 -> no corrector) */
    int have_hist = 0;     /* model_outputs history present */

    O1_TIMING_BEGIN("DENOISE_TOTAL");
    for (int i = 0; i < req->steps; i++) {
        float t_pixeldit = 1.0f - sched.sigmas[i];
        float sigma = sched.sigmas[i];
        if (sigma < T_EPS) sigma = T_EPS;
        cudaMemcpy(tsd, &t_pixeldit, 4, cudaMemcpyHostToDevice);

        /* ---- conditional forward ---- */
        st = hd_forward(&bw, &ws, (const int64_t *)idsd[0], text_len,
                        (const float *)posd[0], maskd[0], z_prev_dev, IMG, tsd,
                        secd, S, NH, NKV, H, I, HD, TMS_ID, NULL, out_dev,
                        NULL);
        if (st != HD_OK) {
            hd_set_error("generate: forward step %d: %s", i, hd_last_error());
            free(noise_step);
            goto fail;
        }
        cudaMemcpy(xp_dev, (const char *)out_dev + (size_t)text_len * FF * 2,
                   nimg * 2, cudaMemcpyDeviceToDevice);

        /* v_cond = (xp - z)/sigma; model_output = -v_guided */
        hd_sched_vcond(z_prev_dev, xp_dev, sigma, mo_dev, (int)nimg);

        if (req->scheduler == HD_SCHED_DEFAULT) {
            /* ---- base/default: CFG dual forward + FlowUniPC ----
               If guidance>1 run the unconditional branch and combine
               pred = uncond + g*(cond - uncond)  (= -v_guided). */
            if (use_cfg) {
                st = hd_forward(&bw, &ws_uncond,
                                (const int64_t *)idsd[1], seq_text_len[1],
                                (const float *)posd[1], maskd[1], z_prev_dev,
                                IMG, tsd, secd, seq_S_uncond, NH, NKV, H, I,
                                HD, TMS_ID, NULL, out_uncond_dev, NULL);
                if (st != HD_OK) {
                    hd_set_error("generate: forward(uncond) step %d: %s",
                                 i, hd_last_error());
                    free(noise_step);
                    goto fail;
                }
                cudaMemcpy(xp_uncond_dev,
                           (const char *)out_uncond_dev +
                               (size_t)seq_text_len[1] * FF * 2,
                           nimg * 2, cudaMemcpyDeviceToDevice);
                hd_sched_vcond(z_prev_dev, xp_uncond_dev, sigma,
                               mo_uncond_dev, (int)nimg);
                hd_sched_cfg_guided(mo_dev, mo_uncond_dev, req->guidance_scale,
                                    mo_dev, (int)nimg);
            }

            /* plan UniPC corrective step (host scalars) */
            hd_unipc_plan plan;
            hd_scheduler_unipc_plan(&sched, i, unipc_this, unipc_lower, &plan);

            /* upcast current sample z (bf16 -> fp32) into unipc_scratch's
               sample slot; use mo_dev (guided) as the current model output. */
            float *z_f32 = unipc_prev;   /* reuse predictor out buffer after */
            hd_sched_bf16_upcast(z_prev_dev, z_f32, (int)nimg);

            /* convert_model_output: conv = sample - sigma_cur*mo.
               Write conv_t into slot 0 (rotated below to the right history). */
            hd_unipc_convert(z_f32, mo_dev, sched.sigmas[i], unipc_cur,
                             (int)nimg);
            cudaMemcpy(unipc_slot_0, unipc_cur, nimg * 4, cudaMemcpyDeviceToDevice);

            /* corrector (si>0 && this_order>0 && history present): uses
               m0=conv_{t-1}, m_old=conv_{t-2}, model_t=conv_t. */
            if (i > 0 && unipc_this > 0 && have_hist) {
                hd_unipc_correct((float *)unipc_last_dev, unipc_slot_1,
                                 unipc_slot_2, unipc_cur,
                                 (float)plan.c_sig_t, (float)plan.c_sig_s0,
                                 (float)plan.c_alpha_t,
                                 (float)plan.c_h_phi_1, (float)plan.c_B_h,
                                 (float)plan.c_rhos0, (float)plan.c_rhos1,
                                 (float)plan.c_inv_rks0, plan.corr_order,
                                 unipc_corr, (int)nimg);
            } else {
                cudaMemcpy(unipc_corr, z_f32, nimg * 4, cudaMemcpyDeviceToDevice);
            }

            /* store last_sample = post-corrector sample (pre-predictor) */
            cudaMemcpy((float *)unipc_last_dev, unipc_corr, nimg * 4,
                       cudaMemcpyDeviceToDevice);

            /* predictor: m0=conv_t, m_old=conv_{t-1} */
            hd_unipc_predict(unipc_corr, unipc_slot_0, unipc_slot_1,
                             (float)plan.p_sig_t,
                             (float)plan.p_sig_s0, (float)plan.p_alpha_t,
                             (float)plan.p_h_phi_1, (float)plan.p_B_h,
                             (float)plan.p_rhos_p, (float)plan.p_inv_rks0,
                             plan.pred_order, unipc_prev, (int)nimg);

            /* cast prev_sample to bf16 z_next */
            hd_f32_convert_bf16(unipc_prev, z_next_dev, (int)nimg);

            /* rotate history: conv_{t} -> slot1, conv_{t-1} -> slot2 */
            cudaMemcpy(unipc_slot_2, unipc_slot_1, nimg * 4,
                       cudaMemcpyDeviceToDevice);
            cudaMemcpy(unipc_slot_1, unipc_slot_0, nimg * 4,
                       cudaMemcpyDeviceToDevice);

            /* advance persistent state */
            unipc_this = plan.next_this_order;
            unipc_lower = plan.next_lower_order;
            have_hist = 1;
        } else {
            /* ---- dev/flash: per-step noise + Euler ---- */
            hd_torch_randn_f32(&step_rng, noise_step, (int64_t)nimg);
            float s_noise = req->noise_scale_start +
                            (req->noise_scale_end - req->noise_scale_start) *
                                (float)i / (float)(req->steps - 1);
            float clip = req->noise_clip_std;
            if (clip > 0.0f) {
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

            O1_TIMING_BEGIN_GPU("SCHEDULER");
            st = hd_scheduler_step(&sched, z_prev_dev, mo_dev, noise_dev,
                                   s_noise, z_next_dev, (int)nimg, scratch);
            O1_TIMING_END_GPU("SCHEDULER");
            if (st != HD_OK) {
                hd_set_error("generate: scheduler step %d: %s",
                             i, hd_last_error());
                free(noise_step);
                goto fail;
            }
        }
        cudaMemcpy(z_prev_dev, z_next_dev, nimg * 2, cudaMemcpyDeviceToDevice);

        if (req->progress_cb) req->progress_cb(i + 1, req->steps, req->progress_user);
    }
    O1_TIMING_END("DENOISE_TOTAL");
    free(noise_step);

    /* ---- decode final z to RGB ---- */
    O1_TIMING_BEGIN("OUTPUT_RECONSTRUCTION");
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
    z_final = NULL;   /* avoid double-free in the fail cleanup path */
    O1_TIMING_END("OUTPUT_RECONSTRUCTION");

    *out_rgb = rgb;
    *out_w = W;
    *out_h = Hh;

    dev_free(wsbase);
    dev_free(posd[0]); dev_free(maskd[0]); dev_free(idsd[0]);
    dev_free(posd[1]); dev_free(maskd[1]); dev_free(idsd[1]);
    dev_free(secd); dev_free(z_prev_dev); dev_free(z_next_dev);
    dev_free(z_f32_dev);
    dev_free(mo_dev); dev_free(mo_uncond_dev); dev_free(noise_dev);
    dev_free(scratch); dev_free(tsd);
    dev_free(out_dev); dev_free(xp_dev);
    dev_free(out_uncond_dev); dev_free(xp_uncond_dev);
    dev_free(scratch_uncond); dev_free(z_cfg_dev);
    if (req->scheduler == HD_SCHED_DEFAULT) {
        dev_free(unipc_hist_dev); dev_free(unipc_last_dev);
        if (ws.sdpa) hd_sdpa_destroy(ws.sdpa);
        if (use_cfg && ws_uncond.sdpa) hd_sdpa_destroy(ws_uncond.sdpa);
    } else {
        if (ws.sdpa) hd_sdpa_destroy(ws.sdpa);
    }
    hd_forward_binding_free(&bw);
    hd_weight_store_free(&store);
    hd_sequence_free(&decks[0]);
    if (nbranch > 1) hd_sequence_free(&decks[1]);
    return HD_OK;

fail:
    dev_free(wsbase);
    dev_free(posd[0]); dev_free(maskd[0]); dev_free(idsd[0]);
    dev_free(posd[1]); dev_free(maskd[1]); dev_free(idsd[1]);
    dev_free(secd); dev_free(z_prev_dev); dev_free(z_next_dev);
    dev_free(z_f32_dev);
    dev_free(mo_dev); dev_free(mo_uncond_dev); dev_free(noise_dev);
    dev_free(scratch); dev_free(tsd);
    dev_free(out_dev); dev_free(xp_dev);
    dev_free(out_uncond_dev); dev_free(xp_uncond_dev);
    dev_free(scratch_uncond); dev_free(z_cfg_dev);
    dev_free(unipc_hist_dev); dev_free(unipc_last_dev);
    if (ws.sdpa) hd_sdpa_destroy(ws.sdpa);
    if (use_cfg && ws_uncond.sdpa) hd_sdpa_destroy(ws_uncond.sdpa);
    if (z_final) free(z_final);
    if (rgb) free(rgb);
    hd_forward_binding_free(&bw);
    hd_weight_store_free(&store);
    hd_sequence_free(&decks[0]);
    if (nbranch > 1) hd_sequence_free(&decks[1]);
    return HD_ERR_MISSING;
}