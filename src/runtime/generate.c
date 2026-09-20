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
#include "engine.h"
#include "engine_priv.h"
#include "vision.h"
#include "vision_kernels.h"
#include "hd_image.h"

#include <cuda_runtime.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hd_cuda.h"
#include "decode.h"
#include "forward.h"
#include "hd_image.h"
#include "layout.h"
#include "ref_alias.h"
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

/*
 * Reference-mode generation (edit / personalize / layout / skeleton).
 *
 * Oracle pipeline.py ref path parity:
 *   - each ref resized to max_size (K==1: max(h,w); K==2: max*48/64;
 *     K<=4: max/2; K<=8: max*24/64; else max/4), patch-aligned,
 *     normalized [-1,1], pixel_unshuffled to [tokens, 3072]
 *   - sequence built by hd_seq_build (template + vision blocks)
 *   - per step: vinputs = cat([z, ref_patches]); forward; select target rows
 *
 * Dev/flash scheduler only (editing default per upstream). The known
 * pre-existing multi-step "double free" engine bug is not addressed here.
 */
hd_status hd_engine_generate_ref(hd_generation_engine *e,
                                 const hd_generation_request *req,
                                 unsigned char **out_rgb, int *out_w,
                                 int *out_h) {
    int W = req->width, Hh = req->height;
    int K = (int)req->reference_count;
    if (K <= 0 || K > HD_SEQ_MAX_REFS) {
        hd_set_error("generate: ref count %d out of range", K);
        return HD_ERR_MISSING;
    }

    /* keep_original_aspect: single ref -> derive output dims from the ref */
    if (req->keep_original_aspect && K == 1) {
        hd_image ref_img;
        hd_status kst = hd_image_load(req->references[0].path, &ref_img);
        if (kst != HD_OK) {
            hd_set_error("generate: ref load: %s", hd_last_error());
            return kst;
        }
        hd_image_keep_aspect(&ref_img, W, Hh, PATCH, &W, &Hh);
        hd_image_free(&ref_img);
    }

    int grid_h = Hh / PATCH, grid_w = W / PATCH;
    if (grid_h <= 0 || grid_w <= 0 || Hh % PATCH || W % PATCH) {
        hd_set_error("generate: dimensions %dx%d not multiple of patch %d",
                     W, Hh, PATCH);
        return HD_ERR_MISSING;
    }
    int IMG = grid_h * grid_w;
    size_t nimg = (size_t)IMG * FF;

    /* ---- load + preprocess references ---- */
    O1_TIMING_BEGIN("REF_PREPROCESS");
    int max_size = (W > Hh ? W : Hh);
    if (K == 2) max_size = max_size * 48 / 64;
    else if (K <= 4) max_size = max_size / 2;
    else if (K <= 8) max_size = max_size * 24 / 64;
    else max_size = max_size / 4;
    if (K == 1) max_size = (W > Hh ? W : Hh); /* K==1: no downscale */

    hd_ref_geom refs[HD_SEQ_MAX_REFS];
    memset(refs, 0, sizeof(refs));
    float *ref_patches = NULL;
    size_t total_ref_tokens = 0;
    hd_status st = HD_OK;

    /* VLM conditioning images (PATH B): one per reference, resized to the
     * Qwen processor grid (patch 16, temporal 2, merge 2). Kept alive until
     * the vision tower runs once after weight load. */
    hd_image vlm_imgs[HD_SEQ_MAX_REFS];
    int vlm_n[HD_SEQ_MAX_REFS];      /* raw vision patch count N = gh*gw */
    int vlm_gh[HD_SEQ_MAX_REFS], vlm_gw[HD_SEQ_MAX_REFS];
    for (int r = 0; r < K; r++) {
        vlm_imgs[r].rgb = NULL;
        vlm_n[r] = 0; vlm_gh[r] = 0; vlm_gw[r] = 0;
    }

    for (int r = 0; r < K; r++) {
        hd_image img;
        st = hd_image_load(req->references[r].path, &img);
        if (st != HD_OK) {
            hd_set_error("generate: ref %d load: %s", r, hd_last_error());
            return st;
        }
        /* Keep a deep copy of the original for the VLM conditioning
         * resize (PATH B); PATH A frees `img` below. */
        size_t npx = (size_t)img.width * img.height;
        float *orig = malloc(npx * 3 * sizeof(float));
        if (!orig) {
            hd_image_free(&img);
            hd_set_error("generate: ref %d orig copy oom", r);
            return HD_ERR_OOM;
        }
        memcpy(orig, img.rgb, npx * 3 * sizeof(float));
        vlm_imgs[r].width = img.width;
        vlm_imgs[r].height = img.height;
        vlm_imgs[r].rgb = orig;
        hd_image resized;
        st = hd_image_resize(&img, max_size, PATCH, &resized);
        hd_image_free(&img);
        if (st != HD_OK) {
            hd_set_error("generate: ref %d resize: %s", r, hd_last_error());
            return st;
        }
        int gh = resized.height / PATCH, gw = resized.width / PATCH;
        size_t tokens = (size_t)gh * gw;
        float *patches = malloc(tokens * FF * sizeof(float));
        if (!patches) { hd_image_free(&resized); return HD_ERR_OOM; }
        st = hd_image_to_patches(&resized, PATCH, patches);
        hd_image_free(&resized);
        if (st != HD_OK) { free(patches); return st; }
        for (size_t i = 0; i < tokens * FF; i++) patches[i] = patches[i] * 2.0f - 1.0f;

        /* VLM conditioning grid, computed ONCE from the ORIGINAL image
         * (oracle pipeline.py): cond_img_size = 384 (K<=4), 288 (K<=8),
         * 192 (else); calculate_dimensions(cond_img_size, ratio) snaps to
         * 32-px multiples. cond_w/cond_h = post-merge token grid, so the
         * VLM resize target is (cond_w*32, cond_h*32) px. Every consumer
         * below (resize, patchify, placeholders, tower output) derives
         * from these same two numbers. */
        int cw, ch;
        int cond_img_size = CONDITION_IMAGE_SIZE;
        if (K > 4 && K <= 8) cond_img_size = CONDITION_IMAGE_SIZE * 48 / 64;
        else if (K > 8) cond_img_size = CONDITION_IMAGE_SIZE / 2;
        hd_image_calc_dims(cond_img_size,
                           (float)vlm_imgs[r].width,
                           (float)vlm_imgs[r].height,
                           PATCH, &cw, &ch);
        int cond_w = cw / (PATCH == 32 ? 32 : PATCH);
        int cond_h = ch / (PATCH == 32 ? 32 : PATCH);
        if (cond_w < 1) cond_w = 1;
        if (cond_h < 1) cond_h = 1;
        /* VLM pixel dims == cond grid * 32 (must be well-formed) */
        {
            int vlm_w = cond_w * 32, vlm_h = cond_h * 32;
            if (vlm_w % 32 != 0 || vlm_h % 32 != 0 ||
                (vlm_w / 16) % 2 != 0 || (vlm_h / 16) % 2 != 0) {
                hd_set_error("generate: ref %d broken vlm grid %dx%d",
                             r, vlm_w, vlm_h);
                free(patches);
                return HD_ERR_MISMATCH;
            }
        }

        refs[r].tokens = (int)tokens;
        refs[r].grid_h = gh;
        refs[r].grid_w = gw;
        refs[r].cond_h = cond_h;
        refs[r].cond_w = cond_w;
        vlm_n[r] = cond_h * cond_w * 4;   /* raw patches N = (2*cond_h)*(2*cond_w) */
        vlm_gh[r] = cond_h * 2;
        vlm_gw[r] = cond_w * 2;

        /* append to ref_patches */
        float *np = realloc(ref_patches, (total_ref_tokens + tokens) * FF * sizeof(float));
        if (!np) { free(patches); free(ref_patches); return HD_ERR_OOM; }
        ref_patches = np;
        memcpy(ref_patches + total_ref_tokens * FF, patches, tokens * FF * sizeof(float));
        free(patches);
        total_ref_tokens += tokens;
    }
    O1_TIMING_END("REF_PREPROCESS");

    /* ---- build sequence ---- */
    O1_TIMING_BEGIN("PROMPT_TOKENIZE");
    hd_sequence seq;
    memset(&seq, 0, sizeof(seq));
    /* Named reference aliases: expand `@name` in the prompt into explicit
     * "reference image N" phrases BEFORE tokenization. If the prompt has no
     * alias, the original prompt is used byte-identically. The alias table is
     * frontend-only and never affects the reference tensor order. */
    hd_ref_alias_table alias_tbl;
    {
        const char *upaths[HD_SEQ_MAX_REFS];
        const char *ualiases[HD_SEQ_MAX_REFS];
        for (int r = 0; r < K; r++) {
            upaths[r] = req->references[r].path;
            ualiases[r] = req->references[r].alias;
        }
        int arst = hd_ref_alias_build(&alias_tbl, upaths, ualiases, K, NULL, 0);
        if (arst != HD_OK) {
            hd_set_error("generate: %s", hd_ref_alias_error());
            free(ref_patches);
            return arst;
        }
        char *expanded = NULL;
        arst = hd_ref_alias_expand(&alias_tbl, req->prompt, &expanded);
        if (arst != HD_OK) {
            hd_set_error("generate: %s", hd_ref_alias_error());
            free(ref_patches);
            return arst;
        }
        if (expanded) {
            if (getenv("O1_VERBOSE_REF")) {
                hd_ref_alias_dump(&alias_tbl, stderr);
                fprintf(stderr, "expanded prompt:\n%s\n", expanded);
            }
            /* Borrow a temporary request with the expanded prompt. */
            hd_generation_request req2 = *req;
            req2.prompt = expanded;
            st = hd_seq_build(&req2, PATCH, 151655, 151656, 151652, TMS_ID,
                              1, 1, 4096, Hh, W, refs, &seq);
            free(expanded);
        } else {
            st = hd_seq_build(req, PATCH, 151655, 151656, 151652, TMS_ID, 1, 1,
                              4096, Hh, W, refs, &seq);
        }
    }
    if (st != HD_OK) {
        hd_set_error("generate: sequence: %s", hd_last_error());
        free(ref_patches);
        return st;
    }
    if (seq.image_len != IMG) {
        hd_set_error("generate: sequence image_len %d != %d", seq.image_len, IMG);
        hd_sequence_free(&seq);
        free(ref_patches);
        return HD_ERR_MISMATCH;
    }
    int S = seq.S;
    int text_len = seq.text_len;
    O1_TIMING_END("PROMPT_TOKENIZE");

    /* ---- resident weights (loaded once by the engine) ---- */
    O1_TIMING_BEGIN("INPUT_PREPARE");
    hd_forward_binding *bw = &e->bw;

    /* ---- PATH B: Qwen-VL semantic conditioning (computed ONCE) ---- */
    /* Resolve the vision tower weights and run the tower on every
     * reference, concatenating image_embeds / deepstack in reference order
     * (matching the <image_pad> template order). */
    hd_visual_cond visual;
    memset(&visual, 0, sizeof(visual));
    void *vimg_emb = NULL, *vds0 = NULL, *vds1 = NULL, *vds2 = NULL;
    uint8_t *vmask_h = NULL, *vmask_dev = NULL;
    hd_vision_binding *vb = &e->vb;
    hd_vision_workspace vws;
    memset(&vws, 0, sizeof(vws));
    {
        /* V = total merged vision tokens = sum(cond_h*cond_w) */
        int V_total = 0;
        int max_N = 0;
        for (int r = 0; r < K; r++) {
            V_total += refs[r].cond_h * refs[r].cond_w;
            if (vlm_n[r] > max_N) max_N = vlm_n[r];
        }
        visual.v_tokens = V_total;
        if (V_total <= 0) {
            hd_set_error("generate: zero vision tokens");
            hd_sequence_free(&seq);
            free(ref_patches);
            for (int r = 0; r < K; r++) if (vlm_imgs[r].rgb) hd_image_free(&vlm_imgs[r]);
            return HD_ERR_MISSING;
        }

        /* Allocate the vision workspace (sized for max_N raw patches) and
         * the concatenated output buffers. */
        int64_t vws_bytes = hd_vision_workspace_bytes(max_N);
        void *vwsbase = dev_alloc((size_t)vws_bytes);
        if (!vwsbase) {
            hd_set_error("generate: vision workspace alloc %lld bytes",
                         (long long)vws_bytes);
            hd_sequence_free(&seq);
            free(ref_patches);
            for (int r = 0; r < K; r++) if (vlm_imgs[r].rgb) hd_image_free(&vlm_imgs[r]);
            return HD_ERR_OOM;
        }
        vws.patch_out = vwsbase;
        vws.bytes = vws_bytes;

        vimg_emb = dev_alloc((size_t)V_total * HD_VISION_OUT_HIDDEN * 2);
        vds0 = dev_alloc((size_t)V_total * HD_VISION_OUT_HIDDEN * 2);
        vds1 = dev_alloc((size_t)V_total * HD_VISION_OUT_HIDDEN * 2);
        vds2 = dev_alloc((size_t)V_total * HD_VISION_OUT_HIDDEN * 2);
        vmask_h = malloc((size_t)S);
        if (!vimg_emb || !vds0 || !vds1 || !vds2 || !vmask_h) {
            hd_set_error("generate: vision buffers oom");
            goto vision_fail;
        }

        /* Run the tower once per reference; concatenate in ref order. */
        int vis_off = 0;
        for (int r = 0; r < K; r++) {
            if (!vlm_imgs[r].rgb) continue;
            int N = vlm_n[r], gh = vlm_gh[r], gw = vlm_gw[r];
            int V = refs[r].cond_h * refs[r].cond_w;
            int t = HD_VISION_TEMPORAL_PATCH, m = HD_VISION_MERGE_SIZE;
            int p = HD_VISION_PATCH_SIZE;
            /* Resize the original ref EXACTLY to the VLM conditioning grid
             * (cond_w*32, cond_h*32), matching the oracle
             * img.resize((cw,ch), LANCZOS). No aspect-preserving scale,
             * no center crop, no intermediate oversize. The grid is the
             * same one used for placeholders / patchify / tower output. */
            int vlm_w = refs[r].cond_w * 32;
            int vlm_h = refs[r].cond_h * 32;
            if (vlm_w % 32 != 0 || vlm_h % 32 != 0 ||
                (vlm_w / 16) % 2 != 0 || (vlm_h / 16) % 2 != 0) {
                hd_set_error("generate: ref %d bad vlm resize %dx%d",
                             r, vlm_w, vlm_h);
                goto vision_fail;
            }
            hd_image vlm_resized;
            memset(&vlm_resized, 0, sizeof(vlm_resized));
            hd_status vst = hd_image_resize_exact(&vlm_imgs[r], vlm_w, vlm_h,
                                                  &vlm_resized);
            hd_image_free(&vlm_imgs[r]);
            if (vst != HD_OK) {
                hd_set_error("generate: ref %d vlm resize: %s", r,
                             hd_last_error());
                goto vision_fail;
            }
            /* normalize to [-1,1] and patchify to [N, C*t*p*p] */
            float *pv_f32 = malloc((size_t)N * HD_VISION_PATCH_DIM * sizeof(float));
            if (!pv_f32) {
                hd_image_free(&vlm_resized);
                hd_set_error("generate: vlm pv oom");
                goto vision_fail;
            }
            for (int i = 0; i < vlm_resized.width * vlm_resized.height * 3; i++)
                vlm_resized.rgb[i] = (vlm_resized.rgb[i] - 0.5f) / 0.5f;
            st = hd_image_to_vlm_patches(&vlm_resized, p, t, m, pv_f32);
            hd_image_free(&vlm_resized);
            if (st != HD_OK) {
                free(pv_f32);
                hd_set_error("generate: vlm patchify: %s", hd_last_error());
                goto vision_fail;
            }
            void *pv_f32_dev = dev_alloc((size_t)N * HD_VISION_PATCH_DIM * 4);
            void *pv_dev = dev_alloc((size_t)N * HD_VISION_PATCH_DIM * 2);
            if (!pv_f32_dev || !pv_dev) {
                free(pv_f32);
                hd_set_error("generate: vlm pv dev alloc oom");
                goto vision_fail;
            }
            cudaMemcpy(pv_f32_dev, pv_f32,
                       (size_t)N * HD_VISION_PATCH_DIM * 4,
                       cudaMemcpyHostToDevice);
            hd_f32_convert_bf16(pv_f32_dev, pv_dev,
                                N * HD_VISION_PATCH_DIM);
            free(pv_f32);
            dev_free(pv_f32_dev);
            if (N % 4 != 0) {
                hd_set_error("generate: vision N %d not divisible by 4", N);
                dev_free(pv_dev);
                goto vision_fail;
            }
            /* cuDNN SDPA plan for this N (vision attention, head_dim 72) */
            if (!vws.sdpa) {
                hd_sdpa_plan *vplan = NULL;
                float vscale = (float)(1.0 / sqrt((double)HD_VISION_HEAD_DIM));
                int vrc = hd_sdpa_create(&vplan, 1, HD_VISION_HEADS,
                                         HD_VISION_HEADS, N, N,
                                         HD_VISION_HEAD_DIM, vscale);
                if (vrc == 0) vws.sdpa = vplan;
            }
            void *vds_out[HD_VISION_NUM_DS] = {
                (uint8_t *)vds0 + (size_t)vis_off * HD_VISION_OUT_HIDDEN * 2,
                (uint8_t *)vds1 + (size_t)vis_off * HD_VISION_OUT_HIDDEN * 2,
                (uint8_t *)vds2 + (size_t)vis_off * HD_VISION_OUT_HIDDEN * 2,
            };
            st = hd_vision_forward(vb, &vws, pv_dev, N, gh, gw,
                                   (uint8_t *)vimg_emb +
                                       (size_t)vis_off * HD_VISION_OUT_HIDDEN * 2,
                                   vds_out);
            dev_free(pv_dev);
            cudaDeviceSynchronize();
            if (st != HD_OK) {
                hd_set_error("generate: vision forward ref %d: %s", r,
                             hd_last_error());
                goto vision_fail;
            }
            vis_off += V;
        }

        /* Build the visual mask: mask[i] = 1 iff i < text_len and
         * input_ids[i] == 151655 (<image_pad>). Rows >= text_len are 0. */
        {
            int count = 0;
            for (int i = 0; i < S; i++) {
                int is_pad = (i < text_len && seq.input_ids[i] == 151655);
                vmask_h[i] = is_pad ? 1 : 0;
                if (is_pad) count++;
            }
            if (count != V_total) {
                hd_set_error("generate: visual placeholder count %d != %d",
                             count, V_total);
                goto vision_fail;
            }
            vmask_dev = dev_alloc((size_t)S);
            if (!vmask_dev) {
                hd_set_error("generate: visual mask dev alloc oom");
                goto vision_fail;
            }
            cudaMemcpy(vmask_dev, vmask_h, (size_t)S, cudaMemcpyHostToDevice);
        }

        visual.image_embeds = vimg_emb;
        visual.deepstack[0] = vds0;
        visual.deepstack[1] = vds1;
        visual.deepstack[2] = vds2;
        visual.visual_mask = vmask_dev;
        goto vision_done;
    vision_fail:
        hd_sequence_free(&seq);
        free(ref_patches);
        if (vwsbase) dev_free(vwsbase);
        if (vimg_emb) dev_free(vimg_emb);
        if (vds0) dev_free(vds0);
        if (vds1) dev_free(vds1);
        if (vds2) dev_free(vds2);
        if (vmask_h) free(vmask_h);
        if (vmask_dev) dev_free(vmask_dev);
        if (vws.sdpa) hd_sdpa_destroy(vws.sdpa);
        for (int r = 0; r < K; r++) if (vlm_imgs[r].rgb) hd_image_free(&vlm_imgs[r]);
        return st;
    vision_done:;
    }

    /* ---- workspace ---- */
    int total_img = IMG + (int)total_ref_tokens;
    int64_t scratch_bytes = 0;
    int64_t ws_bytes = hd_forward_workspace_bytes(S, total_img, NH, NKV, H, I,
                                                  HD, &scratch_bytes);
    void *wsbase = dev_alloc((size_t)ws_bytes);
    if (!wsbase) {
        hd_set_error("generate: workspace alloc %lld bytes", (long long)ws_bytes);
        hd_sequence_free(&seq);
        free(ref_patches);
        return HD_ERR_OOM;
    }
    hd_forward_workspace ws;
    memset(&ws, 0, sizeof(ws));
    ws.hidden_a = wsbase;
    ws.block_scratch_bytes = scratch_bytes;

    /* cuDNN SDPA plan for the full sequence */
    {
        hd_sdpa_plan *plan = NULL;
        float attn_scale = (float)(1.0 / sqrt((double)HD));
        int rc = hd_sdpa_create(&plan, 1, NH, NKV, S, S, HD, attn_scale);
        if (rc == 0) ws.sdpa = plan;
    }

    /* ---- stage device inputs ---- */
    void *posd = dev_alloc((size_t)3 * S * 4);
    void *maskd = dev_alloc((size_t)S * S * 2);
    void *idsd = dev_alloc((size_t)text_len * 8);
    int64_t sec_host[3] = {24, 20, 20};
    void *secd = dev_alloc(sizeof(sec_host));
    if (!posd || !maskd || !idsd || !secd) {
        hd_set_error("generate: input alloc oom");
        hd_sequence_free(&seq);
        free(ref_patches);
        dev_free(wsbase); dev_free(posd); dev_free(maskd); dev_free(idsd);
        dev_free(secd);
        return HD_ERR_OOM;
    }
    cudaMemcpy(posd, seq.pos_f32, (size_t)3 * S * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(maskd, seq.mask_bf16, (size_t)S * S * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(idsd, seq.input_ids, (size_t)text_len * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(secd, sec_host, sizeof(sec_host), cudaMemcpyHostToDevice);

    /* ---- device buffers ---- */
    void *z_prev_dev = NULL, *z_next_dev = NULL;
    float *mo_dev = NULL, *noise_dev = NULL, *scratch = NULL;
    float *tsd = NULL;
    void *out_dev = NULL, *xp_dev = NULL;
    void *ref_dev = NULL, *vinput_dev = NULL;
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
    ref_dev = dev_alloc(total_ref_tokens * FF * 2);
    vinput_dev = dev_alloc((size_t)total_img * FF * 2);
    if (!z_prev_dev || !z_next_dev || !mo_dev || !noise_dev || !scratch ||
        !tsd || !out_dev || !xp_dev || !ref_dev || !vinput_dev) {
        hd_set_error("generate: device alloc oom");
        goto fail;
    }

    /* upload ref patches (fixed) as bf16 */
    {
        if (total_ref_tokens > (size_t)INT_MAX / FF) {
            hd_set_error("generate: reference patch element count overflow");
            goto fail;
        }
        size_t ref_elems = total_ref_tokens * FF;
        float *ref_f32 = malloc(ref_elems * sizeof(float));
        void *ref_f32_dev = dev_alloc(ref_elems * sizeof(float));
        if (!ref_f32 || !ref_f32_dev) {
            free(ref_f32);
            hd_set_error("generate: oom ref f32");
            goto fail;
        }
        memcpy(ref_f32, ref_patches, ref_elems * sizeof(float));
        cudaMemcpy(ref_f32_dev, ref_f32, ref_elems * sizeof(float),
                   cudaMemcpyHostToDevice);
        hd_f32_convert_bf16(ref_f32_dev, ref_dev, (int)ref_elems);
        dev_free(ref_f32_dev);
        free(ref_f32);
    }

    /* ---- initial noise (target latent) ---- */
    O1_TIMING_BEGIN("INITIAL_NOISE");
    float *noise_h = malloc((size_t)3 * Hh * W * sizeof(float));
    float *z_h = malloc(nimg * sizeof(float));
    void *z_f32_dev = dev_alloc(nimg * 4);
    if (!noise_h || !z_h || !z_f32_dev) {
        free(noise_h); free(z_h);
        hd_set_error("generate: oom noise");
        goto fail;
    }
    gen_initial_noise(req->seed, W, Hh, noise_h);
    pixel_unshuffle(noise_h, Hh, W, grid_h, grid_w, z_h);
    for (size_t i = 0; i < nimg; i++) z_h[i] *= req->noise_scale_start;
    cudaMemcpy(z_f32_dev, z_h, nimg * sizeof(float), cudaMemcpyHostToDevice);
    hd_f32_convert_bf16(z_f32_dev, z_prev_dev, (int)nimg);
    dev_free(z_f32_dev);
    free(noise_h); free(z_h);
    O1_TIMING_END("INITIAL_NOISE");

    /* ---- scheduler ---- */
    hd_scheduler sched;
    int n_sigmas;
    /* Dev/Flash recipe: the frozen DEFAULT_TIMESTEPS list is the canonical
     * 28-step schedule; any other step count derives a linspace ramp
     * (flash/flow_match), matching pipeline.build_scheduler. */
    if (req->scheduler == HD_SCHED_DEFAULT)
        n_sigmas = hd_scheduler_derive_default(&sched, req->steps, req->shift,
                                               req->noise_clip_std);
    else if (req->steps <= 1 || req->steps == 28)
        n_sigmas = hd_scheduler_derive_dev(&sched, req->noise_clip_std);
    else if (req->scheduler == HD_SCHED_FLOW_MATCH)
        n_sigmas = hd_scheduler_derive_flow_match(&sched, req->steps, req->shift,
                                                  req->noise_clip_std);
    else
        n_sigmas = hd_scheduler_derive_flash(&sched, req->steps, req->shift,
                                             req->noise_clip_std);
    if (n_sigmas < req->steps + 1) {
        hd_set_error("generate: derive returned %d sigmas (need >= %d)",
                     n_sigmas, req->steps + 1);
        goto fail;
    }
    O1_TIMING_END("INPUT_PREPARE");

    /* ---- denoising chain (dev/flash Euler) ---- */
    float *noise_step = malloc(nimg * sizeof(float));
    if (!noise_step) { hd_set_error("generate: oom noise_step"); goto fail; }
    hd_torch_rng step_rng;
    hd_torch_rng_seed(&step_rng, req->seed + 1);
    O1_TIMING_BEGIN("DENOISE_TOTAL");
    for (int i = 0; i < req->steps; i++) {
        float t_pixeldit = 1.0f - sched.sigmas[i];
        float sigma = sched.sigmas[i];
        if (sigma < T_EPS) sigma = T_EPS;
        cudaMemcpy(tsd, &t_pixeldit, 4, cudaMemcpyHostToDevice);

        /* vinputs = cat([z, ref_patches]) */
        cudaMemcpy(vinput_dev, z_prev_dev, nimg * 2, cudaMemcpyDeviceToDevice);
        cudaMemcpy((char *)vinput_dev + nimg * 2, ref_dev,
                   total_ref_tokens * FF * 2, cudaMemcpyDeviceToDevice);

        st = hd_forward(bw, &ws, (const int64_t *)idsd, text_len,
                        (const float *)posd, maskd, vinput_dev, total_img,
                        &visual, tsd, secd, S, NH, NKV, H, I, HD, TMS_ID,
                        NULL, out_dev, NULL);
        if (st != HD_OK) {
            hd_set_error("generate: forward step %d: %s", i, hd_last_error());
            free(noise_step);
            goto fail;
        }
        /* target rows: out_dev[text_len : text_len+IMG] */
        cudaMemcpy(xp_dev, (const char *)out_dev + (size_t)text_len * FF * 2,
                   nimg * 2, cudaMemcpyDeviceToDevice);
#ifdef O1_DEBUG_TIMING
        {
            uint16_t h[4];
            cudaMemcpy(h, xp_dev, 8, cudaMemcpyDeviceToHost);
            fprintf(stderr, "[ref-dbg] xp[0..3]=%04x %04x %04x %04x\n",
                    h[0], h[1], h[2], h[3]);
            uint16_t zz[4];
            cudaMemcpy(zz, z_prev_dev, 8, cudaMemcpyDeviceToHost);
            fprintf(stderr, "[ref-dbg] z[0..3]=%04x %04x %04x %04x\n",
                    zz[0], zz[1], zz[2], zz[3]);
        }
#endif

        hd_sched_vcond(z_prev_dev, xp_dev, sigma, mo_dev, (int)nimg);

        O1_TIMING_BEGIN_GPU("SCHEDULER");
        if (req->scheduler == HD_SCHED_FLOW_MATCH) {
            hd_sched_flow_match_step(z_prev_dev, mo_dev, sched.sigmas[i],
                                     sched.sigmas[i + 1], z_next_dev,
                                     (int)nimg);
            sched.step_index++;
        } else {
            /* per-step noise + Euler (dev/flash) */
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

            st = hd_scheduler_step(&sched, z_prev_dev, mo_dev, noise_dev,
                                   s_noise, z_next_dev, (int)nimg, scratch);
        }
        O1_TIMING_END_GPU("SCHEDULER");
        if (st != HD_OK) {
            hd_set_error("generate: scheduler step %d: %s", i, hd_last_error());
            free(noise_step);
            goto fail;
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
#ifdef O1_DEBUG_TIMING
    {
        float mn = z_final[0], mx = z_final[0];
        for (size_t i = 1; i < nimg; i++) {
            if (z_final[i] < mn) mn = z_final[i];
            if (z_final[i] > mx) mx = z_final[i];
        }
        fprintf(stderr, "[ref-dbg] z_final[0..3]=%g %g %g %g range=[%g,%g]\n",
                z_final[0], z_final[1], z_final[2], z_final[3], mn, mx);
    }
#endif
    int bad = 0;
    for (size_t i = 0; i < nimg; i++) {
        if (isnan(z_final[i]) || isinf(z_final[i])) { bad = 1; break; }
    }
    if (bad) {
        hd_set_error("generate: final latent contains NaN/Inf");
        free(z_final); free(rgb);
        z_final = NULL; rgb = NULL;
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
        z_final = NULL; rgb = NULL;
        goto fail;
    }
    hd_decode_to_rgb(z_final, grid_h, grid_w, PATCH, 3, rgb);
    free(z_final);
    z_final = NULL;
    O1_TIMING_END("OUTPUT_RECONSTRUCTION");

    *out_rgb = rgb;
    *out_w = W;
    *out_h = Hh;

    dev_free(wsbase); dev_free(posd); dev_free(maskd); dev_free(idsd);
    dev_free(secd); dev_free(z_prev_dev); dev_free(z_next_dev);
    dev_free(mo_dev); dev_free(noise_dev); dev_free(scratch); dev_free(tsd);
    dev_free(out_dev); dev_free(xp_dev); dev_free(ref_dev); dev_free(vinput_dev);
    if (vimg_emb) dev_free(vimg_emb);
    if (vds0) dev_free(vds0);
    if (vds1) dev_free(vds1);
    if (vds2) dev_free(vds2);
    if (vmask_dev) dev_free(vmask_dev);
    if (vmask_h) free(vmask_h);
    if (vws.sdpa) hd_sdpa_destroy(vws.sdpa);
    if (vws.patch_out) dev_free(vws.patch_out);
    hd_sequence_free(&seq);
    free(ref_patches);
    return HD_OK;

fail:
    dev_free(wsbase); dev_free(posd); dev_free(maskd); dev_free(idsd);
    dev_free(secd); dev_free(z_prev_dev); dev_free(z_next_dev);
    dev_free(mo_dev); dev_free(noise_dev); dev_free(scratch); dev_free(tsd);
    dev_free(out_dev); dev_free(xp_dev); dev_free(ref_dev); dev_free(vinput_dev);
    if (vimg_emb) dev_free(vimg_emb);
    if (vds0) dev_free(vds0);
    if (vds1) dev_free(vds1);
    if (vds2) dev_free(vds2);
    if (vmask_dev) dev_free(vmask_dev);
    if (vmask_h) free(vmask_h);
    if (vws.sdpa) hd_sdpa_destroy(vws.sdpa);
    if (vws.patch_out) dev_free(vws.patch_out);
    if (z_final) free(z_final);
    if (rgb) free(rgb);
    hd_sequence_free(&seq);
    free(ref_patches);
    return HD_ERR_MISSING;
}

hd_status hd_engine_generate_t2i(hd_generation_engine *e,
                                 const hd_generation_request *req,
                                 unsigned char **out_rgb, int *out_w,
                                 int *out_h) {
    hd_status st = HD_OK;
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

    /* Aliases require references; with none, any `@alias` is an error. */
    {
        hd_ref_alias_table empty_tbl;
        hd_ref_alias_build(&empty_tbl, NULL, NULL, 0, NULL, 0);
        char *expanded = NULL;
        if (hd_ref_alias_expand(&empty_tbl, cap[0], &expanded) != HD_OK) {
            hd_set_error("generate: %s", hd_ref_alias_error());
            return HD_ERR_MISSING;
        }
        free(expanded);
    }

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

    /* ---- resident weights (loaded once by the engine) ---- */
    O1_TIMING_BEGIN("INPUT_PREPARE");
    hd_forward_binding *bw = &e->bw;

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
            /* hd_seq_t2i's mask is causal on rows [0, text_len-1) and fully
             * unmasked from text_len-1 onward, so the two-pass split at
             * ar_len = text_len-1 is exactly equivalent. */
            int ar_len = seq_text_len[b] - 1;
            float attn_scale = (float)(1.0 / sqrt((double)HD));
            int rc = hd_sdpa_create_prod(&plan, 1, NH, NKV, Ss, ar_len,
                                         HD, attn_scale);
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
    else if (req->steps <= 1 || req->steps == 28)
        n_sigmas = hd_scheduler_derive_dev(&sched, req->noise_clip_std);
    else
        n_sigmas = hd_scheduler_derive_flash(&sched, req->steps, req->shift,
                                             req->noise_clip_std);
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
        st = hd_forward(bw, &ws, (const int64_t *)idsd[0], text_len,
                        (const float *)posd[0], maskd[0], z_prev_dev, IMG,
                        NULL, tsd, secd, S, NH, NKV, H, I, HD, TMS_ID, NULL,
                        out_dev, NULL);
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
                st = hd_forward(bw, &ws_uncond,
                                (const int64_t *)idsd[1], seq_text_len[1],
                                (const float *)posd[1], maskd[1], z_prev_dev,
                                IMG, NULL, tsd, secd, seq_S_uncond, NH, NKV, H,
                                I, HD, TMS_ID, NULL, out_uncond_dev, NULL);
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
        z_final = NULL; rgb = NULL;
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
        z_final = NULL; rgb = NULL;
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
    hd_sequence_free(&decks[0]);
    if (nbranch > 1) hd_sequence_free(&decks[1]);
    return HD_ERR_MISSING;
}
/*
 * One-shot compatibility wrapper: open a resident engine, run one
 * generation, close it. The CLI uses this; the server keeps the engine
 * open across requests.
 */
hd_status hd_generate(const hd_generation_request *req, const char *model_dir,
                      int device_id, unsigned char **out_rgb,
                      int *out_w, int *out_h) {
    if (!req || !model_dir || !out_rgb || !out_w || !out_h) {
        hd_set_error("generate: null argument");
        return HD_ERR_MISSING;
    }
    *out_rgb = NULL;
    *out_w = *out_h = 0;

    hd_generation_engine_options opts = {0};
    opts.profile = req->profile;
    opts.model_path = model_dir;
    opts.device_id = device_id;
    opts.lora = req->lora;
    hd_generation_engine *e = hd_generation_engine_open(&opts);
    if (!e) return HD_ERR_MISSING;

    hd_status st = hd_generation_engine_generate(e, req, out_rgb, out_w, out_h);
    hd_generation_engine_close(e);
    return st;
}
