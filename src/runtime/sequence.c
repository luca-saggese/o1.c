/*
 * M1-post unified sequence builder (see sequence.h).
 *
 * Reproduces the oracle's build_t2i_text_sample + get_rope_index_fix_point
 * semantics for the T2I path in pure C, verified against the frozen M1.4
 * fixture (artifacts/m1/golden/M1_V3_DEV_FORWARD_0/inputs.bin): for the
 * 64x64 canonical prompt the builder must produce byte-identical
 * pos_f32/mask/vinput_mask for the same input_ids.
 */

#include "sequence.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "request.h"

/* bf16 min value (0xFF7F pattern -> -3.3895314e+38) */
#define BF16_MIN_BITS 0xFF7Fu

/* ------------------------------------------------------------------ */
/* Resolution snapping (oracle find_closest_resolution parity)         */
/* ------------------------------------------------------------------ */

static const struct { int w, h; } k_resolutions[] = {
    { 2048, 2048 }, { 2304, 1728 }, { 1728, 2304 }, { 2560, 1440 },
    { 1440, 2560 }, { 2496, 1664 }, { 1664, 2496 }, { 3104, 1312 },
    { 1312, 3104 }, { 2304, 1792 }, { 1792, 2304 },
};
#define NUM_RES (int)(sizeof(k_resolutions) / sizeof(k_resolutions[0]))

void hd_resolution_snap(int width, int height, int *out_w, int *out_h) {
    if (width <= 0 || height <= 0) {
        if (out_w) *out_w = 2048;
        if (out_h) *out_h = 2048;
        return;
    }
    double ratio = (double)width / (double)height;
    int best = 0;
    double min_diff = INFINITY;
    for (int i = 0; i < NUM_RES; i++) {
        double r = (double)k_resolutions[i].w / (double)k_resolutions[i].h;
        double diff = fabs(r - ratio);
        if (diff < min_diff) { min_diff = diff; best = i; }
    }
    if (out_w) *out_w = k_resolutions[best].w;
    if (out_h) *out_h = k_resolutions[best].h;
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) fprintf(stderr, "sequence: OOM (%zu bytes)\n", n);
    return p;
}

/* ------------------------------------------------------------------ */
/* T2I sequence (no references)                                        */
/* ------------------------------------------------------------------ */

hd_status hd_seq_t2i(const int64_t *input_ids, int text_len,
                     int height, int width, int patch_size,
                     int image_token_id, int video_token_id,
                     int vision_start_token_id, int tms_token_id,
                     int timestep_token_num, int spatial_merge_size,
                     int fix_point, hd_sequence *out) {
    if (!input_ids || !out || text_len <= 0 || height <= 0 || width <= 0 ||
        patch_size <= 0) {
        hd_set_error("sequence: bad t2i args");
        return HD_ERR_MISSING;
    }
    memset(out, 0, sizeof(*out));

    int grid_h = height / patch_size;
    int grid_w = width / patch_size;
    int image_len = grid_h * grid_w;
    int S = text_len + image_len;
    if (S <= 0) { hd_set_error("sequence: empty seq"); return HD_ERR_MISSING; }

    /* --- input_ids: text only (template + boi + tms) --- */
    int64_t *ids = xmalloc((size_t)text_len * sizeof(int64_t));
    if (!ids) return HD_ERR_OOM;
    memcpy(ids, input_ids, (size_t)text_len * sizeof(int64_t));

    /*
     * --- position_ids [3,1,S] ---
     * get_rope_index_fix_point with spatial_merge_size=1 (oracle T2I call):
     *   text rows:       0..text_len-1 (all 3 dims)
     *   image rows:      fix_point + grid offsets; t=0, h=0..grid_h-1,
     *                    w=0..grid_w-1 (row-major over grid_w)
     * Verified against M1.4 fixture (S=23, T=19, 2x2 grid):
     *   d0: 0..18, 4096,4096,4096,4096
     *   d1: 0..18, 4096,4096,4097,4097
     *   d2: 0..18, 4096,4097,4096,4097
     */
    float *pos = xmalloc(3u * (size_t)S * sizeof(float));
    if (!pos) { free(ids); return HD_ERR_OOM; }
    for (int i = 0; i < text_len; i++) {
        pos[0 * S + i] = (float)i;
        pos[1 * S + i] = (float)i;
        pos[2 * S + i] = (float)i;
    }
    for (int t = 0; t < 1; t++) {
        for (int h = 0; h < grid_h; h++) {
            for (int w = 0; w < grid_w; w++) {
                int idx = text_len + t * grid_h * grid_w + h * grid_w + w;
                pos[0 * S + idx] = (float)(fix_point + t);
                pos[1 * S + idx] = (float)(fix_point + h);
                pos[2 * S + idx] = (float)(fix_point + w);
            }
        }
    }

    /*
     * --- attention mask [1,1,S,S] bf16 ---
     * causal = triu(full(S,S,min), diagonal=1); causal[token_types, :] = 0
     * token_types = 1 on rows [text_len-1 .. S-1] (tms + image).
     * So bf16 min above the diagonal; rows >= text_len-1 fully zeroed.
     */
    unsigned char *mask = xmalloc((size_t)S * (size_t)S * 2);
    if (!mask) { free(ids); free(pos); return HD_ERR_OOM; }
    for (int r = 0; r < S; r++) {
        for (int c = 0; c < S; c++) {
            uint16_t bits;
            if (r >= text_len - 1) {
                bits = 0x0000u;          /* token row: full attention */
            } else if (c > r) {
                bits = BF16_MIN_BITS;    /* above diagonal */
            } else {
                bits = 0x0000u;
            }
            size_t off = ((size_t)r * S + c) * 2;
            mask[off] = (unsigned char)(bits & 0xFF);
            mask[off + 1] = (unsigned char)(bits >> 8);
        }
    }

    /* --- vinput_mask [S]: image rows (start of image block onward) --- */
    unsigned char *vmask = xmalloc((size_t)S);
    if (!vmask) { free(ids); free(pos); free(mask); return HD_ERR_OOM; }
    for (int i = 0; i < S; i++)
        vmask[i] = (i >= text_len) ? 1 : 0;

    out->input_ids = ids;
    out->pos_f32 = pos;
    out->mask_bf16 = mask;
    out->vinput_mask = vmask;
    out->sec[0] = 24; out->sec[1] = 20; out->sec[2] = 20;
    out->text_len = text_len;
    out->image_len = image_len;
    out->img_begin = text_len;
    out->S = S;
    return HD_OK;
}

/* ------------------------------------------------------------------ */
/* Generic build (edit/personalize/layout/skeleton)                    */
/* ------------------------------------------------------------------ */

hd_status hd_seq_build(const hd_generation_request *req, int patch_size,
                       int image_token_id, int video_token_id,
                       int vision_start_token_id, int tms_token_id,
                       int timestep_token_num, int spatial_merge_size,
                       int fix_point, int height, int width,
                       const int *ref_lens, hd_sequence *out) {
    if (!req || !out) {
        hd_set_error("sequence: null request");
        return HD_ERR_MISSING;
    }
    /*
     * Reference-bearing modes require the reference pixel pipeline (native
     * image decode + resize + pixel_unshuffle) and the template with
     * <image> placeholders. The T2I path (hd_seq_t2i) is fully implemented;
     * ref modes are staged behind the native image pipeline (M1-post.3).
     */
    if (req->reference_count > 0) {
        hd_set_error("sequence: ref modes staged behind native image "
                     "pipeline (M1-post.3); use hd_seq_t2i for T2I");
        return HD_ERR_MISSING;
    }
    (void)image_token_id; (void)video_token_id; (void)vision_start_token_id;
    (void)tms_token_id; (void)timestep_token_num; (void)spatial_merge_size;
    (void)fix_point; (void)ref_lens;
    return hd_seq_t2i(NULL, 0, height, width, patch_size,
                      image_token_id, video_token_id, vision_start_token_id,
                      tms_token_id, timestep_token_num, spatial_merge_size,
                      fix_point, out);
}

void hd_sequence_free(hd_sequence *s) {
    if (!s) return;
    free(s->input_ids);
    free(s->pos_f32);
    free(s->mask_bf16);
    free(s->vinput_mask);
    memset(s, 0, sizeof(*s));
}

int hd_seq_diag(const hd_sequence *s, const char *tag) {
    if (!s) return -1;
    printf("[seq:%s] text_len=%d image_len=%d S=%d img_begin=%d\n",
           tag ? tag : "-", s->text_len, s->image_len, s->S, s->img_begin);
    printf("[seq:%s] sec=[%lld %lld %lld] n_refs=%d\n", tag ? tag : "-",
           (long long)s->sec[0], (long long)s->sec[1], (long long)s->sec[2],
           s->n_refs);
    for (int i = 0; i < s->n_refs && i < HD_SEQ_MAX_REFS; i++)
        printf("[seq:%s] ref[%d] len=%d begin=%d\n", tag ? tag : "-", i,
               s->ref_len[i], s->ref_begin[i]);
    printf("[seq:%s] pos=[3,1,%d] mask=[1,1,%d,%d] vinput=[%d..%d)\n",
           tag ? tag : "-", s->S, s->S, s->S, s->img_begin, s->S);
    return 0;
}