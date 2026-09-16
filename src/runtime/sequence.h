#ifndef HD_SEQUENCE_H
#define HD_SEQUENCE_H

/*
 * M1-post unified sequence builder (M1_POST_FULL_FEATURE_PARITY section 55).
 *
 * One request-to-sequence layer lowering ANY hd_generation_request into the
 * explicit token/position/mask/vinput contract consumed by hd_forward.
 * The transformer remains mode-agnostic.
 *
 * Outputs (all host arrays, caller-freed):
 *   input_ids     [text_len] int64   text tokens (template+boi+tms)
 *   pos_f32       [3,1,S] float32    position ids (MRoPE fix_point semantics)
 *   mask          [1,1,S,S] bf16     attention mask (min-val bf16 above diag,
 *                                    token_rows zeroed)
 *   vinput_mask   [S] uint8          image-row flags (1 = pixel input row)
 *   image_range   [img_begin, img_end)
 *   S             total sequence length
 *   image_len     target image token count
 *   text_len      template token count
 *   sec           [3] int64 MRoPE section ids (fixed [24,20,20])
 *
 * Pixel/vision input tensors (z, reference pixels) are staged separately by
 * the runtime; the builder only computes sequence metadata.
 */

#include <stddef.h>
#include <stdint.h>

#include "hidream.h"
#include "request.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HD_SEQ_MAX_REFS 8

/* Per-reference geometry for hd_seq_build (oracle pipeline.py ref path). */
typedef struct {
    int tokens;   /* ref_image_lens: token count = grid_h*grid_w (resized) */
    int grid_h;   /* resized ref grid height in patches (rh/PATCH) */
    int grid_w;   /* resized ref grid width  in patches (rw/PATCH) */
    int cond_h;   /* VLM cond grid height in patches (after spatial_merge) */
    int cond_w;   /* VLM cond grid width  in patches (after spatial_merge) */
} hd_ref_geom;

typedef struct {
    int64_t *input_ids;       /* [text_len] int64  (malloc'd) */
    float *pos_f32;           /* [3,1,S] float32   (malloc'd) */
    unsigned char *mask_bf16; /* [1,1,S,S] bf16    (malloc'd) */
    unsigned char *vinput_mask; /* [S] uint8       (malloc'd) */
    int64_t sec[3];           /* MRoPE section ids */

    int text_len;
    int image_len;
    int img_begin;            /* first image row in seq */
    int S;                    /* total seq length */

    int n_refs;
    int ref_len[HD_SEQ_MAX_REFS];      /* reference image token counts */
    int ref_begin[HD_SEQ_MAX_REFS];    /* reference token begin offsets */
} hd_sequence;

/*
 * Snap requested dimensions to the frozen PREDEFINED_RESOLUTIONS buckets by
 * closest aspect ratio (oracle find_closest_resolution parity). out_w/out_h
 * receive the snapped bucket. The bucket areas are all near 2048x2048.
 */
void hd_resolution_snap(int width, int height, int *out_w, int *out_h);

/*
 * Build the T2I sequence (no references) exactly like the oracle's
 * build_t2i_text_sample + get_rope_index_fix_point for a prompt already
 * tokenized by the oracle. `input_ids` is the oracle-encoded template
 * (apply_chat_template + boi_token + tms_token), length `text_len`.
 *
 * For edit/personalize modes (references present), use hd_seq_build which
 * appends reference vision-token blocks and applies the K>0 skip semantics.
 */
hd_status hd_seq_t2i(const int64_t *input_ids, int text_len,
                     int height, int width, int patch_size,
                     int image_token_id, int video_token_id,
                     int vision_start_token_id, int tms_token_id,
                     int timestep_token_num, int spatial_merge_size,
                     int fix_point, hd_sequence *out);

/*
 * Build a sequence for any request mode (T2I / edit / personalize / layout /
 * skeleton). References append vision-token blocks (token_types 2),
 * skip_vision_start_token = [0]*K + [1] for target. The target image always
 * follows the text/tms block (token_types 1).
 *
 * `refs` carries per-reference geometry: tokens (resized ref token count),
 * grid_h/grid_w (resized ref grid in patches) and cond_h/cond_w (VLM cond
 * grid after spatial_merge). The first K vision-start tokens (in the
 * template) use the cond grid; the target uses height/width grid; the K
 * vision-block refs use grid_h/grid_w.
 */
hd_status hd_seq_build(const hd_generation_request *req, int patch_size,
                       int image_token_id, int video_token_id,
                       int vision_start_token_id, int tms_token_id,
                       int timestep_token_num, int spatial_merge_size,
                       int fix_point, int height, int width,
                       const hd_ref_geom *refs, hd_sequence *out);

/* Free all arrays owned by `out` (from hd_seq_t2i / hd_seq_build). */
void hd_sequence_free(hd_sequence *s);

/*
 * Sequence manifest diagnostic (contract section 56). Prints text tokens,
 * reference counts/lengths, target token count, special-token indices,
 * total S, position-ID shape, MRoPE sections, prediction-mask spans,
 * workspace estimate. Returns 0 on success.
 */
int hd_seq_diag(const hd_sequence *s, const char *tag);

#ifdef __cplusplus
}
#endif

#endif /* HD_SEQUENCE_H */