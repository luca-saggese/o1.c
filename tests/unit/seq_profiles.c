/*
 * M1-post sequence profile geometry probe (§70).
 *
 * Builds each real sequence profile (T2I 1024, T2I 2048, edit 1ref,
 * edit 1ref keep-aspect, subject 2ref, subject many-ref, layout,
 * skeleton, storyboard) with the unified sequence builder and prints
 * total S, attention shape, workspace estimate, ref-token count and
 * target-token count. The numbers feed docs/M1_POST_PERF_FREEZE.md.
 *
 * CPU-only: no CUDA, no GPU. The workspace estimate mirrors the
 * documented formula in sequence.c hd_seq_diag (persistent buffers +
 * block scratch; scores+probs = 2*NH*S^2 bf16 = 128*S^2 bytes).
 */

#include "hidream.h"
#include "sequence.h"
#include "request.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PATCH 32
#define IMG_TOKEN_ID   151655
#define VIDEO_TOKEN_ID 151656
#define VIS_START_ID   151652
#define TMS_TOKEN_ID   151673
#define TIMESTEP_TOKENS 1
#define SPATIAL_MERGE 1
#define FIX_POINT 4096

/* Frozen model dims (tests/unit/sanity_gen.c) */
#define H       4096
#define NH      32
#define NKV     8
#define HD      128
#define FF      12288
#define HEADOUT 3072

/* Canonical T2I prompt ids (M1.4 golden, 19 tokens incl boi+tms) */
static const int64_t k_t2i_ids[19] = {
    151644, 872, 198, 64, 2518, 38835, 23011, 1212, 264, 40880,
    88758, 4916, 151645, 198, 151644, 77091, 198, 151669, 151673
};

static hd_generation_request make_req(const char *prompt, int n_refs) {
    hd_generation_request r;
    memset(&r, 0, sizeof(r));
    r.prompt = prompt;
    r.reference_count = n_refs;
    return r;
}

/* Documented workspace estimate (mirrors sequence.c diag_workspace_bytes) */
static long long workspace_bytes(int S, int text_len, int image_len) {
    /* scores + probs, bf16: 2*NH*S^2*2 = 128*S^2 bytes (ITEM-06 parity) */
    long long o = 0;
    o += (long long)S * H * 2;             /* hidden_a  */
    o += (long long)S * H * 2;             /* hidden_b  */
    o += (long long)text_len * H * 2;      /* h_text    */
    o += (long long)S * H * 2;             /* norm_out  */
    o += (long long)S * HEADOUT * 2;       /* head_out  */
    o += (long long)H * 2;                 /* t_emb     */
    o += (long long)H * 2;                 /* te_hidden */
    o += 256 * 4;                          /* freq fp32 */
    o += 256 * 2;                          /* freq_bf16 */
    o += 4;                                /* t_scaled  */
    o += (long long)image_len * 1024 * 2;  /* xe_stage  */
    o += (long long)image_len * H * 2;     /* xe_out    */
    o = (o + 7) & ~7LL;

    long long e = 0;
    e += (long long)S * H;                 /* in_ln       */
    e += 3LL * S * NH * HD;                /* qp, q, qr   */
    e += 3LL * S * NKV * HD;               /* kp, k, kr   */
    e += 2LL * S * NKV * HD;               /* vp, v       */
    e += (long long)NH * S * S;            /* scores      */
    e += (long long)NH * S * S;            /* probs       */
    e += (long long)S * NH * HD;           /* attn_sm     */
    e += 3LL * S * H;                      /* attn_m/h/r  */
    e += (long long)S * H;                 /* post        */
    e += 3LL * S * FF;                     /* gate, up, swi */
    e += 2LL * S * H;                      /* mlp, mlp_r  */
    long long bs = e * 2 + 2LL * S * HD * 4 + 256;
    return o + bs;
}

static void print_row(const char *name, const hd_sequence *seq) {
    int ref_tokens = 0;
    for (int i = 0; i < seq->n_refs; i++) ref_tokens += seq->ref_len[i];
    long long ws = workspace_bytes(seq->S, seq->text_len, seq->image_len);
    printf("%-24s S=%-6d attn=%dx%d ws=%-12lld ref=%-5d tgt=%d\n",
           name, seq->S, seq->S, seq->S, ws, ref_tokens, seq->image_len);
}

int main(void) {
    hd_sequence seq;

    /* ---- T2I 1024: canonical prompt, 1024x1024 ---- */
    hd_seq_t2i(k_t2i_ids, 19, 1024, 1024, PATCH, IMG_TOKEN_ID, VIDEO_TOKEN_ID,
               VIS_START_ID, TMS_TOKEN_ID, TIMESTEP_TOKENS, SPATIAL_MERGE,
               FIX_POINT, &seq);
    print_row("T2I 1024", &seq);
    hd_sequence_free(&seq);

    /* ---- T2I 2048: canonical prompt, 2048x2048 ---- */
    hd_seq_t2i(k_t2i_ids, 19, 2048, 2048, PATCH, IMG_TOKEN_ID, VIDEO_TOKEN_ID,
               VIS_START_ID, TMS_TOKEN_ID, TIMESTEP_TOKENS, SPATIAL_MERGE,
               FIX_POINT, &seq);
    print_row("T2I 2048", &seq);
    hd_sequence_free(&seq);

    /* ---- edit 1ref: 1024x1024 tgt, 512x512 ref (16x16=256), cond 12x12 ---- */
    {
        hd_generation_request req = make_req("make the sky red", 1);
        hd_ref_geom refs[1] = { { 256, 16, 16, 12, 12 } };
        hd_seq_build(&req, PATCH, IMG_TOKEN_ID, VIDEO_TOKEN_ID, VIS_START_ID,
                     TMS_TOKEN_ID, TIMESTEP_TOKENS, SPATIAL_MERGE, FIX_POINT,
                     1024, 1024, refs, &seq);
        print_row("edit 1ref", &seq);
        hd_sequence_free(&seq);
    }

    /* ---- edit 1ref keep-aspect: portrait 1024x2048 ref -> tgt 1024x2048,
     * ref 32x64=2048 tokens, cond 8x16=128 (calculate_dimensions(384, 0.5)) ---- */
    {
        hd_generation_request req = make_req("keep the original aspect", 1);
        hd_ref_geom refs[1] = { { 2048, 32, 64, 8, 16 } };
        hd_seq_build(&req, PATCH, IMG_TOKEN_ID, VIDEO_TOKEN_ID, VIS_START_ID,
                     TMS_TOKEN_ID, TIMESTEP_TOKENS, SPATIAL_MERGE, FIX_POINT,
                     1024, 2048, refs, &seq);
        print_row("edit 1ref keep-aspect", &seq);
        hd_sequence_free(&seq);
    }

    /* ---- subject 2ref: 1024x1024 tgt, 2x 384x384 refs (12x12=144), cond 12x12 ---- */
    {
        hd_generation_request req = make_req("two people in a park", 2);
        hd_ref_geom refs[2] = { { 144, 12, 12, 12, 12 }, { 144, 12, 12, 12, 12 } };
        hd_seq_build(&req, PATCH, IMG_TOKEN_ID, VIDEO_TOKEN_ID, VIS_START_ID,
                     TMS_TOKEN_ID, TIMESTEP_TOKENS, SPATIAL_MERGE, FIX_POINT,
                     1024, 1024, refs, &seq);
        print_row("subject 2ref", &seq);
        hd_sequence_free(&seq);
    }

    /* ---- subject many-ref: K=8, 1024x1024 tgt, refs 384x384 (144 tokens),
     * cond 9x9=81 (cond_img_size 288) ---- */
    {
        hd_generation_request req = make_req("eight views of the subject", 8);
        hd_ref_geom refs[8];
        for (int i = 0; i < 8; i++) refs[i] = (hd_ref_geom){ 144, 12, 12, 9, 9 };
        hd_seq_build(&req, PATCH, IMG_TOKEN_ID, VIDEO_TOKEN_ID, VIS_START_ID,
                     TMS_TOKEN_ID, TIMESTEP_TOKENS, SPATIAL_MERGE, FIX_POINT,
                     1024, 1024, refs, &seq);
        print_row("subject many-ref", &seq);
        hd_sequence_free(&seq);
    }

    /* ---- layout: 2 refs + 1 layout canvas = K=3, refs 512x512 (256 tokens),
     * cond 12x12 ---- */
    {
        hd_generation_request req = make_req("compose the scene with layout", 3);
        hd_ref_geom refs[3];
        for (int i = 0; i < 3; i++) refs[i] = (hd_ref_geom){ 256, 16, 16, 12, 12 };
        hd_seq_build(&req, PATCH, IMG_TOKEN_ID, VIDEO_TOKEN_ID, VIS_START_ID,
                     TMS_TOKEN_ID, TIMESTEP_TOKENS, SPATIAL_MERGE, FIX_POINT,
                     1024, 1024, refs, &seq);
        print_row("layout", &seq);
        hd_sequence_free(&seq);
    }

    /* ---- skeleton: K=4 (subject photo + 3 poses), refs 512x512 (256 tokens),
     * cond 12x12 ---- */
    {
        hd_generation_request req = make_req("skeleton poses of the subject", 4);
        hd_ref_geom refs[4];
        for (int i = 0; i < 4; i++) refs[i] = (hd_ref_geom){ 256, 16, 16, 12, 12 };
        hd_seq_build(&req, PATCH, IMG_TOKEN_ID, VIDEO_TOKEN_ID, VIS_START_ID,
                     TMS_TOKEN_ID, TIMESTEP_TOKENS, SPATIAL_MERGE, FIX_POINT,
                     1024, 1024, refs, &seq);
        print_row("skeleton", &seq);
        hd_sequence_free(&seq);
    }

    /* ---- storyboard: multi-panel single-pass T2I from a long sequential
     * prompt (no dedicated code path; M1_POST_STORYBOARD_FINAL_AUDIT).
     * Representative 3-panel prompt, 1024x1024 panels. ---- */
    {
        const char *sb = "panel 1: a red fox sits under a cherry blossom "
                         "tree, wide shot; panel 2: the fox stands up and "
                         "looks at the camera, medium shot; panel 3: the fox "
                         "walks away into the forest, long shot";
        int *ids = NULL; size_t n = 0;
        hd_status st = hd_tokenizer_encode_prompt(sb, &ids, &n);
        if (st != HD_OK) { printf("storyboard: encode failed\n"); return 1; }
        int64_t *ids64 = malloc((n + 2) * sizeof(int64_t));
        for (size_t i = 0; i < n; i++) ids64[i] = ids[i];
        ids64[n] = 151669;                 /* boi */
        ids64[n + 1] = TMS_TOKEN_ID;       /* tms */
        hd_tokenizer_free_ids(ids);
        hd_seq_t2i(ids64, (int)n + 2, 1024, 1024, PATCH, IMG_TOKEN_ID,
                   VIDEO_TOKEN_ID, VIS_START_ID, TMS_TOKEN_ID,
                   TIMESTEP_TOKENS, SPATIAL_MERGE, FIX_POINT, &seq);
        print_row("storyboard", &seq);
        hd_sequence_free(&seq);
        free(ids64);
    }

    return 0;
}