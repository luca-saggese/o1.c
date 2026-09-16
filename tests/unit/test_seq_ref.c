/*
 * M1-post ref-mode sequence builder gate (structural).
 *
 * Verifies hd_seq_build geometry and invariants for edit (K=1) and
 * personalize (K=2) modes against the oracle contract (verified against
 * AutoProcessor + get_rope_index_fix_point on models/dev, transformers
 * 4.57.1):
 *
 *   - template = <|im_start|>user\n + K*(<|vision_start|><|image_pad|>
 *     <|vision_end|>) + caption + <|im_end|>\n<|im_start|>assistant\n
 *     with each <|image_pad|> placeholder EXPANDED to cond_h*cond_w pads
 *   - + boi + tms*N
 *   - vision tokens: tgt first (vision_start + (image_len-1)*image_pad),
 *     then refs
 *   - token_types: 1 tgt, 2 refs, 3 tms; vinput_mask = types 1|2
 *   - pos (K=1, 1024x1024 tgt, 512x512 ref, cond 12x12):
 *       text rows 0..3 at 0..3, 16..27 at 16..27 (template pads occupy
 *       4..15); tgt0 = 4096; tgt_last d1/d2 = 4127; ref1_0 = 4128
 *   - mask: causal triu min above diagonal on text rows, token rows zeroed
 *
 * Bit-exact comparison against the oracle fixture is added by the
 * ref-fixtures gate (tests/unit/tools/dump_ref_oracle.py).
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
#define VIS_END_ID     151653
#define TMS_TOKEN_ID   151673
#define BOI_TOKEN_ID   151669
#define TIMESTEP_TOKENS 1
#define SPATIAL_MERGE 1
#define FIX_POINT 4096

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
    else printf("ok: %s\n", msg); \
} while (0)

static hd_generation_request make_req(const char *prompt, int n_refs) {
    hd_generation_request r;
    memset(&r, 0, sizeof(r));
    r.prompt = prompt;
    r.reference_count = n_refs;
    return r;
}

int main(void) {
    /* ---- K=1 edit: 1024x1024, ref 512x512 (16x16 grid = 256 tokens) ---- */
    {
        const char *cap = "make the sky red";
        hd_generation_request req = make_req(cap, 1);
        hd_ref_geom refs[1] = { { 256, 16, 16, 12, 12 } };
        hd_sequence seq;
        hd_status st = hd_seq_build(&req, PATCH, IMG_TOKEN_ID, VIDEO_TOKEN_ID,
                                    VIS_START_ID, TMS_TOKEN_ID,
                                    TIMESTEP_TOKENS, SPATIAL_MERGE,
                                    FIX_POINT, 1024, 1024, refs, &seq);
        CHECK(st == HD_OK, "hd_seq_build K=1 returns OK");
        if (st != HD_OK) { printf("  error: %s\n", hd_last_error()); return 1; }

        int image_len = 32 * 32;          /* 1024/32 grid */
        int ref_len = 256;
        /* template with expanded cond grid: 3 + 1 + 144 + 1 + caption + 5 */
        char *tpl = NULL;
        hd_tokenizer_build_ref_template(cap, 1, &tpl);
        int *tpl_ids = NULL; size_t tpl_n = 0;
        hd_tokenizer_encode(tpl, &tpl_ids, &tpl_n);
        /* expanded: replace the single image_pad with cond 12x12=144 pads */
        int exp_n = (int)tpl_n + 144 - 1;
        int text_len = exp_n + 1 + TIMESTEP_TOKENS;   /* + boi + tms */
        int S = text_len + image_len + ref_len;
        CHECK(seq.text_len == text_len, "text_len = expanded tpl + boi + tms");
        CHECK(seq.image_len == image_len, "image_len = 32x32");
        CHECK(seq.S == S, "S = text + tgt + ref");
        CHECK(seq.n_refs == 1 && seq.ref_len[0] == ref_len, "ref_len");
        CHECK(seq.ref_begin[0] == text_len + image_len, "ref_begin after tgt");

        /* vinput_mask: tgt rows + ref rows = 1 */
        int vmask_ones = 0;
        for (int i = 0; i < S; i++) vmask_ones += seq.vinput_mask[i];
        CHECK(vmask_ones == image_len + ref_len, "vinput_mask covers tgt+refs");

        /* pos: text rows 0..3 at 0..3 (template grid occupies 4..147),
         * then 16..27 at 148..159 (first vision_end at 148) */
        CHECK(seq.pos_f32[0 * S + 0] == 0.0f, "text row0 d0 = 0");
        CHECK(seq.pos_f32[0 * S + 3] == 3.0f, "text row3 d0 = 3");
        CHECK(seq.pos_f32[0 * S + 148] == 16.0f, "text row16 d0 = 16");
        CHECK(seq.pos_f32[0 * S + text_len - 1] == 27.0f, "last text row d0 = 27");

        /* tgt rows at fix_point + grid offsets */
        int tgt0 = text_len;
        CHECK(seq.pos_f32[0 * S + tgt0] == FIX_POINT, "tgt d0 = fix_point");
        CHECK(seq.pos_f32[1 * S + tgt0] == FIX_POINT, "tgt d1 = fix_point");
        CHECK(seq.pos_f32[2 * S + tgt0] == FIX_POINT, "tgt d2 = fix_point");
        int tgt_last = text_len + image_len - 1;
        CHECK(seq.pos_f32[1 * S + tgt_last] == FIX_POINT + 31, "tgt d1 max = 4127");
        CHECK(seq.pos_f32[2 * S + tgt_last] == FIX_POINT + 31, "tgt d2 max = 4127");

        /* ref rows: fix_point(=0) + st_idx = 4128 */
        int ref0 = text_len + image_len;
        CHECK(seq.pos_f32[0 * S + ref0] == 4128.0f, "ref d0 = 4128");
        CHECK(seq.pos_f32[1 * S + ref0] == 4128.0f, "ref d1 = 4128");
        CHECK(seq.pos_f32[2 * S + ref0] == 4128.0f, "ref d2 = 4128");
        /* ref last row: d1/d2 = 4128 + 15 */
        int ref_last = ref0 + ref_len - 1;
        CHECK(seq.pos_f32[1 * S + ref_last] == 4128.0f + 15, "ref d1 max = 4143");
        CHECK(seq.pos_f32[2 * S + ref_last] == 4128.0f + 15, "ref d2 max = 4143");

        /* mask: token rows (tms + tgt + ref) fully zeroed */
        int token_row_zero = 1;
        for (int r = text_len - 1; r < S; r++) {
            for (int c = 0; c < S; c++) {
                size_t off = ((size_t)r * S + c) * 2;
                if (seq.mask_bf16[off] != 0 || seq.mask_bf16[off + 1] != 0) {
                    token_row_zero = 0; break;
                }
            }
            if (!token_row_zero) break;
        }
        CHECK(token_row_zero, "token rows zeroed in mask");

        /* text rows: causal min above diagonal (bf16 little-endian 0x7F,0xFF) */
        int causal_ok = 1;
        for (int r = 0; r < text_len - 1; r++) {
            for (int c = r + 1; c < S; c++) {
                size_t off = ((size_t)r * S + c) * 2;
                if (seq.mask_bf16[off] != 0x7F || seq.mask_bf16[off + 1] != 0xFF) {
                    causal_ok = 0; break;
                }
            }
            if (!causal_ok) break;
        }
        CHECK(causal_ok, "causal min above diagonal on text rows");

        /* template structure: vision_start + image_pad + vision_end */
        int vs0 = -1;
        for (int i = 0; i < text_len; i++)
            if (seq.input_ids[i] == VIS_START_ID) { vs0 = i; break; }
        CHECK(vs0 == 3, "template vision_start at index 3");
        CHECK(seq.input_ids[vs0 + 1] == IMG_TOKEN_ID, "image_pad after vision_start");
        CHECK(seq.input_ids[vs0 + 145] == VIS_END_ID, "vision_end after 144 pads");

        /* boi + tms at end of template */
        CHECK(seq.input_ids[exp_n] == BOI_TOKEN_ID, "boi after template");
        CHECK(seq.input_ids[exp_n + 1] == TMS_TOKEN_ID, "tms after boi");

        /* tgt block starts at text_len with vision_start */
        CHECK(seq.input_ids[text_len] == VIS_START_ID, "tgt starts with vision_start");
        CHECK(seq.input_ids[text_len + 1] == IMG_TOKEN_ID, "tgt image_pad");
        /* ref block */
        CHECK(seq.input_ids[ref0] == VIS_START_ID, "ref starts with vision_start");
        CHECK(seq.input_ids[ref0 + 1] == IMG_TOKEN_ID, "ref image_pad");

        hd_sequence_free(&seq);
        free(tpl_ids); free(tpl);
    }

    /* ---- K=2 personalize: 1024x1024, refs 384x384 (12x12=144) each ---- */
    {
        const char *cap = "two people in a park";
        hd_generation_request req = make_req(cap, 2);
        hd_ref_geom refs[2] = { { 144, 12, 12, 12, 12 }, { 144, 12, 12, 12, 12 } };
        hd_sequence seq;
        hd_status st = hd_seq_build(&req, PATCH, IMG_TOKEN_ID, VIDEO_TOKEN_ID,
                                    VIS_START_ID, TMS_TOKEN_ID,
                                    TIMESTEP_TOKENS, SPATIAL_MERGE,
                                    FIX_POINT, 1024, 1024, refs, &seq);
        CHECK(st == HD_OK, "hd_seq_build K=2 returns OK");
        if (st != HD_OK) { printf("  error: %s\n", hd_last_error()); return 1; }

        int image_len = 32 * 32;
        char *tpl = NULL;
        hd_tokenizer_build_ref_template(cap, 2, &tpl);
        int *tpl_ids = NULL; size_t tpl_n = 0;
        hd_tokenizer_encode(tpl, &tpl_ids, &tpl_n);
        /* expanded: 2 placeholders -> 2*144 pads */
        int exp_n = (int)tpl_n + 2 * 144 - 2;
        int text_len = exp_n + 1 + TIMESTEP_TOKENS;
        int S = text_len + image_len + 144 + 144;
        CHECK(seq.text_len == text_len, "K=2 text_len = expanded tpl + boi + tms");
        CHECK(seq.S == S, "S = text + tgt + 2 refs");
        CHECK(seq.n_refs == 2, "n_refs = 2");
        CHECK(seq.ref_begin[0] == text_len + image_len, "ref1 begin");
        CHECK(seq.ref_begin[1] == text_len + image_len + 144, "ref2 begin");

        /* pos: text rows 0..3 at 0..3, 16..42 at 148..306 (oracle K=2) */
        CHECK(seq.pos_f32[0 * S + 0] == 0.0f, "K2 text row0 d0 = 0");
        CHECK(seq.pos_f32[0 * S + 148] == 16.0f, "K2 text row16 d0 = 16");
        CHECK(seq.pos_f32[0 * S + text_len - 1] == 42.0f, "K2 last text row d0 = 42");

        /* tgt at fix_point */
        int tgt0 = text_len;
        CHECK(seq.pos_f32[0 * S + tgt0] == FIX_POINT, "K2 tgt d0 = fix_point");
        int tgt_last = text_len + image_len - 1;
        CHECK(seq.pos_f32[1 * S + tgt_last] == FIX_POINT + 31, "K2 tgt d1 max = 4127");

        /* ref1 at 4128, ref2 at 4140 (oracle K=2) */
        int r1 = text_len + image_len;
        CHECK(seq.pos_f32[0 * S + r1] == 4128.0f, "K2 ref1 d0 = 4128");
        int r2 = r1 + 144;
        CHECK(seq.pos_f32[0 * S + r2] == 4140.0f, "K2 ref2 d0 = 4140");

        /* vinput_mask covers tgt + 2 refs */
        int vmask_ones = 0;
        for (int i = 0; i < S; i++) vmask_ones += seq.vinput_mask[i];
        CHECK(vmask_ones == image_len + 288, "K2 vinput_mask covers tgt+refs");

        /* template structure: two vision blocks */
        int vs0 = -1;
        for (int i = 0; i < text_len; i++)
            if (seq.input_ids[i] == VIS_START_ID) { vs0 = i; break; }
        CHECK(vs0 == 3, "K2 first vision_start at index 3");
        CHECK(seq.input_ids[vs0 + 145] == VIS_END_ID, "K2 vision_end after 144 pads");
        /* second vision block starts after first vision_end */
        int vs1 = vs0 + 146;
        CHECK(seq.input_ids[vs1] == VIS_START_ID, "K2 second vision_start");
        CHECK(seq.input_ids[vs1 + 145] == VIS_END_ID, "K2 second vision_end");

        /* boi + tms */
        CHECK(seq.input_ids[exp_n] == BOI_TOKEN_ID, "K2 boi after template");
        CHECK(seq.input_ids[exp_n + 1] == TMS_TOKEN_ID, "K2 tms after boi");

        /* tgt + ref blocks */
        CHECK(seq.input_ids[text_len] == VIS_START_ID, "K2 tgt starts with vision_start");
        CHECK(seq.input_ids[r1] == VIS_START_ID, "K2 ref1 starts with vision_start");
        CHECK(seq.input_ids[r2] == VIS_START_ID, "K2 ref2 starts with vision_start");

        hd_sequence_free(&seq);
        free(tpl_ids); free(tpl);
    }

    printf("\n%d assertions passed, %d failed\n", 0, failures);
    if (failures) return 1;
    return 0;
}
