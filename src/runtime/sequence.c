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
#include "tokenizer.h"

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
                       const hd_ref_geom *refs, hd_sequence *out) {
    if (!req || !out) {
        hd_set_error("sequence: null request");
        return HD_ERR_MISSING;
    }
    memset(out, 0, sizeof(*out));

    int K = req->reference_count;
    if (K < 0 || K > HD_SEQ_MAX_REFS) {
        hd_set_error("sequence: ref count out of range");
        return HD_ERR_MISSING;
    }
    if (K > 0 && !refs) {
        hd_set_error("sequence: refs geometry required");
        return HD_ERR_MISSING;
    }
    if (height <= 0 || width <= 0 || patch_size <= 0) {
        hd_set_error("sequence: bad dims");
        return HD_ERR_MISSING;
    }

    int grid_h = height / patch_size;
    int grid_w = width / patch_size;
    int image_len = grid_h * grid_w;   /* target image token count */

    /* --- tokenize the ref-mode template --- */
    /* <|im_start|>user\n + K*(<|vision_start|><|image_pad|><|vision_end|>)
       + caption + <|im_end|>\n<|im_start|>assistant\n */
    char *tpl = NULL;
    hd_status st = hd_tokenizer_build_ref_template(req->prompt, K, &tpl);
    if (st != HD_OK) return st;
    int *tpl_ids = NULL;
    size_t tpl_n = 0;
    st = hd_tokenizer_encode(tpl, &tpl_ids, &tpl_n);
    free(tpl);
    if (st != HD_OK) return st;

    /*
     * Expand each template image placeholder to its VLM cond grid
     * (cond_h*cond_w image_pad tokens), exactly like the oracle processor
     * (`proc(text=[tpl], images=cond_pils, ...)` replaces the single
     * <|image_pad|> per <|image|> with cond_h*cond_w pads). The template
     * string carries one placeholder pad per reference; the tokenized
     * sequence must carry the full cond grid.
     */
    size_t exp_n = tpl_n;
    for (int i = 0; i < K; i++) {
        if (refs[i].cond_h <= 0 || refs[i].cond_w <= 0) {
            hd_set_error("sequence: bad ref cond grid");
            hd_tokenizer_free_ids(tpl_ids);
            return HD_ERR_MISSING;
        }
        exp_n += (size_t)refs[i].cond_h * refs[i].cond_w - 1;
    }
    int *exp_ids = xmalloc(exp_n * sizeof(int));
    if (!exp_ids) { hd_tokenizer_free_ids(tpl_ids); return HD_ERR_OOM; }
    size_t eo = 0;
    int ri = 0;
    for (size_t i = 0; i < tpl_n; i++) {
        if (tpl_ids[i] == vision_start_token_id && i + 1 < tpl_n &&
            tpl_ids[i + 1] == image_token_id) {
            exp_ids[eo++] = vision_start_token_id;
            int n = refs[ri].cond_h * refs[ri].cond_w;
            for (int j = 0; j < n; j++) exp_ids[eo++] = image_token_id;
            ri++;
            i++;                       /* skip the single placeholder pad */
        } else {
            exp_ids[eo++] = tpl_ids[i];
        }
    }
    hd_tokenizer_free_ids(tpl_ids);
    if (ri != K) {
        hd_set_error("sequence: expanded %d/%d ref placeholders", ri, K);
        free(exp_ids);
        return HD_ERR_MISSING;
    }
    tpl_ids = exp_ids;
    tpl_n = eo;

    /* --- append boi + tms*N --- */
    int text_len = (int)tpl_n + 1 + timestep_token_num;
    int64_t *ids = xmalloc((size_t)text_len * sizeof(int64_t));
    if (!ids) { hd_tokenizer_free_ids(tpl_ids); return HD_ERR_OOM; }
    for (int i = 0; i < (int)tpl_n; i++) ids[i] = tpl_ids[i];
    hd_tokenizer_free_ids(tpl_ids);
    ids[tpl_n] = 151669;                       /* <|boi_token|> */
    for (int i = 0; i < timestep_token_num; i++)
        ids[tpl_n + 1 + i] = tms_token_id;

    /* --- vision tokens: tgt first, then refs --- */
    int ref_total = 0;
    for (int i = 0; i < K; i++) {
        if (refs[i].tokens <= 0) {
            hd_set_error("sequence: bad ref token count");
            free(ids); return HD_ERR_MISSING;
        }
        ref_total += refs[i].tokens;
    }
    int S = text_len + image_len + ref_total;
    int64_t *all_ids = xmalloc((size_t)S * sizeof(int64_t));
    if (!all_ids) { free(ids); return HD_ERR_OOM; }
    memcpy(all_ids, ids, (size_t)text_len * sizeof(int64_t));
    free(ids);

    int off = text_len;
    /* target block: vision_start + (image_len-1)*image_pad */
    all_ids[off++] = vision_start_token_id;
    for (int i = 1; i < image_len; i++) all_ids[off++] = image_token_id;
    /* reference blocks */
    for (int r = 0; r < K; r++) {
        all_ids[off++] = vision_start_token_id;
        for (int i = 1; i < refs[r].tokens; i++) all_ids[off++] = image_token_id;
    }

    /*
     * --- position_ids [3,1,S] ---
     * get_rope_index_fix_point with skip_vision_start_token =
     * [0]*K + [1] + [1]*K. Vision blocks in input_ids order:
     *   v=0..K-1  refs in template (skip 0): text continuous, grid at
     *             text_len+st_idx (cond grid)
     *   v=K       tgt (skip 1): text continuous, grid at fix_point
     *   v=K+1..2K refs in vision blocks (skip 1): grid at st_idx
     * st_idx = max(assigned positions)+1 before each block (oracle).
     */
    float *pos = xmalloc(3u * (size_t)S * sizeof(float));
    if (!pos) { free(all_ids); return HD_ERR_OOM; }

    /* Vision-start positions in input_ids order. */
    int vs_pos[1 + 2 * HD_SEQ_MAX_REFS];
    int vs_n = 0;
    for (int i = 0; i < S; i++)
        if (all_ids[i] == vision_start_token_id) vs_pos[vs_n++] = i;
    /* Expect 2K+1 vision starts (K template + tgt + K vision blocks). */
    if (vs_n != 2 * K + 1) {
        hd_set_error("sequence: vision-start count mismatch");
        free(all_ids); free(pos); return HD_ERR_MISSING;
    }

    int max_pos = -1;
    int fp = fix_point;
    int st_pos = 0;
    int n_blocks = 2 * K + 1;
    for (int v = 0; v < n_blocks; v++) {
        int ed = vs_pos[v] + 1;          /* first image_pad after vision_start */
        int skip = (v < K) ? 0 : 1;
        int tlen = ed - st_pos;
        if (skip) tlen -= 1;
        if (tlen < 0) tlen = 0;
        int st_idx = (max_pos >= 0) ? max_pos + 1 : 0;
        for (int i = 0; i < tlen; i++) {
            pos[0 * S + st_pos + i] = (float)(st_idx + i);
            pos[1 * S + st_pos + i] = (float)(st_idx + i);
            pos[2 * S + st_pos + i] = (float)(st_idx + i);
        }
        max_pos = st_idx + tlen - 1;

        int gh, gw, gt = 1;
        if (v < K) { gh = refs[v].cond_h; gw = refs[v].cond_w; }
        else if (v == K) { gh = grid_h; gw = grid_w; }
        else { int r = v - K - 1; gh = refs[r].grid_h; gw = refs[r].grid_w; }
        if (gh <= 0 || gw <= 0) {
            hd_set_error("sequence: bad ref grid");
            free(all_ids); free(pos); return HD_ERR_MISSING;
        }
        int vlen = gt * gh * gw;
        int base;
        if (skip) {
            if (fp > 0) fp = fp - st_idx;
            base = fp + st_idx;
            fp = 0;
        } else {
            base = tlen + st_idx;
        }
        for (int t = 0; t < gt; t++)
            for (int h = 0; h < gh; h++)
                for (int w = 0; w < gw; w++) {
                    int idx = st_pos + tlen + t * gh * gw + h * gw + w;
                    pos[0 * S + idx] = (float)(base + t);
                    pos[1 * S + idx] = (float)(base + h);
                    pos[2 * S + idx] = (float)(base + w);
                }
        int gmax = base + (gt - 1 > gh - 1 ? (gt - 1 > gw - 1 ? gt - 1 : gw - 1)
                                           : (gh - 1 > gw - 1 ? gh - 1 : gw - 1));
        if (gmax > max_pos) max_pos = gmax;
        /*
         * The block occupies [st_pos, st_pos + tlen + vlen): text region
         * first, then the vision grid (which for skip=1 blocks includes the
         * vision_start token). The oracle's internal `st = ed + grid` cursor
         * is off by one for skip=1 blocks; the concatenation semantics give
         * the exact end as st_pos + tlen + vlen.
         */
        st_pos += tlen + vlen;
    }
    /* trailing text (none in ref modes: vision blocks end the sequence) */
    if (st_pos < S) {
        int tlen = S - st_pos;
        int st_idx = max_pos + 1;
        for (int i = 0; i < tlen; i++) {
            pos[0 * S + st_pos + i] = (float)(st_idx + i);
            pos[1 * S + st_pos + i] = (float)(st_idx + i);
            pos[2 * S + st_pos + i] = (float)(st_idx + i);
        }
    }

    /*
     * --- attention mask [1,1,S,S] bf16 ---
     * causal triu min above diagonal; token rows (types 1,2,3) zeroed.
     * token_types: 1 on tgt rows, 2 on ref rows, 3 on tms rows.
     */
    unsigned char *mask = xmalloc((size_t)S * (size_t)S * 2);
    if (!mask) { free(all_ids); free(pos); return HD_ERR_OOM; }
    unsigned char *types = xmalloc((size_t)S);
    if (!types) { free(all_ids); free(pos); free(mask); return HD_ERR_OOM; }
    for (int i = 0; i < S; i++) types[i] = 0;
    /* tms rows: [tpl_n+1, text_len) */
    for (int i = (int)tpl_n + 1; i < text_len; i++) types[i] = 3;
    /* tgt rows: [text_len, text_len+image_len) */
    for (int i = text_len; i < text_len + image_len; i++) types[i] = 1;
    /* ref rows */
    int roff = text_len + image_len;
    for (int r = 0; r < K; r++) {
        for (int i = 0; i < refs[r].tokens; i++) types[roff + i] = 2;
        roff += refs[r].tokens;
    }
    for (int r = 0; r < S; r++) {
        int is_token_row = (types[r] != 0);
        for (int c = 0; c < S; c++) {
            uint16_t bits;
            if (is_token_row) {
                bits = 0x0000u;
            } else if (c > r) {
                bits = BF16_MIN_BITS;
            } else {
                bits = 0x0000u;
            }
            size_t off = ((size_t)r * S + c) * 2;
            mask[off] = (unsigned char)(bits & 0xFF);
            mask[off + 1] = (unsigned char)(bits >> 8);
        }
    }

    /* --- vinput_mask [S]: image rows (types 1 or 2) --- */
    unsigned char *vmask = xmalloc((size_t)S);
    if (!vmask) { free(all_ids); free(pos); free(mask); free(types); return HD_ERR_OOM; }
    for (int i = 0; i < S; i++)
        vmask[i] = (types[i] == 1 || types[i] == 2) ? 1 : 0;
    free(types);

    out->input_ids = all_ids;
    out->pos_f32 = pos;
    out->mask_bf16 = mask;
    out->vinput_mask = vmask;
    out->sec[0] = 24; out->sec[1] = 20; out->sec[2] = 20;
    out->text_len = text_len;
    out->image_len = image_len;
    out->img_begin = text_len;
    out->S = S;
    out->n_refs = K;
    int acc = text_len + image_len;
    for (int r = 0; r < K; r++) {
        out->ref_len[r] = refs[r].tokens;
        out->ref_begin[r] = acc;
        acc += refs[r].tokens;
    }
    return HD_OK;
}

void hd_sequence_free(hd_sequence *s) {
    if (!s) return;
    free(s->input_ids);
    free(s->pos_f32);
    free(s->mask_bf16);
    free(s->vinput_mask);
    memset(s, 0, sizeof(*s));
}

/* ------------------------------------------------------------------ */
/* Sequence manifest diagnostics (M1-post §56)                         */
/* ------------------------------------------------------------------ */

/*
 * Frozen model dims for the workspace estimate (mirror tests/unit/
 * sanity_gen.c and src/model/forward.c layout): H=4096, NH=32, NKV=8,
 * HD=128, ff_hidden=12288, head_out=3072.
 */
#define DIAG_H       4096
#define DIAG_NH      32
#define DIAG_NKV     8
#define DIAG_HD      128
#define DIAG_FF      12288
#define DIAG_HEADOUT 3072

/* Known special-token ids (tokenizer.h + sequence builder). */
static const struct { int id; const char *name; } k_diag_special[] = {
    { 151644, "im_start"   },
    { 151645, "im_end"     },
    { 151652, "vision_start" },
    { 151653, "vision_end" },
    { 151655, "image_pad"  },
    { 151656, "video_pad"  },
    { 151669, "boi"        },
    { 151673, "tms"        },
};

/*
 * Documented workspace estimate (forward.c forward_layout_offsets +
 * block.c hd_decoder_block_scratch_bytes, both CPU-reproducible):
 *   persistent regions: hidden_a/b, h_text, norm_out, head_out, t_emb,
 *   te_hidden, freq fp32/bf16, t_scaled, xe_stage, xe_out
 *   block scratch: bf16 q/k/v/scores/probs/attn/mlp buffers + fp32
 *   cos/sin + mrope slack
 * Attention scores+probs alone are 2*NH*S^2 bf16 elements = 128*S^2
 * bytes (reproduces ITEM-06: S=4115 -> 2,167,452,800 B = 2165.9 MB).
 */
static int64_t diag_workspace_bytes(const hd_sequence *s,
                                    int64_t *scores_probs) {
    int64_t S = s->S, img = s->image_len, text = s->text_len;
    int64_t H = DIAG_H, NH = DIAG_NH, NKV = DIAG_NKV, HD = DIAG_HD;
    int64_t FF = DIAG_FF;

    int64_t sp = 2 * NH * S * S * 2;   /* scores + probs, bf16 */
    if (scores_probs) *scores_probs = sp;

    int64_t o = 0;
    o += S * H * 2;                    /* hidden_a  */
    o += S * H * 2;                    /* hidden_b  */
    o += text * H * 2;                 /* h_text    */
    o += S * H * 2;                    /* norm_out  */
    o += S * DIAG_HEADOUT * 2;         /* head_out  */
    o += H * 2;                        /* t_emb     */
    o += H * 2;                        /* te_hidden */
    o += 256 * 4;                      /* freq fp32 */
    o += 256 * 2;                      /* freq_bf16 */
    o += 4;                            /* t_scaled  */
    o += img * 1024 * 2;               /* xe_stage  */
    o += img * H * 2;                  /* xe_out    */
    o = (o + 7) & ~(int64_t)7;

    int64_t e = 0;
    e += S * H;                        /* in_ln       */
    e += 3 * S * NH * HD;              /* qp, q, qr   */
    e += 3 * S * NKV * HD;             /* kp, k, kr   */
    e += 2 * S * NKV * HD;             /* vp, v       */
    e += NH * S * S;                   /* scores      */
    e += NH * S * S;                   /* probs       */
    e += S * NH * HD;                  /* attn_sm     */
    e += 3 * S * H;                    /* attn_m/h/r  */
    e += S * H;                        /* post        */
    e += 3 * S * FF;                   /* gate, up, swi */
    e += 2 * S * H;                    /* mlp, mlp_r  */
    int64_t bs = e * 2 + 2 * S * HD * 4 + 256;
    return o + bs;
}

int hd_seq_diag(const hd_sequence *s, const char *tag) {
    if (!s) return -1;
    const char *t = tag ? tag : "-";
    printf("[seq:%s] text_len=%d image_len=%d S=%d img_begin=%d\n",
           t, s->text_len, s->image_len, s->S, s->img_begin);
    printf("[seq:%s] sec=[%lld %lld %lld] n_refs=%d\n", t,
           (long long)s->sec[0], (long long)s->sec[1], (long long)s->sec[2],
           s->n_refs);
    for (int i = 0; i < s->n_refs && i < HD_SEQ_MAX_REFS; i++)
        printf("[seq:%s] ref[%d] len=%d begin=%d\n", t, i,
               s->ref_len[i], s->ref_begin[i]);
    printf("[seq:%s] pos=[3,1,%d] mask=[1,1,%d,%d] vinput=[%d..%d)\n",
           t, s->S, s->S, s->S, s->img_begin, s->S);

    /* text tokens (input_ids[0..text_len)) */
    printf("[seq:%s] text_tokens[%d]=", t, s->text_len);
    for (int i = 0; i < s->text_len; i++)
        printf("%s%lld", i ? " " : "", (long long)s->input_ids[i]);
    printf("\n");

    /* special-token indices (first occurrence + count per known id).
     * hd_seq_t2i stores only the text tokens in input_ids (vision tokens
     * are implicit); hd_seq_build stores the full S-token sequence. */
    int scan_n = (s->n_refs > 0) ? s->S : s->text_len;
    printf("[seq:%s] special_tokens:", t);
    int printed = 0;
    for (size_t k = 0; k < sizeof(k_diag_special) / sizeof(k_diag_special[0]);
         k++) {
        int first = -1, count = 0;
        for (int i = 0; i < scan_n; i++) {
            if (s->input_ids[i] == k_diag_special[k].id) {
                if (first < 0) first = i;
                count++;
            }
        }
        if (count > 0) {
            printf("%s %s@%d(x%d)", printed ? "," : "",
                   k_diag_special[k].name, first, count);
            printed = 1;
        }
    }
    if (!printed) printf(" none");
    printf("\n");

    /* MRoPE sections */
    printf("[seq:%s] mrope_sections=[%lld %lld %lld] total=%lld\n", t,
           (long long)s->sec[0], (long long)s->sec[1], (long long)s->sec[2],
           (long long)(s->sec[0] + s->sec[1] + s->sec[2]));

    /* prediction-mask spans: rows with any bf16-min above the diagonal are
     * causal text rows; fully-zeroed rows are full-attention token rows. */
    printf("[seq:%s] mask_spans:", t);
    int in_causal = 0, in_full = 0;
    for (int r = 0; r < s->S; r++) {
        int causal = 0;
        for (int c = r + 1; c < s->S; c++) {
            size_t off = ((size_t)r * s->S + c) * 2;
            if (s->mask_bf16[off] == 0x7Fu && s->mask_bf16[off + 1] == 0xFFu) {
                causal = 1;
                break;
            }
        }
        if (causal && !in_causal) {
            if (in_full) printf("..%d]", r - 1);
            printf(" causal[%d", r);
            in_causal = 1; in_full = 0;
        } else if (!causal && !in_full) {
            if (in_causal) printf("..%d]", r - 1);
            printf(" full[%d", r);
            in_full = 1; in_causal = 0;
        }
    }
    if (in_causal) printf("..%d]", s->S - 1);
    if (in_full) printf("..%d]", s->S - 1);
    if (!in_causal && !in_full) printf(" none");
    printf("\n");

    /* workspace estimate */
    int64_t sp = 0;
    int64_t total = diag_workspace_bytes(s, &sp);
    printf("[seq:%s] workspace_estimate: scores+probs=%lld B, total=%lld B "
           "(%.1f MB)\n", t, (long long)sp, (long long)total,
           (double)total / (1024.0 * 1024.0));
    return 0;
}