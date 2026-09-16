/*
 * M1.6 native prompt tokenizer / input path.
 *
 * Pure-C reproduction of the frozen Qwen2Tokenizer (GPT-2 byte-level BPE)
 * used to build the canonical HiDream input_ids. No Python oracle, no model
 * weights, no network. All vocabulary/merge/special-token data is embedded at
 * build time from the frozen manifests (see tokenizer_tables.h).
 *
 * The oracle freeze path was:
 *   template = processor.apply_chat_template(
 *       messages=[{"role":"user","content":prompt}],
 *       tokenize=False, add_generation_prompt=True)
 *   input_ids = tokenizer.encode(template, add_special_tokens=False)
 *
 * We reproduce exactly that: build the im-chat template for a single user
 * message plus the assistant generation prompt, then byte-level-BPE it with
 * GPT-2 pre-tokenization and special-token (added-token) handling.
 */

#include "tokenizer.h"

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

/* Binary search in the (piece-sorted) full vocab for a piece string. */
static int vocab_lookup(const char *piece) {
    int lo = 0, hi = hd_tok_vocab_count - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int c = strcmp(hd_tok_vocab_str[mid], piece);
        if (c == 0) return hd_tok_vocab_id[mid];
        if (c < 0) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

/* Linear search in the special-token table. */
int hd_tokenizer_special_id(const char *content) {
    if (!content) return -1;
    for (int i = 0; i < hd_tok_special_count; i++) {
        if (strcmp(hd_tok_special_str[i], content) == 0)
            return hd_tok_special_id[i];
    }
    return -1;
}

/* ------------------------------------------------------------------ */
/* UTF-8 helpers                                                       */
/* ------------------------------------------------------------------ */

static size_t utf8_next(const char *s, unsigned *cp) {
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80) { *cp = c; return 1; }
    int n; unsigned v;
    if ((c & 0xE0) == 0xC0)      { n = 2; v = c & 0x1F; }
    else if ((c & 0xF0) == 0xE0) { n = 3; v = c & 0x0F; }
    else if ((c & 0xF8) == 0xF0) { n = 4; v = c & 0x07; }
    else { *cp = 0xFFFFFFFFu; return 1; }
    for (int i = 1; i < n; i++) {
        unsigned char cc = (unsigned char)s[i];
        if ((cc & 0xC0) != 0x80) { *cp = 0xFFFFFFFFu; return (size_t)i; }
        v = (v << 6) | (cc & 0x3F);
    }
    *cp = v;
    return (size_t)n;
}

static int cp_is_alpha(unsigned cp) {
    return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
}
static int cp_is_digit(unsigned cp) {
    return cp >= '0' && cp <= '9';
}
static int cp_is_ws(unsigned cp) {
    return cp == ' ' || cp == '\n' || cp == '\t' ||
           cp == '\r' || cp == '\f' || cp == '\v';
}

/* ------------------------------------------------------------------ */
/* GPT-2 pre-tokenizer regex                                          */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *text;
    size_t len;
} hd_mctx;

/* Number of pattern chars forming one atom at index p. */
static size_t atom_plen(const char *pat, size_t plen, size_t p) {
    if (pat[p] == '\\') return 2;
    if (pat[p] == '[') {
        size_t i = p + 1;
        while (i < plen && pat[i] != ']') i++;
        return (i + 1) - p;
    }
    return 1;
}

/* Match one atom at text position pos; returns bytes consumed or -1. */
static long atom_match(const hd_mctx *m, size_t pos, const char *pat, size_t p) {
    if (pos >= m->len) return -1;
    unsigned cp; utf8_next(m->text + pos, &cp);
    size_t plen = strlen(pat);
    if (pat[p] == '\\') {
        char e = pat[p + 1];
        if (e == 's') return cp_is_ws(cp) ? (long)utf8_next(m->text + pos, &cp) : -1;
        if (e == 'S') return cp_is_ws(cp) ? -1 : (long)utf8_next(m->text + pos, &cp);
        return -1;
    }
    if (pat[p] == '[') {
        size_t i = p + 1;
        int neg = 0;
        if (pat[i] == '^') { neg = 1; i++; }
        int in_class = 0;
        while (pat[i] != ']') {
            if (pat[i] == '\\' && i + 1 < plen) {
                char e = pat[i + 1];
                if (e == 's') { if (cp_is_ws(cp)) in_class = 1; }
                else if (e == 'S') { if (!cp_is_ws(cp)) in_class = 1; }
                else if (e == 'd') { if (cp_is_digit(cp)) in_class = 1; }
                else if (e == 'D') { if (!cp_is_digit(cp)) in_class = 1; }
                else if (e == 'w') { if (cp_is_alpha(cp) || cp_is_digit(cp) || cp == '_') in_class = 1; }
                else if (e == 'W') { if (!(cp_is_alpha(cp) || cp_is_digit(cp) || cp == '_')) in_class = 1; }
                i += 2;
                continue;
            }
            char lo = pat[i];
            if (pat[i + 1] == '-' && pat[i + 2] != ']' &&
                pat[i + 2] && pat[i + 2] != ']') {
                char hi = pat[i + 2];
                if (cp_is_alpha(lo) && cp_is_alpha(hi)) {
                    if ((unsigned char)lo <= cp && cp <= (unsigned char)hi) in_class = 1;
                } else if (cp_is_digit(lo) && cp_is_digit(hi)) {
                    if ((unsigned char)lo <= cp && cp <= (unsigned char)hi) in_class = 1;
                }
                i += 3;
            } else {
                if (cp == (unsigned char)lo) in_class = 1;
                i++;
            }
        }
        if (neg) in_class = !in_class;
        return in_class ? (long)utf8_next(m->text + pos, &cp) : -1;
    }
    if ((unsigned char)pat[p] == cp && cp < 0x80) return 1;
    return -1;
}

static long match_seq(const hd_mctx *m, size_t pos, const char *pat, size_t plen, size_t p);

static long match_seq(const hd_mctx *m, size_t pos, const char *pat, size_t plen, size_t p) {
    if (p >= plen) return (long)pos;
    char c = pat[p];

    if (c == '(') {
        if (pat[p + 1] == '?' && pat[p + 2] == '!') {
            size_t q = p + 3;
            while (q < plen && pat[q] != ')') q++;
            int inner_ws = 1;
            if (q - (p + 3) == 2 && pat[p + 3] == '\\' && pat[p + 4] == 'S')
                inner_ws = 0;
            if (!inner_ws && pos < m->len) {
                unsigned cp; utf8_next(m->text + pos, &cp);
                if (!cp_is_ws(cp)) return -1;
            }
            return match_seq(m, pos, pat, plen, q + 1);
        }
        return -1;
    }

    size_t alen = atom_plen(pat, plen, p);
    char quant = (p + alen < plen) ? pat[p + alen] : '\0';

    if (quant == '+') {
        /* one or more: consume at least one atom first, then greedily extend
           (backtracking) and keep the longest full match. */
        long first = atom_match(m, pos, pat, p);
        if (first < 0) return -1;
        long best = -1;
        long cur = pos + (size_t)first; /* we have consumed >=1 atom */
        for (;;) {
            long rest = match_seq(m, (size_t)cur, pat, plen, p + alen + 1);
            if (rest >= 0 && (best < 0 || rest > best)) best = rest;
            long more = atom_match(m, (size_t)cur, pat, p);
            if (more < 0) break;
            cur += more;
        }
        return best;
    } else if (quant == '?') {
        long best = match_seq(m, pos, pat, plen, p + alen + 1);
        long ate = atom_match(m, pos, pat, p);
        if (ate >= 0) {
            long r2 = match_seq(m, pos + (size_t)ate, pat, plen, p + alen + 1);
            if (r2 >= 0 && r2 > best) best = r2;
        }
        return best;
    } else {
        long ate = atom_match(m, pos, pat, p);
        if (ate < 0) return -1;
        return match_seq(m, pos + (size_t)ate, pat, plen, p + alen);
    }
}

static size_t match_pattern(const char *text, size_t len, size_t pos,
                            const char *pat) {
    hd_mctx m = { text, len };
    long e = match_seq(&m, pos, pat, strlen(pat), 0);
    if (e < 0 || (size_t)e <= pos) return 0;
    return (size_t)e - pos;
}

/* ------------------------------------------------------------------ */
/* BPE merge                                                           */
/* ------------------------------------------------------------------ */

typedef struct {
    size_t count;
    const char **str;
    unsigned char *owned;
} hd_pieces;

static void pieces_clear(hd_pieces *p) {
    for (size_t i = 0; i < p->count; i++)
        if (p->owned[i]) free((void *)p->str[i]);
    free(p->str);
    free(p->owned);
}

/* Binary search in the (left,right)-sorted merge table. */
static int merge_rank_of(const char *l, const char *r) {
    int lo = 0, hi = hd_tok_merge_count - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int c = strcmp(hd_tok_merge_left[mid], l);
        if (c == 0) c = strcmp(hd_tok_merge_right[mid], r);
        if (c == 0) return hd_tok_merge_rank[mid];
        if (c < 0) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

static int bpe_apply_merge(hd_pieces *p) {
    int best_rank = -1, best_at = -1;
    for (size_t i = 0; i + 1 < p->count; i++) {
        int r = merge_rank_of(p->str[i], p->str[i + 1]);
        if (r >= 0 && (best_rank < 0 || r < best_rank)) { best_rank = r; best_at = (int)i; }
    }
    if (best_rank < 0) return 0;

    const char *l = p->str[best_at], *r = p->str[best_at + 1];
    size_t nl = strlen(l), nr = strlen(r);
    char *merged = malloc(nl + nr + 1);
    if (!merged) return -1;
    memcpy(merged, l, nl);
    memcpy(merged + nl, r, nr);
    merged[nl + nr] = '\0';

    if (p->owned[best_at + 1]) free((void *)p->str[best_at + 1]);
    p->str[best_at] = merged;
    p->owned[best_at] = 1;
    for (size_t i = (size_t)best_at + 1; i + 1 < p->count; i++) {
        p->str[i] = p->str[i + 1];
        p->owned[i] = p->owned[i + 1];
    }
    p->count--;
    return 1;
}

static char **bpe_encode(const char *text, size_t nbytes, size_t *out_count) {
    *out_count = 0;
    if (nbytes == 0) return NULL;

    hd_pieces p;
    p.count = nbytes;
    p.str = malloc(nbytes * sizeof(char *));
    p.owned = malloc(nbytes);
    if (!p.str || !p.owned) {
        free(p.str); free(p.owned);
        hd_set_error("tokenizer oom (pieces)");
        return NULL;
    }
    for (size_t i = 0; i < nbytes; i++) {
        p.str[i] = hd_tok_byte_piece[(unsigned char)text[i]];
        p.owned[i] = 0;
    }

    for (;;) {
        int r = bpe_apply_merge(&p);
        if (r < 0) { pieces_clear(&p); hd_set_error("tokenizer oom (merge)"); return NULL; }
        if (r == 0) break;
    }

    char **out = malloc(p.count * sizeof(char *));
    if (!out) { pieces_clear(&p); hd_set_error("tokenizer oom (result)"); return NULL; }
    for (size_t i = 0; i < p.count; i++) {
        size_t n = strlen(p.str[i]) + 1;
        out[i] = malloc(n);
        if (!out[i]) {
            for (size_t j = 0; j < i; j++) free(out[j]);
            free(out);
            pieces_clear(&p);
            hd_set_error("tokenizer oom (result copy)");
            return NULL;
        }
        memcpy(out[i], p.str[i], n);
    }
    size_t cnt = p.count;
    pieces_clear(&p);
    *out_count = cnt;
    return out;
}

/* ------------------------------------------------------------------ */
/* encode: special tokens + pre-tokenize + BPE                        */
/* ------------------------------------------------------------------ */

static int ensure_cap(int **ids, size_t *cap, size_t n_ids, size_t need) {
    if (n_ids + need <= *cap) return 1;
    size_t ncap = *cap ? *cap : 16;
    while (n_ids + need > ncap) ncap *= 2;
    int *ni = realloc(*ids, ncap * sizeof(int));
    if (!ni) { hd_set_error("tokenizer oom (ids)"); return 0; }
    *ids = ni; *cap = ncap;
    return 1;
}

static size_t pretokenize(const char *text, size_t len, size_t pos) {
    static const char *const PATS[] = {
        "'s", "'t", "'re", "'ve", "'m", "'ll", "'d",
        " ?[A-Za-z]+",
        " ?[0-9]+",
        " ?[^\\sA-Za-z0-9]+",
        "\\s+(?!\\S)",
        "\\s+",
    };
    for (size_t pi = 0; pi < sizeof(PATS) / sizeof(PATS[0]); pi++) {
        size_t mlen = match_pattern(text, len, pos, PATS[pi]);
        if (mlen > 0) return mlen;
    }
    return 0;
}

static int special_at(const char *text, size_t at, size_t *len, int *id) {
    for (int i = 0; i < hd_tok_special_count; i++) {
        const char *s = hd_tok_special_str[i];
        size_t sl = strlen(s);
        if (strncmp(text + at, s, sl) == 0) { *len = sl; *id = hd_tok_special_id[i]; return 1; }
    }
    return 0;
}

hd_status hd_tokenizer_encode(const char *text, int **out_ids, size_t *out_count) {
    *out_ids = NULL; *out_count = 0;
    if (!text) { hd_set_error("tokenizer: null input"); return HD_ERR_PARSE; }

    int *ids = NULL; size_t n_ids = 0, cap = 0;
    size_t len = strlen(text), i = 0;
    hd_status status = HD_OK;

    /* Split-isolated semantics: scan for special (added) tokens atomically;
       only plain runs between special tokens get byte-level BPE. */
    while (i < len) {
        size_t sl; int sid;
        if (special_at(text, i, &sl, &sid)) {
            if (!ensure_cap(&ids, &cap, n_ids, 1)) { status = HD_ERR_OOM; break; }
            ids[n_ids++] = sid;
            i += sl;
            continue;
        }

        /* Consume a plain segment that contains no special token. */
        size_t start = i;
        while (i < len && !special_at(text, i, &sl, &sid)) i++;
        size_t end = i;

        /* Split the plain segment with the GPT-2 regex pre-tokenizer and
           byte-level BPE each token separately (merges never cross PAT
           token boundaries), exactly like the oracle. */
        size_t pos = start;
        while (pos < end) {
            size_t mlen = pretokenize(text, end, pos);
            if (mlen == 0) mlen = 1;

            size_t np; char **pieces = bpe_encode(text + pos, mlen, &np);
            if (!pieces) { status = HD_ERR_OOM; goto out; }
            for (size_t k = 0; k < np; k++) {
                int vid = vocab_lookup(pieces[k]);
                if (vid < 0) {
                    hd_set_error("tokenizer: piece '%s' not in vocab", pieces[k]);
                    for (size_t j = 0; j < np; j++) free(pieces[j]);
                    free(pieces);
                    status = HD_ERR_PARSE;
                    goto out;
                }
                if (!ensure_cap(&ids, &cap, n_ids, 1)) {
                    for (size_t j = 0; j < np; j++) free(pieces[j]);
                    free(pieces);
                    status = HD_ERR_OOM;
                    goto out;
                }
                ids[n_ids++] = vid;
            }
            for (size_t j = 0; j < np; j++) free(pieces[j]);
            free(pieces);
            pos += mlen;
        }
    }

out:
    if (status != HD_OK) { free(ids); return status; }
    *out_ids = ids; *out_count = n_ids;
    return HD_OK;
}

hd_status hd_tokenizer_build_template(const char *prompt, char **out) {
    *out = NULL;
    if (!prompt) { hd_set_error("tokenizer: null prompt"); return HD_ERR_PARSE; }
    static const char pre[] = "<|im_start|>user\n";
    static const char post[] = "<|im_end|>\n<|im_start|>assistant\n";
    size_t nl = strlen(prompt);
    size_t total = sizeof(pre) - 1 + nl + sizeof(post) - 1;
    char *buf = malloc(total + 1);
    if (!buf) { hd_set_error("tokenizer oom (template)"); return HD_ERR_OOM; }
    memcpy(buf, pre, sizeof(pre) - 1);
    memcpy(buf + sizeof(pre) - 1, prompt, nl);
    memcpy(buf + sizeof(pre) - 1 + nl, post, sizeof(post) - 1);
    buf[total] = '\0';
    *out = buf;
    return HD_OK;
}

/*
 * Builds the ref-mode im-chat template: K <|vision_start|><|image_pad|>
 * <|vision_end|> placeholders followed by the caption, then the assistant
 * generation prompt. Matches the oracle's apply_chat_template for
 * content=[{"type":"image"}]*K + [{"type":"text","text":caption}].
 */
hd_status hd_tokenizer_build_ref_template(const char *caption, int k,
                                          char **out) {
    *out = NULL;
    if (!caption || k < 0) { hd_set_error("tokenizer: bad ref template args"); return HD_ERR_PARSE; }
    static const char pre[] = "<|im_start|>user\n";
    static const char post[] = "<|im_end|>\n<|im_start|>assistant\n";
    static const char img[] = "<|vision_start|><|image_pad|><|vision_end|>";
    size_t nl = strlen(caption);
    size_t total = sizeof(pre) - 1 + (size_t)k * (sizeof(img) - 1) + nl + sizeof(post) - 1;
    char *buf = malloc(total + 1);
    if (!buf) { hd_set_error("tokenizer oom (ref template)"); return HD_ERR_OOM; }
    size_t off = 0;
    memcpy(buf + off, pre, sizeof(pre) - 1); off += sizeof(pre) - 1;
    for (int i = 0; i < k; i++) {
        memcpy(buf + off, img, sizeof(img) - 1); off += sizeof(img) - 1;
    }
    memcpy(buf + off, caption, nl); off += nl;
    memcpy(buf + off, post, sizeof(post) - 1); off += sizeof(post) - 1;
    buf[total] = '\0';
    *out = buf;
    return HD_OK;
}

hd_status hd_tokenizer_encode_prompt(const char *prompt, int **out_ids, size_t *out_count) {
    char *tpl = NULL;
    hd_status st = hd_tokenizer_build_template(prompt, &tpl);
    if (st != HD_OK) return st;
    st = hd_tokenizer_encode(tpl, out_ids, out_count);
    free(tpl);
    return st;
}

void hd_tokenizer_free_ids(int *ids) { free(ids); }
