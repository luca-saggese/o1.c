/*
 * Named reference aliases: frontend-only prompt expansion.
 *
 * See ref_alias.h. No model, tokenizer or CUDA dependency: this is pure
 * string/metadata handling that runs before hd_seq_build.
 */

#include "ref_alias.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "hidream.h"

/* ------------------------------------------------------------------ */
/* Error buffer                                                        */
/* ------------------------------------------------------------------ */

static char g_err[512];

const char *hd_ref_alias_error(void) { return g_err; }

static void set_err(const char *fmt, const char *a, const char *b) {
    snprintf(g_err, sizeof(g_err), fmt, a ? a : "", b ? b : "");
}

/* ------------------------------------------------------------------ */
/* Alias token validation                                              */
/* ------------------------------------------------------------------ */

int hd_ref_alias_valid(const char *alias) {
    if (!alias || !alias[0]) return 0;
    /* reserved internal prefix */
    if (alias[0] == '_' && alias[1] == '_') return 0;
    unsigned char c0 = (unsigned char)alias[0];
    if (!((c0 >= 'A' && c0 <= 'Z') || (c0 >= 'a' && c0 <= 'z') ||
          c0 == '_'))
        return 0;
    for (const char *p = alias; *p; p++) {
        unsigned char c = (unsigned char)*p;
        int ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                 (c >= '0' && c <= '9') || c == '_' || c == '-';
        if (!ok) return 0; /* rejects whitespace and '=' too */
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* --ref-image parsing                                                 */
/* ------------------------------------------------------------------ */

int hd_ref_alias_split(const char *arg, const char **alias, const char **path) {
    if (alias) *alias = NULL;
    if (path) *path = NULL;
    if (!arg || !arg[0]) return HD_ERR_MISSING;

    const char *eq = strchr(arg, '=');
    if (eq && eq != arg) {
        size_t n = (size_t)(eq - arg);
        char name[128];
        if (n < sizeof(name)) {
            memcpy(name, arg, n);
            name[n] = '\0';
            if (hd_ref_alias_valid(name)) {
                if (alias) *alias = arg;   /* caller re-reads the prefix */
                if (path) *path = eq + 1;
                return HD_OK;
            }
        }
    }
    /* No valid alias prefix: the whole argument is the path. */
    if (path) *path = arg;
    return HD_OK;
}

/* ------------------------------------------------------------------ */
/* Table construction                                                  */
/* ------------------------------------------------------------------ */

int hd_ref_alias_build(hd_ref_alias_table *tbl,
                       const char *const *user_paths,
                       const char *const *user_aliases,
                       int n_user,
                       const char *const *internal_paths,
                       int n_internal) {
    if (!tbl) return HD_ERR_MISSING;
    memset(tbl, 0, sizeof(*tbl));
    g_err[0] = '\0';

    if (n_user < 0) n_user = 0;
    if (n_internal < 0) n_internal = 0;
    if (n_user + n_internal > HD_REF_ALIAS_MAX) {
        set_err("too many references (max %s)", "16", NULL);
        return HD_ERR_MISMATCH;
    }

    for (int i = 0; i < n_user; i++) {
        hd_ref_alias_entry *e = &tbl->entries[tbl->count];
        e->path = user_paths ? user_paths[i] : NULL;
        e->user_alias = user_aliases ? user_aliases[i] : NULL;
        e->is_internal = 0;
        e->user_index = i;
        e->effective_index = tbl->count;
        if (e->user_alias && e->user_alias[0]) {
            if (!hd_ref_alias_valid(e->user_alias)) {
                set_err("invalid reference alias: %s", e->user_alias, NULL);
                return HD_ERR_MISMATCH;
            }
            for (int j = 0; j < tbl->count; j++) {
                const hd_ref_alias_entry *p = &tbl->entries[j];
                if (p->user_alias && p->user_alias[0] &&
                    strcmp(p->user_alias, e->user_alias) == 0) {
                    set_err("duplicate reference alias: %s", e->user_alias,
                            NULL);
                    return HD_ERR_MISMATCH;
                }
            }
        }
        tbl->count++;
    }

    for (int j = 0; j < n_internal; j++) {
        hd_ref_alias_entry *e = &tbl->entries[tbl->count];
        e->path = internal_paths ? internal_paths[j] : NULL;
        e->user_alias = NULL;
        e->is_internal = 1;
        e->user_index = -1;
        e->effective_index = tbl->count;
        tbl->count++;
    }
    return HD_OK;
}

/* ------------------------------------------------------------------ */
/* Lookup                                                              */
/* ------------------------------------------------------------------ */

int hd_ref_alias_lookup(const hd_ref_alias_table *tbl, const char *name) {
    if (!tbl || !name || !name[0]) return 0;
    for (int i = 0; i < tbl->count; i++) {
        const hd_ref_alias_entry *e = &tbl->entries[i];
        if (e->user_alias && e->user_alias[0] &&
            strcmp(e->user_alias, name) == 0)
            return i + 1; /* 1-based reference image number */
    }
    /* automatic @refN (N = 1-based effective index) */
    if (strncmp(name, "ref", 3) == 0 && name[3]) {
        char *end = NULL;
        long n = strtol(name + 3, &end, 10);
        if (end && *end == '\0' && n >= 1 && n <= tbl->count)
            return (int)n;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Prompt expansion                                                    */
/* ------------------------------------------------------------------ */

/* Growable string. */
typedef struct {
    char *p;
    size_t len, cap;
} sbuf;

static int sb_put(sbuf *s, const char *str, size_t n) {
    if (s->len + n + 1 > s->cap) {
        size_t nc = s->cap ? s->cap * 2 : 256;
        while (nc < s->len + n + 1) nc *= 2;
        char *np = realloc(s->p, nc);
        if (!np) return 0;
        s->p = np;
        s->cap = nc;
    }
    memcpy(s->p + s->len, str, n);
    s->len += n;
    s->p[s->len] = '\0';
    return 1;
}

/* Detect whether the prompt contains at least one KNOWN alias token. Only
 * then is the prompt expanded (and the mapping header injected). Unknown
 * `@tokens` are left literal and never cause a failure. */
static int has_alias(const hd_ref_alias_table *tbl, const char *prompt) {
    for (const char *p = prompt; *p; p++) {
        if (*p != '@') continue;
        const char *q = p + 1;
        size_t n = 0;
        while (q[n] && (isalnum((unsigned char)q[n]) || q[n] == '_' ||
                        q[n] == '-'))
            n++;
        if (n == 0 || n >= 128) continue;
        char name[128];
        memcpy(name, q, n);
        name[n] = '\0';
        if (hd_ref_alias_lookup(tbl, name) > 0) return 1;
    }
    return 0;
}

int hd_ref_alias_expand(const hd_ref_alias_table *tbl, const char *prompt,
                        char **out_expanded) {
    if (out_expanded) *out_expanded = NULL;
    if (!tbl || !prompt) return HD_ERR_MISSING;
    g_err[0] = '\0';

    /* No alias -> leave the prompt byte-identical (compatibility). */
    if (!has_alias(tbl, prompt)) return HD_OK;

    sbuf body = {0};
    const char *p = prompt;
    while (*p) {
        if (*p != '@') {
            if (!sb_put(&body, p, 1)) { free(body.p); return HD_ERR_OOM; }
            p++;
            continue;
        }
        const char *q = p + 1;
        size_t n = 0;
        while (q[n] && (isalnum((unsigned char)q[n]) || q[n] == '_' ||
                        q[n] == '-'))
            n++;
        char name[128];
        if (n == 0 || n >= sizeof(name)) {
            /* lone '@' or overlong token: emit literally */
            if (!sb_put(&body, p, 1)) { free(body.p); return HD_ERR_OOM; }
            p++;
            continue;
        }
        memcpy(name, q, n);
        name[n] = '\0';
        int ref = hd_ref_alias_lookup(tbl, name);
        if (ref <= 0) {
            /* Unknown alias: leave the `@token` verbatim in the prompt. */
            if (!sb_put(&body, p, n + 1)) { free(body.p); return HD_ERR_OOM; }
            p = q + n;
            continue;
        }
        char phrase[160];
        int m = snprintf(phrase, sizeof(phrase),
                         "reference image %d (\"%s\")", ref, name);
        if (m < 0 || !sb_put(&body, phrase, (size_t)m)) {
            free(body.p);
            return HD_ERR_OOM;
        }
        p = q + n;
    }

    /* Header: name -> image mapping (explicit + automatic aliases). */
    sbuf out = {0};
    const char *hdr = "Reference image mapping:\n";
    if (!sb_put(&out, hdr, strlen(hdr))) { free(body.p); return HD_ERR_OOM; }
    for (int i = 0; i < tbl->count; i++) {
        const hd_ref_alias_entry *e = &tbl->entries[i];
        char line[192];
        int m;
        if (e->user_alias && e->user_alias[0])
            m = snprintf(line, sizeof(line), "- \"%s\" = reference image %d\n",
                         e->user_alias, i + 1);
        else
            m = snprintf(line, sizeof(line), "- \"ref%d\" = reference image %d\n",
                         i + 1, i + 1);
        if (m < 0 || !sb_put(&out, line, (size_t)m)) {
            free(body.p);
            free(out.p);
            return HD_ERR_OOM;
        }
    }
    const char *sep = "\nUser request:\n";
    if (!sb_put(&out, sep, strlen(sep)) ||
        !sb_put(&out, body.p ? body.p : "", body.len)) {
        free(body.p);
        free(out.p);
        return HD_ERR_OOM;
    }
    free(body.p);
    if (out_expanded) *out_expanded = out.p;
    else free(out.p);
    return HD_OK;
}

/* ------------------------------------------------------------------ */
/* Introspection                                                       */
/* ------------------------------------------------------------------ */

void hd_ref_alias_dump(const hd_ref_alias_table *tbl, void *f) {
    FILE *fp = f ? (FILE *)f : stderr;
    if (!tbl) return;
    fprintf(fp, "reference mapping:\n");
    for (int i = 0; i < tbl->count; i++) {
        const hd_ref_alias_entry *e = &tbl->entries[i];
        fprintf(fp, "  @ref%-3d -> ref[%d] %s%s\n", i + 1, i,
                e->path ? e->path : "(none)",
                e->is_internal ? " (internal)" : "");
        if (e->user_alias && e->user_alias[0])
            fprintf(fp, "  @%-7s -> ref[%d] %s\n", e->user_alias, i,
                    e->path ? e->path : "(none)");
    }
}
