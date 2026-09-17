#include "safetensors.h"
#include "json.h"
#include "o1_timing.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char g_st_error[512] = "";

static void st_err(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_st_error, sizeof(g_st_error), fmt, ap);
    va_end(ap);
}

const char *hd_st_last_error(void) { return g_st_error; }

static char *st_strdup(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s) + 1;
    char *p = malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

static hd_dtype st_dtype(const char *s) {
    if (!s) return HD_DTYPE_UNKNOWN;
    if (strcmp(s, "F32") == 0) return HD_DTYPE_F32;
    if (strcmp(s, "BF16") == 0) return HD_DTYPE_BF16;
    if (strcmp(s, "F16") == 0) return HD_DTYPE_F16;
    if (strcmp(s, "F64") == 0) return HD_DTYPE_F64;
    if (strcmp(s, "I64") == 0) return HD_DTYPE_I64;
    if (strcmp(s, "I32") == 0) return HD_DTYPE_I32;
    return HD_DTYPE_UNKNOWN;
}

static int64_t st_dtype_size(hd_dtype d) {
    switch (d) {
        case HD_DTYPE_F32: return 4;
        case HD_DTYPE_BF16: return 2;
        case HD_DTYPE_F16: return 2;
        case HD_DTYPE_F64: return 8;
        case HD_DTYPE_I64: return 8;
        case HD_DTYPE_I32: return 4;
        default: return -1;
    }
}

static char *st_read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { st_err("cannot open %s", path); return NULL; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); st_err("seek %s", path); return NULL; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); st_err("ftell %s", path); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); st_err("seek %s", path); return NULL; }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); st_err("oom %s", path); return NULL; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) { free(buf); st_err("short read %s", path); return NULL; }
    buf[sz] = '\0';
    if (out_len) *out_len = (size_t)sz;
    return buf;
}

static int st_cmp(const void *a, const void *b) {
    const hd_st_tensor *x = a, *y = b;
    return strcmp(x->name, y->name);
}

/* Parses one shard header and appends its tensors to out. */
static hd_status st_parse_shard(const char *dir, const char *shard,
                                hd_st_index *out, int64_t *cap) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/%s", dir, shard);

    FILE *f = fopen(path, "rb");
    if (!f) { st_err("cannot open shard %s", path); return HD_ERR_IO; }

    uint8_t lenbuf[8];
    if (fread(lenbuf, 1, 8, f) != 8) { fclose(f); st_err("short header len %s", path); return HD_ERR_IO; }
    uint64_t hlen = 0;
    for (int i = 0; i < 8; i++) hlen |= (uint64_t)lenbuf[i] << (i * 8);
    if (hlen == 0 || hlen > (uint64_t)1 << 30) {
        fclose(f); st_err("implausible header length %llu in %s", (unsigned long long)hlen, path);
        return HD_ERR_PARSE;
    }

    char *hdr = malloc((size_t)hlen + 1);
    if (!hdr) { fclose(f); st_err("oom header %s", path); return HD_ERR_OOM; }
    if (fread(hdr, 1, (size_t)hlen, f) != (size_t)hlen) {
        free(hdr); fclose(f); st_err("short header %s", path); return HD_ERR_IO;
    }
    hdr[hlen] = '\0';
    int64_t payload_base = 8 + (int64_t)hlen;
    fclose(f);

    const char *jerr = NULL;
    hd_json *root = hd_json_parse(hdr, &jerr);
    free(hdr);
    if (!root) { st_err("parse header %s: %s", path, jerr ? jerr : "?"); return HD_ERR_PARSE; }

    hd_status st = HD_OK;
    for (size_t i = 0; i < root->u.object.count; i++) {
        const char *name = root->u.object.keys[i];
        if (strcmp(name, "__metadata__") == 0) continue;
        const hd_json *t = root->u.object.values[i];

        const char *dt = hd_json_string(hd_json_get(t, "dtype"));
        const hd_json *shape = hd_json_get(t, "shape");
        const hd_json *offs = hd_json_get(t, "data_offsets");
        if (!dt || !shape || shape->type != HD_JSON_ARRAY ||
            !offs || offs->type != HD_JSON_ARRAY || offs->u.array.count != 2) {
            st_err("shard %s tensor %s malformed", shard, name);
            st = HD_ERR_PARSE;
            break;
        }

        if (out->n_tensors == *cap) {
            int64_t ncap = *cap ? *cap * 2 : 1024;
            hd_st_tensor *nt = realloc(out->tensors, (size_t)ncap * sizeof(*nt));
            if (!nt) { st_err("oom tensor table"); st = HD_ERR_OOM; break; }
            out->tensors = nt;
            *cap = ncap;
        }

        hd_st_tensor *e = &out->tensors[out->n_tensors];
        memset(e, 0, sizeof(*e));
        e->name = st_strdup(name);
        e->shard = st_strdup(shard);
        e->dtype = st_dtype(dt);
        e->rank = (int)shape->u.array.count;
        e->shape = calloc((size_t)(e->rank > 0 ? e->rank : 1), sizeof(int64_t));
        if (!e->name || !e->shard || !e->shape) { st_err("oom tensor meta"); st = HD_ERR_OOM; break; }
        for (size_t d = 0; d < shape->u.array.count; d++) {
            e->shape[d] = hd_json_int(shape->u.array.items[d], 0);
        }
        e->numel = hd_numel(e->shape, e->rank);
        int64_t begin = hd_json_int(offs->u.array.items[0], -1);
        int64_t end = hd_json_int(offs->u.array.items[1], -1);
        if (begin < 0 || end < begin) { st_err("bad offsets for %s", name); st = HD_ERR_PARSE; break; }
        e->data_begin = payload_base + begin;
        e->nbytes = end - begin;

        int64_t expect = e->numel * st_dtype_size(e->dtype);
        if (e->dtype == HD_DTYPE_UNKNOWN || expect != e->nbytes) {
            st_err("tensor %s: dtype %s numel %lld -> %lld bytes but header says %lld",
                   name, dt, (long long)e->numel, (long long)expect, (long long)e->nbytes);
            st = HD_ERR_PARSE;
            break;
        }
        out->total_bytes += e->nbytes;
        out->n_tensors++;
    }

    hd_json_free(root);
    return st;
}

hd_status hd_st_index_load(const char *dir, hd_st_index *out) {
    O1_TIMING_BEGIN("METADATA_PARSE");
    memset(out, 0, sizeof(*out));
    out->dir = st_strdup(dir);

    char path[2048];
    snprintf(path, sizeof(path), "%s/model.safetensors.index.json", dir);
    out->index_path = st_strdup(path);

    size_t len = 0;
    char *text = st_read_file(path, &len);
    if (!text) return HD_ERR_IO;

    const char *jerr = NULL;
    hd_json *root = hd_json_parse(text, &jerr);
    free(text);
    if (!root) { st_err("parse %s: %s", path, jerr ? jerr : "?"); return HD_ERR_PARSE; }

    const hd_json *wm = hd_json_get(root, "weight_map");
    if (!wm || wm->type != HD_JSON_OBJECT) {
        st_err("%s: missing weight_map", path);
        hd_json_free(root);
        return HD_ERR_PARSE;
    }

    /* Collect the distinct shard names referenced by the weight map. */
    char **shards = NULL;
    size_t n_shards = 0, cap_shards = 0;
    for (size_t i = 0; i < wm->u.object.count; i++) {
        const char *s = hd_json_string(wm->u.object.values[i]);
        if (!s) continue;
        int seen = 0;
        for (size_t k = 0; k < n_shards; k++) {
            if (strcmp(shards[k], s) == 0) { seen = 1; break; }
        }
        if (seen) continue;
        if (n_shards == cap_shards) {
            size_t nc = cap_shards ? cap_shards * 2 : 8;
            char **ns = realloc(shards, nc * sizeof(*ns));
            if (!ns) { st_err("oom shard list"); hd_json_free(root); free(shards); return HD_ERR_OOM; }
            shards = ns; cap_shards = nc;
        }
        shards[n_shards++] = st_strdup(s);
    }
    hd_json_free(root);

    int64_t cap = 0;
    hd_status st = HD_OK;
    for (size_t i = 0; i < n_shards && st == HD_OK; i++) {
        st = st_parse_shard(dir, shards[i], out, &cap);
    }
    for (size_t i = 0; i < n_shards; i++) free(shards[i]);
    free(shards);

    if (st != HD_OK) { hd_st_index_free(out); return st; }

    qsort(out->tensors, (size_t)out->n_tensors, sizeof(*out->tensors), st_cmp);
    O1_TIMING_END("METADATA_PARSE");
    return HD_OK;
}

void hd_st_index_free(hd_st_index *idx) {
    if (!idx) return;
    free(idx->dir);
    free(idx->index_path);
    for (int64_t i = 0; i < idx->n_tensors; i++) {
        free(idx->tensors[i].name);
        free(idx->tensors[i].shard);
        free(idx->tensors[i].shape);
    }
    free(idx->tensors);
    memset(idx, 0, sizeof(*idx));
}

const hd_st_tensor *hd_st_index_find(const hd_st_index *idx, const char *name) {
    int64_t lo = 0, hi = idx->n_tensors - 1;
    while (lo <= hi) {
        int64_t mid = lo + (hi - lo) / 2;
        int c = strcmp(idx->tensors[mid].name, name);
        if (c == 0) return &idx->tensors[mid];
        if (c < 0) lo = mid + 1;
        else hi = mid - 1;
    }
    return NULL;
}

hd_status hd_st_read(const char *dir, const char *shard, int64_t offset,
                     int64_t nbytes, void *dst) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/%s", dir, shard);
    FILE *f = fopen(path, "rb");
    if (!f) { st_err("cannot open shard %s", path); return HD_ERR_IO; }
    if (fseek(f, (long)offset, SEEK_SET) != 0) { fclose(f); st_err("seek %s", path); return HD_ERR_IO; }
    size_t rd = fread(dst, 1, (size_t)nbytes, f);
    fclose(f);
    if (rd != (size_t)nbytes) { st_err("short read %s (%zu/%lld)", path, rd, (long long)nbytes); return HD_ERR_IO; }
    return HD_OK;
}

hd_status hd_st_read_tensor(const hd_st_index *idx, const hd_st_tensor *t, void *dst) {
    return hd_st_read(idx->dir, t->shard, t->data_begin, t->nbytes, dst);
}

hd_status hd_st_header_scan(const char *path, hd_st_index *out) {
    memset(out, 0, sizeof(*out));
    const char *slash = strrchr(path, '/');
    char dir[2048];
    if (slash) {
        size_t n = (size_t)(slash - path);
        if (n >= sizeof(dir)) n = sizeof(dir) - 1;
        memcpy(dir, path, n);
        dir[n] = '\0';
    } else {
        strcpy(dir, ".");
    }
    out->dir = st_strdup(dir);
    int64_t cap = 0;
    return st_parse_shard(dir, slash ? slash + 1 : path, out, &cap);
}