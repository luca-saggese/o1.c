#include "hidream.h"
#include "json.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static char g_error[512] = "";

const char *hd_last_error(void) { return g_error; }

void hd_set_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error, sizeof(g_error), fmt, ap);
    va_end(ap);
}

static void set_err(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error, sizeof(g_error), fmt, ap);
    va_end(ap);
}

static char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { set_err("cannot open %s", path); return NULL; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); set_err("seek failed on %s", path); return NULL; }
    long sz = ftell(f);
    if (sz < 0) { fclose(f); set_err("ftell failed on %s", path); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); set_err("seek failed on %s", path); return NULL; }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); set_err("oom reading %s", path); return NULL; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) { free(buf); set_err("short read on %s", path); return NULL; }
    buf[sz] = '\0';
    if (out_len) *out_len = (size_t)sz;
    return buf;
}

static char *str_dup(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s) + 1;
    char *p = malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

hd_dtype hd_dtype_from_string(const char *s) {
    if (!s) return HD_DTYPE_UNKNOWN;
    if (strncmp(s, "torch.", 6) == 0) s += 6;
    if (strcmp(s, "float32") == 0 || strcmp(s, "fp32") == 0) return HD_DTYPE_F32;
    if (strcmp(s, "bfloat16") == 0 || strcmp(s, "bf16") == 0) return HD_DTYPE_BF16;
    if (strcmp(s, "float16") == 0 || strcmp(s, "fp16") == 0 || strcmp(s, "half") == 0) return HD_DTYPE_F16;
    if (strcmp(s, "float64") == 0 || strcmp(s, "fp64") == 0 || strcmp(s, "double") == 0) return HD_DTYPE_F64;
    if (strcmp(s, "int64") == 0) return HD_DTYPE_I64;
    if (strcmp(s, "int32") == 0) return HD_DTYPE_I32;
    return HD_DTYPE_UNKNOWN;
}

const char *hd_dtype_to_string(hd_dtype d) {
    switch (d) {
        case HD_DTYPE_F32: return "float32";
        case HD_DTYPE_BF16: return "bfloat16";
        case HD_DTYPE_F16: return "float16";
        case HD_DTYPE_F64: return "float64";
        case HD_DTYPE_I64: return "int64";
        case HD_DTYPE_I32: return "int32";
        default: return "unknown";
    }
}

int64_t hd_numel(const int64_t *shape, int rank) {
    int64_t n = 1;
    for (int i = 0; i < rank; i++) {
        if (shape[i] < 0) return -1;
        n *= shape[i];
    }
    return n;
}

/* ------------------------------------------------------------------ */
/* Profile                                                             */
/* ------------------------------------------------------------------ */

hd_status hd_profile_load(const char *profile_name, const char *config_dir,
                          hd_profile *out) {
    memset(out, 0, sizeof(*out));

    if (!profile_name || !*profile_name ||
        strchr(profile_name, '/') || strchr(profile_name, '\\') ||
        strcmp(profile_name, ".") == 0 || strcmp(profile_name, "..") == 0) {
        set_err("invalid profile name '%s'", profile_name ? profile_name : "");
        return HD_ERR_PROFILE;
    }

    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.json", config_dir, profile_name);
    size_t len = 0;
    char *text = read_file(path, &len);
    if (!text) return HD_ERR_IO;

    const char *jerr = NULL;
    hd_json *root = hd_json_parse(text, &jerr);
    free(text);
    if (!root) { set_err("parse %s: %s", path, jerr ? jerr : "?"); return HD_ERR_PARSE; }

    const char *p = hd_json_string(hd_json_get(root, "profile"));
    const char *repo = hd_json_string(hd_json_get(root, "hf_repo"));
    const char *rev = hd_json_string(hd_json_get(root, "immutable_revision"));
    const char *lp = hd_json_string(hd_json_get(root, "local_path"));
    const char *mt = hd_json_string(hd_json_get(root, "model_type"));
    const char *dt = hd_json_string(hd_json_get(root, "dtype"));
    const char *mth = hd_json_string(hd_json_get(root, "model_type_hf"));
    int64_t steps = hd_json_int(hd_json_get(root, "num_inference_steps"), -1);

    hd_status st = HD_OK;
    if (!p || strcmp(p, profile_name) != 0) {
        set_err("%s: profile field '%s' != requested '%s'", path, p ? p : "?", profile_name);
        st = HD_ERR_PROFILE;
    } else if (!repo || !rev || !lp || !mt) {
        set_err("%s: missing required field (hf_repo/immutable_revision/local_path/model_type)", path);
        st = HD_ERR_PROFILE;
    } else if (strlen(rev) < 40 || steps < 0) {
        set_err("%s: immutable_revision must be a full SHA and steps >= 0", path);
        st = HD_ERR_PROFILE;
    }

    if (st == HD_OK) {
        out->profile = str_dup(p);
        out->hf_repo = str_dup(repo);
        out->immutable_revision = str_dup(rev);
        out->local_path = str_dup(lp);
        out->model_type = str_dup(mt);
        out->dtype = str_dup(dt ? dt : "");
        out->model_type_hf = str_dup(mth ? mth : "");
        out->num_inference_steps = (int)steps;
    }
    hd_json_free(root);
    return st;
}

void hd_profile_free(hd_profile *p) {
    if (!p) return;
    free(p->profile); free(p->hf_repo); free(p->immutable_revision);
    free(p->local_path); free(p->model_type); free(p->dtype);
    free(p->model_type_hf);
    memset(p, 0, sizeof(*p));
}

/* ------------------------------------------------------------------ */
/* Tensor manifest                                                     */
/* ------------------------------------------------------------------ */

hd_status hd_tensor_manifest_load(const char *path, hd_tensor_manifest *out) {
    memset(out, 0, sizeof(*out));
    size_t len = 0;
    char *text = read_file(path, &len);
    if (!text) return HD_ERR_IO;

    const char *jerr = NULL;
    hd_json *root = hd_json_parse(text, &jerr);
    free(text);
    if (!root) { set_err("parse %s: %s", path, jerr ? jerr : "?"); return HD_ERR_PARSE; }

    hd_status st = HD_OK;
    out->profile = str_dup(hd_json_string(hd_json_get(root, "profile")));
    out->oracle_sha = str_dup(hd_json_string(hd_json_get(root, "oracle_sha")));
    out->dtype = str_dup(hd_json_string(hd_json_get(root, "dtype")));
    out->parameter_count = hd_json_int(hd_json_get(root, "parameter_count"), -1);

    const hd_json *arr = hd_json_get(root, "tensors");
    if (!arr || arr->type != HD_JSON_ARRAY) {
        set_err("%s: missing 'tensors' array", path);
        hd_json_free(root);
        return HD_ERR_MANIFEST;
    }
    out->n_tensors = (int64_t)arr->u.array.count;
    out->tensors = calloc((size_t)out->n_tensors, sizeof(hd_tensor_meta));
    if (!out->tensors) { set_err("oom"); hd_json_free(root); return HD_ERR_OOM; }

    for (size_t i = 0; i < arr->u.array.count; i++) {
        const hd_json *t = arr->u.array.items[i];
        const char *name = hd_json_string(hd_json_get(t, "name"));
        const char *dtype = hd_json_string(hd_json_get(t, "dtype"));
        const hd_json *shape = hd_json_get(t, "shape");
        int64_t numel = hd_json_int(hd_json_get(t, "numel"), -1);

        if (!name || !dtype || !shape || shape->type != HD_JSON_ARRAY) {
            set_err("%s: tensor %zu missing name/dtype/shape", path, i);
            st = HD_ERR_MANIFEST;
            break;
        }
        hd_tensor_meta *m = &out->tensors[i];
        m->name = str_dup(name);
        m->dtype = hd_dtype_from_string(dtype);
        m->rank = (int)shape->u.array.count;
        m->shape = calloc((size_t)(m->rank > 0 ? m->rank : 1), sizeof(int64_t));
        if (!m->shape) { set_err("oom"); st = HD_ERR_OOM; break; }
        for (size_t d = 0; d < shape->u.array.count; d++) {
            m->shape[d] = hd_json_int(shape->u.array.items[d], 0);
        }
        m->numel = numel;
        int64_t calc = hd_numel(m->shape, m->rank);
        if (numel != calc) {
            set_err("%s: tensor %s numel %lld != prod(shape) %lld", path, name,
                    (long long)numel, (long long)calc);
            st = HD_ERR_MANIFEST;
            break;
        }
        if (m->dtype == HD_DTYPE_UNKNOWN) {
            set_err("%s: tensor %s has unknown dtype '%s'", path, name, dtype);
            st = HD_ERR_MANIFEST;
            break;
        }
    }

    hd_json_free(root);
    if (st != HD_OK) { hd_tensor_manifest_free(out); return st; }
    return HD_OK;
}

void hd_tensor_manifest_free(hd_tensor_manifest *m) {
    if (!m) return;
    free(m->profile); free(m->oracle_sha); free(m->dtype);
    for (int64_t i = 0; i < m->n_tensors; i++) {
        free(m->tensors[i].name);
        free(m->tensors[i].shape);
    }
    free(m->tensors);
    memset(m, 0, sizeof(*m));
}

/* ------------------------------------------------------------------ */
/* Validation                                                          */
/* ------------------------------------------------------------------ */

hd_status hd_tensor_manifest_compare(const hd_tensor_manifest *expected,
                                     const hd_tensor_manifest *actual,
                                     hd_manifest_diff *diff) {
    if (diff) memset(diff, 0, sizeof(*diff));
    if (!expected || !actual) { set_err("null manifest in compare"); return HD_ERR_MANIFEST; }

    int64_t i = 0, j = 0;
    while (i < expected->n_tensors || j < actual->n_tensors) {
        if (i >= expected->n_tensors) { if (diff) diff->unexpected++; j++; continue; }
        if (j >= actual->n_tensors) { if (diff) diff->missing++; i++; continue; }
        int cmp = strcmp(expected->tensors[i].name, actual->tensors[j].name);
        if (cmp == 0) {
            const hd_tensor_meta *e = &expected->tensors[i];
            const hd_tensor_meta *a = &actual->tensors[j];
            if (e->rank != a->rank) {
                if (diff) diff->shape_mismatch++;
                else { set_err("rank mismatch: %s", e->name); return HD_ERR_MISMATCH; }
            } else {
                for (int d = 0; d < e->rank; d++) {
                    if (e->shape[d] != a->shape[d]) {
                        if (diff) diff->shape_mismatch++;
                        else { set_err("shape mismatch: %s", e->name); return HD_ERR_MISMATCH; }
                        break;
                    }
                }
            }
            if (e->dtype != a->dtype) {
                if (diff) diff->dtype_mismatch++;
                else { set_err("dtype mismatch: %s", e->name); return HD_ERR_MISMATCH; }
            }
            if (e->numel != a->numel) {
                if (diff) diff->numel_mismatch++;
                else { set_err("numel mismatch: %s", e->name); return HD_ERR_MISMATCH; }
            }
            i++; j++;
        } else if (cmp < 0) {
            if (diff) diff->missing++;
            else { set_err("missing tensor: %s", expected->tensors[i].name); return HD_ERR_MISMATCH; }
            i++;
        } else {
            if (diff) diff->unexpected++;
            else { set_err("unexpected tensor: %s", actual->tensors[j].name); return HD_ERR_MISMATCH; }
            j++;
        }
    }
    if (!diff) return HD_OK;
    if (diff->missing || diff->unexpected || diff->shape_mismatch ||
        diff->dtype_mismatch || diff->numel_mismatch) {
        return HD_ERR_MISMATCH;
    }
    return HD_OK;
}

static int dir_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

hd_status hd_validate_profile(const char *config_dir, const char *profile_name) {
    if (strcmp(profile_name, "dev") != 0 && strcmp(profile_name, "base") != 0) {
        set_err("unknown profile '%s' (expected dev or base)", profile_name);
        return HD_ERR_PROFILE;
    }

    hd_profile p;
    hd_status st = hd_profile_load(profile_name, config_dir, &p);
    if (st != HD_OK) return st;

    if (strcmp(profile_name, "dev") == 0) {
        hd_tensor_manifest authoritative, prof_manifest;
        st = hd_tensor_manifest_load("config/tensor_manifest_dev.json", &authoritative);
        if (st != HD_OK) { hd_profile_free(&p); return st; }
        char path[1024];
        snprintf(path, sizeof(path), "%s/tensor_manifest_dev.json", config_dir);
        st = hd_tensor_manifest_load(path, &prof_manifest);
        if (st != HD_OK) { hd_tensor_manifest_free(&authoritative); hd_profile_free(&p); return st; }

        hd_manifest_diff diff;
        st = hd_tensor_manifest_compare(&authoritative, &prof_manifest, &diff);
        if (st != HD_OK) {
            set_err("dev manifest mismatch: %d missing, %d unexpected, "
                    "%d shape, %d dtype, %d numel",
                    diff.missing, diff.unexpected, diff.shape_mismatch,
                    diff.dtype_mismatch, diff.numel_mismatch);
            hd_tensor_manifest_free(&authoritative);
            hd_tensor_manifest_free(&prof_manifest);
            hd_profile_free(&p);
            return HD_ERR_MISMATCH;
        }
        printf("[hd] dev tensors: %lld/%lld exact (name/rank/shape/dtype/numel), "
               "param_count=%lld\n",
               (long long)authoritative.n_tensors, (long long)prof_manifest.n_tensors,
               (long long)authoritative.parameter_count);
        hd_tensor_manifest_free(&authoritative);
        hd_tensor_manifest_free(&prof_manifest);
    }

    int weights_present = dir_exists(p.local_path);
    if (strcmp(profile_name, "dev") == 0 && !weights_present) {
        set_err("dev profile requires local weights at '%s' but path is absent",
                p.local_path);
        hd_profile_free(&p);
        return HD_ERR_MISSING;
    }

    printf("[hd] profile=%s repo=%s revision=%s local=%s (%s) steps=%d\n",
           p.profile, p.hf_repo, p.immutable_revision, p.local_path,
           weights_present ? "present" : "not_downloaded",
           p.num_inference_steps);
    hd_profile_free(&p);
    return HD_OK;
}
