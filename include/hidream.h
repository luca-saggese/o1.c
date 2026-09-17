#ifndef HIDREAM_H
#define HIDREAM_H

/*
 * Public C ABI for the HiDream GB10 inference engine.
 *
 * M1.0 scope: profile/config/manifest loading and validation. No model math.
 * All functions return hd_status and set an error string retrievable via
 * hd_last_error(). CUDA/cuBLAS status handling arrives with the runtime layer
 * in later milestones; nothing in this header may silently ignore errors.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Status codes                                                        */
/* ------------------------------------------------------------------ */

typedef enum {
    HD_OK = 0,
    HD_ERR_IO = 1,        /* file read/stat failure */
    HD_ERR_PARSE = 2,     /* JSON/text parse failure */
    HD_ERR_PROFILE = 3,   /* unknown/invalid profile */
    HD_ERR_MANIFEST = 4,  /* tensor manifest structural problem */
    HD_ERR_MISMATCH = 5,  /* validation mismatch (counts, shapes, ...) */
    HD_ERR_MISSING = 6,   /* required local path absent (e.g. weights) */
    HD_ERR_OOM = 7,       /* allocation failure */
    HD_ERR_RUNTIME = 8,   /* runtime/backend failure (e.g. cuDNN SDPA) */
} hd_status;

const char *hd_last_error(void);

/* Sets the shared error string (used by sibling modules for hd_last_error). */
void hd_set_error(const char *fmt, ...);

/* ------------------------------------------------------------------ */
/* Dtypes                                                              */
/* ------------------------------------------------------------------ */

typedef enum {
    HD_DTYPE_F32 = 0,
    HD_DTYPE_BF16 = 1,
    HD_DTYPE_F16 = 2,
    HD_DTYPE_F64 = 3,
    HD_DTYPE_I64 = 4,
    HD_DTYPE_I32 = 5,
    HD_DTYPE_UNKNOWN = -1,
} hd_dtype;

hd_dtype hd_dtype_from_string(const char *s);   /* accepts torch.* aliases */
const char *hd_dtype_to_string(hd_dtype d);

int64_t hd_numel(const int64_t *shape, int rank);

/* ------------------------------------------------------------------ */
/* Profile                                                             */
/* ------------------------------------------------------------------ */

typedef struct {
    char *profile;              /* "dev" or "base" */
    char *hf_repo;
    char *immutable_revision;
    char *local_path;           /* relative to project root */
    char *model_type;           /* "dev" or "full" */
    char *dtype;                /* raw dtype string from config */
    char *model_type_hf;        /* e.g. "qwen3_vl" */
    int num_inference_steps;
} hd_profile;

hd_status hd_profile_load(const char *profile_name,
                          const char *config_dir,
                          hd_profile *out);
void hd_profile_free(hd_profile *p);

/* ------------------------------------------------------------------ */
/* Tensor manifest                                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    char *name;
    int64_t *shape;
    int rank;
    hd_dtype dtype;
    int64_t numel;
} hd_tensor_meta;

typedef struct {
    char *profile;
    char *oracle_sha;
    char *dtype;
    int64_t parameter_count;
    int64_t n_tensors;
    hd_tensor_meta *tensors;
} hd_tensor_manifest;

hd_status hd_tensor_manifest_load(const char *path, hd_tensor_manifest *out);
void hd_tensor_manifest_free(hd_tensor_manifest *m);

/* ------------------------------------------------------------------ */
/* Validation                                                          */
/* ------------------------------------------------------------------ */

typedef struct {
    int missing;       /* tensors in expected but not actual */
    int unexpected;    /* tensors in actual but not expected */
    int shape_mismatch;
    int dtype_mismatch;
    int numel_mismatch;
    int name_order_mismatch;
} hd_manifest_diff;

/*
 * Compare two tensor manifests. Both are expected to be name-ordered (the
 * frozen M0 manifest is). Populates *diff with mismatch counts.
 * Returns HD_OK when structurally identical, HD_ERR_MISMATCH otherwise.
 */
hd_status hd_tensor_manifest_compare(const hd_tensor_manifest *expected,
                                     const hd_tensor_manifest *actual,
                                     hd_manifest_diff *diff);

/*
 * Full M1.0 profile validation:
 *   - loads config/<profile_name>.json
 *   - loads config/tensor_manifest_<profile_name>.json and, for dev, compares
 *     it against the authoritative tensor manifest (759 tensors /
 *     8,804,887,792 params)
 *   - reports local weight path presence (missing weights are OK for base)
 * Returns HD_OK only when the profile is valid and dev's manifest matches.
 */
hd_status hd_validate_profile(const char *config_dir, const char *profile_name);

#ifdef __cplusplus
}
#endif

#endif /* HIDREAM_H */
