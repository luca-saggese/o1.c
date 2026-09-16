#ifndef HD_REQUEST_H
#define HD_REQUEST_H

/*
 * M1-post unified generation request ABI.
 *
 * One request representation for every HiDream mode (M1_POST_FULL_FEATURE_PARITY
 * section 7). The request is mode-agnostic at the transformer level: the
 * unified sequence builder (src/runtime/sequence.c) lowers any request into
 * the same token/position/mask/vinput contract consumed by hd_forward.
 *
 * No unordered containers, no per-mode inference engines.
 */

#include <stddef.h>
#include <stdint.h>

#include "hidream.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Modes                                                               */
/* ------------------------------------------------------------------ */

typedef enum {
    HD_MODE_T2I = 0,            /* text -> image (no references) */
    HD_MODE_EDIT,               /* single reference image editing */
    HD_MODE_PERSONALIZE,        /* multi-reference subject personalization */
    HD_MODE_PERSONALIZE_LAYOUT, /* personalization + layout boxes */
    HD_MODE_PERSONALIZE_SKELETON, /* personalization + skeleton/openpose */
    HD_MODE_STORYBOARD,         /* multi-panel storyboard (decomposed) */
} hd_mode;

const char *hd_mode_name(hd_mode m);
hd_mode hd_mode_from_name(const char *s); /* returns HD_MODE_T2I if unknown */

/* ------------------------------------------------------------------ */
/* Schedulers                                                          */
/* ------------------------------------------------------------------ */

typedef enum {
    HD_SCHED_DEFAULT = 0,   /* FlowUniPCMultistepScheduler (Full) */
    HD_SCHED_FLASH,         /* FlashFlowMatchEulerDiscreteScheduler (Dev) */
    HD_SCHED_FLOW_MATCH,    /* FlowMatchEulerDiscreteScheduler (Dev edit) */
} hd_scheduler_kind;

const char *hd_scheduler_name(hd_scheduler_kind k);
hd_scheduler_kind hd_scheduler_from_name(const char *s);

/* ------------------------------------------------------------------ */
/* Conditioning inputs                                                 */
/* ------------------------------------------------------------------ */

/* Reference image role (typed reference roles, contract section 7). */
typedef enum {
    HD_REF_SUBJECT = 0,   /* subject/identity reference */
    HD_REF_LAYOUT,        /* layout-conditioned reference */
    HD_REF_SKELETON,      /* skeleton/openpose reference */
} hd_reference_role;

typedef struct {
    const char *path;     /* image path (native decode) */
    hd_reference_role role;
} hd_reference_image;

/* Layout bbox in relative xxyy coordinates [0,1] (oracle parse_layout_bboxes). */
typedef struct {
    float x1, y1, x2, y2;
    const char *text;     /* optional label */
} hd_layout_condition;

/* Skeleton/openpose metadata (JSON string, oracle load_layout_bboxes style). */
typedef struct {
    const char *json;
} hd_skeleton_condition;

/* ------------------------------------------------------------------ */
/* Progress                                                            */
/* ------------------------------------------------------------------ */

typedef void (*hd_progress_callback)(int step, int total, void *user);

/* ------------------------------------------------------------------ */
/* Unified request                                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *prompt;           /* user prompt (may be " " for uncond) */
    hd_mode mode;
    const char *profile;          /* "dev" | "base" */

    int width;                    /* requested output width  (px) */
    int height;                   /* requested output height (px) */
    uint64_t seed;

    int steps;                    /* num_inference_steps */
    float guidance_scale;         /* CFG scale (0.0 = no CFG) */
    float shift;                  /* scheduler shift */
    hd_scheduler_kind scheduler;

    const hd_reference_image *references;
    size_t reference_count;

    const hd_layout_condition *layout;   /* may be NULL */
    const hd_skeleton_condition *skeleton; /* may be NULL */

    int keep_original_aspect;     /* 1 ref: derive dims from ref aspect */

    float noise_scale_start;      /* default 8.0 */
    float noise_scale_end;        /* default 8.0 */
    float noise_clip_std;         /* default 8.0 */

    hd_progress_callback progress_cb;
    void *progress_user;
} hd_generation_request;

/*
 * Fill profile defaults per the frozen recipe (docs/M1_POST_SCHEDULER_MATRIX):
 *   dev:  steps=28, guidance=0.0, shift=1.0, scheduler=flash
 *   base: steps=50, guidance=5.0, shift=3.0, scheduler=default
 * noise_scale_start=end=8.0, noise_clip_std=8.0.
 * Does not touch prompt/mode/dimensions/seed/references.
 */
void hd_request_defaults(hd_generation_request *req);

/* Validate a request fails closed (mode/profile/dims/steps/references). */
hd_status hd_request_validate(const hd_generation_request *req);

#ifdef __cplusplus
}
#endif

#endif /* HD_REQUEST_H */