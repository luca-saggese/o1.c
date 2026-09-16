/*
 * M1-post unified request ABI implementation (see request.h).
 */

#include "request.h"

#include <string.h>

const char *hd_mode_name(hd_mode m) {
    switch (m) {
    case HD_MODE_T2I: return "t2i";
    case HD_MODE_EDIT: return "edit";
    case HD_MODE_PERSONALIZE: return "personalize";
    case HD_MODE_PERSONALIZE_LAYOUT: return "personalize_layout";
    case HD_MODE_PERSONALIZE_SKELETON: return "personalize_skeleton";
    case HD_MODE_STORYBOARD: return "storyboard";
    }
    return "unknown";
}

hd_mode hd_mode_from_name(const char *s) {
    if (!s) return HD_MODE_T2I;
    if (!strcmp(s, "edit")) return HD_MODE_EDIT;
    if (!strcmp(s, "personalize") || !strcmp(s, "multi-ref") ||
        !strcmp(s, "multi_ref"))
        return HD_MODE_PERSONALIZE;
    if (!strcmp(s, "personalize_layout") || !strcmp(s, "layout"))
        return HD_MODE_PERSONALIZE_LAYOUT;
    if (!strcmp(s, "personalize_skeleton") || !strcmp(s, "skeleton"))
        return HD_MODE_PERSONALIZE_SKELETON;
    if (!strcmp(s, "storyboard")) return HD_MODE_STORYBOARD;
    return HD_MODE_T2I;
}

const char *hd_scheduler_name(hd_scheduler_kind k) {
    switch (k) {
    case HD_SCHED_DEFAULT: return "default";
    case HD_SCHED_FLASH: return "flash";
    case HD_SCHED_FLOW_MATCH: return "flow_match";
    }
    return "unknown";
}

hd_scheduler_kind hd_scheduler_from_name(const char *s) {
    if (!s) return HD_SCHED_DEFAULT;
    if (!strcmp(s, "flash")) return HD_SCHED_FLASH;
    if (!strcmp(s, "flow_match")) return HD_SCHED_FLOW_MATCH;
    return HD_SCHED_DEFAULT;
}

void hd_request_defaults(hd_generation_request *req) {
    if (!req) return;
    if (req->profile && !strcmp(req->profile, "base")) {
        if (req->steps <= 0) req->steps = 50;
        if (req->guidance_scale < 0.0f) req->guidance_scale = 5.0f;
        if (req->shift < 0.0f) req->shift = 3.0f;
        if (req->scheduler == HD_SCHED_DEFAULT) req->scheduler = HD_SCHED_DEFAULT;
    } else {
        if (req->steps <= 0) req->steps = 28;
        if (req->guidance_scale < 0.0f) req->guidance_scale = 0.0f;
        if (req->shift < 0.0f) req->shift = 1.0f;
        if (req->scheduler == HD_SCHED_DEFAULT) req->scheduler = HD_SCHED_FLASH;
    }
    if (req->noise_scale_start <= 0.0f) req->noise_scale_start = 8.0f;
    if (req->noise_scale_end <= 0.0f) req->noise_scale_end = 8.0f;
    if (req->noise_clip_std <= 0.0f) req->noise_clip_std = 8.0f;
}

hd_status hd_request_validate(const hd_generation_request *req) {
    if (!req) { hd_set_error("request: null request"); return HD_ERR_MISSING; }
    if (!req->prompt) { hd_set_error("request: null prompt"); return HD_ERR_MISSING; }
    if (!req->profile) { hd_set_error("request: null profile"); return HD_ERR_MISSING; }
    if (strcmp(req->profile, "dev") && strcmp(req->profile, "base")) {
        hd_set_error("request: unsupported profile '%s'", req->profile);
        return HD_ERR_PROFILE;
    }
    if (req->width <= 0 || req->height <= 0) {
        hd_set_error("request: invalid dimensions %dx%d", req->width, req->height);
        return HD_ERR_MISSING;
    }
    if (req->steps <= 0 || req->steps > 64) {
        hd_set_error("request: invalid steps %d", req->steps);
        return HD_ERR_MISSING;
    }
    if (req->mode == HD_MODE_EDIT && req->reference_count != 1) {
        hd_set_error("request: edit mode requires exactly 1 reference");
        return HD_ERR_MISSING;
    }
    if ((req->mode == HD_MODE_PERSONALIZE ||
         req->mode == HD_MODE_PERSONALIZE_LAYOUT ||
         req->mode == HD_MODE_PERSONALIZE_SKELETON) &&
        req->reference_count < 1) {
        hd_set_error("request: personalization requires >= 1 reference");
        return HD_ERR_MISSING;
    }
    if (req->mode == HD_MODE_PERSONALIZE_LAYOUT && !req->layout) {
        hd_set_error("request: layout mode requires layout condition");
        return HD_ERR_MISSING;
    }
    if (req->mode == HD_MODE_PERSONALIZE_SKELETON && !req->skeleton) {
        hd_set_error("request: skeleton mode requires skeleton condition");
        return HD_ERR_MISSING;
    }
    return HD_OK;
}