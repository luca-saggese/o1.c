/*
 * M1-post resident generation engine (see engine.h).
 *
 * Loads the device weights and resolves the forward/vision bindings once,
 * then serves any number of generations without reloading. The per-request
 * paths live in generate.c and consume the resident store read-only.
 */

#include "engine.h"
#include "engine_priv.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "forward.h"
#include "hd_lora.h"
#include "safetensors.h"
#include "vision.h"
#include "weights.h"

#define NLAYERS 36

static hd_status load_weights(const char *model_path, int device_id,
                              hd_weight_store *store) {
    size_t mdlen = strlen(model_path);
    int is_gguf = mdlen > 5 && strcmp(model_path + mdlen - 5, ".gguf") == 0;
    if (is_gguf) {
        hd_status st = hd_weights_to_device_gguf(model_path, device_id, store);
        if (st != HD_OK)
            hd_set_error("engine: weights: %s", hd_weights_last_error());
        return st;
    }
    hd_st_index idx;
    if (hd_st_index_load(model_path, &idx) != HD_OK) {
        hd_set_error("engine: index: %s", hd_st_last_error());
        return HD_ERR_IO;
    }
    hd_status st = hd_weights_to_device(model_path, &idx, device_id, store);
    hd_st_index_free(&idx);
    if (st != HD_OK)
        hd_set_error("engine: weights: %s", hd_weights_last_error());
    return st;
}

hd_generation_engine *hd_generation_engine_open(
    const hd_generation_engine_options *opts) {
    if (!opts || !opts->model_path || !opts->profile) {
        hd_set_error("engine: null options");
        return NULL;
    }
    if (strcmp(opts->profile, "dev") && strcmp(opts->profile, "base")) {
        hd_set_error("engine: unsupported profile '%s'", opts->profile);
        return NULL;
    }

    hd_generation_engine *e = calloc(1, sizeof(*e));
    if (!e) {
        hd_set_error("engine: oom");
        return NULL;
    }
    e->device_id = opts->device_id;

    if (load_weights(opts->model_path, opts->device_id, &e->store) != HD_OK)
        goto fail;
    e->load_count = 1;

    if (hd_forward_resolve(&e->store, NLAYERS, &e->bw) != HD_OK) {
        hd_set_error("engine: resolve: %s", hd_last_error());
        goto fail;
    }
    e->forward_resolve_count = 1;

    if (opts->lora && opts->lora->count > 0) {
        if (hd_lora_apply(opts->lora, &e->store, opts->device_id) != HD_OK) {
            hd_set_error("engine: lora: %s", hd_lora_last_error());
            goto fail;
        }
    }

    /* The vision tower is only needed by reference-bearing modes. Resolve it
     * eagerly so the first edit/personalize request does not pay the cost;
     * a missing tower is not fatal for pure T2I. */
    if (hd_vision_resolve(&e->store, &e->vb) == HD_OK) {
        e->has_vision = 1;
        e->vision_resolve_count = 1;
    } else {
        memset(&e->vb, 0, sizeof(e->vb));
    }

    return e;

fail:
    hd_generation_engine_close(e);
    return NULL;
}

hd_status hd_generation_engine_generate(hd_generation_engine *e,
                                        const hd_generation_request *req,
                                        unsigned char **out_rgb, int *out_w,
                                        int *out_h) {
    if (!e || !req || !out_rgb || !out_w || !out_h) {
        hd_set_error("engine: null argument");
        return HD_ERR_MISSING;
    }
    *out_rgb = NULL;
    *out_w = *out_h = 0;

    hd_status st;
    if (req->reference_count > 0) {
        if (!e->has_vision) {
            hd_set_error("engine: vision tower unavailable for reference mode");
            return HD_ERR_MISSING;
        }
        st = hd_engine_generate_ref(e, req, out_rgb, out_w, out_h);
    } else {
        st = hd_engine_generate_t2i(e, req, out_rgb, out_w, out_h);
    }
    if (st == HD_OK) e->request_count++;
    return st;
}

void hd_generation_engine_close(hd_generation_engine *e) {
    if (!e) return;
    /* The vision binding only holds pointers into the resident store; the
     * store free below releases them. */
    hd_forward_binding_free(&e->bw);
    hd_weight_store_free(&e->store);
    free(e);
}

int hd_generation_engine_load_count(const hd_generation_engine *e) {
    return e ? e->load_count : 0;
}

int hd_generation_engine_forward_resolve_count(const hd_generation_engine *e) {
    return e ? e->forward_resolve_count : 0;
}

int hd_generation_engine_vision_resolve_count(const hd_generation_engine *e) {
    return e ? e->vision_resolve_count : 0;
}

int hd_generation_engine_request_count(const hd_generation_engine *e) {
    return e ? e->request_count : 0;
}