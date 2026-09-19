#ifndef HD_ENGINE_H
#define HD_ENGINE_H

/*
 * M1-post resident generation engine.
 *
 * The model weights are the expensive, shape-independent part of a
 * generation: loading them from disk and resolving the forward/vision
 * bindings costs seconds and tens of GB of device memory. The engine
 * loads them ONCE and keeps them resident, so a server (or any caller
 * issuing several generations) never reloads the model per request.
 *
 * Per-request state (workspaces, SDPA plans, staging buffers) stays
 * per-request: it is shape-dependent and cheap relative to the weights.
 *
 * hd_generate() remains the one-shot compatibility wrapper
 * (open -> generate -> close) used by the CLI.
 */

#include <stddef.h>
#include <stdint.h>

#include "hidream.h"
#include "request.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque resident engine. */
typedef struct hd_generation_engine hd_generation_engine;

typedef struct {
    const char *profile;    /* "dev" | "base" (required) */
    const char *config_dir; /* reserved; may be NULL */
    const char *model_path; /* weights dir or .gguf (required) */
    int device_id;          /* CUDA device index */
    const hd_lora_config *lora; /* startup-global adapters; may be NULL */
} hd_generation_engine_options;

/*
 * Load the weights and resolve the forward/vision bindings once. The
 * returned engine owns the resident weight store; it must be released
 * with hd_generation_engine_close(). Returns NULL on failure (see
 * hd_last_error()).
 */
hd_generation_engine *hd_generation_engine_open(
    const hd_generation_engine_options *opts);

/*
 * Run one generation against the resident weights. Same contract as
 * hd_generate(): out_rgb is malloc'd and owned by the caller.
 */
hd_status hd_generation_engine_generate(hd_generation_engine *engine,
                                        const hd_generation_request *req,
                                        unsigned char **out_rgb, int *out_w,
                                        int *out_h);

/* Release the resident weights and bindings. NULL is a no-op. */
void hd_generation_engine_close(hd_generation_engine *engine);

/* Diagnostics: number of weight loads performed by this engine (always 1
 * for a live engine). Used by tests to prove the model is not reloaded. */
int hd_generation_engine_load_count(const hd_generation_engine *engine);

/* Diagnostics for the preload lifecycle release gate: the forward and
 * vision bindings must be resolved exactly once, and the request counter
 * must advance once per successful generation. */
int hd_generation_engine_forward_resolve_count(const hd_generation_engine *engine);
int hd_generation_engine_vision_resolve_count(const hd_generation_engine *engine);
int hd_generation_engine_request_count(const hd_generation_engine *engine);

#ifdef __cplusplus
}
#endif

#endif /* HD_ENGINE_H */