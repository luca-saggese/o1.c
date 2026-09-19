#ifndef HD_ENGINE_PRIV_H
#define HD_ENGINE_PRIV_H

/*
 * Private layout of the resident generation engine. Shared between
 * engine.c (owner) and generate.c (the per-request paths that consume
 * the resident weights). Not part of the public ABI.
 */

#include "engine.h"
#include "forward.h"
#include "vision.h"
#include "weights.h"

struct hd_generation_engine {
    hd_weight_store store;      /* resident device weights (arena) */
    hd_forward_binding bw;      /* resolved once */
    hd_vision_binding vb;       /* resolved once (has_vision) */
    int has_vision;
    int device_id;
    int load_count;             /* weight loads performed (== 1) */
    int forward_resolve_count;  /* forward binding resolves (== 1) */
    int vision_resolve_count;   /* vision binding resolves (== 1) */
    int request_count;          /* generations served by this engine */
};

/* Per-request paths, implemented in generate.c. They consume the resident
 * weights and never free them. */
hd_status hd_engine_generate_ref(hd_generation_engine *e,
                                 const hd_generation_request *req,
                                 unsigned char **out_rgb, int *out_w,
                                 int *out_h);
hd_status hd_engine_generate_t2i(hd_generation_engine *e,
                                 const hd_generation_request *req,
                                 unsigned char **out_rgb, int *out_w,
                                 int *out_h);

#endif /* HD_ENGINE_PRIV_H */