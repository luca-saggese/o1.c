#ifndef HD_GENERATE_H
#define HD_GENERATE_H

/*
 * M1.8 production native generation engine (M1_POST_FULL_FEATURE_PARITY
 * section 48-53). One engine for every mode: the unified request ABI is
 * lowered by the unified sequence builder into the single transformer
 * contract; the scheduler abstraction drives the denoising chain; the
 * unified output path decodes to RGB. No Python, no network, no golden
 * input required for normal inference.
 *
 * The engine is mode-agnostic: reference-bearing modes (edit/personalize/
 * layout/skeleton) stage reference pixel patches through the same
 * vinputs contract as the target latent.
 */

#include <stddef.h>
#include <stdint.h>

#include "hidream.h"
#include "request.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Run one full generation.
 *
 *   req        unified request (validated; defaults applied by caller)
 *   model_dir  local weights directory (profile local_path)
 *   device_id  CUDA device index
 *   out_rgb    [out_h * out_w * 3] uint8 RGB (malloc'd, caller frees)
 *   out_w/out_h decoded image dimensions
 *
 * Returns HD_OK on success. Fails closed on any NaN/Inf in the final
 * latent, on unsupported mode/scheduler, or on any stage error.
 */
hd_status hd_generate(const hd_generation_request *req, const char *model_dir,
                      int device_id, unsigned char **out_rgb,
                      int *out_w, int *out_h);

#ifdef __cplusplus
}
#endif

#endif /* HD_GENERATE_H */