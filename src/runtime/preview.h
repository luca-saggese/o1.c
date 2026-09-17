#ifndef HD_PREVIEW_H
#define HD_PREVIEW_H

/*
 * M1-post preview helper (contract section 8).
 *
 * Parity target: python/app.py:801-846 — previews at 1/4, 1/2, 3/4
 * milestones, downscaled JPEG with max side <= 384px.
 *
 * This is a standalone helper: a NULL/disabled preview path simply never
 * calls it, so a disabled preview costs nothing.
 */

#include <stddef.h>

#include "hidream.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Downscale an RGB image (row-major, 3 bytes per pixel, R,G,B) so that the
 * larger side is <= max_dim, then encode it as a baseline JPEG.
 *
 * On success returns HD_OK and sets *out_jpeg to a malloc'd buffer of
 * *out_len bytes (caller frees). Returns HD_ERR_OOM on allocation failure.
 * max_dim <= 0 means "no downscale" (encode at native size).
 */
hd_status hd_preview_extract(const unsigned char *rgb, int w, int h,
                             int max_dim, unsigned char **out_jpeg,
                             size_t *out_len);

#ifdef __cplusplus
}
#endif

#endif /* HD_PREVIEW_H */