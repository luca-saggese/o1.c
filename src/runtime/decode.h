#ifndef HD_DECODE_H
#define HD_DECODE_H

#include <stddef.h>

/*
 * Unified native output decode (oracle pipeline.py final rearrange):
 *
 *   z bf16 [img_tokens, C*p1*p2]  (pixel_unshuffled latent)
 *     -> float in [-1,1]
 *     -> (x+1)/2
 *     -> rearrange 'B (H W) (C p1 p2) -> B C (H p1) (W p2)'
 *     -> *255, round, clip -> RGB uint8 [h, w, 3]
 *
 * No VAE: HiDream O1 produces pixels directly via the patch/pixel head.
 * This is the single output path used by the CLI and by tests.
 */

/*
 * Decode a pixel-unshuffled latent to RGB.
 *
 *   z_f32    [img_tokens * C * patch * patch] fp32 host latent in [-1,1]
 *   grid_h   image rows in tokens
 *   grid_w   image cols in tokens
 *   patch    patch size (32 for HiDream O1)
 *   channels 3
 *   out      [grid_h*patch * grid_w*patch * 3] uint8, row-major RGB
 */
void hd_decode_to_rgb(const float *z_f32, int grid_h, int grid_w, int patch,
                      int channels, unsigned char *out);

#endif /* HD_DECODE_H */
