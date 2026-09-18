#ifndef HD_IMAGE_H
#define HD_IMAGE_H

/*
 * M1-post native image pipeline (contract section 54).
 *
 * Parity target: python/models/utils.py (resize_pilimage, calculate_dimensions,
 * keep_original_aspect) + pipeline.py TENSOR_TRANSFORM + pixel_unshuffle.
 *
 * All pixel buffers are float32 RGB in [0,1], row-major [3*H*W] (R plane,
 * G plane, B plane). No Python, no PIL, no network.
 */

#include <stddef.h>
#include <stdint.h>

#include "hidream.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int width;                 /* decoded pixel width  */
    int height;                /* decoded pixel height */
    float *rgb;                /* [3*width*height] float32 in [0,1] */
} hd_image;

/*
 * Decode PNG or JPEG from path into RGB float [0,1].
 * Applies EXIF orientation (oracle Image.open().convert("RGB") semantics).
 * Returns HD_ERR_MISSING on unsupported format / decode failure.
 */
hd_status hd_image_load(const char *path, hd_image *out);

/*
 * Oracle resize_pilimage parity (python/models/utils.py):
 *   while min(w,h) >= 2*image_size: halve with BOX resample
 *   S_max = image_size^2; scale = sqrt(S_max / (w*h))
 *   try 4 candidate sizes (round/floor combos, patch-aligned),
 *   pick largest with area <= S_max, then resize + center-crop.
 *   resampler = BICUBIC (oracle default).
 * Returns HD_ERR_MISSING if src is NULL or image_size < patch_size.
 */
hd_status hd_image_resize(const hd_image *src, int image_size, int patch_size,
                          hd_image *out);

/*
 * Oracle direct resize for the VLM conditioning path:
 *   pil_cond = img.resize((cw, ch), Image.LANCZOS)
 * Resizes exactly to (new_w, new_h) with an antialiased Lanczos-3 resampler.
 * No aspect-preserving scale, no center crop, no intermediate oversize.
 */
hd_status hd_image_resize_exact(const hd_image *src, int new_w, int new_h,
                                hd_image *out);

/*
 * Oracle calculate_dimensions parity (python/models/utils.py):
 *   width  = sqrt(max_size^2 * ratio)   (ratio = aspect_w/aspect_h)
 *   height = width / ratio
 *   both dims snapped down to multiples of 32 (oracle hardcodes 32).
 * NOTE: oracle snaps to 32, not patch_size; keep patch_size param for
 * callers that need a different alignment, but default callers pass 32.
 */
void hd_image_calc_dims(int max_size, float aspect_w, float aspect_h,
                        int patch_size, int *out_w, int *out_h);

/*
 * Oracle pixel_unshuffle parity (einops rearrange):
 *   "C (H p1) (W p2) -> (H W) (C p1 p2)"
 * patches_out must hold grid_h*grid_w * 3*patch_size*patch_size floats.
 * grid_h = height/patch_size, grid_w = width/patch_size.
 */
hd_status hd_image_to_patches(const hd_image *img, int patch_size,
                              float *patches_out);

/*
 * Oracle keep_original_aspect parity (python/models/pipeline.py):
 *   resize the single reference to max_size=2048 (patch-aligned) and use
 *   its resulting size as the target output dims. Returns the snapped
 *   output dims (already patch-aligned by resize_pilimage).
 */
void hd_image_keep_aspect(const hd_image *ref, int req_w, int req_h,
                          int patch_size, int *out_w, int *out_h);

/*
 * Qwen2VLImageProcessor patchify parity for the VLM vision tower input.
 * Produces pixel_values [n, C*t*p*p] bf16-ready floats from a normalized
 * [-1,1] RGB image, matching the processor's patchify:
 *
 *   patches = reshape(grid_t, t, C, gh//m, m, p, gw//m, m, p)
 *             .transpose(0,3,6,4,7,2,1,5,8).flatten
 *
 * with grid_t=1, t=temporal_patch_size, m=merge_size, p=patch_size.
 * Token order: (bh, bw, m_h, m_w); within a token:
 *   index = c*(t*p*p) + tt*(p*p) + p1*p + p2.
 * img must be [0,1] float32 RGB (hd_image layout); the caller normalizes
 * to [-1,1] (the processor does rescale 1/255 + normalize mean/std 0.5).
 * out must hold n * C*t*p*p floats.
 */
hd_status hd_image_to_vlm_patches(const hd_image *img, int patch_size,
                                  int temporal_patch_size, int merge_size,
                                  float *out);

void hd_image_free(hd_image *img);

#ifdef __cplusplus
}
#endif

#endif /* HD_IMAGE_H */