/*
 * TEMPORARY test stub for hd_image.c (src/image/hd_image.c is being written in
 * parallel by another agent and is not yet available).
 *
 * This stub implements the subset of the hd_image API that the layout unit
 * test needs: alloc/free, resize, and keep_aspect. It is NOT the production
 * implementation — it only needs to let tests/unit/layout_pipe.c link and run.
 *
 * hd_image_resize here implements oracle resize_pilimage parity (BICUBIC via
 * a simple separable cubic resampler), which is good enough to feed a bordered
 * reference image into the comparison. The layout test compares native bordered
 * output against the oracle's bordered output, so the resampler must match the
 * oracle's resize within the test epsilon.
 *
 * Remove this file once src/image/hd_image.c exists.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "hd_image.h"

/* ---- resize (oracle resize_pilimage, BICUBIC) ---- */

static double cubic_w(double t) {
    double t2 = t * t;
    double t3 = t2 * t;
    if (t < 0) t = -t;
    if (t >= 0 && t < 1) return 1.5 * t3 - 2.5 * t2 + 1.0;
    if (t < 2) return -0.5 * t3 + 2.5 * t2 - 4.0 * t + 2.0;
    return 0.0;
}

static void stub_resample(const hd_image *src, int new_w, int new_h,
                          hd_image *dst) {
    for (int y = 0; y < new_h; y++) {
        double fy = (y + 0.5) * src->height / new_h - 0.5;
        int iy0 = (int)floor(fy);
        double ty = fy - iy0;
        for (int x = 0; x < new_w; x++) {
            double fx = (x + 0.5) * src->width / new_w - 0.5;
            int ix0 = (int)floor(fx);
            double tx = fx - ix0;
            for (int c = 0; c < 3; c++) {
                double acc = 0.0;
                for (int j = -1; j <= 2; j++) {
                    int yy = iy0 + j;
                    if (yy < 0) yy = 0;
                    if (yy >= src->height) yy = src->height - 1;
                    double wy = cubic_w(ty - j);
                    for (int i = -1; i <= 2; i++) {
                        int xx = ix0 + i;
                        if (xx < 0) xx = 0;
                        if (xx >= src->width) xx = src->width - 1;
                        double wx = cubic_w(tx - i);
                        double val = src->rgb[((size_t)yy * src->width + xx) * 3 + c];
                        acc += val * wy * wx;
                    }
                }
                dst->rgb[((size_t)y * new_w + x) * 3 + c] = (float)acc;
            }
        }
    }
}

/* TEMPORARY stub — see header comment. */
hd_status hd_image_resize(const hd_image *src, int max_size, int patch_size,
                          hd_image *out) {
    if (!src || !out || max_size < patch_size || max_size <= 0) {
        return HD_ERR_MISSING;
    }
    double scale = (double)max_size / (double)(src->width > src->height
                                               ? src->width : src->height);
    int nw = (int)round(src->width * scale);
    int nh = (int)round(src->height * scale);
    /* snap down to multiple of patch_size, >= patch_size */
    if (nw % patch_size != 0) nw = (nw / patch_size) * patch_size;
    if (nh % patch_size != 0) nh = (nh / patch_size) * patch_size;
    if (nw < patch_size) nw = patch_size;
    if (nh < patch_size) nh = patch_size;

    out->width = nw;
    out->height = nh;
    if (out->rgb) free(out->rgb);
    out->rgb = (float *)malloc((size_t)3 * nw * nh * sizeof(float));
    if (!out->rgb) return HD_ERR_OOM;
    stub_resample(src, nw, nh, out);
    return HD_OK;
}

void hd_image_keep_aspect(const hd_image *ref, int req_w, int req_h,
                          int patch_size, int *out_w, int *out_h) {
    (void)ref; (void)req_w; (void)req_h; (void)patch_size;
    if (out_w) *out_w = 0;
    if (out_h) *out_h = 0;
}

hd_status hd_image_load(const char *path, hd_image *out) {
    (void)path; (void)out;
    return HD_ERR_MISSING;
}

void hd_image_calc_dims(int max_size, float aspect_w, float aspect_h,
                        int patch_size, int *out_w, int *out_h) {
    (void)max_size; (void)aspect_w; (void)aspect_h; (void)patch_size;
    if (out_w) *out_w = 0;
    if (out_h) *out_h = 0;
}

hd_status hd_image_to_patches(const hd_image *img, int patch_size,
                              float *patches_out) {
    (void)img; (void)patch_size; (void)patches_out;
    return HD_ERR_MISSING;
}

void hd_image_free(hd_image *img) {
    if (!img) return;
    free(img->rgb);
    free(img);
}
