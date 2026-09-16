/*
 * M1-post native image pipeline (contract section 54).
 *
 * Parity target: python/models/utils.py (resize_pilimage, calculate_dimensions,
 * keep_original_aspect) + pipeline.py TENSOR_TRANSFORM + pixel_unshuffle.
 *
 * All pixel buffers are float32 RGB in [0,1], row-major [3*H*W] (R plane,
 * G plane, B plane). No Python, no PIL, no network.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hd_image.h"
#include "png.h"

/* libjpeg vendored headers (src/image/jpeglib.h etc.) */
#include "jpeglib.h"

/* ------------------------------------------------------------------ */
/* EXIF orientation helpers                                            */
/* ------------------------------------------------------------------ */

/* Minimal EXIF orientation reader: scans APP1 (0xFFE1) for the "Exif\0\0"
 * header and the 0x0112 (Orientation) tag. Returns 1..8, or 0 when absent. */
static int exif_orientation(const unsigned char *d, size_t n) {
    size_t i = 2; /* skip SOI */
    while (i + 4 <= n) {
        if (d[i] != 0xFF) break;
        unsigned char marker = d[i + 1];
        if (marker == 0xD8 || (marker >= 0xD0 && marker <= 0xD7)) { i += 2; continue; }
        if (marker == 0xDA || marker == 0xD9) break; /* SOS/EOI: no more metadata */
        if (i + 4 > n) break;
        size_t seg_len = ((size_t)d[i + 2] << 8) | d[i + 3];
        if (seg_len < 2) break;
        size_t payload = i + 4;
        if (marker == 0xE1 && payload + 6 <= n &&
            d[payload] == 'E' && d[payload + 1] == 'x' &&
            d[payload + 2] == 'i' && d[payload + 3] == 'f' &&
            d[payload + 4] == 0 && d[payload + 5] == 0) {
            /* TIFF header at payload+6 */
            size_t t = payload + 6;
            if (t + 8 > n) break;
            int le = (d[t] == 'I' && d[t + 1] == 'I');
            int be = (d[t] == 'M' && d[t + 1] == 'M');
            if (!le && !be) break;
            size_t ifd0 = 0;
            if (le) ifd0 = (size_t)d[t + 4] | ((size_t)d[t + 5] << 8) |
                           ((size_t)d[t + 6] << 16) | ((size_t)d[t + 7] << 24);
            else    ifd0 = ((size_t)d[t + 4] << 24) | ((size_t)d[t + 5] << 16) |
                           ((size_t)d[t + 6] << 8) | (size_t)d[t + 7];
            size_t e = t + ifd0;
            if (e + 2 > n) break;
            int nent = le ? (d[e] | (d[e + 1] << 8))
                          : ((d[e] << 8) | d[e + 1]);
            for (int k = 0; k < nent; k++) {
                size_t ent = e + 2 + (size_t)k * 12;
                if (ent + 12 > n) break;
                unsigned tag = le ? ((unsigned)d[ent] | ((unsigned)d[ent + 1] << 8))
                                  : (((unsigned)d[ent] << 8) | d[ent + 1]);
                if (tag == 0x0112) {
                    unsigned val = le ? ((unsigned)d[ent + 8] | ((unsigned)d[ent + 9] << 8))
                                      : (((unsigned)d[ent + 8] << 8) | d[ent + 9]);
                    return (int)val;
                }
            }
            break;
        }
        i = payload + seg_len;
    }
    return 0;
}

/* Apply EXIF orientation to an RGB float image (in place).
 * Orientation values follow the standard 1..8 mapping. */
static void apply_orientation(float *rgb, int w, int h, int orient) {
    if (orient <= 1) return;
    int n = w * h;
    float *tmp = malloc((size_t)n * 3 * sizeof(float));
    if (!tmp) return;
    memcpy(tmp, rgb, (size_t)n * 3 * sizeof(float));
    /* tmp is source, rgb is destination */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int sx = x, sy = y;
            switch (orient) {
                case 2: sx = w - 1 - x; break;                 /* flip H */
                case 3: sx = w - 1 - x; sy = h - 1 - y; break; /* 180 */
                case 4: sy = h - 1 - y; break;                 /* flip V */
                case 5: { int t = sx; sx = sy; sy = t; sx = w - 1 - sx; break; } /* transpose+flip */
                case 6: { int t = sx; sx = sy; sy = t; break; } /* 90 CW */
                case 7: { int t = sx; sx = sy; sy = t; sy = h - 1 - sy; break; } /* transpose */
                case 8: { int t = sx; sx = sy; sy = t; sx = w - 1 - sx; sy = h - 1 - sy; break; } /* 90 CCW */
                default: break;
            }
            if (sx < 0 || sx >= w || sy < 0 || sy >= h) continue;
            for (int c = 0; c < 3; c++)
                rgb[((size_t)y * w + x) * 3 + c] = tmp[((size_t)sy * w + sx) * 3 + c];
        }
    }
    free(tmp);
}

/* ------------------------------------------------------------------ */
/* Decoders                                                            */
/* ------------------------------------------------------------------ */

static hd_status decode_png(const char *path, hd_image *out) {
    png_image *img = png_load(path);
    if (!img) { hd_set_error("hd_image: png decode failed"); return HD_ERR_MISSING; }
    int w = img->width, h = img->height;
    float *rgb = malloc((size_t)w * h * 3 * sizeof(float));
    if (!rgb) { png_free(img); hd_set_error("hd_image: oom"); return HD_ERR_OOM; }
    for (int i = 0; i < w * h; i++) {
        int ch = img->channels;
        const unsigned char *p = &img->data[(size_t)i * ch];
        rgb[(size_t)i * 3 + 0] = p[0] / 255.0f;
        rgb[(size_t)i * 3 + 1] = (ch >= 3 ? p[1] : p[0]) / 255.0f;
        rgb[(size_t)i * 3 + 2] = (ch >= 3 ? p[2] : p[0]) / 255.0f;
    }
    png_free(img);
    out->width = w; out->height = h; out->rgb = rgb;
    return HD_OK;
}

static hd_status decode_jpeg(const char *path, hd_image *out) {
    FILE *f = fopen(path, "rb");
    if (!f) { hd_set_error("hd_image: cannot open jpeg"); return HD_ERR_MISSING; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); hd_set_error("hd_image: empty jpeg"); return HD_ERR_MISSING; }
    unsigned char *buf = malloc((size_t)sz);
    if (!buf) { fclose(f); hd_set_error("hd_image: oom"); return HD_ERR_OOM; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf); fclose(f); hd_set_error("hd_image: jpeg read failed"); return HD_ERR_MISSING;
    }
    fclose(f);

    int orient = exif_orientation(buf, (size_t)sz);

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, buf, (size_t)sz);
    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        jpeg_destroy_decompress(&cinfo);
        free(buf); hd_set_error("hd_image: jpeg header parse failed"); return HD_ERR_MISSING;
    }
    jpeg_start_decompress(&cinfo);
    int w = cinfo.output_width, h = cinfo.output_height;
    int ch = cinfo.output_components;
    float *rgb = malloc((size_t)w * h * 3 * sizeof(float));
    if (!rgb) {
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        free(buf); hd_set_error("hd_image: oom"); return HD_ERR_OOM;
    }
    unsigned char *row = malloc((size_t)w * ch);
    if (!row) {
        free(rgb); jpeg_finish_decompress(&cinfo); jpeg_destroy_decompress(&cinfo);
        free(buf); hd_set_error("hd_image: oom"); return HD_ERR_OOM;
    }
    for (int y = 0; y < h; y++) {
        jpeg_read_scanlines(&cinfo, &row, 1);
        for (int x = 0; x < w; x++) {
            const unsigned char *p = &row[(size_t)x * ch];
            rgb[((size_t)y * w + x) * 3 + 0] = p[0] / 255.0f;
            rgb[((size_t)y * w + x) * 3 + 1] = (ch >= 3 ? p[1] : p[0]) / 255.0f;
            rgb[((size_t)y * w + x) * 3 + 2] = (ch >= 3 ? p[2] : p[0]) / 255.0f;
        }
    }
    free(row);
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    free(buf);

    if (orient > 1) {
        /* orientation 5-8 swap dims */
        if (orient >= 5) { int t = w; w = h; h = t; }
        apply_orientation(rgb, w, h, orient);
    }
    out->width = w; out->height = h; out->rgb = rgb;
    return HD_OK;
}

hd_status hd_image_load(const char *path, hd_image *out) {
    out->width = 0; out->height = 0; out->rgb = NULL;
    if (!path) { hd_set_error("hd_image: null path"); return HD_ERR_PARSE; }
    FILE *f = fopen(path, "rb");
    if (!f) { hd_set_error("hd_image: cannot open '%s'", path); return HD_ERR_MISSING; }
    unsigned char magic[8];
    size_t got = fread(magic, 1, 8, f);
    fclose(f);
    if (got < 8) { hd_set_error("hd_image: file too small"); return HD_ERR_MISSING; }
    if (magic[0] == 0x89 && magic[1] == 'P' && magic[2] == 'N' && magic[3] == 'G')
        return decode_png(path, out);
    if (magic[0] == 0xFF && magic[1] == 0xD8)
        return decode_jpeg(path, out);
    hd_set_error("hd_image: unsupported format (png/jpeg only)");
    return HD_ERR_MISSING;
}

void hd_image_free(hd_image *img) {
    if (!img) return;
    free(img->rgb);
    img->rgb = NULL;
    img->width = img->height = 0;
}

/* ------------------------------------------------------------------ */
/* Resampling (BICUBIC, PIL parity)                                    */
/* ------------------------------------------------------------------ */

static double cubic_w(double t) {
    if (t < 0) t = -t;
    double t2 = t * t, t3 = t2 * t;
    if (t < 1) return 1.5 * t3 - 2.5 * t2 + 1.0;
    if (t < 2) return -0.5 * t3 + 2.5 * t2 - 4.0 * t + 2.0;
    return 0.0;
}

/* PIL BICUBIC: source coordinate = (dst + 0.5) * scale - 0.5, 4-tap. */
static void bicubic_resample(const hd_image *src, int new_w, int new_h,
                             hd_image *dst) {
    double sx = (double)src->width / new_w;
    double sy = (double)src->height / new_h;
    float *out = malloc((size_t)new_w * new_h * 3 * sizeof(float));
    if (!out) { dst->rgb = NULL; return; }
    for (int y = 0; y < new_h; y++) {
        double fy = (y + 0.5) * sy - 0.5;
        int iy0 = (int)floor(fy);
        double ty = fy - iy0;
        for (int x = 0; x < new_w; x++) {
            double fx = (x + 0.5) * sx - 0.5;
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
                        double wgt = wy * cubic_w(tx - i);
                        acc += wgt * src->rgb[((size_t)yy * src->width + xx) * 3 + c];
                    }
                }
                out[((size_t)y * new_w + x) * 3 + c] = (float)acc;
            }
        }
    }
    dst->width = new_w; dst->height = new_h; dst->rgb = out;
}

/* PIL BOX resample: average of source pixels mapping to each dst pixel. */
static void box_resample(const hd_image *src, int new_w, int new_h,
                         hd_image *dst) {
    float *out = malloc((size_t)new_w * new_h * 3 * sizeof(float));
    if (!out) { dst->rgb = NULL; return; }
    double sx = (double)src->width / new_w;
    double sy = (double)src->height / new_h;
    for (int y = 0; y < new_h; y++) {
        int y0 = (int)floor(y * sy), y1 = (int)ceil((y + 1) * sy);
        if (y1 > src->height) y1 = src->height;
        if (y1 <= y0) y1 = y0 + 1;
        for (int x = 0; x < new_w; x++) {
            int x0 = (int)floor(x * sx), x1 = (int)ceil((x + 1) * sx);
            if (x1 > src->width) x1 = src->width;
            if (x1 <= x0) x1 = x0 + 1;
            for (int c = 0; c < 3; c++) {
                double acc = 0.0;
                for (int yy = y0; yy < y1; yy++)
                    for (int xx = x0; xx < x1; xx++)
                        acc += src->rgb[((size_t)yy * src->width + xx) * 3 + c];
                out[((size_t)y * new_w + x) * 3 + c] = (float)(acc / ((y1 - y0) * (x1 - x0)));
            }
        }
    }
    dst->width = new_w; dst->height = new_h; dst->rgb = out;
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */

hd_status hd_image_resize(const hd_image *src, int image_size, int patch_size,
                          hd_image *out) {
    out->width = 0; out->height = 0; out->rgb = NULL;
    if (!src || !src->rgb || image_size < patch_size) {
        hd_set_error("hd_image: resize bad args");
        return HD_ERR_MISSING;
    }
    hd_image cur = *src;
    hd_image tmp;
    /* Oracle: while min(w,h) >= 2*image_size, halve with BOX. */
    while (cur.width >= 2 * image_size && cur.height >= 2 * image_size) {
        box_resample(&cur, cur.width / 2, cur.height / 2, &tmp);
        if (!tmp.rgb) { hd_set_error("hd_image: oom"); return HD_ERR_OOM; }
        if (cur.rgb != src->rgb) free(cur.rgb);
        cur = tmp;
    }

    int m = patch_size;
    int width = cur.width, height = cur.height;
    double S_max = (double)image_size * image_size;
    double scale = sqrt(S_max / ((double)width * height));

    /* 4 candidate sizes, patch-aligned (Python: round(x)//m*m, floor(x)//m*m) */
    int cand[4][2];
    int rw = (int)round(width * scale), fw = (int)floor(width * scale);
    int rh = (int)round(height * scale), fh = (int)floor(height * scale);
    cand[0][0] = rw / m * m; cand[0][1] = rh / m * m;
    cand[1][0] = rw / m * m; cand[1][1] = fh / m * m;
    cand[2][0] = fw / m * m; cand[2][1] = rh / m * m;
    cand[3][0] = fw / m * m; cand[3][1] = fh / m * m;
    /* sort by area desc (stable: keep first on ties) */
    for (int i = 0; i < 4; i++)
        for (int j = i + 1; j < 4; j++) {
            long ai = (long)cand[i][0] * cand[i][1], aj = (long)cand[j][0] * cand[j][1];
            if (aj > ai) { int t0 = cand[i][0], t1 = cand[i][1]; cand[i][0] = cand[j][0]; cand[i][1] = cand[j][1]; cand[j][0] = t0; cand[j][1] = t1; }
        }
    int nw = cand[0][0], nh = cand[0][1];
    for (int i = 0; i < 4; i++) {
        if ((long)cand[i][0] * cand[i][1] <= (long)S_max) { nw = cand[i][0]; nh = cand[i][1]; break; }
    }
    if (nw <= 0 || nh <= 0) { nw = m; nh = m; }

    /* resize + center-crop */
    double s1 = (double)width / nw;
    double s2 = (double)height / nh;
    hd_image resized;
    if (s1 < s2) {
        int rh = (int)round(height / s1);
        bicubic_resample(&cur, nw, rh, &resized);
        if (!resized.rgb) { if (cur.rgb != src->rgb) free(cur.rgb); hd_set_error("hd_image: oom"); return HD_ERR_OOM; }
        int top = (rh - nh) / 2;
        float *crop = malloc((size_t)nw * nh * 3 * sizeof(float));
        if (!crop) { free(resized.rgb); if (cur.rgb != src->rgb) free(cur.rgb); hd_set_error("hd_image: oom"); return HD_ERR_OOM; }
        for (int y = 0; y < nh; y++)
            memcpy(&crop[(size_t)y * nw * 3], &resized.rgb[((size_t)(top + y) * nw) * 3], (size_t)nw * 3 * sizeof(float));
        free(resized.rgb);
        out->width = nw; out->height = nh; out->rgb = crop;
    } else {
        int rw = (int)round(width / s2);
        bicubic_resample(&cur, rw, nh, &resized);
        if (!resized.rgb) { if (cur.rgb != src->rgb) free(cur.rgb); hd_set_error("hd_image: oom"); return HD_ERR_OOM; }
        int left = (rw - nw) / 2;
        float *crop = malloc((size_t)nw * nh * 3 * sizeof(float));
        if (!crop) { free(resized.rgb); if (cur.rgb != src->rgb) free(cur.rgb); hd_set_error("hd_image: oom"); return HD_ERR_OOM; }
        for (int y = 0; y < nh; y++)
            for (int x = 0; x < nw; x++)
                for (int c = 0; c < 3; c++)
                    crop[((size_t)y * nw + x) * 3 + c] = resized.rgb[((size_t)y * rw + (left + x)) * 3 + c];
        free(resized.rgb);
        out->width = nw; out->height = nh; out->rgb = crop;
    }
    if (cur.rgb != src->rgb) free(cur.rgb);
    return HD_OK;
}

void hd_image_calc_dims(int max_size, float aspect_w, float aspect_h,
                        int patch_size, int *out_w, int *out_h) {
    double ratio = (double)aspect_w / aspect_h;
    double width = sqrt((double)max_size * max_size * ratio);
    double height = width / ratio;
    int m = patch_size > 0 ? patch_size : 32;
    *out_w = (int)(width / m) * m;
    *out_h = (int)(height / m) * m;
    if (*out_w < m) *out_w = m;
    if (*out_h < m) *out_h = m;
}

hd_status hd_image_to_patches(const hd_image *img, int patch_size,
                              float *patches_out) {
    if (!img || !img->rgb || !patches_out || patch_size <= 0) {
        hd_set_error("hd_image: to_patches bad args");
        return HD_ERR_MISSING;
    }
    if (img->width % patch_size != 0 || img->height % patch_size != 0) {
        hd_set_error("hd_image: dims not patch-aligned");
        return HD_ERR_MISSING;
    }
    int gh = img->height / patch_size, gw = img->width / patch_size;
    /* einops "C (H p1) (W p2) -> (H W) (C p1 p2)" */
    for (int ph = 0; ph < gh; ph++) {
        for (int pw = 0; pw < gw; pw++) {
            size_t dst = ((size_t)ph * gw + pw) * 3 * patch_size * patch_size;
            for (int c = 0; c < 3; c++) {
                for (int j = 0; j < patch_size; j++) {
                    for (int i = 0; i < patch_size; i++) {
                        int sy = ph * patch_size + j, sx = pw * patch_size + i;
                        patches_out[dst + (size_t)c * patch_size * patch_size +
                                    (size_t)j * patch_size + i] =
                            img->rgb[((size_t)sy * img->width + sx) * 3 + c];
                    }
                }
            }
        }
    }
    return HD_OK;
}

void hd_image_keep_aspect(const hd_image *ref, int req_w, int req_h,
                          int patch_size, int *out_w, int *out_h) {
    (void)req_w; (void)req_h;
    /* Oracle: resize the single reference to max_size=2048 (patch-aligned)
     * and use its resulting size as the target output dims. */
    hd_image resized;
    hd_status st = hd_image_resize(ref, 2048, patch_size, &resized);
    if (st != HD_OK) { *out_w = req_w; *out_h = req_h; return; }
    *out_w = resized.width;
    *out_h = resized.height;
    hd_image_free(&resized);
}