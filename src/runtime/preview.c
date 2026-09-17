/*
 * preview.c — preview extraction helper (contract section 8).
 *
 * Parity target: python/app.py:801-846 — downscale to max side <= 384px,
 * encode JPEG. Uses the vendored libjpeg headers (src/image/jpeglib.h)
 * against the system libjpeg.so.8, same as src/image/hd_image.c.
 *
 * Standalone helper: a NULL/disabled preview path never calls it, so a
 * disabled preview costs nothing.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "preview.h"

/* libjpeg vendored headers (src/image/jpeglib.h etc.) */
#include "jpeglib.h"

/* ── Downscale (box filter, integer math) ──────────────────────────────── */

static unsigned char *downscale(const unsigned char *rgb, int w, int h,
                                int max_dim, int *out_w, int *out_h) {
    int dw = w, dh = h;
    if (max_dim > 0 && (w > max_dim || h > max_dim)) {
        /* keep aspect ratio; larger side becomes exactly max_dim */
        if (w >= h) {
            dw = max_dim;
            dh = (int)(((int64_t)h * max_dim + w / 2) / w);
        } else {
            dh = max_dim;
            dw = (int)(((int64_t)w * max_dim + h / 2) / h);
        }
        if (dw < 1) dw = 1;
        if (dh < 1) dh = 1;
    }

    unsigned char *out = (unsigned char *)malloc((size_t)dw * dh * 3);
    if (!out) return NULL;

    for (int y = 0; y < dh; y++) {
        int y0 = (int)(((int64_t)y * h) / dh);
        int y1 = (int)(((int64_t)(y + 1) * h) / dh);
        if (y1 <= y0) y1 = y0 + 1;
        if (y1 > h) y1 = h;
        for (int x = 0; x < dw; x++) {
            int x0 = (int)(((int64_t)x * w) / dw);
            int x1 = (int)(((int64_t)(x + 1) * w) / dw);
            if (x1 <= x0) x1 = x0 + 1;
            if (x1 > w) x1 = w;
            int r = 0, g = 0, b = 0;
            int n = 0;
            for (int yy = y0; yy < y1; yy++) {
                const unsigned char *row = rgb + (size_t)yy * w * 3;
                for (int xx = x0; xx < x1; xx++) {
                    r += row[xx * 3 + 0];
                    g += row[xx * 3 + 1];
                    b += row[xx * 3 + 2];
                    n++;
                }
            }
            unsigned char *p = out + ((size_t)y * dw + x) * 3;
            p[0] = (unsigned char)((r + n / 2) / n);
            p[1] = (unsigned char)((g + n / 2) / n);
            p[2] = (unsigned char)((b + n / 2) / n);
        }
    }
    *out_w = dw;
    *out_h = dh;
    return out;
}

/* ── JPEG encode via vendored libjpeg headers ──────────────────────────── */

static hd_status encode_jpeg(const unsigned char *rgb, int w, int h,
                             unsigned char **out_jpeg, size_t *out_len) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    unsigned char *buf = NULL;
    unsigned long buf_len = 0;
    jpeg_mem_dest(&cinfo, &buf, &buf_len);

    cinfo.image_width = (JDIMENSION)w;
    cinfo.image_height = (JDIMENSION)h;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 72, TRUE); /* oracle app.py quality=72 */

    jpeg_start_compress(&cinfo, TRUE);

    JSAMPROW row_pointer[1];
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = (JSAMPROW)(rgb + (size_t)cinfo.next_scanline * w * 3);
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);

    if (!buf || buf_len == 0) {
        free(buf);
        return HD_ERR_OOM;
    }
    *out_jpeg = buf;
    *out_len = (size_t)buf_len;
    return HD_OK;
}

hd_status hd_preview_extract(const unsigned char *rgb, int w, int h,
                             int max_dim, unsigned char **out_jpeg,
                             size_t *out_len) {
    if (!rgb || w <= 0 || h <= 0 || !out_jpeg || !out_len) return HD_ERR_PARSE;
    *out_jpeg = NULL;
    *out_len = 0;

    int dw = w, dh = h;
    unsigned char *scaled = NULL;
    if (max_dim > 0 && (w > max_dim || h > max_dim)) {
        scaled = downscale(rgb, w, h, max_dim, &dw, &dh);
        if (!scaled) return HD_ERR_OOM;
    }

    hd_status st = encode_jpeg(scaled ? scaled : rgb, dw, dh, out_jpeg,
                               out_len);
    free(scaled);
    return st;
}