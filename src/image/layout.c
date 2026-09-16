/*
 * M1-post layout conditioning (contract section 55).
 *
 * Native C parity for python/models/utils.py layout functions and
 * pipeline.py K / max_size / cond_img_size integration.
 *
 * No Python, no PIL, no network at runtime.
 */

#define _POSIX_C_SOURCE 200809L /* strdup */

#include "layout.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"

#define HD_LAYOUT_MAX_BOX 5

/* Oracle DEFAULT_COLORS (10 entries). */
static const int kDefaultColors[10][3] = {
    {255, 0, 0},
    {0, 180, 0},
    {0, 0, 255},
    {204, 180, 0},
    {255, 0, 255},
    {0, 255, 255},
    {128, 0, 0},
    {0, 128, 0},
    {0, 0, 128},
    {128, 128, 0},
};

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */

/* Python round(): round-half-even for the float -> int conversion. */
static int py_round(double x) {
    if (x >= 0.0) {
        double f = floor(x);
        double d = x - f;
        if (d < 0.5) return (int)f;
        if (d > 0.5) return (int)(f + 1.0);
        /* d == 0.5 : round half to even */
        return (fmod(f, 2.0) == 0.0) ? (int)f : (int)(f + 1.0);
    }
    return -py_round(-x);
}

static int imax(int a, int b) { return a > b ? a : b; }
static int imin(int a, int b) { return a < b ? a : b; }

/* A parsed, absolute-pixel bbox (from parse_layout_bboxes). */
typedef struct {
    int x1, y1, x2, y2;
    int orig_idx;
} hd_layout_abs_box;

/* Allocate an hd_image with a zeroed RGB buffer. */
static hd_image *alloc_image(int width, int height) {
    if (width <= 0 || height <= 0) return NULL;
    size_t n = (size_t)width * (size_t)height;
    hd_image *img = (hd_image *)calloc(1, sizeof(hd_image));
    if (!img) return NULL;
    img->width = width;
    img->height = height;
    img->rgb = (float *)calloc(3 * n, sizeof(float));
    if (!img->rgb) {
        free(img);
        return NULL;
    }
    return img;
}

static hd_status copy_image(const hd_image *src, hd_image **out) {
    if (!src || !out) return HD_ERR_MISSING;
    hd_image *dst = alloc_image(src->width, src->height);
    if (!dst) return HD_ERR_OOM;
    size_t n = (size_t)3 * src->width * src->height;
    memcpy(dst->rgb, src->rgb, n * sizeof(float));
    *out = dst;
    return HD_OK;
}

/* ------------------------------------------------------------------ */
/* JSON parsing (load_layout_bboxes / parse_layout_bboxes)             */
/* ------------------------------------------------------------------ */

/*
 * Convert the JSON value at *vptr (relative xxyy bbox) into a condition.
 * Matches oracle _as_bbox_and_text + the relative stage of
 * _xxyy_relative_to_absolute_bbox (sorting; scaling happens later in
 * create_reference_images since it needs image dims).
 */
static hd_status parse_bbox_value(const hd_json *v, hd_layout_condition *out) {
    const hd_json *bbox_node = NULL;
    const char *text = "";

    if (v && v->type == HD_JSON_OBJECT) {
        const hd_json *b = hd_json_get(v, "bbox");
        if (!b) b = hd_json_get(v, "box");
        if (!b) return HD_ERR_PARSE; /* oracle: Missing bbox in layout item */
        bbox_node = b;

        const hd_json *t = hd_json_get(v, "text");
        if (!t) t = hd_json_get(v, "label");
        /* oracle: str(item.get("text") or item.get("label") or "") */
        if (t) {
            const char *s = hd_json_string(t);
            if (s) {
                text = s;
            } else if (t->type == HD_JSON_INT || t->type == HD_JSON_DOUBLE) {
                char buf[64];
                if (t->type == HD_JSON_INT) {
                    snprintf(buf, sizeof(buf), "%lld", (long long)t->u.integer);
                } else {
                    double dv = t->u.number;
                    if (dv == (double)(long long)dv) {
                        snprintf(buf, sizeof(buf), "%lld", (long long)dv);
                    } else {
                        snprintf(buf, sizeof(buf), "%g", dv);
                    }
                }
                text = buf;
            }
        }
        if (!text) text = "";
    } else if (v && v->type == HD_JSON_ARRAY) {
        bbox_node = v;
        text = "";
    } else {
        return HD_ERR_PARSE;
    }

    if (hd_json_array_len(bbox_node) != 4) return HD_ERR_PARSE;

    double x1 = hd_json_double(hd_json_array_at(bbox_node, 0), 0.0);
    double x2 = hd_json_double(hd_json_array_at(bbox_node, 1), 0.0);
    double y1 = hd_json_double(hd_json_array_at(bbox_node, 2), 0.0);
    double y2 = hd_json_double(hd_json_array_at(bbox_node, 3), 0.0);

    double a = fabs(x1), b_ = fabs(y1), c = fabs(x2), d = fabs(y2);
    double max_abs = a;
    if (b_ > max_abs) max_abs = b_;
    if (c > max_abs) max_abs = c;
    if (d > max_abs) max_abs = d;
    /* max_abs <= 1.0 -> relative [0,1]; <= 100.0 -> legacy [0,100].
       We keep the raw values here; scaling to pixels happens later. */

    /* Sort endpoints (x1<=x2, y1<=y2). */
    double tx1 = x1 < x2 ? x1 : x2;
    double tx2 = x1 < x2 ? x2 : x1;
    double ty1 = y1 < y2 ? y1 : y2;
    double ty2 = y1 < y2 ? y2 : y1;
    x1 = tx1; x2 = tx2; y1 = ty1; y2 = ty2;

    out->x1 = x1;
    out->x2 = x2;
    out->y1 = y1;
    out->y2 = y2;
    out->text = text[0] ? strdup(text) : strdup("");
    if (!out->text) return HD_ERR_OOM;
    return HD_OK;
}

hd_status hd_layout_parse(const char *json, hd_layout_condition **out,
                          size_t *count) {
    if (!json || !out || !count) return HD_ERR_MISSING;
    *out = NULL;
    *count = 0;

    const char *err = NULL;
    hd_json *root = hd_json_parse(json, &err);
    if (!root) return HD_ERR_PARSE;

    const hd_json *boxes = root;
    if (root->type == HD_JSON_OBJECT) {
        const char *keys[] = {"layout_bboxes", "bboxes", "boxes", "bbox_list"};
        boxes = NULL;
        for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); i++) {
            boxes = hd_json_get(root, keys[i]);
            if (boxes) break;
        }
        if (!boxes) {
            hd_json_free(root);
            return HD_ERR_PARSE;
        }
    }

    if (!boxes || boxes->type != HD_JSON_ARRAY) {
        hd_json_free(root);
        return HD_ERR_PARSE;
    }

    size_t n = hd_json_array_len(boxes);
    hd_layout_condition *conds = (hd_layout_condition *)
        calloc(n ? n : 1, sizeof(hd_layout_condition));
    if (!conds) {
        hd_json_free(root);
        return HD_ERR_OOM;
    }

    for (size_t i = 0; i < n; i++) {
        const hd_json *item = hd_json_array_at(boxes, i);
        hd_status st = parse_bbox_value(item, &conds[i]);
        if (st != HD_OK) {
            hd_layout_conditions_free(conds, n);
            hd_json_free(root);
            return st;
        }
    }

    hd_json_free(root);
    *out = conds;
    *count = n;
    return HD_OK;
}

void hd_layout_conditions_free(hd_layout_condition *conds, size_t count) {
    if (!conds) return;
    for (size_t i = 0; i < count; i++) free(conds[i].text);
    free(conds);
}

/* ------------------------------------------------------------------ */
/* absolute bbox conversion (parse_layout_bboxes)                      */
/* ------------------------------------------------------------------ */

/*
 * Mirror _xxyy_relative_to_absolute_bbox: scale relative coords to pixels,
 * clamp to [0, w-1]/[0, h-1] with Python round(), error if degenerate.
 */
static hd_status to_absolute_bbox(const hd_layout_condition *c, int width,
                                  int height, int *x1o, int *y1o, int *x2o,
                                  int *y2o) {
    double x1 = c->x1, x2 = c->x2, y1 = c->y1, y2 = c->y2;

    double a = fabs(x1), b_ = fabs(y1), cc = fabs(x2), d = fabs(y2);
    double max_abs = a;
    if (b_ > max_abs) max_abs = b_;
    if (cc > max_abs) max_abs = cc;
    if (d > max_abs) max_abs = d;

    if (max_abs <= 1.0) {
        x1 *= width;  x2 *= width;
        y1 *= height; y2 *= height;
    } else if (max_abs <= 100.0) {
        x1 = x1 / 100.0 * width;
        x2 = x2 / 100.0 * width;
        y1 = y1 / 100.0 * height;
        y2 = y2 / 100.0 * height;
    }

    int ix1 = py_round(x1), iy1 = py_round(y1);
    int ix2 = py_round(x2), iy2 = py_round(y2);
    ix1 = imax(0, imin(width - 1, ix1));
    iy1 = imax(0, imin(height - 1, iy1));
    ix2 = imax(0, imin(width - 1, ix2));
    iy2 = imax(0, imin(height - 1, iy2));

    if (ix2 <= ix1 || iy2 <= iy1) return HD_ERR_PARSE;

    *x1o = ix1; *y1o = iy1; *x2o = ix2; *y2o = iy2;
    return HD_OK;
}

static int bbox_area(const hd_layout_abs_box *b) {
    int w = imax(0, b->x2 - b->x1);
    int h = imax(0, b->y2 - b->y1);
    return w * h;
}

/* ------------------------------------------------------------------ */
/* PIL rectangle outline parity                                        */
/* ------------------------------------------------------------------ */

static void set_pixel(hd_image *im, int x, int y, const float rgb[3]) {
    if (x < 0 || x >= im->width || y < 0 || y >= im->height) return;
    float *p = im->rgb + (size_t)y * im->width * 3 + (size_t)x * 3;
    p[0] = rgb[0];
    p[1] = rgb[1];
    p[2] = rgb[2];
}

/* Oracle: hline(im, x0, y, x1) inclusive [x0,x1]. */
static void pil_hline(hd_image *im, int x0, int y, int x1, const float rgb[3]) {
    if (y < 0 || y >= im->height) return;
    int lo = x0 < 0 ? 0 : x0;
    if (lo >= im->width) return;
    int hi = x1 >= im->width ? im->width - 1 : x1;
    if (hi < 0) return;
    if (lo > hi) return;
    float *row = im->rgb + (size_t)y * im->width * 3;
    for (int x = lo; x <= hi; x++) {
        row[(size_t)x * 3 + 0] = rgb[0];
        row[(size_t)x * 3 + 1] = rgb[1];
        row[(size_t)x * 3 + 2] = rgb[2];
    }
}

/* Oracle: line(im, x0,y0, x1,y1) vertical (dx==0) drawn as points y0..y1. */
static void pil_vline(hd_image *im, int x, int y0, int y1, const float rgb[3]) {
    if (x < 0 || x >= im->width) return;
    int lo = y0 < 0 ? 0 : y0;
    if (lo >= im->height) return;
    int hi = y1 >= im->height ? im->height - 1 : y1;
    if (hi < 0) return;
    if (lo > hi) return;
    for (int y = lo; y <= hi; y++) set_pixel(im, x, y, rgb);
}

/*
 * PIL ImagingDrawRectangle outline, exactly:
 *   for i in 0..width-1:
 *     hline(x0, y0+i, x1)
 *     hline(x0, y1-i, x1)
 *     line(x1-i, y0+width, x1-i, y1-width+1)
 *     line(x0+i, y0+width, x0+i, y1-width+1)
 */
static void draw_rectangle_outline(hd_image *im, int x0, int y0, int x1, int y1,
                                   const float rgb[3], int width) {
    if (y0 > y1) { int t = y0; y0 = y1; y1 = t; }
    if (width == 0) width = 1;
    for (int i = 0; i < width; i++) {
        pil_hline(im, x0, y0 + i, x1, rgb);
        pil_hline(im, x0, y1 - i, x1, rgb);
        pil_vline(im, x1 - i, y0 + width, y1 - width + 1, rgb);
        pil_vline(im, x0 + i, y0 + width, y1 - width + 1, rgb);
    }
}

/* add_outer_border_keep_size: draw `width` nested 1px outline rectangles. */
static void add_outer_border(hd_image *img, const int color[3], int width) {
    float rgb[3] = {color[0] / 255.0f, color[1] / 255.0f, color[2] / 255.0f};
    if (width <= 0) return;
    int w = img->width, h = img->height;
    for (int t = 0; t < width; t++) {
        draw_rectangle_outline(img, t, t, w - 1 - t, h - 1 - t, rgb, 1);
    }
}

/* ------------------------------------------------------------------ */
/* create_layout_reference_images                                      */
/* ------------------------------------------------------------------ */

hd_status hd_layout_create_reference_images(const hd_image **refs, size_t n_refs,
        const hd_layout_condition *layout, size_t n_layout,
        int image_width, int image_height, int ref_max_size, int patch_size,
        hd_image ***out_images, size_t *out_count) {
    if (!out_images || !out_count) return HD_ERR_MISSING;
    *out_images = NULL;
    *out_count = 0;
    if (n_refs > 0 && !refs) return HD_ERR_MISSING;
    if (n_layout > 0 && !layout) return HD_ERR_MISSING;

    /* ---- 1. parse_layout_bboxes -> absolute boxes ---- */
    hd_layout_abs_box *abs_boxes = NULL;
    if (n_layout > 0) {
        abs_boxes = (hd_layout_abs_box *)calloc(n_layout, sizeof(*abs_boxes));
        if (!abs_boxes) return HD_ERR_OOM;
        for (size_t i = 0; i < n_layout; i++) {
            hd_status st = to_absolute_bbox(&layout[i], image_width, image_height,
                                            &abs_boxes[i].x1, &abs_boxes[i].y1,
                                            &abs_boxes[i].x2, &abs_boxes[i].y2);
            if (st != HD_OK) {
                free(abs_boxes);
                return st;
            }
            abs_boxes[i].orig_idx = (int)i;
        }
    }

    /* ---- 2. draw_bbox_layout (black canvas + colored outlines) ---- */
    hd_image *canvas = alloc_image(image_width, image_height);
    if (!canvas) { free(abs_boxes); return HD_ERR_OOM; }

    /* color_list maps orig_idx -> color (or NULL). */
    float *color_list = NULL;
    if (n_layout > 0) {
        color_list = (float *)calloc(3 * n_layout, sizeof(float));
        if (!color_list) {
            hd_image_free(canvas);
            free(abs_boxes);
            return HD_ERR_OOM;
        }
        /* Need per-slot "is set" marker: use -1 sentinel in red channel. */
        for (size_t i = 0; i < n_layout; i++) color_list[3 * i] = -1.0f;
    }

    double edge = sqrt((double)image_width * (double)image_height);
    int max_bbox_line_width = (int)(edge * 0.05);
    int bbox_line_gap = imax(1, max_bbox_line_width / HD_LAYOUT_MAX_BOX);

    if (n_layout > 0) {
        /* sort copy by area desc, keep only top MAX_BOX */
        hd_layout_abs_box *sorted = (hd_layout_abs_box *)
            malloc(n_layout * sizeof(*sorted));
        if (!sorted) {
            free(color_list); hd_image_free(canvas); free(abs_boxes);
            return HD_ERR_OOM;
        }
        memcpy(sorted, abs_boxes, n_layout * sizeof(*sorted));
        for (size_t i = 1; i < n_layout; i++) {
            hd_layout_abs_box key = sorted[i];
            int key_area = bbox_area(&key);
            size_t j = i;
            while (j > 0 && bbox_area(&sorted[j - 1]) < key_area) {
                sorted[j] = sorted[j - 1];
                j--;
            }
            sorted[j] = key;
        }
        size_t top = n_layout < (size_t)HD_LAYOUT_MAX_BOX ? n_layout
                                                          : (size_t)HD_LAYOUT_MAX_BOX;
        for (size_t s = 0; s < top; s++) {
            int sorted_idx = (int)s;
            const int *c = kDefaultColors[sorted_idx % 10];
            float rgb[3] = {c[0] / 255.0f, c[1] / 255.0f, c[2] / 255.0f};
            int oi = sorted[s].orig_idx;
            if (oi >= 0 && oi < (int)n_layout) {
                color_list[3 * oi + 0] = rgb[0];
                color_list[3 * oi + 1] = rgb[1];
                color_list[3 * oi + 2] = rgb[2];
            }
            int line_width = imax(max_bbox_line_width - sorted_idx * bbox_line_gap, 5);
            draw_rectangle_outline(canvas, sorted[s].x1, sorted[s].y1,
                                   sorted[s].x2, sorted[s].y2, rgb, line_width);
        }
        free(sorted);
    }

    /* ---- 3. bordered refs + layout image ---- */
    size_t total = n_refs + 1;
    hd_image **imgs = (hd_image **)calloc(total, sizeof(hd_image *));
    if (!imgs) {
        free(color_list); hd_image_free(canvas); free(abs_boxes);
        return HD_ERR_OOM;
    }

    hd_status rc = HD_OK;
    for (size_t idx = 0; idx < n_refs; idx++) {
        const hd_image *ref = refs[idx];
        hd_image *working = NULL;

        if (ref_max_size > 0) {
            working = alloc_image(ref->width, ref->height);
            if (!working) { rc = HD_ERR_OOM; break; }
            hd_status rs = hd_image_resize(ref, ref_max_size, patch_size, working);
            if (rs != HD_OK) {
                hd_image_free(working);
                rc = rs;
                break;
            }
        } else {
            hd_status cs = copy_image(ref, &working);
            if (cs != HD_OK) { rc = cs; break; }
        }

        int color_r, color_g, color_b;
        int set = 0;
        if (idx < n_layout && color_list && color_list[3 * idx] >= 0.0f) {
            color_r = (int)(color_list[3 * idx + 0] * 255.0f + 0.5f);
            color_g = (int)(color_list[3 * idx + 1] * 255.0f + 0.5f);
            color_b = (int)(color_list[3 * idx + 2] * 255.0f + 0.5f);
            set = 1;
        }
        if (!set) {
            const int *c = kDefaultColors[idx % 10];
            color_r = c[0]; color_g = c[1]; color_b = c[2];
        }
        int line_width = (int)(sqrt((double)working->width * (double)working->height) * 0.04);
        int col[3] = {color_r, color_g, color_b};
        add_outer_border(working, col, line_width);
        imgs[idx] = working;
    }

    if (rc == HD_OK) {
        imgs[n_refs] = canvas;
        *out_images = imgs;
        *out_count = total;
        free(color_list);
        free(abs_boxes);
        return HD_OK;
    }

    /* error path */
    hd_image_free(canvas);
    for (size_t i = 0; i < n_refs; i++) hd_image_free(imgs[i]);
    free(imgs);
    free(color_list);
    free(abs_boxes);
    return rc;
}

void hd_layout_images_free(hd_image **imgs, size_t count) {
    if (!imgs) return;
    for (size_t i = 0; i < count; i++) hd_image_free(imgs[i]);
    free(imgs);
}

int hd_layout_max_size(int K, int max_dim) {
    if (K == 1) return max_dim;
    if (K == 2) return max_dim * 48 / 64;
    if (K <= 4) return max_dim / 2;
    if (K <= 8) return max_dim * 24 / 64;
    return max_dim / 4;
}
