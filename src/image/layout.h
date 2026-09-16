#ifndef HD_LAYOUT_H
#define HD_LAYOUT_H

/*
 * M1-post layout conditioning (contract section 55).
 *
 * Parity target: python/models/utils.py
 *   load_layout_bboxes / parse_layout_bboxes / draw_bbox_layout /
 *   add_outer_border_keep_size / create_layout_reference_images
 * and pipeline.py integration (K / max_size / cond_img_size formulas).
 *
 * No Python, no PIL, no network at runtime.
 */

#include <stddef.h>

#include "hd_image.h"
#include "hidream.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * One parsed layout bbox. Coordinates are the raw *relative* xxyy values from
 * the layout JSON: [x1, x2, y1, y2], each in [0,1] (or [0,100] legacy). They are
 * sorted (x1<=x2, y1<=y2) at parse time. They are converted to absolute pixels
 * (against image_width/image_height) inside hd_layout_create_reference_images,
 * matching oracle parse_layout_bboxes -> _xxyy_relative_to_absolute_bbox.
 */
typedef struct {
    double x1, x2;   /* sorted relative x endpoints (x1 <= x2) */
    double y1, y2;   /* sorted relative y endpoints (y1 <= y2) */
    char *text;      /* label (owned by the condition array); may be "" */
} hd_layout_condition;

/*
 * Parse a layout bbox JSON string (or a path to a JSON file) into an array of
 * hd_layout_condition. Mirrors oracle load_layout_bboxes + parse_layout_bboxes
 * semantics for the relative-coordinate stage:
 *
 *   - The top-level value may be a list, or a dict holding one of the keys
 *     "layout_bboxes", "bboxes", "boxes", "bbox_list" (first match wins).
 *   - Each item is either a dict with a "bbox"/"box" 4-element list plus an
 *     optional "text"/"label", or a bare 4-element list (text == "").
 *
 * Returns HD_OK and sets *out (caller frees via hd_layout_conditions_free) and
 * *count. Returns HD_ERR_PARSE on malformed JSON / structure.
 */
hd_status hd_layout_parse(const char *json, hd_layout_condition **out,
                          size_t *count);

/*
 * Reproduce oracle create_layout_reference_images:
 *
 *   - parse_layout_bboxes(layout, image_width, image_height) -> absolute boxes
 *   - draw_bbox_layout -> black image of size image_width x image_height with
 *     up to MAX_BOX=5 colored outline rectangles (no text is drawn by the
 *     oracle; see NOTES).
 *   - for each reference image: optionally resize to ref_max_size (via
 *     hd_image_resize), then add_outer_border_keep_size with the color assigned
 *     to that reference's layout box (or DEFAULT_COLORS[idx % 8]).
 *
 * Returns *out_images (an array of n_refs+1 images: n_refs bordered refs
 * followed by the layout canvas) and *out_count = n_refs+1. This matches the
 * oracle exactly. The layout canvas is the final element (out_images[n_refs]).
 *
 * NOTES / DEVIATIONS from the task's literal wording:
 *   - The task text says "composite the reference images onto a canvas" and
 *     "draw label text". The frozen oracle does neither: it returns a LIST of
 *     n_refs+1 independent images and does not draw any text. We match the
 *     oracle (the source of truth). Each bordered ref keeps its own dimensions.
 *   - Set ref_max_size <= 0 to skip the resize step (oracle ref_max_size=None).
 *
 * Returns HD_OK, or HD_ERR_PARSE / HD_ERR_OOM / HD_ERR_MISSING on failure.
 * On failure *out_images may be partially allocated; call
 * hd_layout_images_free on it (count of successfully built entries is not
 * reported; simplest is to not free on error in a hot path, but callers that
 * care can free via hd_layout_images_free with out_count).
 */
hd_status hd_layout_create_reference_images(const hd_image **refs, size_t n_refs,
        const hd_layout_condition *layout, size_t n_layout,
        int image_width, int image_height, int ref_max_size, int patch_size,
        hd_image ***out_images, size_t *out_count);

/*
 * Oracle pipeline.py max_size formula given the number of reference images K
 * (after layout adds one) and max(w, h) of the target image:
 *   K==1: max_dim; K==2: max_dim*48//64; K<=4: max_dim//2;
 *   K<=8: max_dim*24//64; else: max_dim//4.
 */
int hd_layout_max_size(int K, int max_dim);

void hd_layout_conditions_free(hd_layout_condition *conds, size_t count);

void hd_layout_images_free(hd_image **imgs, size_t count);

#ifdef __cplusplus
}
#endif

#endif /* HD_LAYOUT_H */
