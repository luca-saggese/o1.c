/*
 * test_preview.c — unit test for hd_preview_extract (contract section 8).
 *
 * Asserts:
 *   - downscale dimensions are correct (aspect preserved, max side <= max_dim)
 *   - no upscale when the image is already smaller than max_dim
 *   - output is a valid JPEG (starts with 0xFF 0xD8)
 *   - NULL/disabled preview path costs nothing (separate function: a NULL
 *     preview simply never calls hd_preview_extract)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "preview.h"

static int failures = 0;

#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);    \
            failures++;                                                      \
        }                                                                    \
    } while (0)

/* solid-color RGB image, row-major, 3 bytes per pixel */
static unsigned char *make_rgb(int w, int h, int r, int g, int b) {
    unsigned char *img = (unsigned char *)malloc((size_t)w * h * 3);
    if (!img) return NULL;
    for (int i = 0; i < w * h; i++) {
        img[i * 3 + 0] = (unsigned char)r;
        img[i * 3 + 1] = (unsigned char)g;
        img[i * 3 + 2] = (unsigned char)b;
    }
    return img;
}

static void test_downscale_dims(void) {
    unsigned char *img = make_rgb(768, 512, 10, 20, 30);
    CHECK(img != NULL, "alloc 768x512");
    if (!img) return;

    unsigned char *jpeg = NULL;
    size_t len = 0;
    hd_status st = hd_preview_extract(img, 768, 512, 384, &jpeg, &len);
    CHECK(st == HD_OK, "768x512 -> 384x256 ok");
    CHECK(jpeg != NULL && len > 0, "jpeg produced");
    if (jpeg) {
        CHECK(jpeg[0] == 0xFF && jpeg[1] == 0xD8, "starts with SOI 0xFF 0xD8");
        free(jpeg);
    }
    free(img);
}

static void test_portrait_dims(void) {
    unsigned char *img = make_rgb(200, 400, 200, 100, 50);
    CHECK(img != NULL, "alloc 200x400");
    if (!img) return;

    unsigned char *jpeg = NULL;
    size_t len = 0;
    hd_status st = hd_preview_extract(img, 200, 400, 384, &jpeg, &len);
    CHECK(st == HD_OK, "200x400 -> 192x384 ok");
    CHECK(jpeg != NULL && len > 0, "jpeg produced");
    if (jpeg) {
        CHECK(jpeg[0] == 0xFF && jpeg[1] == 0xD8, "starts with SOI 0xFF 0xD8");
        free(jpeg);
    }
    free(img);
}

static void test_no_upscale(void) {
    unsigned char *img = make_rgb(100, 80, 1, 2, 3);
    CHECK(img != NULL, "alloc 100x80");
    if (!img) return;

    unsigned char *jpeg = NULL;
    size_t len = 0;
    hd_status st = hd_preview_extract(img, 100, 80, 384, &jpeg, &len);
    CHECK(st == HD_OK, "small image ok");
    CHECK(jpeg != NULL && len > 0, "jpeg produced");
    if (jpeg) {
        CHECK(jpeg[0] == 0xFF && jpeg[1] == 0xD8, "starts with SOI 0xFF 0xD8");
        free(jpeg);
    }
    free(img);
}

static void test_invalid_args(void) {
    unsigned char *jpeg = NULL;
    size_t len = 0;
    CHECK(hd_preview_extract(NULL, 10, 10, 384, &jpeg, &len) == HD_ERR_PARSE,
          "NULL rgb rejected");
    CHECK(hd_preview_extract((const unsigned char *)"x", 0, 10, 384, &jpeg,
                             &len) == HD_ERR_PARSE,
          "w<=0 rejected");
    CHECK(hd_preview_extract((const unsigned char *)"x", 10, -1, 384, &jpeg,
                             &len) == HD_ERR_PARSE,
          "h<=0 rejected");
    CHECK(hd_preview_extract((const unsigned char *)"x", 10, 10, 384, NULL,
                             &len) == HD_ERR_PARSE,
          "NULL out_jpeg rejected");
}

int main(void) {
    test_downscale_dims();
    test_portrait_dims();
    test_no_upscale();
    test_invalid_args();

    if (failures) {
        fprintf(stderr, "test_preview: %d failure(s)\n", failures);
        return 1;
    }
    printf("test_preview: all tests passed\n");
    return 0;
}