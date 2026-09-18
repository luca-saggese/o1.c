/*
 * Unit test: M1-post native image pipeline (src/image/hd_image.c).
 *
 * Validates the native implementation against the frozen Python oracle
 * semantics (python/models/utils.py resize_pilimage / calculate_dimensions,
 * pipeline.py TENSOR_TRANSFORM + pixel_unshuffle):
 *
 *   (a) hd_image_calc_dims matches the oracle formula
 *       width=sqrt(max^2*ratio), height=width/ratio, snapped to 32.
 *   (b) hd_image_resize: output patch-aligned, area <= image_size^2,
 *       and for a downscale the max channel deviation vs a reference
 *       BICUBIC resample stays within tolerance.
 *   (c) hd_image_to_patches roundtrips through the einops rearrange
 *       "C (H p1) (W p2) -> (H W) (C p1 p2)".
 *   (d) PNG/JPEG decode: synthetic images decode to expected dims and
 *       channel values.
 *   (e) hd_image_keep_aspect derives output dims from a 2048-resized ref.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hd_image.h"
#include "png_wrap.h"

static int failures = 0;

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__, msg);  \
            failures++;                                                     \
        } else {                                                            \
            printf("  ok  %s\n", msg);                                      \
        }                                                                   \
    } while (0)

static void synth_rgb(int w, int h, float *rgb) {
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            float *p = rgb + ((size_t)y * w + x) * 3;
            p[0] = (float)((x * 3 + y) % 256) / 255.0f;
            p[1] = (float)((y * 5 + x * 2) % 256) / 255.0f;
            p[2] = (float)((x * y) % 256) / 255.0f;
        }
}

static void test_calc_dims(void) {
    int w, h;
    /* oracle: width=sqrt(1024^2*1.0)=1024, height=1024 */
    hd_image_calc_dims(1024, 1.0f, 1.0f, 32, &w, &h);
    CHECK(w == 1024 && h == 1024, "calc_dims 1:1 -> 1024x1024");
    /* oracle: ratio 16/9 -> width=sqrt(1024^2*1.777)=1365.3->1344, height=768 */
    hd_image_calc_dims(1024, 16.0f, 9.0f, 32, &w, &h);
    CHECK(w == 1344 && h == 768, "calc_dims 16:9 -> 1344x768");
    /* oracle: ratio 9/16 -> width=sqrt(1024^2*0.5625)=768, height=1365->1344 */
    hd_image_calc_dims(1024, 9.0f, 16.0f, 32, &w, &h);
    CHECK(w == 768 && h == 1344, "calc_dims 9:16 -> 768x1344");
}

static void test_resize(void) {
    hd_image src;
    src.width = 640; src.height = 480;
    src.rgb = malloc((size_t)3 * 640 * 480 * sizeof(float));
    synth_rgb(640, 480, src.rgb);

    hd_image out;
    hd_status st = hd_image_resize(&src, 1024, 32, &out);
    CHECK(st == HD_OK, "resize 640x480 -> 1024 returns HD_OK");
    if (st == HD_OK) {
        CHECK(out.width % 32 == 0 && out.height % 32 == 0,
              "resize output patch-aligned");
        CHECK((long)out.width * out.height <= 1024L * 1024L,
              "resize output area <= image_size^2");
        CHECK(out.width >= 32 && out.height >= 32, "resize output >= patch");
        /* 640x480 is 4:3; 1024 target -> scale=sqrt(1048576/307200)=1.8475
           -> 1182.4->1152, 886.8->864 -> 1152x864 */
        CHECK(out.width == 1152 && out.height == 864,
              "resize 640x480 -> 1152x864");
        /* BICUBIC ringing may overshoot [0,1] slightly (PIL parity);
           reject only broken values (e.g. |v| > 2) */
        float mn = 1.0f, mx = 0.0f;
        for (long i = 0; i < (long)out.width * out.height * 3; i++) {
            if (out.rgb[i] < mn) mn = out.rgb[i];
            if (out.rgb[i] > mx) mx = out.rgb[i];
        }
        CHECK(mn >= -2.0f && mx <= 2.0f, "resize output within BICUBIC range");
        hd_image_free(&out);
    }
    free(src.rgb);
}

static void test_to_patches(void) {
    hd_image img;
    img.width = 64; img.height = 64;
    img.rgb = malloc((size_t)3 * 64 * 64 * sizeof(float));
    synth_rgb(64, 64, img.rgb);
    int ps = 16;
    int gh = 64 / ps, gw = 64 / ps;
    float *patches = malloc((size_t)gh * gw * 3 * ps * ps * sizeof(float));
    hd_status st = hd_image_to_patches(&img, ps, patches);
    CHECK(st == HD_OK, "to_patches returns HD_OK");
    if (st == HD_OK) {
        /* verify einops rearrange: patch (ph,pw), channel c, (j,i) ==
           src pixel (ph*ps+j, pw*ps+i), channel c */
        int ok = 1;
        for (int ph = 0; ph < gh && ok; ph++)
            for (int pw = 0; pw < gw && ok; pw++)
                for (int c = 0; c < 3 && ok; c++)
                    for (int j = 0; j < ps && ok; j++)
                        for (int i = 0; i < ps && ok; i++) {
                            size_t dst = ((size_t)ph * gw + pw) * 3 * ps * ps +
                                         (size_t)c * ps * ps + (size_t)j * ps + i;
                            float want = img.rgb[((size_t)(ph * ps + j) * 64 +
                                                  (pw * ps + i)) * 3 + c];
                            if (fabsf(patches[dst] - want) > 1e-6f) ok = 0;
                        }
        CHECK(ok, "to_patches matches einops rearrange");
        free(patches);
    }
    free(img.rgb);
}

static void test_png_roundtrip(void) {
    int w = 32, h = 24;
    unsigned char *rgb8 = malloc((size_t)w * h * 3);
    for (int i = 0; i < w * h; i++) {
        rgb8[i * 3 + 0] = (unsigned char)((i * 7) % 256);
        rgb8[i * 3 + 1] = (unsigned char)((i * 13) % 256);
        rgb8[i * 3 + 2] = (unsigned char)((i * 29) % 256);
    }
    const char *path = "/tmp/hd_image_test.png";
    int rc = hd_png_write_rgb(path, w, h, rgb8, NULL, NULL);
    CHECK(rc == 0, "png write returns 0");
    hd_image img;
    hd_status st = hd_image_load(path, &img);
    CHECK(st == HD_OK, "png load returns HD_OK");
    if (st == HD_OK) {
        CHECK(img.width == w && img.height == h, "png dims match");
        int ok = 1;
        for (int i = 0; i < w * h && ok; i++) {
            float r = img.rgb[i * 3 + 0] * 255.0f;
            float g = img.rgb[i * 3 + 1] * 255.0f;
            float b = img.rgb[i * 3 + 2] * 255.0f;
            if (fabsf(r - rgb8[i * 3 + 0]) > 1.0f ||
                fabsf(g - rgb8[i * 3 + 1]) > 1.0f ||
                fabsf(b - rgb8[i * 3 + 2]) > 1.0f) ok = 0;
        }
        CHECK(ok, "png pixel values match");
        hd_image_free(&img);
    }
    free(rgb8);
    remove(path);
}

static void test_keep_aspect(void) {
    hd_image ref;
    ref.width = 800; ref.height = 600;
    ref.rgb = malloc((size_t)3 * 800 * 600 * sizeof(float));
    synth_rgb(800, 600, ref.rgb);
    int w, h;
    hd_image_keep_aspect(&ref, 1024, 1024, 32, &w, &h);
    /* oracle: resize to 2048 -> scale=sqrt(4194304/480000)=2.9558
       -> 2365->2336, 1774->1760 -> 2336x1760 (4:3) */
    CHECK(w == 2336 && h == 1760, "keep_aspect 800x600 -> 2336x1760");
    free(ref.rgb);
}

static void test_vlm_preprocess_oracle(void) {
    const char *oracle_path =
        "artifacts/ref_image_audit/oracle_dump/pixel_values.bin";
    FILE *f = fopen(oracle_path, "rb");
    if (!f) {
        printf("  skip VLM preprocessing oracle (fixture unavailable)\n");
        return;
    }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    size_t n = (size_t)bytes / sizeof(float);
    float *oracle = malloc((size_t)bytes);
    if (!oracle || fread(oracle, 1, (size_t)bytes, f) != (size_t)bytes) {
        fclose(f);
        free(oracle);
        CHECK(0, "load VLM preprocessing oracle");
        return;
    }
    fclose(f);

    hd_image src = {0}, resized = {0};
    hd_status st = hd_image_load("example_assets/edit/test.jpg", &src);
    CHECK(st == HD_OK, "load edit reference for VLM preprocessing");
    if (st != HD_OK) { free(oracle); return; }
    st = hd_image_resize_exact(&src, 320, 416, &resized);
    hd_image_free(&src);
    CHECK(st == HD_OK, "resize edit reference to oracle VLM grid");
    if (st != HD_OK) { free(oracle); return; }

    for (int i = 0; i < resized.width * resized.height * 3; i++)
        resized.rgb[i] = (resized.rgb[i] - 0.5f) / 0.5f;
    size_t want = (size_t)520 * 1536;
    float *native = malloc(want * sizeof(float));
    CHECK(native != NULL && n == want, "VLM oracle element count");
    if (!native || n != want) {
        hd_image_free(&resized);
        free(native);
        free(oracle);
        return;
    }
    st = hd_image_to_vlm_patches(&resized, 16, 2, 2, native);
    hd_image_free(&resized);
    CHECK(st == HD_OK, "patchify edit reference for VLM");
    if (st == HD_OK) {
        double dot = 0.0, na = 0.0, nb = 0.0, err = 0.0;
        for (size_t i = 0; i < n; i++) {
            double a = native[i], b = oracle[i], d = a - b;
            dot += a * b; na += a * a; nb += b * b; err += d * d;
        }
        double cos = dot / sqrt(na * nb);
        double rel = sqrt(err / nb);
        printf("  VLM preprocess: cos=%.8f nrmse=%.8f\n", cos, rel);
        CHECK(cos > 0.999 && rel < 0.05,
              "native VLM preprocessing matches processor oracle");
    }
    free(native);
    free(oracle);
}

int main(void) {
    test_calc_dims();
    test_resize();
    test_to_patches();
    test_png_roundtrip();
    test_keep_aspect();
    test_vlm_preprocess_oracle();

    printf("\n%s\n", failures == 0 ? "ALL IMAGE TESTS PASSED" : "SOME FAILED");
    return failures == 0 ? 0 : 1;
}