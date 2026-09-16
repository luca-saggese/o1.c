/*
 * Decode parity gate: hd_decode_to_rgb must reproduce the M1.7 test's
 * decode_z semantics exactly (docs/M1_7_TOLERANCES.md section 4).
 *
 * Builds a synthetic latent with a known pattern and checks the output
 * against a reference implementation of the same loop.
 */
#include "decode.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
    else printf("ok: %s\n", msg); \
} while (0)

/* Reference decode (identical loop to test_m1_7_dev.c decode_z). */
static void ref_decode(const float *zf, int grid_h, int grid_w, int patch,
                       int channels, unsigned char *out) {
    const int img_tokens = grid_h * grid_w;
    const int feats = channels * patch * patch;
    const int h = grid_h * patch;
    const int w = grid_w * patch;
    for (int tok = 0; tok < img_tokens; tok++) {
        for (int c = 0; c < channels; c++) {
            for (int p1 = 0; p1 < patch; p1++) {
                for (int p2 = 0; p2 < patch; p2++) {
                    float v = (zf[(size_t)tok * feats + (size_t)c * patch * patch +
                                  (size_t)p1 * patch + p2] + 1.0f) / 2.0f;
                    float px = roundf(v * 255.0f);
                    if (px < 0.0f) px = 0.0f;
                    if (px > 255.0f) px = 255.0f;
                    out[((size_t)(tok / grid_w * patch + p1) * w +
                         (size_t)(tok % grid_w * patch + p2)) * (size_t)channels + (size_t)c] =
                        (unsigned char)px;
                }
            }
        }
    }
}

int main(void) {
    /* 64x64 image: 2x2 grid, patch 32, 3 channels. */
    const int grid_h = 2, grid_w = 2, patch = 32, channels = 3;
    const int img_tokens = grid_h * grid_w;
    const int feats = channels * patch * patch;
    const int h = grid_h * patch, w = grid_w * patch;

    float *z = malloc((size_t)img_tokens * feats * sizeof(float));
    unsigned char *got = malloc((size_t)h * w * 3);
    unsigned char *ref = malloc((size_t)h * w * 3);
    if (!z || !got || !ref) { printf("FAIL: oom\n"); return 1; }

    /* Deterministic pattern covering [-1,1], edges, and out-of-range. */
    for (int i = 0; i < img_tokens * feats; i++) {
        int tok = i / feats, f = i % feats;
        float v = (float)((tok * 7 + f * 13) % 101) / 50.0f - 1.0f;
        if (f == 0) v = -1.0f;
        if (f == 1) v = 1.0f;
        if (f == 2) v = -2.0f;   /* clamps to 0 */
        if (f == 3) v = 2.0f;    /* clamps to 255 */
        z[i] = v;
    }

    hd_decode_to_rgb(z, grid_h, grid_w, patch, channels, got);
    ref_decode(z, grid_h, grid_w, patch, channels, ref);

    int same = memcmp(got, ref, (size_t)h * w * 3) == 0;
    CHECK(same, "decode matches reference loop byte-for-byte");

    /* Spot checks: output is RGB-packed; feature f maps to pixel offset f*3.
   f=0 (ch0 p2=0) -> out[0], f=1 -> out[3], f=2 -> out[6], f=3 -> out[9]. */
    CHECK(got[0] == 0, "latent -1 decodes to 0");
    CHECK(got[3] == 255, "latent +1 decodes to 255");
    CHECK(got[6] == 0, "latent -2 clamps to 0");
    CHECK(got[9] == 255, "latent +2 clamps to 255");

    /* 1024x1024 geometry: 32x32 grid, patch 32. */
    const int gh = 32, gw = 32;
    const int n1024 = gh * gw * feats;
    float *z1024 = malloc((size_t)n1024 * sizeof(float));
    unsigned char *img1024 = malloc((size_t)(gh * patch) * (gw * patch) * 3);
    if (!z1024 || !img1024) { printf("FAIL: oom\n"); return 1; }
    for (int i = 0; i < n1024; i++) z1024[i] = 0.0f;   /* mid-gray */
    hd_decode_to_rgb(z1024, gh, gw, patch, channels, img1024);
    CHECK(img1024[0] == 127 || img1024[0] == 128, "1024x1024 mid-gray decodes to ~127");
    CHECK(img1024[(size_t)(1024 * 1024 - 1) * 3] == 127 ||
          img1024[(size_t)(1024 * 1024 - 1) * 3] == 128,
          "1024x1024 last pixel decodes");

    free(z); free(got); free(ref); free(z1024); free(img1024);
    printf("\n%d assertions passed, %d failed\n", failures == 0 ? 6 : 6 - failures, failures);
    return failures ? 1 : 0;
}