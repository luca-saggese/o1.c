#include "decode.h"

#include <math.h>

void hd_decode_to_rgb(const float *z_f32, int grid_h, int grid_w, int patch,
                      int channels, unsigned char *out) {
    const int img_tokens = grid_h * grid_w;
    const int feats = channels * patch * patch;
    const int w = grid_w * patch;
    for (int tok = 0; tok < img_tokens; tok++) {
        const int tok_row = tok / grid_w;
        const int tok_col = tok % grid_w;
        for (int c = 0; c < channels; c++) {
            for (int p1 = 0; p1 < patch; p1++) {
                for (int p2 = 0; p2 < patch; p2++) {
                    float v = (z_f32[(size_t)tok * feats + (size_t)c * patch * patch +
                                     (size_t)p1 * patch + p2] + 1.0f) / 2.0f;
                    float px = roundf(v * 255.0f);
                    if (px < 0.0f) px = 0.0f;
                    if (px > 255.0f) px = 255.0f;
                    out[((size_t)(tok_row * patch + p1) * w +
                         (size_t)(tok_col * patch + p2)) * (size_t)channels + (size_t)c] =
                        (unsigned char)px;
                }
            }
        }
    }
}