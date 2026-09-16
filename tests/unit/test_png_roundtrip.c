/*
 * Round-trip test for the vendored single-header PNG encoder.
 *
 * Generates a 64x64 RGB gradient, writes it through hd_png_write_rgb
 * (with a tEXt chunk), reads the file back and validates the PNG
 * signature, the IHDR fields (64x64, 8-bit, color type 2 = RGB) and a
 * sane file size. No external PNG library is used.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "png_wrap.h"

#define TEST_W 64
#define TEST_H 64
#define TEST_PATH "/tmp/png_roundtrip_test.png"

static int fail(const char *reason)
{
    fprintf(stderr, "PNG_ROUNDTRIP_FAIL: %s\n", reason);
    return 1;
}

int main(void)
{
    unsigned char *rgb = malloc((size_t)TEST_W * TEST_H * 3u);
    if (rgb == NULL) {
        return fail("out of memory");
    }

    /* Deterministic gradient: r=x, g=y, b=(x+y)%256. */
    for (int y = 0; y < TEST_H; y++) {
        for (int x = 0; x < TEST_W; x++) {
            unsigned char *p = rgb + ((size_t)y * TEST_W + (size_t)x) * 3u;
            p[0] = (unsigned char)x;
            p[1] = (unsigned char)y;
            p[2] = (unsigned char)((x + y) % 256);
        }
    }

    const char *meta = "{\"model\":\"qwen3-vl\",\"step\":0,\"seed\":42}";
    if (hd_png_write_rgb(TEST_PATH, TEST_W, TEST_H, rgb, "test", meta) != 0) {
        free(rgb);
        return fail("hd_png_write_rgb failed");
    }
    free(rgb);

    FILE *f = fopen(TEST_PATH, "rb");
    if (f == NULL) {
        return fail("cannot open written file");
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return fail("fseek failed");
    }
    long size = ftell(f);
    if (size < 0) {
        fclose(f);
        return fail("ftell failed");
    }
    if (size <= 100) {
        fclose(f);
        return fail("file too small");
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return fail("fseek failed");
    }

    unsigned char *buf = malloc((size_t)size);
    if (buf == NULL) {
        fclose(f);
        return fail("out of memory");
    }
    if (fread(buf, 1, (size_t)size, f) != (size_t)size) {
        free(buf);
        fclose(f);
        return fail("short read");
    }
    fclose(f);

    /* PNG signature: 137 80 78 71 13 10 26 10 */
    static const unsigned char sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (memcmp(buf, sig, 8) != 0) {
        free(buf);
        return fail("bad PNG signature");
    }

    /* IHDR: length(4) "IHDR"(4) width(4) height(4) bitdepth(1) colortype(1) */
    if (size < 8 + 25) {
        free(buf);
        return fail("file too short for IHDR");
    }
    if (memcmp(buf + 12, "IHDR", 4) != 0) {
        free(buf);
        return fail("missing IHDR chunk");
    }
    unsigned int w = ((unsigned int)buf[16] << 24) | ((unsigned int)buf[17] << 16) |
                     ((unsigned int)buf[18] << 8) | (unsigned int)buf[19];
    unsigned int h = ((unsigned int)buf[20] << 24) | ((unsigned int)buf[21] << 16) |
                     ((unsigned int)buf[22] << 8) | (unsigned int)buf[23];
    if (w != TEST_W || h != TEST_H) {
        free(buf);
        return fail("IHDR dimensions mismatch");
    }
    if (buf[24] != 8) {
        free(buf);
        return fail("IHDR bit depth is not 8");
    }
    if (buf[25] != 2) {
        free(buf);
        return fail("IHDR color type is not RGB(2)");
    }

    free(buf);
    printf("PNG_ROUNDTRIP_OK\n");
    return 0;
}