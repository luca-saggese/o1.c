#ifndef HD_PNG_WRAP_H
#define HD_PNG_WRAP_H

/*
 * Thin wrapper around the vendored single-header PNG encoder (src/io/png.h).
 * Exposes a single entry point for writing an RGB image plus an optional
 * tEXt chunk (keyword/text). text_keyword/text_meta may be NULL to omit the
 * text chunk.
 */

/* Writes a width*height RGB image (3 channels, 8-bit) to path.
 * Returns 0 on success, nonzero on failure. */
int hd_png_write_rgb(const char *path, int width, int height,
                     const unsigned char *rgb,
                     const char *text_keyword, const char *text_meta);

/* Encodes a width*height RGB image (3 channels, 8-bit) to a freshly
 * malloc'd PNG byte buffer (no filesystem access). On success *out points
 * to the PNG bytes and *out_len to their length; the caller frees *out.
 * Returns 0 on success, nonzero on failure. */
int hd_png_encode_rgb(const unsigned char *rgb, int width, int height,
                      unsigned char **out, size_t *out_len);

#endif /* HD_PNG_WRAP_H */