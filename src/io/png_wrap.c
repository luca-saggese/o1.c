#include <string.h>

#include "png_wrap.h"

#define PNG_IMPLEMENTATION
#include "png.h"

int hd_png_write_rgb(const char *path, int width, int height,
                     const unsigned char *rgb,
                     const char *text_keyword, const char *text_meta)
{
    if (path == NULL || rgb == NULL || width <= 0 || height <= 0) {
        return 1;
    }

    png_image *img = png_create(width, height, 3);
    if (img == NULL) {
        return 1;
    }

    size_t nbytes = (size_t)width * (size_t)height * 3u;
    memcpy(img->data, rgb, nbytes);

    int rc;
    if (text_keyword != NULL && text_meta != NULL) {
        rc = png_save_with_text(img, path, text_keyword, text_meta);
    } else {
        rc = png_save(img, path);
    }

    png_free(img);
    return rc == 0 ? 0 : 1;
}