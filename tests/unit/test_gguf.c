/*
 * M3: GGUF reader validation.
 *
 * Opens a GGUF pack produced by tools/hidream_convert.py and checks:
 *   - header/metadata parse (alignment, profile, revision, num_layers)
 *   - tensor table integrity (count, names, offsets, sizes)
 *   - payload round-trip: read a few tensors and compare against the
 *     original safetensors payloads (F32 cast to BF16).
 */

#include "gguf.h"
#include "safetensors.h"
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        printf("FAIL: " __VA_ARGS__); \
        printf("\n"); \
        fails++; \
    } \
} while (0)

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: %s <model.gguf> [safetensors_dir]\n", argv[0]);
        return 2;
    }
    const char *gguf_path = argv[1];
    const char *st_dir = argc > 2 ? argv[2] : NULL;

    hd_gguf_file f;
    hd_status st = hd_gguf_open(gguf_path, &f);
    if (st != HD_OK) {
        printf("FAIL: hd_gguf_open: %s\n", hd_gguf_last_error());
        return 1;
    }
    printf("GGUF: %lld tensors, alignment=%llu, payload=%llu bytes\n",
           (long long)f.n_tensors, (unsigned long long)f.alignment,
           (unsigned long long)f.payload_bytes);
    printf("  arch=%s profile=%s revision=%s dtype=%s layers=%lld\n",
           f.arch ? f.arch : "?", f.profile ? f.profile : "?",
           f.revision ? f.revision : "?", f.dtype ? f.dtype : "?",
           (long long)f.num_layers);
    printf("  name=%s variant=%s quantization=%s layout_version=%lld\n",
           f.name ? f.name : "?", f.variant ? f.variant : "?",
           f.quantization ? f.quantization : "?",
           (long long)f.layout_version);

    CHECK(f.alignment == 256, "alignment %llu != 256", (unsigned long long)f.alignment);
    CHECK(f.n_tensors > 0, "no tensors");
    CHECK(f.arch && strcmp(f.arch, "hidream_o1") == 0, "arch mismatch");
    CHECK(f.profile && (!strcmp(f.profile, "dev") || !strcmp(f.profile, "base")),
          "profile mismatch");
    CHECK(f.dtype && strcmp(f.dtype, "bf16") == 0, "dtype mismatch");
    CHECK(f.num_layers == 36, "num_layers %lld != 36", (long long)f.num_layers);
    CHECK(f.name && f.name[0], "general.name missing");
    /* variant/quantization are optional in legacy packs: the reader falls
     * back to profile/bf16. When present they must be non-empty. */
    CHECK(!f.variant || f.variant[0], "hidream.variant present but empty");
    CHECK(!f.quantization || f.quantization[0], "hidream.quantization present but empty");
    CHECK(f.layout_version > 0, "hidream.layout_version missing");

    /* Offsets must be 256-aligned and monotonically increasing. */
    uint64_t prev_end = 0;
    for (int64_t i = 0; i < f.n_tensors; i++) {
        const hd_gguf_tensor *t = &f.tensors[i];
        CHECK(t->offset % 256 == 0, "tensor %s offset %llu not 256-aligned",
              t->name, (unsigned long long)t->offset);
        CHECK(t->offset >= prev_end, "tensor %s offset %llu < prev end %llu",
              t->name, (unsigned long long)t->offset,
              (unsigned long long)prev_end);
        prev_end = t->offset + t->nbytes;
    }

    /* Payload round-trip against safetensors (if dir given). */
    if (st_dir) {
        hd_st_index idx;
        st = hd_st_index_load(st_dir, &idx);
        if (st != HD_OK) {
            printf("FAIL: hd_st_index_load: %s\n", hd_st_last_error());
            return 1;
        }
        int checked = 0;
        for (int64_t i = 0; i < f.n_tensors && checked < 8; i++) {
            const hd_gguf_tensor *gt = &f.tensors[i];
            const hd_st_tensor *stt = hd_st_index_find(&idx, gt->name);
            if (!stt) continue;
            uint8_t *gguf_buf = malloc((size_t)gt->nbytes);
            uint8_t *st_buf = malloc((size_t)stt->nbytes);
            if (!gguf_buf || !st_buf) { printf("oom\n"); return 1; }
            st = hd_gguf_read_tensor(&f, gt, gguf_buf);
            CHECK(st == HD_OK, "read %s: %s", gt->name, hd_gguf_last_error());
            st = hd_st_read_tensor(&idx, stt, st_buf);
            CHECK(st == HD_OK, "st read %s", gt->name);
            if (st == HD_OK) {
                /* F32 source -> BF16 in GGUF. */
                if (stt->dtype == HD_DTYPE_F32) {
                    uint8_t *cast = malloc((size_t)gt->nbytes);
                    hd_f32_buf_to_bf16((const float *)st_buf, cast,
                                       (size_t)stt->numel);
                    int same = memcmp(gguf_buf, cast, (size_t)gt->nbytes) == 0;
                    CHECK(same, "tensor %s payload mismatch (F32->BF16)", gt->name);
                    free(cast);
                } else {
                    int same = memcmp(gguf_buf, st_buf, (size_t)gt->nbytes) == 0;
                    CHECK(same, "tensor %s payload mismatch", gt->name);
                }
            }
            free(gguf_buf);
            free(st_buf);
            checked++;
        }
        hd_st_index_free(&idx);
        printf("  round-trip checked %d tensors\n", checked);
    }

    hd_gguf_close(&f);
    if (fails) {
        printf("GGUF TEST: %d FAILURES\n", fails);
        return 1;
    }
    printf("GGUF TEST: PASS\n");
    return 0;
}