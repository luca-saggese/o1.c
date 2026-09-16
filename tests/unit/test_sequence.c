/*
 * M1-post sequence builder gate: byte-for-byte parity with the frozen M1.4
 * fixture for the canonical 64x64 T2I prompt.
 *
 * Rebuilds the T2I sequence (S=23, T=19, image_len=4) from the same
 * input_ids the fixture used and compares pos_f32 (float32), mask (bf16)
 * and vinput_mask (int64) bytes against artifacts/m1/golden/
 * M1_V3_DEV_FORWARD_0/inputs.bin. A mismatch at this gate blocks any
 * production run built on the unified sequence builder.
 */

#include "hidream.h"
#include "sequence.h"
#include "json.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define M14_DIR   "artifacts/m1/golden/M1_V3_DEV_FORWARD_0"
#define T  19
#define TOT_S 23
#define H  64
#define W  64
#define PATCH 32
#define FIX_POINT 4096

#define IMG_TOKEN_ID   151655
#define VIDEO_TOKEN_ID 151656
#define VIS_START_ID   151652
#define TMS_TOKEN_ID   151673
#define TIMESTEP_TOKENS 1
#define SPATIAL_MERGE 1   /* oracle T2I calls with spatial_merge_size=1 */

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
    else printf("ok: %s\n", msg); \
} while (0)

static void *read_file(const char *path, size_t *out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    void *buf = malloc((size_t)sz + 1);
    fseek(f, 0, SEEK_SET);
    fread(buf, 1, (size_t)sz, f);
    fclose(f);
    ((char *)buf)[sz] = '\0';
    *out = (size_t)sz;
    return buf;
}

int main(void) {
    /* input_ids from M1.4 golden 01_model_input (int64 [1,19]) */
    size_t msz = 0;
    void *meta_b = read_file(M14_DIR "/M1_V3_DEV_FORWARD_0.json", &msz);
    if (!meta_b) { printf("FAIL: cannot load golden layout\n"); return 1; }
    const char *jerr = NULL;
    hd_json *meta = hd_json_parse((const char *)meta_b, &jerr);
    free(meta_b);
    if (!meta) { printf("FAIL: parse golden layout: %s\n", jerr ? jerr : "?"); return 1; }
    const hd_json *arr = hd_json_get(meta, "tensors");
    int64_t off_in = -1;
    size_t n = arr ? hd_json_array_len(arr) : 0;
    for (size_t i = 0; i < n; i++) {
        const hd_json *t = hd_json_array_at(arr, i);
        const char *nm = hd_json_string(hd_json_get(t, "name"));
        if (nm && !strcmp(nm, "01_model_input"))
            off_in = hd_json_int(hd_json_get(t, "offset"), -1);
    }
    hd_json_free(meta);
    if (off_in < 0) { printf("FAIL: golden missing 01_model_input\n"); return 1; }

    size_t bsz = 0;
    void *bin = read_file(M14_DIR "/M1_V3_DEV_FORWARD_0.bin", &bsz);
    if (!bin) { printf("FAIL: golden bin not found\n"); return 1; }
    int64_t ids[T];
    memcpy(ids, (const char *)bin + off_in, T * sizeof(int64_t));
    free(bin);

    hd_sequence seq;
    hd_status st = hd_seq_t2i(ids, T, H, W, PATCH, IMG_TOKEN_ID,
                              VIDEO_TOKEN_ID, VIS_START_ID, TMS_TOKEN_ID,
                              TIMESTEP_TOKENS, SPATIAL_MERGE, FIX_POINT, &seq);
    CHECK(st == HD_OK, "hd_seq_t2i returns OK");
    if (st != HD_OK) { printf("  error: %s\n", hd_last_error()); return 1; }
    CHECK(seq.text_len == T && seq.image_len == 4 && seq.S == TOT_S,
          "sequence geometry S=23 T=19 image=4");
    hd_seq_diag(&seq, "t2i-64");

    /* ---- golden inputs.bin ---- */
    size_t isz = 0;
    void *inputs = read_file(M14_DIR "/inputs.bin", &isz);
    if (!inputs) { printf("FAIL: inputs.bin not found\n"); return 1; }
    size_t imsz = 0;
    void *imb = read_file(M14_DIR "/inputs.json", &imsz);
    if (!imb) { printf("FAIL: inputs.json not found\n"); free(inputs); return 1; }
    const char *ierr = NULL;
    hd_json *imeta = hd_json_parse((const char *)imb, &ierr);
    free(imb);
    if (!imeta) { printf("FAIL: parse inputs.json: %s\n", ierr ? ierr : "?"); free(inputs); return 1; }
    const hd_json *iarr = hd_json_get(imeta, "tensors");
    int64_t off_pos = 0, off_mask = 0, off_vin = 0, off_vm = 0;
    n = iarr ? hd_json_array_len(iarr) : 0;
    for (size_t i = 0; i < n; i++) {
        const hd_json *t = hd_json_array_at(iarr, i);
        const char *nm = hd_json_string(hd_json_get(t, "name"));
        int64_t off = hd_json_int(hd_json_get(t, "offset"), 0);
        if (!nm) continue;
        if (!strcmp(nm, "pos_f32")) off_pos = off;
        else if (!strcmp(nm, "mask")) off_mask = off;
        else if (!strcmp(nm, "vinputs")) off_vin = off;
        else if (!strcmp(nm, "vinput_mask")) off_vm = off;
    }
    hd_json_free(imeta);

    /* pos_f32 [3,1,23] = 276 bytes */
    int pos_ok = memcmp(seq.pos_f32, (const char *)inputs + off_pos, 276) == 0;
    printf("  pos_f32[0..2]  = %.0f %.0f %.0f (golden %.0f %.0f %.0f)\n",
           seq.pos_f32[0], seq.pos_f32[1], seq.pos_f32[2],
           ((const float *)((const char *)inputs + off_pos))[0],
           ((const float *)((const char *)inputs + off_pos))[1],
           ((const float *)((const char *)inputs + off_pos))[2]);
    printf("  pos_f32 img d0 = %.0f (golden %.0f)\n",
           seq.pos_f32[3 * TOT_S - 1],
           ((const float *)((const char *)inputs + off_pos))[3 * TOT_S - 1]);
    CHECK(pos_ok, "pos_f32 byte-identical to M1.4 fixture");

    /* mask [1,1,23,23] bf16 = 1058 bytes */
    int mask_ok = memcmp(seq.mask_bf16, (const char *)inputs + off_mask, 1058) == 0;
    CHECK(mask_ok, "mask byte-identical to M1.4 fixture");

    /* vinput_mask derived from seq.vinput_mask vs golden int64 [23] */
    /* golden vinput_mask: 0..18 -> 0, 19..22 -> 1 */
    int vm_ok = 1;
    for (int i = 0; i < TOT_S; i++) {
        int64_t gold = 0;
        memcpy(&gold, (const char *)inputs + off_vm + i * 8, 8);
        if ((int64_t)seq.vinput_mask[i] != gold) { vm_ok = 0; break; }
    }
    CHECK(vm_ok, "vinput_mask matches golden (19..22 image rows)");

    hd_sequence_free(&seq);
    free(inputs);

    printf("\n%d assertions passed, %d failed\n", (pos_ok && mask_ok && vm_ok) ? 4 : 0, failures);
    if (failures) return 1;
    return 0;
}