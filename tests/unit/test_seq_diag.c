/*
 * M1-post sequence manifest diagnostics gate (§56).
 *
 * Builds the canonical 64x64 T2I sequence (S=23, T=19, image_len=4) from
 * the frozen M1.4 input_ids (embedded below; verified against
 * artifacts/m1/golden/M1_V3_DEV_FORWARD_0/01_model_input) and asserts
 * hd_seq_diag returns 0 and prints the expected geometry, text tokens,
 * special-token indices, mask spans and workspace estimate.
 *
 * Self-contained: does not require the golden artifacts on disk.
 */

#include "hidream.h"
#include "sequence.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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

/* Frozen M1.4 golden input_ids (int64 [1,19]): encode_prompt(canonical)
 * + <|boi_token|>(151669) + <|tms_token|>(151673). */
static const int64_t k_golden_ids[T] = {
    151644, 872, 198, 64, 2518, 38835, 23011, 1212, 264, 40880,
    88758, 4916, 151645, 198, 151644, 77091, 198, 151669, 151673
};

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
    else printf("ok: %s\n", msg); \
} while (0)

int main(void) {
    hd_sequence seq;
    hd_status st = hd_seq_t2i(k_golden_ids, T, H, W, PATCH, IMG_TOKEN_ID,
                              VIDEO_TOKEN_ID, VIS_START_ID, TMS_TOKEN_ID,
                              TIMESTEP_TOKENS, SPATIAL_MERGE, FIX_POINT, &seq);
    CHECK(st == HD_OK, "hd_seq_t2i returns OK");
    if (st != HD_OK) { printf("  error: %s\n", hd_last_error()); return 1; }
    CHECK(seq.text_len == T && seq.image_len == 4 && seq.S == TOT_S,
          "sequence geometry S=23 T=19 image=4");

    /* capture hd_seq_diag stdout */
    char tmp[] = "build/diag_capture_XXXXXX";
    int fd = mkstemp(tmp);
    if (fd < 0) { printf("FAIL: mkstemp\n"); return 1; }
    fflush(stdout);
    int saved = dup(1);
    dup2(fd, 1);
    int rc = hd_seq_diag(&seq, "t2i-64");
    fflush(stdout);
    dup2(saved, 1);
    close(saved);
    lseek(fd, 0, SEEK_SET);
    char buf[65536];
    ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    remove(tmp);
    if (n < 0) { printf("FAIL: read capture\n"); return 1; }
    buf[n] = '\0';

    CHECK(rc == 0, "hd_seq_diag returns 0");
    CHECK(strstr(buf, "[seq:t2i-64] text_len=19 image_len=4 S=23 img_begin=19") != NULL,
          "diag prints text_len/image_len/S/img_begin");
    CHECK(strstr(buf, "text_tokens[19]=") != NULL &&
          strstr(buf, "151669 151673") != NULL,
          "diag prints text token ids incl boi+tms");
    CHECK(strstr(buf, "special_tokens:") != NULL &&
          strstr(buf, "boi@17") != NULL && strstr(buf, "tms@18") != NULL,
          "diag prints special-token indices");
    CHECK(strstr(buf, "mrope_sections=[24 20 20] total=64") != NULL,
          "diag prints MRoPE sections");
    CHECK(strstr(buf, "mask_spans: causal[0..17] full[18..22]") != NULL,
          "diag prints prediction-mask spans");
    CHECK(strstr(buf, "workspace_estimate: scores+probs=") != NULL,
          "diag prints workspace estimate");

    hd_sequence_free(&seq);

    printf("\n%d assertions passed, %d failed\n", 7 - failures, failures);
    if (failures) return 1;
    return 0;
}