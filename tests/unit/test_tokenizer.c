#include "hidream.h"
#include "tokenizer.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s (at %s:%d)\n", msg, __FILE__, __LINE__); \
        failures++; \
    } else { \
        printf("ok: %s\n", msg); \
    } \
} while (0)

/* Frozen canonical prompt + its exact token ids from
 * config/startup_manifest_dev.json (canonical_input_ids, 17 tokens). */
static const char *CANONICAL_PROMPT =
    "a red fox sits under a cherry blossom tree";
static const int FROZEN_IDS[17] = {
    151644, 872, 198, 64, 2518, 38835, 23011, 1212, 264,
    40880, 88758, 4916, 151645, 198, 151644, 77091, 198
};
#define N_FROZEN (sizeof(FROZEN_IDS) / sizeof(FROZEN_IDS[0]))

static void test_canonical(void) {
    int *ids = NULL; size_t cnt = 0;
    hd_status st = hd_tokenizer_encode_prompt(CANONICAL_PROMPT, &ids, &cnt);
    CHECK(st == HD_OK, "canonical prompt encode returns HD_OK");
    if (st != HD_OK) { printf("  err: %s\n", hd_last_error()); return; }
    CHECK(cnt == N_FROZEN, "canonical prompt -> 17 ids");
    if (cnt != N_FROZEN) {
        printf("  got %zu ids, want %zu\n", cnt, N_FROZEN);
        goto check_free;
    }
    for (size_t i = 0; i < N_FROZEN; i++) {
        if (ids[i] != FROZEN_IDS[i]) {
            failures++;
            printf("FAIL: token %zu: got %d, want %d\n", i, ids[i], FROZEN_IDS[i]);
        }
    }
    if (failures == 0) printf("ok: canonical prompt maps to frozen ids exactly\n");
check_free:
    hd_tokenizer_free_ids(ids);
}

static void test_determinism(void) {
    int *a = NULL, *b = NULL; size_t na = 0, nb = 0;
    hd_status sa = hd_tokenizer_encode_prompt(CANONICAL_PROMPT, &a, &na);
    hd_status sb = hd_tokenizer_encode_prompt(CANONICAL_PROMPT, &b, &nb);
    CHECK(sa == HD_OK && sb == HD_OK, "two encodes both HD_OK");
    CHECK(na == nb, "identical id counts");
    if (sa == HD_OK && sb == HD_OK) {
        int same = (na == nb);
        for (size_t i = 0; same && i < na; i++) if (a[i] != b[i]) same = 0;
        CHECK(same, "same prompt -> same ids (deterministic)");
    }
    hd_tokenizer_free_ids(a);
    hd_tokenizer_free_ids(b);
}

static void test_empty_prompt(void) {
    int *ids = NULL; size_t cnt = 0;
    hd_status st = hd_tokenizer_encode_prompt("", &ids, &cnt);
    CHECK(st == HD_OK, "empty prompt encode returns HD_OK");
    if (st == HD_OK) {
        /* Template "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n" */
        static const int want[8] = {
            151644, 872, 198, 151645, 198, 151644, 77091, 198
        };
        CHECK(cnt == 8, "empty prompt -> 8 template ids");
        if (cnt == 8) {
            int same = 1;
            for (size_t i = 0; i < 8; i++) if (ids[i] != want[i]) same = 0;
            CHECK(same, "empty prompt reproduces frozen template ids");
        }
    }
    hd_tokenizer_free_ids(ids);
}

static void test_special_tokens(void) {
    CHECK(hd_tokenizer_special_id("<|im_start|>") == 151644, "im_start id");
    CHECK(hd_tokenizer_special_id("<|im_end|>") == 151645, "im_end id");
    CHECK(hd_tokenizer_special_id("<|vision_start|>") == 151652, "vision_start id");
    CHECK(hd_tokenizer_special_id("<|vision_end|>") == 151653, "vision_end id");
    CHECK(hd_tokenizer_special_id("<|image_pad|>") == 151655, "image_pad id");
    CHECK(hd_tokenizer_special_id("<|video_pad|>") == 151656, "video_pad id");
    CHECK(hd_tokenizer_special_id("definitely_not_a_special") == -1, "unknown -> -1");
    CHECK(hd_tokenizer_special_id(NULL) == -1, "NULL -> -1");
}

static void test_identity(void) {
    const char *id = HD_TOK_IDENTITY;
    CHECK(id != NULL && strstr(id, "Qwen2Tokenizer") != NULL,
          "tokenizer identity string present");
}

static void test_template_builder(void) {
    char *tpl = NULL;
    hd_status st = hd_tokenizer_build_template("hi", &tpl);
    CHECK(st == HD_OK && tpl != NULL, "template builder HD_OK");
    if (st == HD_OK) {
        CHECK(strcmp(tpl, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n") == 0,
              "template string exact");
        free(tpl);
    }
}

static void test_teapot_full_vocab(void) {
    /* M1-post gate: full-vocab encode of the release-sanity prompt must be
       bit-exact with the frozen oracle (transformers AutoTokenizer on
       models/dev, apply_chat_template + encode(add_special_tokens=False)). */
    static const char *PROMPT =
        "A red ceramic teapot on a wooden table, next to a yellow lemon and a "
        "blue ceramic cup. Soft natural window light from the left, realistic "
        "photography, shallow depth of field, clean background, high detail.";
    static const int FROZEN[51] = {
        151644, 872, 198, 32, 2518, 42024, 1013, 89901, 389, 264, 22360, 1965,
        11, 1790, 311, 264, 13753, 29464, 323, 264, 6303, 42024, 10525, 13,
        24079, 5810, 3241, 3100, 504, 279, 2115, 11, 25489, 23751, 11, 25600,
        7990, 315, 2070, 11, 4240, 4004, 11, 1550, 7716, 13, 151645, 198,
        151644, 77091, 198
    };
    int *ids = NULL; size_t cnt = 0;
    hd_status st = hd_tokenizer_encode_prompt(PROMPT, &ids, &cnt);
    CHECK(st == HD_OK, "teapot prompt encode returns HD_OK");
    if (st != HD_OK) { printf("  err: %s\n", hd_last_error()); return; }
    CHECK(cnt == 51, "teapot prompt -> 51 ids");
    if (cnt != 51) {
        printf("  got %zu ids, want 51\n", cnt);
        goto check_free;
    }
    for (size_t i = 0; i < 51; i++) {
        if (ids[i] != FROZEN[i]) {
            failures++;
            printf("FAIL: teapot token %zu: got %d, want %d\n", i, ids[i], FROZEN[i]);
        }
    }
    if (failures == 0) printf("ok: teapot prompt maps to frozen oracle ids exactly\n");
check_free:
    hd_tokenizer_free_ids(ids);
}

int main(void) {
    printf("tokenizer identity: %s\n", HD_TOK_IDENTITY);
    printf("special tokens: im_start=%d im_end=%d vision_start=%d vision_end=%d "
           "image_pad=%d video_pad=%d\n",
           HD_TOK_IM_START, HD_TOK_IM_END, HD_TOK_VISION_START, HD_TOK_VISION_END,
           HD_TOK_IMAGE_PAD, HD_TOK_VIDEO_PAD);

    test_identity();
    test_special_tokens();
    test_template_builder();
    test_canonical();
    test_teapot_full_vocab();
    test_determinism();
    test_empty_prompt();

    if (failures == 0) {
        printf("ALL TOKENIZER TESTS PASSED\n");
        return 0;
    }
    printf("%d TOKENIZER TEST FAILURE(S)\n", failures);
    return 1;
}
