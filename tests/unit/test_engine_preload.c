/*
 * M1-post resident engine: preload lifecycle gate (spec section 32.2).
 *
 * Proves that the expensive, shape-independent part of a generation --
 * loading the device weights and resolving the forward/vision bindings --
 * happens exactly ONCE per engine, and that two sequential generations
 * against one open engine do not reload the model.
 *
 * Usage:
 *   test_engine_preload <model_dir_or_gguf> [device_id]
 *
 * The test is intentionally cheap: it runs two 1-step generations at a
 * small resolution. The point is the load count, not image quality.
 */

#include "engine.h"
#include "request.h"

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

static hd_status run_one(hd_generation_engine *e, const char *prompt,
                         uint64_t seed, unsigned char **rgb, int *w, int *h) {
    hd_generation_request req;
    memset(&req, 0, sizeof(req));
    /* Mirror the CLI: set the "not set" sentinels BEFORE hd_request_defaults,
     * which only fills shift/guidance when they are negative. Leaving shift
     * at 0.0 yields a degenerate schedule whose output ignores seed/prompt. */
    req.prompt = prompt;
    req.profile = "dev";
    req.mode = HD_MODE_T2I;
    req.width = 1024;
    req.height = 1024;
    req.steps = 2;
    req.seed = seed;
    req.guidance_scale = -1.0f;
    req.shift = -1.0f;
    req.scheduler = HD_SCHED_DEFAULT;
    hd_request_defaults(&req);
    return hd_generation_engine_generate(e, &req, rgb, w, h);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: %s <model_dir_or_gguf> [device_id]\n", argv[0]);
        return 2;
    }
    const char *model_path = argv[1];
    int device_id = argc > 2 ? atoi(argv[2]) : 0;

    hd_generation_engine_options opts = {0};
    opts.profile = "dev";
    opts.model_path = model_path;
    opts.device_id = device_id;

    hd_generation_engine *e = hd_generation_engine_open(&opts);
    if (!e) {
        printf("FAIL: engine open: %s\n", hd_last_error());
        return 1;
    }

    /* Gate 1: exactly one weight load and one binding resolve after open. */
    CHECK(hd_generation_engine_load_count(e) == 1,
          "load_count after open = %d, expected 1",
          hd_generation_engine_load_count(e));
    CHECK(hd_generation_engine_forward_resolve_count(e) == 1,
          "forward_resolve_count after open = %d, expected 1",
          hd_generation_engine_forward_resolve_count(e));
    CHECK(hd_generation_engine_vision_resolve_count(e) == 1,
          "vision_resolve_count after open = %d, expected 1",
          hd_generation_engine_vision_resolve_count(e));
    CHECK(hd_generation_engine_request_count(e) == 0,
          "request_count after open = %d, expected 0",
          hd_generation_engine_request_count(e));

    /* Gate 2: two sequential generations, still one load. */
    unsigned char *rgb1 = NULL, *rgb2 = NULL;
    int w1 = 0, h1 = 0, w2 = 0, h2 = 0;

    hd_status st = run_one(e, "a red apple on a table", 42, &rgb1, &w1, &h1);
    CHECK(st == HD_OK, "generation 1: %s", hd_last_error());
    CHECK(rgb1 != NULL && w1 == 1024 && h1 == 1024,
          "generation 1 output %dx%d", w1, h1);

    CHECK(hd_generation_engine_load_count(e) == 1,
          "load_count after generation 1 = %d, expected 1",
          hd_generation_engine_load_count(e));
    CHECK(hd_generation_engine_request_count(e) == 1,
          "request_count after generation 1 = %d, expected 1",
          hd_generation_engine_request_count(e));

    st = run_one(e, "a blue cup on a desk", 43, &rgb2, &w2, &h2);
    CHECK(st == HD_OK, "generation 2: %s", hd_last_error());
    CHECK(rgb2 != NULL && w2 == 1024 && h2 == 1024,
          "generation 2 output %dx%d", w2, h2);

    CHECK(hd_generation_engine_load_count(e) == 1,
          "load_count after generation 2 = %d, expected 1",
          hd_generation_engine_load_count(e));
    CHECK(hd_generation_engine_forward_resolve_count(e) == 1,
          "forward_resolve_count after generation 2 = %d, expected 1",
          hd_generation_engine_forward_resolve_count(e));
    CHECK(hd_generation_engine_vision_resolve_count(e) == 1,
          "vision_resolve_count after generation 2 = %d, expected 1",
          hd_generation_engine_vision_resolve_count(e));
    CHECK(hd_generation_engine_request_count(e) == 2,
          "request_count after generation 2 = %d, expected 2",
          hd_generation_engine_request_count(e));

    /* Different seeds must produce different images (no state crossover). */
    if (rgb1 && rgb2) {
        size_t n = (size_t)w1 * h1 * 3;
        CHECK(memcmp(rgb1, rgb2, n) != 0,
              "two different seeds produced identical RGB");
    }

    free(rgb1);
    free(rgb2);
    hd_generation_engine_close(e);

    if (fails == 0) {
        printf("ENGINE_PRELOAD_OK: 2 generations, 1 weight load, "
               "1 forward resolve, 1 vision resolve\n");
        return 0;
    }
    printf("ENGINE_PRELOAD_FAIL: %d check(s) failed\n", fails);
    return 1;
}