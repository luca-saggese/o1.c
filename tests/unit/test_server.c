/*
 * Model-free unit tests for the OpenAI Images API server.
 *
 * The server translation unit is included directly (with O1_SERVER_TEST) so
 * the static parsers can be exercised without opening a socket or loading a
 * model. Nothing here touches CUDA or the weights.
 */

#define O1_SERVER_TEST 1
#include "../../src/server/o1_server.c"

#include <assert.h>

static int g_fail = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            g_fail++;                                                          \
        }                                                                      \
    } while (0)

static void test_base64(void) {
    buf b = {0};
    base64_encode(&b, (const uint8_t *)"", 0);
    CHECK(b.len == 0);
    buf_free(&b);

    memset(&b, 0, sizeof(b));
    base64_encode(&b, (const uint8_t *)"f", 1);
    CHECK(strcmp(b.ptr, "Zg==") == 0);
    buf_free(&b);

    memset(&b, 0, sizeof(b));
    base64_encode(&b, (const uint8_t *)"fo", 2);
    CHECK(strcmp(b.ptr, "Zm8=") == 0);
    buf_free(&b);

    memset(&b, 0, sizeof(b));
    base64_encode(&b, (const uint8_t *)"foo", 3);
    CHECK(strcmp(b.ptr, "Zm9v") == 0);
    buf_free(&b);

    memset(&b, 0, sizeof(b));
    base64_encode(&b, (const uint8_t *)"foobar", 6);
    CHECK(strcmp(b.ptr, "Zm9vYmFy") == 0);
    buf_free(&b);
}

static void test_json_escape(void) {
    buf b = {0};
    json_escape(&b, "a\"b\\c\nd\te\x01");
    CHECK(strcmp(b.ptr, "\"a\\\"b\\\\c\\nd\\te\\u0001\"") == 0);
    buf_free(&b);
}

static void test_parse_size(void) {
    int w = 0, h = 0;
    CHECK(parse_size("1024x1024", &w, &h) && w == 1024 && h == 1024);
    CHECK(parse_size("2048X1536", &w, &h) && w == 2048 && h == 1536);
    CHECK(!parse_size("1024", &w, &h));
    CHECK(!parse_size("1024x", &w, &h));
    CHECK(!parse_size("x1024", &w, &h));
    CHECK(!parse_size("1024x1024x2", &w, &h));
    CHECK(!parse_size("0x100", &w, &h));
    CHECK(!parse_size("-4x100", &w, &h));
    CHECK(!parse_size(NULL, &w, &h));
}

static void test_name_validation(void) {
    hd_mode m;
    CHECK(mode_from_name_checked("edit", &m) && m == HD_MODE_EDIT);
    CHECK(mode_from_name_checked("personalize", &m) && m == HD_MODE_PERSONALIZE);
    CHECK(mode_from_name_checked("multi-ref", &m) && m == HD_MODE_PERSONALIZE);
    CHECK(mode_from_name_checked("layout", &m) && m == HD_MODE_PERSONALIZE_LAYOUT);
    CHECK(!mode_from_name_checked("storyboard", &m));
    CHECK(!mode_from_name_checked("nonsense", &m));
    CHECK(!mode_from_name_checked(NULL, &m));

    hd_scheduler_kind s;
    CHECK(scheduler_from_name_checked("flash", &s) && s == HD_SCHED_FLASH);
    CHECK(scheduler_from_name_checked("flow_match", &s) && s == HD_SCHED_FLOW_MATCH);
    CHECK(scheduler_from_name_checked("default", &s) && s == HD_SCHED_DEFAULT);
    CHECK(!scheduler_from_name_checked("bogus", &s));
    CHECK(!scheduler_from_name_checked(NULL, &s));
}

static void test_http_reason(void) {
    CHECK(strcmp(http_reason(413), "Payload Too Large") == 0);
    CHECK(strcmp(http_reason(415), "Unsupported Media Type") == 0);
    CHECK(strcmp(http_reason(429), "Too Many Requests") == 0);
    CHECK(strcmp(http_reason(503), "Service Unavailable") == 0);
}

static void test_content_length(void) {
    const char *h = "POST /x HTTP/1.1\r\nContent-Length: 42\r\n\r\n";
    CHECK(content_length(h, strlen(h)) == 42);
    const char *bad = "POST /x HTTP/1.1\r\nContent-Length: abc\r\n\r\n";
    CHECK(content_length(bad, strlen(bad)) == -1);
    const char *neg = "POST /x HTTP/1.1\r\nContent-Length: -5\r\n\r\n";
    CHECK(content_length(neg, strlen(neg)) == -1);
    const char *none = "GET /x HTTP/1.1\r\n\r\n";
    CHECK(content_length(none, strlen(none)) == -1);
}

static void test_header_end(void) {
    const char *h = "GET / HTTP/1.1\r\nHost: x\r\n\r\nBODY";
    ssize_t e = header_end(h, strlen(h));
    CHECK(e == 27);
    CHECK(strncmp(h + e, "BODY", 4) == 0);
}

static void test_alias_array(void) {
    char **items = NULL;
    int n = 0;
    CHECK(parse_alias_array("[\"person\",\"shirt\"]", &items, &n));
    CHECK(n == 2);
    CHECK(strcmp(items[0], "person") == 0);
    CHECK(strcmp(items[1], "shirt") == 0);
    for (int i = 0; i < n; i++) free(items[i]);
    free(items);

    items = NULL;
    n = 0;
    CHECK(parse_alias_array("[]", &items, &n));
    CHECK(n == 0);
    free(items);

    items = NULL;
    n = 0;
    CHECK(!parse_alias_array("[\"a\",", &items, &n));
    CHECK(!parse_alias_array("{}", &items, &n));
}

static void test_json_body(void) {
    const char *body =
        "{\"prompt\":\"a cat\",\"model\":\"o1-dev\",\"n\":2,"
        "\"size\":\"1024x1024\",\"o1_seed\":12345678901234,"
        "\"o1_steps\":28,\"o1_scheduler\":\"flash\","
        "\"o1_guidance_scale\":1.5,\"o1_keep_original_aspect\":true,"
        "\"o1_reference_aliases\":[\"a\",\"b\"],\"unknown_field\":{\"x\":[1,2]},"
        "\"o1_exact_size\":false}";
    parsed_fields f;
    memset(&f, 0, sizeof(f));
    CHECK(parse_json_body(body, &f));
    CHECK(f.prompt && strcmp(f.prompt, "a cat") == 0);
    CHECK(f.model && strcmp(f.model, "o1-dev") == 0);
    CHECK(f.n == 2);
    CHECK(f.size && strcmp(f.size, "1024x1024") == 0);
    CHECK(f.has_seed && f.seed == 12345678901234ull);
    CHECK(f.has_steps && f.steps == 28);
    CHECK(f.has_scheduler && strcmp(f.scheduler, "flash") == 0);
    CHECK(f.has_guidance && f.guidance == 1.5);
    CHECK(f.has_keep_aspect && f.keep_aspect);
    CHECK(f.n_aliases == 2 && strcmp(f.aliases[1], "b") == 0);
    CHECK(f.has_exact_size && !f.exact_size);
    parsed_fields_free(&f);

    memset(&f, 0, sizeof(f));
    CHECK(!parse_json_body("{\"prompt\":", &f));
    parsed_fields_free(&f);

    memset(&f, 0, sizeof(f));
    CHECK(parse_json_body("{}", &f));
    parsed_fields_free(&f);
}

static void test_multipart(void) {
    const char *body =
        "--BOUND\r\n"
        "Content-Disposition: form-data; name=\"prompt\"\r\n"
        "\r\n"
        "hello world\r\n"
        "--BOUND\r\n"
        "Content-Disposition: form-data; name=\"image[]\"; filename=\"a.png\"\r\n"
        "Content-Type: image/png\r\n"
        "\r\n"
        "AAA\r\n"
        "--BOUND\r\n"
        "Content-Disposition: form-data; name=\"image[]\"; filename=\"b.png\"\r\n"
        "\r\n"
        "BBB\r\n"
        "--BOUND--\r\n";
    multipart_form form;
    CHECK(multipart_parse(body, strlen(body),
                          "multipart/form-data; boundary=BOUND", &form));
    CHECK(form.count == 3);
    const multipart_part *p = multipart_text(&form, "prompt");
    CHECK(p && p->len == 11 && strncmp(p->data, "hello world", 11) == 0);
    CHECK(form.parts[1].filename && strcmp(form.parts[1].filename, "a.png") == 0);
    CHECK(form.parts[1].len == 3 && strncmp(form.parts[1].data, "AAA", 3) == 0);
    CHECK(form.parts[2].len == 3 && strncmp(form.parts[2].data, "BBB", 3) == 0);
    multipart_form_free(&form);

    /* Missing boundary parameter must fail closed. */
    CHECK(!multipart_parse(body, strlen(body), "multipart/form-data", &form));
    multipart_form_free(&form);

    /* Truncated body (no closing delimiter) must fail closed. */
    const char *trunc =
        "--BOUND\r\n"
        "Content-Disposition: form-data; name=\"prompt\"\r\n"
        "\r\n"
        "hello";
    CHECK(!multipart_parse(trunc, strlen(trunc),
                           "multipart/form-data; boundary=BOUND", &form));
    multipart_form_free(&form);
}

static void test_resolution_snap(void) {
    int w = 0, h = 0;
    hd_resolution_snap(1024, 1024, &w, &h);
    CHECK(w == 2048 && h == 2048);
    hd_resolution_snap(2048, 2048, &w, &h);
    CHECK(w == 2048 && h == 2048);
}

static void test_model_ids(void) {
    CHECK(model_id_matches("dev", "hidream-o1-image-dev"));
    CHECK(model_id_matches("dev", "o1-dev"));
    CHECK(model_id_matches("dev", "dev"));
    CHECK(!model_id_matches("dev", "hidream-o1-image"));
    CHECK(model_id_matches("base", "hidream-o1-image"));
    CHECK(model_id_matches("base", "o1-base"));
    CHECK(!model_id_matches("base", "o1-dev"));
    CHECK(!model_id_matches("dev", NULL));
}

int main(void) {
    test_base64();
    test_json_escape();
    test_parse_size();
    test_name_validation();
    test_http_reason();
    test_content_length();
    test_header_end();
    test_alias_array();
    test_json_body();
    test_multipart();
    test_resolution_snap();
    test_model_ids();
    if (g_fail) {
        fprintf(stderr, "SERVER_UNIT_FAIL: %d\n", g_fail);
        return 1;
    }
    printf("SERVER_UNIT_OK\n");
    return 0;
}
