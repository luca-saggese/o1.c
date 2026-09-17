/*
 * test_refiner.c — mocked-endpoint tests for the native prompt-refiner
 * clients (src/runtime/refiner.c).
 *
 * Spins up a tiny TCP listener on 127.0.0.1:0 (ephemeral port), accepts one
 * connection, reads the HTTP request, asserts the expected path/body fields,
 * and replies with a canned OpenAI-compatible JSON response. Also exercises
 * the error paths: connection refused -> HD_ERR_IO, malformed JSON ->
 * HD_ERR_PARSE.
 */

#define _POSIX_C_SOURCE 200809L

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "refiner.h"

static int g_failures = 0;

#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg);    \
            g_failures++;                                                    \
        }                                                                    \
    } while (0)

/* ── Mock server ────────────────────────────────────────────────────────── */

typedef struct {
    int fd;
    int port;
    char req[16384]; /* last served request, for assertions */
} mock_server;

static int mock_start(mock_server *srv) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0; /* ephemeral */
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    socklen_t alen = sizeof(addr);
    if (getsockname(fd, (struct sockaddr *)&addr, &alen) != 0) {
        close(fd);
        return -1;
    }
    if (listen(fd, 1) != 0) {
        close(fd);
        return -1;
    }
    srv->fd = fd;
    srv->port = ntohs(addr.sin_port);
    return 0;
}

static void mock_stop(mock_server *srv) {
    if (srv->fd >= 0) close(srv->fd);
    srv->fd = -1;
}

/* Accept one connection, read the full request, write `reply`, close. */
static int mock_serve(mock_server *srv, const char *reply, char *req_buf,
                      size_t req_cap) {
    int cfd = accept(srv->fd, NULL, NULL);
    if (cfd < 0) return -1;

    size_t len = 0;
    while (len + 1 < req_cap) {
        ssize_t n = recv(cfd, req_buf + len, req_cap - len - 1, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            close(cfd);
            return -1;
        }
        if (n == 0) break;
        len += (size_t)n;
        /* stop once the full body has arrived (client keeps the socket open
         * until it has read our reply, so EOF will never come) */
        req_buf[len] = '\0';
        const char *hdr_end = strstr(req_buf, "\r\n\r\n");
        if (hdr_end) {
            const char *cl = strstr(req_buf, "Content-Length:");
            if (cl && cl < hdr_end) {
                long body_len = strtol(cl + strlen("Content-Length:"), NULL, 10);
                size_t body_off = (size_t)(hdr_end + 4 - req_buf);
                if (len >= body_off + (size_t)body_len) break;
            }
        }
    }
    req_buf[len] = '\0';

    size_t reply_len = strlen(reply);
    size_t sent = 0;
    while (sent < reply_len) {
        ssize_t n = send(cfd, reply + sent, reply_len - sent, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            close(cfd);
            return -1;
        }
        sent += (size_t)n;
    }
    close(cfd);
    return 0;
}

static void mock_endpoint(mock_server *srv, char *out, size_t out_cap) {
    snprintf(out, out_cap, "http://127.0.0.1:%d/v1", srv->port);
}

typedef struct {
    const char *endpoint;
    const char *raw;
    char *refined;
    hd_status status;
} client_ctx;

/*
 * Run `client` in a forked child while the parent serves one request.
 * The child writes its result (status + refined string) back through a pipe
 * so the parent can assert on it. Returns 0 on success.
 */
static int run_with_server(mock_server *srv, const char *reply,
                           int (*client)(void *), void *arg) {
    int pipefd[2];
    if (pipe(pipefd) != 0) return -1;

    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        return -1;
    }
    if (pid == 0) {
        /* child: run the client, report result through the pipe */
        close(pipefd[0]);
        client_ctx *c = (client_ctx *)arg;
        int rc = client(c);
        size_t refined_len = c->refined ? strlen(c->refined) : 0;
        if (write(pipefd[1], &c->status, sizeof(c->status)) !=
                (ssize_t)sizeof(c->status) ||
            write(pipefd[1], &refined_len, sizeof(refined_len)) !=
                (ssize_t)sizeof(refined_len) ||
            (refined_len &&
             write(pipefd[1], c->refined, refined_len) !=
                 (ssize_t)refined_len)) {
            _exit(2);
        }
        _exit(rc == 0 ? 0 : 1);
    }
    /* parent: serve exactly one request */
    close(pipefd[1]);
    char req[16384];
    memset(req, 0, sizeof(req));
    int rc = mock_serve(srv, reply, req, sizeof(req));
    memcpy(srv->req, req, sizeof(req));

    /* read the child's result */
    client_ctx *c = (client_ctx *)arg;
    hd_status status;
    size_t refined_len = 0;
    ssize_t got = read(pipefd[0], &status, sizeof(status));
    if (got == (ssize_t)sizeof(status)) {
        ssize_t got2 = read(pipefd[0], &refined_len, sizeof(refined_len));
        if (got2 != (ssize_t)sizeof(refined_len)) refined_len = 0;
        c->status = status;
        if (refined_len) {
            char *buf = (char *)malloc(refined_len + 1);
            if (buf) {
                size_t off = 0;
                while (off < refined_len) {
                    ssize_t n = read(pipefd[0], buf + off, refined_len - off);
                    if (n <= 0) break;
                    off += (size_t)n;
                }
                buf[off] = '\0';
                c->refined = buf;
            }
        }
    }
    close(pipefd[0]);

    int wstatus = -1;
    waitpid(pid, &wstatus, 0);
    if (rc != 0) return -1;
    if (!WIFEXITED(wstatus) || WEXITSTATUS(wstatus) != 0) return -1;
    return 0;
}

/* ── Tests ──────────────────────────────────────────────────────────────── */

static const char kCannedOk[] =
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: application/json\r\n"
    "Content-Length: 200\r\n"
    "\r\n"
    "{\"id\":\"chatcmpl-1\",\"object\":\"chat.completion\","
    "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
    "\"content\":\"A serene mountain lake at dawn.\"},"
    "\"finish_reason\":\"stop\"}]}";

static int client_legacy(void *arg) {
    client_ctx *c = (client_ctx *)arg;
    c->status = hd_refiner_refine_legacy(c->endpoint, "test-key", c->raw,
                                         &c->refined);
    return c->status == HD_OK && c->refined != NULL ? 0 : 1;
}

static int client_v2(void *arg) {
    client_ctx *c = (client_ctx *)arg;
    c->status = hd_refiner_refine_v2(c->endpoint, c->raw, &c->refined);
    return c->status == HD_OK && c->refined != NULL ? 0 : 1;
}

static int client_legacy_io(void *arg) {
    client_ctx *c = (client_ctx *)arg;
    c->refined = (char *)0x1; /* sentinel: must be reset to NULL */
    c->status = hd_refiner_refine_legacy(c->endpoint, NULL, c->raw,
                                         &c->refined);
    return c->status == HD_ERR_IO && c->refined == NULL ? 0 : 1;
}

static int client_v2_parse(void *arg) {
    client_ctx *c = (client_ctx *)arg;
    c->status = hd_refiner_refine_v2(c->endpoint, c->raw, &c->refined);
    return c->status == HD_ERR_PARSE && c->refined == NULL ? 0 : 1;
}

static void test_legacy_ok(void) {
    mock_server srv;
    CHECK(mock_start(&srv) == 0, "mock_start");
    if (srv.fd < 0) return;

    char endpoint[128];
    mock_endpoint(&srv, endpoint, sizeof(endpoint));

    client_ctx ctx = {endpoint, "a cat on the moon", NULL, HD_ERR_IO};
    CHECK(run_with_server(&srv, kCannedOk, client_legacy, &ctx) == 0,
          "legacy client succeeded");
    CHECK(ctx.status == HD_OK, "legacy returns HD_OK");
    CHECK(ctx.refined != NULL, "legacy out_refined set");
    if (ctx.refined) {
        CHECK(strcmp(ctx.refined, "A serene mountain lake at dawn.") == 0,
              "legacy parses choices[0].message.content");
        free(ctx.refined);
    }

    /* request shape: path, model, system prompt, user = raw */
    CHECK(strstr(srv.req, "POST /v1/chat/completions HTTP/1.1") != NULL,
          "legacy request path");
    CHECK(strstr(srv.req, "\"model\":\"HiDream-ai/Prompt-Refine\"") != NULL,
          "legacy request model");
    CHECK(strstr(srv.req, "\"role\":\"system\"") != NULL, "legacy system role");
    CHECK(strstr(srv.req, "\"role\":\"user\"") != NULL, "legacy user role");
    CHECK(strstr(srv.req, "\"content\":\"a cat on the moon\"") != NULL,
          "legacy user = raw prompt");
    CHECK(strstr(srv.req, "SCALIST") != NULL, "legacy system prompt text");
    CHECK(strstr(srv.req, "max_tokens") == NULL, "legacy has no max_tokens");
    CHECK(strstr(srv.req, "temperature") == NULL, "legacy has no temperature");

    mock_stop(&srv);
}

static void test_v2_ok(void) {
    mock_server srv;
    CHECK(mock_start(&srv) == 0, "mock_start");
    if (srv.fd < 0) return;

    char endpoint[128];
    mock_endpoint(&srv, endpoint, sizeof(endpoint));

    client_ctx ctx = {endpoint, "a cat on the moon", NULL, HD_ERR_IO};
    CHECK(run_with_server(&srv, kCannedOk, client_v2, &ctx) == 0,
          "v2 client succeeded");
    CHECK(ctx.status == HD_OK, "v2 returns HD_OK");
    CHECK(ctx.refined != NULL, "v2 out_refined set");
    if (ctx.refined) {
        CHECK(strcmp(ctx.refined, "A serene mountain lake at dawn.") == 0,
              "v2 parses choices[0].message.content");
        free(ctx.refined);
    }

    CHECK(strstr(srv.req, "POST /v1/chat/completions HTTP/1.1") != NULL,
          "v2 request path");
    CHECK(strstr(srv.req, "\"model\":\"HiDream-ai/Prompt-Refine\"") != NULL,
          "v2 request model");
    CHECK(strstr(srv.req, "\"max_tokens\":2048") != NULL, "v2 max_tokens");
    CHECK(strstr(srv.req, "SCALIST") != NULL, "v2 system prompt text");
    CHECK(strstr(srv.req, "temperature") == NULL, "v2 has no temperature");

    mock_stop(&srv);
}

static void test_connection_refused(void) {
    /* grab an ephemeral port, close it, then connect -> refused */
    mock_server srv;
    CHECK(mock_start(&srv) == 0, "mock_start");
    if (srv.fd < 0) return;
    int port = srv.port;
    mock_stop(&srv);

    char endpoint[128];
    snprintf(endpoint, sizeof(endpoint), "http://127.0.0.1:%d/v1", port);

    client_ctx ctx = {endpoint, "hello", NULL, HD_OK};
    CHECK(client_legacy_io(&ctx) == 0,
          "connection refused -> HD_ERR_IO, out untouched");
}

static void test_malformed_json(void) {
    mock_server srv;
    CHECK(mock_start(&srv) == 0, "mock_start");
    if (srv.fd < 0) return;

    char endpoint[128];
    mock_endpoint(&srv, endpoint, sizeof(endpoint));

    static const char kBad[] =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/json\r\n"
        "\r\n"
        "{not valid json";
    client_ctx ctx = {endpoint, "hello", NULL, HD_OK};
    CHECK(run_with_server(&srv, kBad, client_v2_parse, &ctx) == 0,
          "malformed JSON -> HD_ERR_PARSE");

    mock_stop(&srv);
}

static void test_missing_content(void) {
    mock_server srv;
    CHECK(mock_start(&srv) == 0, "mock_start");
    if (srv.fd < 0) return;

    char endpoint[128];
    mock_endpoint(&srv, endpoint, sizeof(endpoint));

    static const char kNoContent[] =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/json\r\n"
        "\r\n"
        "{\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\"}}]}";
    client_ctx ctx = {endpoint, "hello", NULL, HD_OK};
    CHECK(run_with_server(&srv, kNoContent, client_v2_parse, &ctx) == 0,
          "missing content -> HD_ERR_PARSE");

    mock_stop(&srv);
}

int main(void) {
    test_legacy_ok();
    test_v2_ok();
    test_connection_refused();
    test_malformed_json();
    test_missing_content();

    if (g_failures == 0) {
        printf("test_refiner: all tests passed\n");
        return 0;
    }
    fprintf(stderr, "test_refiner: %d failure(s)\n", g_failures);
    return 1;
}