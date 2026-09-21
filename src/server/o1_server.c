/*
 * o1.c — OpenAI-compatible Images API server.
 *
 * The HTTP transport, socket handling, thread/queue discipline and JSON
 * helpers are PORTED from the ds4 reference server (MIT License,
 * Copyright (c) 2026 Salvatore Sanfilippo) — see _reference/ds4-server.c.
 * The model/chat/KV/tool layers of that server are NOT ported: this server
 * exposes one resident HiDream-O1 image model through the OpenAI Images API
 * and reuses the validated o1.c generation runtime unchanged.
 *
 * Architecture (see O1_OPENAI_IMAGE_SERVER_IMPLEMENTATION.md section 13):
 *
 *   client connection thread                single model worker
 *   -------------------------               -------------------
 *   parse/validate HTTP                     dequeue job
 *   parse JSON or multipart                 hd_generation_engine_generate()
 *   build job, enqueue, wait                PNG encode + base64
 *   serialize + send response               signal client
 *
 * The generation engine owns the CUDA device, the resident weights and the
 * large workspaces. Exactly one thread may call it at a time, so GPU work is
 * serialized through a bounded job queue while HTTP connections stay
 * concurrent.
 */

#define _GNU_SOURCE

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "engine.h"
#include "hd_image.h"
#include "hidream.h"
#include "hd_lora.h"
#include "layout.h"
#include "png_wrap.h"
#include "request.h"
#include "sequence.h"

/* ------------------------------------------------------------------ */
/* Process globals                                                     */
/* ------------------------------------------------------------------ */

static volatile sig_atomic_t g_stop_requested = 0;
static volatile sig_atomic_t g_listen_fd = -1;

#define O1_SERVER_IO_TIMEOUT_SEC 10
#define O1_SERVER_SEND_STALL_TIMEOUT_MS 2000
#define O1_SERVER_MAX_N 4
#define O1_SERVER_MAX_IMAGE_BYTES (50u * 1024u * 1024u)
#define O1_SERVER_MAX_PROMPT_BYTES (64u * 1024u)
#define O1_SERVER_MAX_HEADER_BYTES (64u * 1024u)
#define O1_SERVER_MAX_MULTIPART_PARTS 64

static void stop_signal_handler(int sig) {
    (void)sig;
    if (g_stop_requested) _exit(130);
    g_stop_requested = 1;
    if (g_listen_fd >= 0) {
        int fd = (int)g_listen_fd;
        g_listen_fd = -1;
        close(fd);
    }
}

/* ------------------------------------------------------------------ */
/* Buffers and allocation helpers                                      */
/* ------------------------------------------------------------------ */

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} buf;

static void die(const char *msg) {
    fprintf(stderr, "hidream-server: %s\n", msg);
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xrealloc(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static char *xstrndup(const char *s, size_t n) {
    char *p = xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

static bool random_bytes(void *dst, size_t len) {
    unsigned char *p = dst;
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd < 0) return false;
    while (len) {
        ssize_t n = read(fd, p, len);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { close(fd); return false; }
        p += (size_t)n;
        len -= (size_t)n;
    }
    close(fd);
    return true;
}

static uint64_t random_u64(void) {
    uint64_t v = 0;
    if (random_bytes(&v, sizeof(v))) return v;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    v = (uint64_t)tv.tv_sec * 1000000ull + (uint64_t)tv.tv_usec;
    v ^= (uint64_t)getpid() << 40;
    return v;
}

static void buf_reserve(buf *b, size_t add) {
    if (add > SIZE_MAX - b->len - 1) die("buffer overflow");
    size_t need = b->len + add + 1;
    if (need <= b->cap) return;
    size_t cap = b->cap ? b->cap * 2 : 256;
    while (cap < need) {
        if (cap > SIZE_MAX / 2) { cap = need; break; }
        cap *= 2;
    }
    b->ptr = xrealloc(b->ptr, cap);
    b->cap = cap;
}

static void buf_append(buf *b, const void *p, size_t n) {
    if (n == 0) return;
    buf_reserve(b, n);
    memcpy(b->ptr + b->len, p, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static void buf_putc(buf *b, char c) { buf_append(b, &c, 1); }

static void buf_puts(buf *b, const char *s) { buf_append(b, s, strlen(s)); }

static void buf_printf(buf *b, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    int n = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);
    if (n < 0) die("vsnprintf failed");
    buf_reserve(b, (size_t)n);
    vsnprintf(b->ptr + b->len, b->cap - b->len, fmt, ap2);
    va_end(ap2);
    b->len += (size_t)n;
}

static char *buf_take(buf *b) {
    if (!b->ptr) return xstrdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}

static void buf_free(buf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}

/* ------------------------------------------------------------------ */
/* Minimal JSON reader (ported from ds4-server.c)                      */
/* ------------------------------------------------------------------ */

static void json_ws(const char **p) {
    while (**p && isspace((unsigned char)**p)) (*p)++;
}

static bool json_lit(const char **p, const char *lit) {
    size_t n = strlen(lit);
    if (strncmp(*p, lit, n) != 0) return false;
    *p += n;
    return true;
}

static int json_hex(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static void utf8_put(buf *b, uint32_t cp) {
    if (cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) cp = 0xfffd;
    if (cp <= 0x7f) {
        buf_putc(b, (char)cp);
    } else if (cp <= 0x7ff) {
        buf_putc(b, (char)(0xc0 | (cp >> 6)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        buf_putc(b, (char)(0xe0 | (cp >> 12)));
        buf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    } else {
        buf_putc(b, (char)(0xf0 | (cp >> 18)));
        buf_putc(b, (char)(0x80 | ((cp >> 12) & 0x3f)));
        buf_putc(b, (char)(0x80 | ((cp >> 6) & 0x3f)));
        buf_putc(b, (char)(0x80 | (cp & 0x3f)));
    }
}

static bool json_u16(const char **p, uint32_t *out) {
    if ((*p)[0] != '\\' || (*p)[1] != 'u') return false;
    uint32_t cp = 0;
    for (int i = 0; i < 4; i++) {
        int h = json_hex((*p)[2 + i]);
        if (h < 0) return false;
        cp = (cp << 4) | (uint32_t)h;
    }
    *p += 6;
    *out = cp;
    return true;
}

static bool json_string(const char **p, char **out) {
    /* Always define *out: several callers reparse in place with
     * `free(x); json_string(&p, &x)`, and a stale freed pointer would be a
     * double free. */
    *out = NULL;
    json_ws(p);
    if (**p != '"') return false;
    (*p)++;
    buf b = {0};
    while (**p && **p != '"') {
        unsigned char c = (unsigned char)*(*p)++;
        if (c != '\\') { buf_putc(&b, (char)c); continue; }
        c = (unsigned char)*(*p)++;
        switch (c) {
        case '"': buf_putc(&b, '"'); break;
        case '\\': buf_putc(&b, '\\'); break;
        case '/': buf_putc(&b, '/'); break;
        case 'b': buf_putc(&b, '\b'); break;
        case 'f': buf_putc(&b, '\f'); break;
        case 'n': buf_putc(&b, '\n'); break;
        case 'r': buf_putc(&b, '\r'); break;
        case 't': buf_putc(&b, '\t'); break;
        case 'u': {
            *p -= 2;
            uint32_t cp = 0, lo = 0;
            if (!json_u16(p, &cp)) goto fail;
            if (cp >= 0xd800 && cp <= 0xdbff) {
                const char *low_start = *p;
                if (json_u16(p, &lo) && lo >= 0xdc00 && lo <= 0xdfff) {
                    cp = 0x10000u + ((cp - 0xd800u) << 10) + (lo - 0xdc00u);
                } else {
                    *p = low_start;
                    cp = 0xfffd;
                }
            }
            utf8_put(&b, cp);
            break;
        }
        default: goto fail;
        }
    }
    if (**p != '"') goto fail;
    (*p)++;
    *out = buf_take(&b);
    return true;
fail:
    buf_free(&b);
    return false;
}

static bool json_number(const char **p, double *out) {
    json_ws(p);
    char *end = NULL;
    double v = strtod(*p, &end);
    if (end == *p) return false;
    *p = end;
    *out = v;
    return true;
}

static bool json_int(const char **p, int *out) {
    double v = 0.0;
    if (!json_number(p, &v)) return false;
    /* strtod() accepts NaN/Infinity; NaN fails every comparison, so
     * `!(v >= 0)` folds NaN (and negatives) to 0 before the (int) cast. */
    if (!(v >= 0)) v = 0;
    if (v > INT_MAX) v = INT_MAX;
    *out = (int)v;
    return true;
}

static bool json_bool(const char **p, bool *out) {
    json_ws(p);
    if (json_lit(p, "true")) { *out = true; return true; }
    if (json_lit(p, "false")) { *out = false; return true; }
    return false;
}

/* Ignored fields still nest, so skipping is recursive with an explicit
 * ceiling: without it a useless field like {"x":[[[...]]]} can exhaust the
 * C stack before the request is rejected. */
#define JSON_MAX_NESTING 256

static bool json_skip_value_depth(const char **p, int depth);

static bool json_skip_array_depth(const char **p, int depth) {
    if (depth >= JSON_MAX_NESTING) return false;
    json_ws(p);
    if (**p != '[') return false;
    (*p)++;
    json_ws(p);
    if (**p == ']') { (*p)++; return true; }
    for (;;) {
        if (!json_skip_value_depth(p, depth + 1)) return false;
        json_ws(p);
        if (**p == ']') { (*p)++; return true; }
        if (**p != ',') return false;
        (*p)++;
    }
}

static bool json_skip_object_depth(const char **p, int depth) {
    if (depth >= JSON_MAX_NESTING) return false;
    json_ws(p);
    if (**p != '{') return false;
    (*p)++;
    json_ws(p);
    if (**p == '}') { (*p)++; return true; }
    for (;;) {
        char *key = NULL;
        if (!json_string(p, &key)) return false;
        free(key);
        json_ws(p);
        if (**p != ':') return false;
        (*p)++;
        if (!json_skip_value_depth(p, depth + 1)) return false;
        json_ws(p);
        if (**p == '}') { (*p)++; return true; }
        if (**p != ',') return false;
        (*p)++;
    }
}

static bool json_skip_value_depth(const char **p, int depth) {
    json_ws(p);
    if (**p == '"') {
        char *s = NULL;
        bool ok = json_string(p, &s);
        free(s);
        return ok;
    }
    if (**p == '{') return json_skip_object_depth(p, depth);
    if (**p == '[') return json_skip_array_depth(p, depth);
    if (json_lit(p, "true") || json_lit(p, "false") || json_lit(p, "null"))
        return true;
    double v = 0.0;
    return json_number(p, &v);
}

static bool json_skip_value(const char **p) {
    return json_skip_value_depth(p, 0);
}

static bool json_raw_value(const char **p, char **out) {
    *out = NULL;
    json_ws(p);
    const char *start = *p;
    if (!json_skip_value(p)) return false;
    size_t n = (size_t)(*p - start);
    char *s = xmalloc(n + 1);
    memcpy(s, start, n);
    s[n] = '\0';
    *out = s;
    return true;
}

static void json_escape(buf *b, const char *s) {
    buf_putc(b, '"');
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') {
            buf_putc(b, '\\');
            buf_putc(b, (char)c);
        } else if (c == '\n') {
            buf_puts(b, "\\n");
        } else if (c == '\r') {
            buf_puts(b, "\\r");
        } else if (c == '\t') {
            buf_puts(b, "\\t");
        } else if (c < 0x20) {
            buf_printf(b, "\\u%04x", (unsigned)c);
        } else {
            buf_putc(b, (char)c);
        }
    }
    buf_putc(b, '"');
}

/* ------------------------------------------------------------------ */
/* Base64                                                              */
/* ------------------------------------------------------------------ */

static const char k_b64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static void base64_encode(buf *b, const uint8_t *src, size_t n) {
    size_t i = 0;
    for (; i + 3 <= n; i += 3) {
        uint32_t v = ((uint32_t)src[i] << 16) | ((uint32_t)src[i + 1] << 8) |
                     (uint32_t)src[i + 2];
        buf_putc(b, k_b64[(v >> 18) & 63]);
        buf_putc(b, k_b64[(v >> 12) & 63]);
        buf_putc(b, k_b64[(v >> 6) & 63]);
        buf_putc(b, k_b64[v & 63]);
    }
    size_t rem = n - i;
    if (rem == 1) {
        uint32_t v = (uint32_t)src[i] << 16;
        buf_putc(b, k_b64[(v >> 18) & 63]);
        buf_putc(b, k_b64[(v >> 12) & 63]);
        buf_puts(b, "==");
    } else if (rem == 2) {
        uint32_t v = ((uint32_t)src[i] << 16) | ((uint32_t)src[i + 1] << 8);
        buf_putc(b, k_b64[(v >> 18) & 63]);
        buf_putc(b, k_b64[(v >> 12) & 63]);
        buf_putc(b, k_b64[(v >> 6) & 63]);
        buf_putc(b, '=');
    }
}

/* ------------------------------------------------------------------ */
/* HTTP transport (ported from ds4-server.c)                           */
/* ------------------------------------------------------------------ */

static long long wall_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

static bool send_all(int fd, const void *p, size_t n) {
    const char *s = p;
    long long deadline = wall_ms() + O1_SERVER_SEND_STALL_TIMEOUT_MS;
    while (n) {
        if (g_stop_requested) return false;
        ssize_t w = send(fd, s, n, 0);
        if (w < 0 && errno == EINTR) continue;
        if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            long long remaining = deadline - wall_ms();
            if (remaining <= 0) return false;
            struct pollfd pfd = {.fd = fd, .events = POLLOUT};
            int timeout = remaining > 50 ? 50 : (int)remaining;
            int rc;
            do {
                rc = poll(&pfd, 1, timeout);
            } while (rc < 0 && errno == EINTR);
            if (rc < 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)))
                return false;
            continue;
        }
        if (w <= 0) return false;
        s += w;
        n -= (size_t)w;
        deadline = wall_ms() + O1_SERVER_SEND_STALL_TIMEOUT_MS;
    }
    return true;
}

static void append_cors_headers(buf *h) {
    buf_puts(h,
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: *\r\n");
}

static const char *http_reason(int code) {
    switch (code) {
    case 200: return "OK";
    case 204: return "No Content";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 409: return "Conflict";
    case 413: return "Payload Too Large";
    case 415: return "Unsupported Media Type";
    case 429: return "Too Many Requests";
    case 500: return "Internal Server Error";
    case 503: return "Service Unavailable";
    default: return "Error";
    }
}

static bool http_response(int fd, bool enable_cors, int code, const char *type,
                          const char *body) {
    const size_t body_len = body ? strlen(body) : 0;
    buf h = {0};
    buf_printf(&h,
        "HTTP/1.1 %d %s\r\n"
        "Content-Length: %zu\r\n",
        code, http_reason(code), body_len);
    if (type && type[0]) {
        buf_puts(&h, "Content-Type: ");
        buf_puts(&h, type);
        buf_puts(&h, "\r\n");
    }
    if (enable_cors) append_cors_headers(&h);
    buf_puts(&h, "Connection: close\r\n\r\n");
    bool ok = send_all(fd, h.ptr, h.len);
    if (ok && body_len) ok = send_all(fd, body, body_len);
    buf_free(&h);
    return ok;
}

/* OpenAI error envelope: {error:{message,type,param,code}}.
 * `param` and `code` may be NULL, in which case they are emitted as null. */
static bool http_error_full(int fd, bool enable_cors, int code,
                            const char *type, const char *msg,
                            const char *param, const char *err_code) {
    buf b = {0};
    buf_puts(&b, "{\"error\":{\"message\":");
    json_escape(&b, msg ? msg : "error");
    buf_puts(&b, ",\"type\":");
    json_escape(&b, type ? type : "invalid_request_error");
    buf_puts(&b, ",\"param\":");
    if (param) json_escape(&b, param); else buf_puts(&b, "null");
    buf_puts(&b, ",\"code\":");
    if (err_code) json_escape(&b, err_code); else buf_puts(&b, "null");
    buf_puts(&b, "}}\n");
    bool ok = http_response(fd, enable_cors, code, "application/json", b.ptr);
    buf_free(&b);
    return ok;
}

static bool http_error(int fd, bool enable_cors, int code, const char *msg) {
    return http_error_full(fd, enable_cors, code, "invalid_request_error", msg,
                           NULL, NULL);
}

typedef struct {
    char method[8];
    char path[256];
    char *body;
    size_t body_len;
    char *content_type;
    char *authorization;
} http_request;

static void http_request_free(http_request *r) {
    free(r->body);
    free(r->content_type);
    free(r->authorization);
    memset(r, 0, sizeof(*r));
}

static ssize_t header_end(const char *p, size_t n) {
    for (size_t i = 3; i < n; i++) {
        if (p[i - 3] == '\r' && p[i - 2] == '\n' && p[i - 1] == '\r' &&
            p[i] == '\n')
            return (ssize_t)(i + 1);
    }
    for (size_t i = 1; i < n; i++) {
        if (p[i - 1] == '\n' && p[i] == '\n') return (ssize_t)(i + 1);
    }
    return -1;
}

/* Returns the Content-Length value, or -1 when the header is absent or
 * malformed. A malformed value must never be treated as 0: that would let a
 * body be silently ignored (or, worse, mis-framed). */
static long content_length(const char *h, size_t n) {
    const char *p = h, *end = h + n;
    while (p < end) {
        const char *line = p;
        while (p < end && *p != '\n') p++;
        size_t len = (size_t)(p - line);
        if (len && line[len - 1] == '\r') len--;
        if (len >= 15 && strncasecmp(line, "Content-Length:", 15) == 0) {
            const char *v = line + 15;
            while (v < line + len && isspace((unsigned char)*v)) v++;
            if (v >= line + len) return -1;
            char *vend = NULL;
            long val = strtol(v, &vend, 10);
            if (vend == v || val < 0) return -1;
            return val;
        }
        if (p < end) p++;
    }
    return -1;
}

static char *header_value(const char *h, size_t n, const char *name) {
    size_t name_len = strlen(name);
    const char *p = h, *end = h + n;
    while (p < end) {
        const char *line = p;
        while (p < end && *p != '\n') p++;
        size_t len = (size_t)(p - line);
        if (len && line[len - 1] == '\r') len--;
        if (len > name_len && strncasecmp(line, name, name_len) == 0 &&
            line[name_len] == ':') {
            const char *v = line + name_len + 1;
            const char *vend = line + len;
            while (v < vend && isspace((unsigned char)*v)) v++;
            while (vend > v && isspace((unsigned char)vend[-1])) vend--;
            return xstrndup(v, (size_t)(vend - v));
        }
        if (p < end) p++;
    }
    return NULL;
}

static bool read_http_request(int fd, http_request *r, size_t max_body,
                              bool *too_large) {
    buf b = {0};
    ssize_t hend = -1;

    while (hend < 0 && b.len < O1_SERVER_MAX_HEADER_BYTES) {
        char tmp[4096];
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) goto fail;
        buf_append(&b, tmp, (size_t)n);
        hend = header_end(b.ptr, b.len);
    }
    if (hend < 0) goto fail;

    char line[512];
    size_t i = 0;
    while (i < b.len && b.ptr[i] != '\n' && i + 1 < sizeof(line)) {
        line[i] = b.ptr[i];
        i++;
    }
    line[i] = '\0';
    if (sscanf(line, "%7s %255s", r->method, r->path) != 2) goto fail;
    char *q = strchr(r->path, '?');
    if (q) *q = '\0';

    r->content_type = header_value(b.ptr, (size_t)hend, "Content-Type");
    r->authorization = header_value(b.ptr, (size_t)hend, "Authorization");

    long clen = content_length(b.ptr, (size_t)hend);
    if (clen < 0) clen = 0;
    if ((size_t)clen > max_body) {
        /* Drain (bounded) so the client can read the 413 response instead of
         * seeing a connection reset while it is still writing the body. */
        if (too_large) *too_large = true;
        size_t remaining = (size_t)clen;
        while (remaining > 0) {
            char tmp[65536];
            size_t want = remaining < sizeof(tmp) ? remaining : sizeof(tmp);
            ssize_t n = recv(fd, tmp, want, 0);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) break;
            remaining -= (size_t)n;
        }
        goto fail;
    }
    while (b.len < (size_t)hend + (size_t)clen) {
        char tmp[8192];
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) goto fail;
        buf_append(&b, tmp, (size_t)n);
    }

    r->body_len = (size_t)clen;
    r->body = xmalloc(r->body_len + 1);
    memcpy(r->body, b.ptr + hend, r->body_len);
    r->body[r->body_len] = '\0';
    buf_free(&b);
    return true;
fail:
    buf_free(&b);
    return false;
}

static void configure_client_socket(int fd) {
    struct timeval tv;
    tv.tv_sec = O1_SERVER_IO_TIMEOUT_SEC;
    tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}

static int listen_on(const char *host, int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons((uint16_t)port);
    if (!strcmp(host, "localhost")) host = "127.0.0.1";
    if (inet_pton(AF_INET, host, &sa.sin_addr) != 1) {
        close(fd);
        errno = EINVAL;
        return -1;
    }
    if (bind(fd, (struct sockaddr *)&sa, sizeof(sa)) != 0) {
        close(fd);
        return -1;
    }
    if (listen(fd, 128) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

/* ------------------------------------------------------------------ */
/* Server configuration and resident engine                            */
/* ------------------------------------------------------------------ */

#define O1_SERVER_MAX_LORAS 8

typedef struct {
    const char *profile;
    const char *config_dir;
    const char *model_path;
    int device_id;
    const char *host;
    int port;
    bool cors;
    size_t max_body;
    int queue_depth;
    const char *api_key;
    hd_lora_spec loras[O1_SERVER_MAX_LORAS];
    int lora_count;
} server_config;

static server_config g_cfg;
static hd_generation_engine *g_engine = NULL;

/* Canonical model id per profile. The OpenAI `model` field is validated
 * against this and never triggers a hot load: one profile per process. */
static const char *canonical_model_id(const char *profile) {
    return strcmp(profile, "base") == 0 ? "hidream-o1-image" : "hidream-o1-image-dev";
}

static bool model_id_matches(const char *profile, const char *id) {
    if (!id || !id[0]) return false;
    if (strcmp(id, canonical_model_id(profile)) == 0) return true;
    if (strcmp(profile, "base") == 0) {
        return strcmp(id, "o1-base") == 0 || strcmp(id, "base") == 0;
    }
    return strcmp(id, "o1-dev") == 0 || strcmp(id, "dev") == 0;
}

/* ------------------------------------------------------------------ */
/* Jobs                                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    hd_generation_request req;
    char *prompt;                 /* owned; req.prompt aliases this */
    char *ref_paths[HD_SEQ_MAX_REFS];
    char *ref_aliases[HD_SEQ_MAX_REFS];
    hd_reference_image refs[HD_SEQ_MAX_REFS];
    hd_layout_condition *layout_conds;
    size_t n_layout;
    int n;
    int requested_w, requested_h;
    bool size_auto;
    bool verbose;

    /* results, produced by the worker */
    int status;
    char *error_type;
    char *error_msg;
    char *error_param;
    char *error_code;
    char *body;

    long long queue_wait_ms;
    long long parse_ms;
    long long preprocess_ms;
    long long generation_ms;
    long long encode_ms;
    long long total_ms;

    int client_fd;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    bool done;
    bool cancelled;
} server_job;

static void server_job_free(server_job *j) {
    if (!j) return;
    free(j->prompt);
    for (int i = 0; i < HD_SEQ_MAX_REFS; i++) {
        /* The paths are private temp copies of the uploaded images; remove
         * them so a long-running server does not accumulate scratch files. */
        if (j->ref_paths[i]) unlink(j->ref_paths[i]);
        free(j->ref_paths[i]);
        free(j->ref_aliases[i]);
    }
    if (j->layout_conds) hd_layout_conditions_free(j->layout_conds, j->n_layout);
    free(j->error_type);
    free(j->error_msg);
    free(j->error_param);
    free(j->error_code);
    free(j->body);
    pthread_mutex_destroy(&j->mu);
    pthread_cond_destroy(&j->cv);
    free(j);
}

static server_job *server_job_new(int client_fd) {
    server_job *j = xmalloc(sizeof(*j));
    memset(j, 0, sizeof(*j));
    j->client_fd = client_fd;
    j->status = 200;
    pthread_mutex_init(&j->mu, NULL);
    pthread_cond_init(&j->cv, NULL);
    return j;
}

static void server_job_fail(server_job *j, int status, const char *type,
                            const char *msg, const char *param,
                            const char *code) {
    j->status = status;
    free(j->error_type);
    free(j->error_msg);
    free(j->error_param);
    free(j->error_code);
    j->error_type = xstrdup(type ? type : "invalid_request_error");
    j->error_msg = xstrdup(msg ? msg : "error");
    j->error_param = param ? xstrdup(param) : NULL;
    j->error_code = code ? xstrdup(code) : NULL;
}

/* ------------------------------------------------------------------ */
/* Bounded job queue                                                   */
/* ------------------------------------------------------------------ */

static server_job **g_queue = NULL;
static int g_queue_cap = 0;
static int g_queue_head = 0;
static int g_queue_len = 0;
static bool g_queue_stopping = false;
static pthread_mutex_t g_queue_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_queue_cv = PTHREAD_COND_INITIALIZER;

static bool enqueue(server_job *j) {
    pthread_mutex_lock(&g_queue_mu);
    if (g_queue_stopping || g_queue_len >= g_queue_cap) {
        pthread_mutex_unlock(&g_queue_mu);
        return false;
    }
    g_queue[(g_queue_head + g_queue_len) % g_queue_cap] = j;
    g_queue_len++;
    pthread_cond_signal(&g_queue_cv);
    pthread_mutex_unlock(&g_queue_mu);
    return true;
}

static server_job *dequeue(void) {
    pthread_mutex_lock(&g_queue_mu);
    while (g_queue_len == 0 && !g_queue_stopping)
        pthread_cond_wait(&g_queue_cv, &g_queue_mu);
    if (g_queue_len == 0) {
        pthread_mutex_unlock(&g_queue_mu);
        return NULL;
    }
    server_job *j = g_queue[g_queue_head];
    g_queue_head = (g_queue_head + 1) % g_queue_cap;
    g_queue_len--;
    pthread_mutex_unlock(&g_queue_mu);
    return j;
}

static void queue_stop(void) {
    pthread_mutex_lock(&g_queue_mu);
    g_queue_stopping = true;
    pthread_cond_broadcast(&g_queue_cv);
    pthread_mutex_unlock(&g_queue_mu);
}

/* ------------------------------------------------------------------ */
/* Worker                                                              */
/* ------------------------------------------------------------------ */

static void job_run(server_job *j);

static void *worker_main(void *arg) {
    (void)arg;
    for (;;) {
        server_job *j = dequeue();
        if (!j) break;
        job_run(j);
        pthread_mutex_lock(&j->mu);
        j->done = true;
        pthread_cond_broadcast(&j->cv);
        pthread_mutex_unlock(&j->mu);
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Client handling                                                     */
/* ------------------------------------------------------------------ */

static void handle_request(int fd, http_request *r);

static bool client_socket_disconnected(int fd) {
    struct pollfd pfd = {.fd = fd, .events = POLLIN};
    int rc = poll(&pfd, 1, 0);
    if (rc < 0) return true;
    if (rc == 0) return false;
    if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) return true;
    if (pfd.revents & POLLIN) {
        char tmp[1];
        ssize_t n = recv(fd, tmp, 1, MSG_PEEK | MSG_DONTWAIT);
        if (n == 0) return true;
        if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK &&
            errno != EINTR)
            return true;
    }
    return false;
}

/* Wait for the worker to finish the job, aborting early if the client goes
 * away. The job is never freed here: the worker may still be inside
 * hd_generation_engine_generate(), so the caller waits for `done` first. */
static void wait_for_job_or_disconnect(server_job *j) {
    pthread_mutex_lock(&j->mu);
    while (!j->done) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 100 * 1000 * 1000;
        if (ts.tv_nsec >= 1000000000) {
            ts.tv_sec++;
            ts.tv_nsec -= 1000000000;
        }
        pthread_cond_timedwait(&j->cv, &j->mu, &ts);
        if (!j->done && client_socket_disconnected(j->client_fd))
            j->cancelled = true;
    }
    pthread_mutex_unlock(&j->mu);
}

static void *client_main(void *arg) {
    int fd = (int)(intptr_t)arg;
    configure_client_socket(fd);

    http_request r;
    memset(&r, 0, sizeof(r));
    bool too_large = false;
    if (!read_http_request(fd, &r, g_cfg.max_body, &too_large)) {
        if (too_large)
            http_error_full(fd, g_cfg.cors, 413, "invalid_request_error",
                            "request body exceeds the configured limit", NULL,
                            "request_too_large");
        else
            http_error(fd, g_cfg.cors, 400, "malformed HTTP request");
        close(fd);
        return NULL;
    }

    handle_request(fd, &r);

    http_request_free(&r);
    close(fd);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Parsed request fields                                               */
/* ------------------------------------------------------------------ */

typedef struct {
    char *prompt;
    char *model;
    int n;
    char *size;
    char *output_format;
    char *user;

    char *o1_mode;
    bool has_seed;
    uint64_t seed;
    bool has_steps;
    int steps;
    bool has_scheduler;
    char *scheduler;
    bool has_guidance;
    double guidance;
    bool has_shift;
    double shift;
    bool has_noise_start;
    double noise_start;
    bool has_noise_end;
    double noise_end;
    bool has_noise_clip;
    double noise_clip;
    bool has_keep_aspect;
    bool keep_aspect;
    char *layout_bboxes;
    char **aliases;
    int n_aliases;
    bool verbose;
    bool has_exact_size;
    bool exact_size;

    /* standard fields that must be validated then refused */
    bool has_mask;
    char *background;
    char *quality;
    char *compression;
    bool has_lora;
    bool response_format_url;
} parsed_fields;

static void parsed_fields_free(parsed_fields *f) {
    free(f->prompt);
    free(f->model);
    free(f->size);
    free(f->output_format);
    free(f->user);
    free(f->o1_mode);
    free(f->scheduler);
    free(f->layout_bboxes);
    for (int i = 0; i < f->n_aliases; i++) free(f->aliases[i]);
    free(f->aliases);
    free(f->background);
    free(f->quality);
    free(f->compression);
    memset(f, 0, sizeof(*f));
}

static bool parse_alias_array(const char *json, char ***out, int *out_n) {
    *out = NULL;
    *out_n = 0;
    const char *p = json;
    json_ws(&p);
    if (*p != '[') return false;
    p++;
    char **items = NULL;
    int n = 0, cap = 0;
    json_ws(&p);
    if (*p == ']') return true;
    for (;;) {
        char *s = NULL;
        if (!json_string(&p, &s)) {
            for (int i = 0; i < n; i++) free(items[i]);
            free(items);
            return false;
        }
        if (n == cap) {
            cap = cap ? cap * 2 : 4;
            items = xrealloc(items, (size_t)cap * sizeof(*items));
        }
        items[n++] = s;
        json_ws(&p);
        if (*p == ']') break;
        if (*p != ',') {
            for (int i = 0; i < n; i++) free(items[i]);
            free(items);
            return false;
        }
        p++;
    }
    *out = items;
    *out_n = n;
    return true;
}

/* Parse the JSON request body. Unknown fields are skipped (forward
 * compatible); fields that are known but unsupported are recorded so the
 * caller can reject them with a precise error. */
static bool parse_json_body(const char *body, parsed_fields *f) {
    const char *p = body;
    json_ws(&p);
    if (*p != '{') return false;
    p++;
    json_ws(&p);
    if (*p == '}') return true;
    for (;;) {
        char *key = NULL;
        if (!json_string(&p, &key)) return false;
        json_ws(&p);
        if (*p != ':') { free(key); return false; }
        p++;
        json_ws(&p);

        bool ok = true;
        if (!strcmp(key, "prompt")) {
            free(f->prompt);
            ok = json_string(&p, &f->prompt);
        } else if (!strcmp(key, "model")) {
            free(f->model);
            ok = json_string(&p, &f->model);
        } else if (!strcmp(key, "n")) {
            ok = json_int(&p, &f->n);
        } else if (!strcmp(key, "size")) {
            free(f->size);
            ok = json_string(&p, &f->size);
        } else if (!strcmp(key, "output_format")) {
            free(f->output_format);
            ok = json_string(&p, &f->output_format);
        } else if (!strcmp(key, "user")) {
            free(f->user);
            ok = json_string(&p, &f->user);
        } else if (!strcmp(key, "response_format")) {
            char *v = NULL;
            ok = json_string(&p, &v);
            if (ok && v && !strcmp(v, "url")) f->response_format_url = true;
            free(v);
        } else if (!strcmp(key, "o1_mode")) {
            free(f->o1_mode);
            ok = json_string(&p, &f->o1_mode);
        } else if (!strcmp(key, "o1_seed")) {
            char *raw = NULL;
            ok = json_raw_value(&p, &raw);
            if (ok) {
                errno = 0;
                char *end = NULL;
                unsigned long long v = strtoull(raw, &end, 10);
                if (end == raw || errno == ERANGE) ok = false;
                else { f->seed = (uint64_t)v; f->has_seed = true; }
            }
            free(raw);
        } else if (!strcmp(key, "o1_steps")) {
            ok = json_int(&p, &f->steps);
            if (ok) f->has_steps = true;
        } else if (!strcmp(key, "o1_scheduler")) {
            free(f->scheduler);
            ok = json_string(&p, &f->scheduler);
            if (ok) f->has_scheduler = true;
        } else if (!strcmp(key, "o1_guidance_scale")) {
            ok = json_number(&p, &f->guidance);
            if (ok) f->has_guidance = true;
        } else if (!strcmp(key, "o1_shift")) {
            ok = json_number(&p, &f->shift);
            if (ok) f->has_shift = true;
        } else if (!strcmp(key, "o1_noise_start")) {
            ok = json_number(&p, &f->noise_start);
            if (ok) f->has_noise_start = true;
        } else if (!strcmp(key, "o1_noise_end")) {
            ok = json_number(&p, &f->noise_end);
            if (ok) f->has_noise_end = true;
        } else if (!strcmp(key, "o1_noise_clip")) {
            ok = json_number(&p, &f->noise_clip);
            if (ok) f->has_noise_clip = true;
        } else if (!strcmp(key, "o1_keep_original_aspect")) {
            ok = json_bool(&p, &f->keep_aspect);
            if (ok) f->has_keep_aspect = true;
        } else if (!strcmp(key, "o1_layout_bboxes")) {
            free(f->layout_bboxes);
            ok = json_raw_value(&p, &f->layout_bboxes);
        } else if (!strcmp(key, "o1_reference_aliases")) {
            char *raw = NULL;
            ok = json_raw_value(&p, &raw);
            if (ok) {
                for (int i = 0; i < f->n_aliases; i++) free(f->aliases[i]);
                free(f->aliases);
                f->aliases = NULL;
                f->n_aliases = 0;
                ok = parse_alias_array(raw, &f->aliases, &f->n_aliases);
            }
            free(raw);
        } else if (!strcmp(key, "o1_verbose")) {
            ok = json_bool(&p, &f->verbose);
        } else if (!strcmp(key, "o1_exact_size")) {
            ok = json_bool(&p, &f->exact_size);
            if (ok) f->has_exact_size = true;
        } else if (!strcmp(key, "o1_lora")) {
            f->has_lora = true;
            ok = json_skip_value(&p);
        } else if (!strcmp(key, "mask")) {
            f->has_mask = true;
            ok = json_skip_value(&p);
        } else if (!strcmp(key, "background")) {
            free(f->background);
            ok = json_string(&p, &f->background);
        } else if (!strcmp(key, "quality")) {
            free(f->quality);
            ok = json_string(&p, &f->quality);
        } else if (!strcmp(key, "compression")) {
            free(f->compression);
            ok = json_string(&p, &f->compression);
        } else {
            ok = json_skip_value(&p);
        }
        free(key);
        if (!ok) return false;
        json_ws(&p);
        if (*p == '}') return true;
        if (*p != ',') return false;
        p++;
    }
}

/* ------------------------------------------------------------------ */
/* Multipart/form-data                                                 */
/* ------------------------------------------------------------------ */

typedef struct {
    char *name;
    char *filename;
    char *data;
    size_t len;
} multipart_part;

typedef struct {
    multipart_part *parts;
    int count;
    int cap;
} multipart_form;

static void multipart_form_free(multipart_form *m) {
    for (int i = 0; i < m->count; i++) {
        free(m->parts[i].name);
        free(m->parts[i].filename);
        free(m->parts[i].data);
    }
    free(m->parts);
    memset(m, 0, sizeof(*m));
}

static void multipart_push(multipart_form *m, multipart_part p) {
    if (m->count == m->cap) {
        m->cap = m->cap ? m->cap * 2 : 8;
        m->parts = xrealloc(m->parts, (size_t)m->cap * sizeof(*m->parts));
    }
    m->parts[m->count++] = p;
}

static char *content_type_param(const char *ct, const char *param) {
    if (!ct) return NULL;
    size_t plen = strlen(param);
    const char *p = ct;
    while ((p = strchr(p, ';')) != NULL) {
        p++;
        while (*p == ' ' || *p == '\t') p++;
        if (strncasecmp(p, param, plen) == 0) {
            const char *v = p + plen;
            while (*v == ' ' || *v == '\t') v++;
            if (*v != '=') continue;
            v++;
            while (*v == ' ' || *v == '\t') v++;
            if (*v == '"') {
                v++;
                const char *end = strchr(v, '"');
                if (!end) return NULL;
                return xstrndup(v, (size_t)(end - v));
            }
            const char *end = v;
            while (*end && *end != ';' && *end != ' ' && *end != '\t') end++;
            if (end == v) return NULL;
            return xstrndup(v, (size_t)(end - v));
        }
    }
    return NULL;
}

static bool multipart_parse(const char *body, size_t body_len,
                            const char *content_type, multipart_form *out) {
    memset(out, 0, sizeof(*out));
    char *boundary = content_type_param(content_type, "boundary");
    if (!boundary || !boundary[0]) {
        free(boundary);
        return false;
    }
    buf delim = {0};
    buf_puts(&delim, "--");
    buf_puts(&delim, boundary);
    free(boundary);

    /* The first delimiter must be found within a small preamble: an
     * unbounded search would accept a body whose "boundary" appears far
     * inside arbitrary data. */
    const char *start = NULL;
    for (size_t i = 0; i + delim.len <= body_len && i < 4096; i++) {
        if (memcmp(body + i, delim.ptr, delim.len) == 0) {
            start = body + i + delim.len;
            break;
        }
    }
    if (!start) {
        buf_free(&delim);
        return false;
    }

    const char *end = body + body_len;
    int guard = 0;
    while (start < end) {
        if (++guard > O1_SERVER_MAX_MULTIPART_PARTS) {
            buf_free(&delim);
            multipart_form_free(out);
            return false;
        }
        /* `start` points at a delimiter (the first iteration already
         * consumed the opening one). A delimiter followed by "--" ends the
         * body; otherwise it is followed by CRLF and the part headers. */
        if (start + delim.len <= end &&
            memcmp(start, delim.ptr, delim.len) == 0) {
            start += delim.len;
            if (start + 2 <= end && start[0] == '-' && start[1] == '-') break;
            if (start + 2 <= end && start[0] == '\r' && start[1] == '\n')
                start += 2;
            else if (start + 1 <= end && start[0] == '\n')
                start += 1;
        }

        const char *hdr_end = NULL;
        for (const char *q = start; q + 4 <= end; q++) {
            if (q[0] == '\r' && q[1] == '\n' && q[2] == '\r' && q[3] == '\n') {
                hdr_end = q;
                break;
            }
        }
        if (!hdr_end) {
            buf_free(&delim);
            multipart_form_free(out);
            return false;
        }

        multipart_part part;
        memset(&part, 0, sizeof(part));
        const char *q = start;
        while (q < hdr_end) {
            const char *line = q;
            while (q < hdr_end && *q != '\n') q++;
            size_t len = (size_t)(q - line);
            if (len && line[len - 1] == '\r') len--;
            if (len >= 19 && strncasecmp(line, "Content-Disposition", 19) == 0) {
                char *disp = xstrndup(line, len);
                char *nm = content_type_param(disp, "name");
                char *fn = content_type_param(disp, "filename");
                if (nm) { free(part.name); part.name = nm; }
                if (fn) { free(part.filename); part.filename = fn; }
                free(disp);
            }
            if (q < hdr_end) q++;
        }

        const char *data = hdr_end + 4;
        const char *next = NULL;
        for (const char *q = data; q + delim.len <= end; q++) {
            if (memcmp(q, delim.ptr, delim.len) != 0) continue;
            /* Only a delimiter that starts a line terminates the part. */
            if (q >= body + 2 && q[-2] == '\r' && q[-1] == '\n') {
                next = q - 2;
                break;
            }
            if (q >= body + 1 && q[-1] == '\n') {
                next = q - 1;
                break;
            }
        }
        if (!next) {
            free(part.name);
            free(part.filename);
            buf_free(&delim);
            multipart_form_free(out);
            return false;
        }
        part.len = (size_t)(next - data);
        part.data = xmalloc(part.len + 1);
        memcpy(part.data, data, part.len);
        part.data[part.len] = '\0';
        if (!part.name) {
            free(part.data);
            free(part.filename);
            buf_free(&delim);
            multipart_form_free(out);
            return false;
        }
        multipart_push(out, part);
        start = next + 2; /* the delimiter itself; the loop re-checks it */
    }
    buf_free(&delim);
    return out->count > 0;
}

static const multipart_part *multipart_text(const multipart_form *m,
                                            const char *name) {
    for (int i = 0; i < m->count; i++) {
        if (m->parts[i].filename) continue;
        if (!strcmp(m->parts[i].name, name)) return &m->parts[i];
    }
    return NULL;
}

static bool multipart_bool(const multipart_part *p, bool *out) {
    if (!p) return false;
    if (!strcmp(p->data, "true") || !strcmp(p->data, "1")) { *out = true; return true; }
    if (!strcmp(p->data, "false") || !strcmp(p->data, "0")) { *out = false; return true; }
    return false;
}

/* ------------------------------------------------------------------ */
/* Request validation and job construction                             */
/* ------------------------------------------------------------------ */

static bool parse_size(const char *s, int *w, int *h) {
    if (!s) return false;
    char *end = NULL;
    long a = strtol(s, &end, 10);
    if (end == s || a <= 0) return false;
    if (*end != 'x' && *end != 'X') return false;
    const char *p = end + 1;
    long b = strtol(p, &end, 10);
    if (end == p || b <= 0) return false;
    while (*end == ' ' || *end == '\t') end++;
    if (*end) return false;
    if (a > 8192 || b > 8192) return false;
    *w = (int)a;
    *h = (int)b;
    return true;
}

/* hd_mode_from_name() silently falls back to T2I, so the server validates
 * the name itself to be able to answer `invalid_mode`. */
static bool mode_from_name_checked(const char *s, hd_mode *out) {
    if (!s || !s[0]) return false;
    if (!strcmp(s, "t2i") || !strcmp(s, "text2image")) { *out = HD_MODE_T2I; return true; }
    if (!strcmp(s, "edit")) { *out = HD_MODE_EDIT; return true; }
    if (!strcmp(s, "personalize") || !strcmp(s, "multi-ref") ||
        !strcmp(s, "multi_ref")) { *out = HD_MODE_PERSONALIZE; return true; }
    if (!strcmp(s, "personalize_layout") || !strcmp(s, "layout")) {
        *out = HD_MODE_PERSONALIZE_LAYOUT;
        return true;
    }
    if (!strcmp(s, "personalize_skeleton") || !strcmp(s, "skeleton")) {
        *out = HD_MODE_PERSONALIZE_SKELETON;
        return true;
    }
    return false;
}

static bool scheduler_from_name_checked(const char *s, hd_scheduler_kind *out) {
    if (!s || !s[0]) return false;
    if (!strcmp(s, "flash")) { *out = HD_SCHED_FLASH; return true; }
    if (!strcmp(s, "flow_match")) { *out = HD_SCHED_FLOW_MATCH; return true; }
    if (!strcmp(s, "default") || !strcmp(s, "unipc")) { *out = HD_SCHED_DEFAULT; return true; }
    return false;
}

static void job_error(server_job *j, int status, const char *type,
                      const char *msg, const char *param, const char *code) {
    server_job_fail(j, status, type, msg, param, code);
}

/* Shared tail of both endpoints: validate the parsed fields, resolve the
 * effective size, fill the generation request and enqueue the job. */
static bool job_submit(server_job *j, parsed_fields *f, int image_count,
                       bool is_edit) {
    if (!f->prompt || !f->prompt[0]) {
        job_error(j, 400, "invalid_request_error", "prompt is required",
                  "prompt", "invalid_request");
        return false;
    }
    if (strlen(f->prompt) > O1_SERVER_MAX_PROMPT_BYTES) {
        job_error(j, 400, "invalid_request_error", "prompt is too long",
                  "prompt", NULL);
        return false;
    }
    if (f->model && f->model[0] && !model_id_matches(g_cfg.profile, f->model)) {
        job_error(j, 404, "invalid_request_error",
                  "the requested model is not loaded by this server",
                  "model", "model_not_found");
        return false;
    }
    if (f->has_lora) {
        job_error(j, 400, "invalid_request_error",
                  "LoRA adapters are configured at server startup, not per request",
                  "o1_lora", "unsupported_parameter");
        return false;
    }
    if (f->response_format_url) {
        job_error(j, 400, "invalid_request_error",
                  "response_format=url is not supported; use b64_json",
                  "response_format", "unsupported_parameter");
        return false;
    }
    if (f->output_format && f->output_format[0] &&
        strcmp(f->output_format, "png") != 0) {
        job_error(j, 400, "invalid_request_error",
                  "only output_format=png is supported", "output_format",
                  "unsupported_output_format");
        return false;
    }
    if (f->compression) {
        job_error(j, 400, "invalid_request_error",
                  "compression is not supported", "compression",
                  "unsupported_parameter");
        return false;
    }
    if (f->background && f->background[0] &&
        strcmp(f->background, "auto") != 0 &&
        strcmp(f->background, "opaque") != 0) {
        job_error(j, 400, "invalid_request_error",
                  "only background=auto or background=opaque is supported",
                  "background", "unsupported_background");
        return false;
    }
    if (f->quality && f->quality[0] && strcmp(f->quality, "auto") != 0) {
        job_error(j, 400, "invalid_request_error",
                  "quality presets are not supported; use quality=auto",
                  "quality", "unsupported_parameter");
        return false;
    }
    if (f->has_mask) {
        job_error(j, 400, "invalid_request_error",
                  "mask-conditioned inpainting is not implemented",
                  "mask", "mask_not_supported");
        return false;
    }

    int n = f->n;
    if (n == 0) n = 1;
    if (n < 1 || n > O1_SERVER_MAX_N) {
        job_error(j, 400, "invalid_request_error",
                  "n must be between 1 and 4", "n", "invalid_request");
        return false;
    }

    /* ---- mode ---- */
    hd_mode mode;
    if (is_edit) {
        if (image_count <= 0) {
            job_error(j, 400, "invalid_request_error",
                      "at least one input image is required", "image",
                      "invalid_image");
            return false;
        }
        mode = image_count == 1 ? HD_MODE_EDIT : HD_MODE_PERSONALIZE;
        if (f->layout_bboxes) mode = HD_MODE_PERSONALIZE_LAYOUT;
    } else {
        mode = HD_MODE_T2I;
    }
    if (f->o1_mode && f->o1_mode[0]) {
        hd_mode explicit_mode;
        if (!mode_from_name_checked(f->o1_mode, &explicit_mode)) {
            job_error(j, 400, "invalid_request_error", "unknown o1_mode",
                      "o1_mode", "invalid_mode");
            return false;
        }
        if (explicit_mode == HD_MODE_STORYBOARD) {
            job_error(j, 400, "invalid_request_error",
                      "storyboard mode is not exposed by this server",
                      "o1_mode", "invalid_mode");
            return false;
        }
        if (!is_edit && explicit_mode != HD_MODE_T2I) {
            job_error(j, 400, "invalid_request_error",
                      "o1_mode requires input images; use /v1/images/edits",
                      "o1_mode", "invalid_mode");
            return false;
        }
        if (is_edit) {
            if (explicit_mode == HD_MODE_EDIT && image_count != 1) {
                job_error(j, 400, "invalid_request_error",
                          "o1_mode=edit requires exactly one input image",
                          "o1_mode", "invalid_mode");
                return false;
            }
            if (explicit_mode == HD_MODE_PERSONALIZE && image_count < 2) {
                job_error(j, 400, "invalid_request_error",
                          "o1_mode=personalize requires at least two input images",
                          "o1_mode", "invalid_mode");
                return false;
            }
            if (explicit_mode == HD_MODE_PERSONALIZE_LAYOUT &&
                !f->layout_bboxes) {
                job_error(j, 400, "invalid_request_error",
                          "o1_mode=personalize_layout requires o1_layout_bboxes",
                          "o1_mode", "invalid_mode");
                return false;
            }
            if (explicit_mode == HD_MODE_PERSONALIZE_SKELETON) {
                job_error(j, 400, "invalid_request_error",
                          "skeleton conditioning is not implemented by this server",
                          "o1_mode", "invalid_mode");
                return false;
            }
            mode = explicit_mode;
        }
    }

    /* ---- aliases ---- */
    if (f->n_aliases != 0 && f->n_aliases != image_count) {
        job_error(j, 400, "invalid_request_error",
                  "o1_reference_aliases must have exactly one entry per input image",
                  "o1_reference_aliases", "reference_alias_count_mismatch");
        return false;
    }

    /* ---- size ---- */
    bool size_auto = !f->size || !f->size[0] || !strcmp(f->size, "auto");
    int req_w = 0, req_h = 0;
    if (!size_auto) {
        if (!parse_size(f->size, &req_w, &req_h)) {
            job_error(j, 400, "invalid_request_error",
                      "size must be WIDTHxHEIGHT or auto", "size",
                      "invalid_size");
            return false;
        }
    }
    if (f->has_keep_aspect && f->keep_aspect && !size_auto) {
        job_error(j, 400, "invalid_request_error",
                  "o1_keep_original_aspect cannot be combined with an explicit size",
                  "o1_keep_original_aspect", "invalid_size");
        return false;
    }

    int eff_w = 2048, eff_h = 2048;
    if (!size_auto) {
        hd_resolution_snap(req_w, req_h, &eff_w, &eff_h);
        if (f->has_exact_size && f->exact_size &&
            (eff_w != req_w || eff_h != req_h)) {
            job_error(j, 400, "invalid_request_error",
                      "requested size is not a supported HiDream bucket",
                      "size", "unsupported_size");
            return false;
        }
        if (eff_w != req_w || eff_h != req_h) {
            fprintf(stderr,
                    "hidream-server: request size %dx%d snapped to %dx%d\n",
                    req_w, req_h, eff_w, eff_h);
        }
    }

    /* ---- scheduler ---- */
    hd_scheduler_kind sched = HD_SCHED_DEFAULT;
    if (f->has_scheduler) {
        if (!scheduler_from_name_checked(f->scheduler, &sched)) {
            job_error(j, 400, "invalid_request_error", "unknown o1_scheduler",
                      "o1_scheduler", "invalid_scheduler");
            return false;
        }
    }

    /* ---- layout ---- */
    if (f->layout_bboxes) {
        hd_layout_condition *conds = NULL;
        size_t n_conds = 0;
        if (hd_layout_parse(f->layout_bboxes, &conds, &n_conds) != HD_OK) {
            job_error(j, 400, "invalid_request_error",
                      "o1_layout_bboxes is not a valid layout description",
                      "o1_layout_bboxes", "invalid_request");
            return false;
        }
        j->layout_conds = conds;
        j->n_layout = n_conds;
    }

    /* ---- fill the generation request ---- */
    j->prompt = xstrdup(f->prompt);
    j->n = n;
    j->requested_w = size_auto ? eff_w : req_w;
    j->requested_h = size_auto ? eff_h : req_h;
    j->size_auto = size_auto;
    j->verbose = f->verbose;

    hd_generation_request *r = &j->req;
    memset(r, 0, sizeof(*r));
    r->prompt = j->prompt;
    r->mode = mode;
    r->profile = g_cfg.profile;
    r->width = eff_w;
    r->height = eff_h;
    r->seed = f->has_seed ? f->seed : random_u64();
    r->steps = f->has_steps ? f->steps : 0;
    r->guidance_scale = f->has_guidance ? (float)f->guidance : -1.0f;
    r->shift = f->has_shift ? (float)f->shift : -1.0f;
    r->scheduler = f->has_scheduler ? sched : HD_SCHED_DEFAULT;
    r->keep_original_aspect = (f->has_keep_aspect && f->keep_aspect) ? 1 : 0;
    r->noise_scale_start = f->has_noise_start ? (float)f->noise_start : 0.0f;
    r->noise_scale_end = f->has_noise_end ? (float)f->noise_end : 0.0f;
    r->noise_clip_std = f->has_noise_clip ? (float)f->noise_clip : 0.0f;
    r->layout = j->layout_conds;
    r->references = j->refs;
    r->reference_count = (size_t)image_count;

    hd_request_defaults(r);

    if (hd_request_validate(r) != HD_OK) {
        job_error(j, 400, "invalid_request_error", hd_last_error(), NULL, NULL);
        return false;
    }
    return true;
}

/* ------------------------------------------------------------------ */
/* Job execution (single model worker)                                 */
/* ------------------------------------------------------------------ */

static void job_run(server_job *j) {
    long long t0 = wall_ms();
    j->queue_wait_ms = t0 - j->total_ms; /* total_ms holds the enqueue stamp */

    /* The reference-alias expansion lives in the generation path and is
     * gated on this environment variable, exactly like the CLI's
     * --verbose. It is process-wide, so it is restored after the job. */
    char *prev_verbose_ref = getenv("O1_VERBOSE_REF");
    char *saved_verbose_ref = prev_verbose_ref ? xstrdup(prev_verbose_ref) : NULL;
    if (j->verbose) setenv("O1_VERBOSE_REF", "1", 1);
    else unsetenv("O1_VERBOSE_REF");

    hd_generation_request req = j->req;
    uint64_t base_seed = req.seed;

    buf data = {0};
    buf body = {0};
    int eff_w = 0, eff_h = 0;
    long long gen_total = 0, enc_total = 0;

    for (int i = 0; i < j->n; i++) {
        req.seed = base_seed + (uint64_t)i;
        unsigned char *rgb = NULL;
        int w = 0, h = 0;
        long long g0 = wall_ms();
        hd_status st = hd_generation_engine_generate(g_engine, &req, &rgb, &w, &h);
        gen_total += wall_ms() - g0;
        if (st != HD_OK) {
            free(rgb);
            job_error(j, 500, "server_error", hd_last_error(), NULL, NULL);
            goto done;
        }
        eff_w = w;
        eff_h = h;

        long long e0 = wall_ms();
        unsigned char *png = NULL;
        size_t png_len = 0;
        int enc_rc = hd_png_encode_rgb(rgb, w, h, &png, &png_len);
        free(rgb);
        if (enc_rc != 0) {
            job_error(j, 500, "server_error", "PNG encoding failed", NULL, NULL);
            goto done;
        }

        if (i) buf_putc(&data, ',');
        buf_puts(&data, "{\"b64_json\":\"");
        base64_encode(&data, png, png_len);
        free(png);
        buf_printf(&data,
                   "\",\"output_format\":\"png\",\"size\":\"%dx%d\","
                   "\"quality\":\"auto\"}",
                   w, h);
        enc_total += wall_ms() - e0;
    }

    buf_printf(&body, "{\"created\":%lld,\"data\":[", (long long)time(NULL));
    buf_append(&body, data.ptr, data.len);
    buf_puts(&body, "],\"o1\":{");
    buf_printf(&body, "\"seed\":%llu,", (unsigned long long)base_seed);
    buf_puts(&body, "\"scheduler\":");
    json_escape(&body, hd_scheduler_name(req.scheduler));
    buf_printf(&body, ",\"steps\":%d,", req.steps);
    buf_printf(&body, "\"requested_size\":\"%dx%d\",", j->requested_w,
               j->requested_h);
    buf_printf(&body, "\"effective_size\":\"%dx%d\"", eff_w, eff_h);
    buf_puts(&body, "}}\n");

    j->body = buf_take(&body);
    j->status = 200;

    j->generation_ms = gen_total;
    j->encode_ms = enc_total;
    j->total_ms = wall_ms() - t0;

done:
    buf_free(&data);
    buf_free(&body);
    if (saved_verbose_ref) {
        setenv("O1_VERBOSE_REF", saved_verbose_ref, 1);
        free(saved_verbose_ref);
    } else {
        unsetenv("O1_VERBOSE_REF");
    }
}

/* ------------------------------------------------------------------ */
/* Endpoint handlers                                                   */
/* ------------------------------------------------------------------ */

static void send_models(int fd) {
    buf b = {0};
    buf_puts(&b, "{\"object\":\"list\",\"data\":[{\"id\":");
    json_escape(&b, canonical_model_id(g_cfg.profile));
    buf_puts(&b, ",\"object\":\"model\",\"created\":0,\"owned_by\":\"local\"}]}\n");
    http_response(fd, g_cfg.cors, 200, "application/json", b.ptr);
    buf_free(&b);
}

static void send_health(int fd) {
    buf b = {0};
    buf_puts(&b, "{\"status\":\"ok\",\"model\":");
    json_escape(&b, canonical_model_id(g_cfg.profile));
    buf_puts(&b, ",\"profile\":");
    json_escape(&b, g_cfg.profile);
    buf_printf(&b, ",\"ready\":%s}\n", g_engine ? "true" : "false");
    http_response(fd, g_cfg.cors, 200, "application/json", b.ptr);
    buf_free(&b);
}

/* Materialize one uploaded image into a private temp file. The client
 * filename is never used: only the bytes matter, and a server-owned path
 * removes any traversal or symlink concern. */
static char *materialize_upload(const char *data, size_t len) {
    const char *dir = getenv("TMPDIR");
    if (!dir || !dir[0]) dir = "/tmp";
    char tmpl[PATH_MAX];
    snprintf(tmpl, sizeof(tmpl), "%s/hidream-srv-XXXXXX", dir);
    int fd = mkstemp(tmpl);
    if (fd < 0) return NULL;
    (void)fchmod(fd, 0600);
    size_t off = 0;
    while (off < len) {
        ssize_t w = write(fd, data + off, len - off);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) { close(fd); unlink(tmpl); return NULL; }
        off += (size_t)w;
    }
    close(fd);
    return xstrdup(tmpl);
}

static void dispatch_job(int fd, server_job *j) {
    j->total_ms = wall_ms(); /* enqueue stamp; job_run converts it to wait */
    if (!enqueue(j)) {
        http_error_full(fd, g_cfg.cors, 429, "server_error",
                        "the generation queue is full, retry later", NULL,
                        "queue_full");
        server_job_free(j);
        return;
    }
    wait_for_job_or_disconnect(j);
    if (j->cancelled) {
        server_job_free(j);
        return;
    }
    if (j->status == 200 && j->body) {
        http_response(fd, g_cfg.cors, 200, "application/json", j->body);
    } else {
        http_error_full(fd, g_cfg.cors, j->status, j->error_type,
                        j->error_msg, j->error_param, j->error_code);
    }
    if (j->verbose) {
        fprintf(stderr,
                "hidream-server: queue_wait_ms=%lld parse_ms=%lld "
                "preprocess_ms=%lld generation_ms=%lld encode_ms=%lld "
                "total_ms=%lld\n",
                j->queue_wait_ms, j->parse_ms, j->preprocess_ms,
                j->generation_ms, j->encode_ms, j->total_ms);
    }
    server_job_free(j);
}

static void handle_generations(int fd, http_request *r) {
    parsed_fields f;
    memset(&f, 0, sizeof(f));
    if (!parse_json_body(r->body, &f)) {
        parsed_fields_free(&f);
        http_error(fd, g_cfg.cors, 400, "malformed JSON request body");
        return;
    }
    server_job *j = server_job_new(fd);
    if (!job_submit(j, &f, 0, false)) {
        parsed_fields_free(&f);
        http_error_full(fd, g_cfg.cors, j->status, j->error_type, j->error_msg,
                        j->error_param, j->error_code);
        server_job_free(j);
        return;
    }
    parsed_fields_free(&f);
    dispatch_job(fd, j);
}

static void handle_edits_json(int fd, http_request *r) {
    parsed_fields f;
    memset(&f, 0, sizeof(f));
    if (!parse_json_body(r->body, &f)) {
        parsed_fields_free(&f);
        http_error(fd, g_cfg.cors, 400, "malformed JSON request body");
        return;
    }
    parsed_fields_free(&f);
    http_error_full(fd, g_cfg.cors, 400, "invalid_request_error",
                    "input images must be uploaded as multipart/form-data",
                    "image", "invalid_image");
}

static void handle_edits_multipart(int fd, http_request *r) {
    multipart_form form;
    if (!multipart_parse(r->body, r->body_len, r->content_type, &form)) {
        multipart_form_free(&form);
        http_error(fd, g_cfg.cors, 400, "malformed multipart/form-data body");
        return;
    }

    parsed_fields f;
    memset(&f, 0, sizeof(f));
    server_job *j = server_job_new(fd);
    bool ok = true;

    const multipart_part *p;
    if ((p = multipart_text(&form, "prompt"))) f.prompt = xstrndup(p->data, p->len);
    if ((p = multipart_text(&form, "model"))) f.model = xstrndup(p->data, p->len);
    if ((p = multipart_text(&form, "size"))) f.size = xstrndup(p->data, p->len);
    if ((p = multipart_text(&form, "output_format")))
        f.output_format = xstrndup(p->data, p->len);
    if ((p = multipart_text(&form, "user"))) f.user = xstrndup(p->data, p->len);
    if ((p = multipart_text(&form, "background")))
        f.background = xstrndup(p->data, p->len);
    if ((p = multipart_text(&form, "quality")))
        f.quality = xstrndup(p->data, p->len);
    if ((p = multipart_text(&form, "compression")))
        f.compression = xstrndup(p->data, p->len);
    if ((p = multipart_text(&form, "o1_mode")))
        f.o1_mode = xstrndup(p->data, p->len);
    if ((p = multipart_text(&form, "o1_scheduler"))) {
        f.scheduler = xstrndup(p->data, p->len);
        f.has_scheduler = true;
    }
    if ((p = multipart_text(&form, "o1_layout_bboxes")))
        f.layout_bboxes = xstrndup(p->data, p->len);
    if ((p = multipart_text(&form, "n"))) {
        char *end = NULL;
        long v = strtol(p->data, &end, 10);
        if (end == p->data || v < 0 || v > INT_MAX) ok = false;
        else f.n = (int)v;
    }
    if ((p = multipart_text(&form, "o1_seed"))) {
        char *end = NULL;
        errno = 0;
        unsigned long long v = strtoull(p->data, &end, 10);
        if (end == p->data || errno == ERANGE) ok = false;
        else { f.seed = (uint64_t)v; f.has_seed = true; }
    }
    if ((p = multipart_text(&form, "o1_steps"))) {
        char *end = NULL;
        long v = strtol(p->data, &end, 10);
        if (end == p->data) ok = false;
        else { f.steps = (int)v; f.has_steps = true; }
    }
    if ((p = multipart_text(&form, "o1_guidance_scale"))) {
        char *end = NULL;
        f.guidance = strtod(p->data, &end);
        if (end == p->data) ok = false; else f.has_guidance = true;
    }
    if ((p = multipart_text(&form, "o1_shift"))) {
        char *end = NULL;
        f.shift = strtod(p->data, &end);
        if (end == p->data) ok = false; else f.has_shift = true;
    }
    if ((p = multipart_text(&form, "o1_noise_start"))) {
        char *end = NULL;
        f.noise_start = strtod(p->data, &end);
        if (end == p->data) ok = false; else f.has_noise_start = true;
    }
    if ((p = multipart_text(&form, "o1_noise_end"))) {
        char *end = NULL;
        f.noise_end = strtod(p->data, &end);
        if (end == p->data) ok = false; else f.has_noise_end = true;
    }
    if ((p = multipart_text(&form, "o1_noise_clip"))) {
        char *end = NULL;
        f.noise_clip = strtod(p->data, &end);
        if (end == p->data) ok = false; else f.has_noise_clip = true;
    }
    if ((p = multipart_text(&form, "o1_keep_original_aspect"))) {
        if (!multipart_bool(p, &f.keep_aspect)) ok = false;
        else f.has_keep_aspect = true;
    }
    if ((p = multipart_text(&form, "o1_exact_size"))) {
        if (!multipart_bool(p, &f.exact_size)) ok = false;
        else f.has_exact_size = true;
    }
    if ((p = multipart_text(&form, "o1_verbose"))) {
        if (!multipart_bool(p, &f.verbose)) ok = false;
    }
    if ((p = multipart_text(&form, "o1_reference_aliases"))) {
        if (!parse_alias_array(p->data, &f.aliases, &f.n_aliases)) ok = false;
    }
    /* `mask` is an uploaded file in the OpenAI API, not a text field; any
     * part with that name (text or file) marks the request as mask-bearing. */
    for (int i = 0; i < form.count; i++)
        if (!strcmp(form.parts[i].name, "mask")) f.has_mask = true;
    if (multipart_text(&form, "o1_lora")) f.has_lora = true;

    if (!ok) {
        parsed_fields_free(&f);
        multipart_form_free(&form);
        server_job_free(j);
        http_error(fd, g_cfg.cors, 400, "malformed multipart field value");
        return;
    }

    /* Collect image parts in upload order. `image` and `image[]` are both
     * accepted; the order of appearance is the reference order. */
    int n_images = 0;
    for (int i = 0; i < form.count; i++) {
        const multipart_part *part = &form.parts[i];
        if (!part->filename) continue;
        if (strcmp(part->name, "image") != 0 &&
            strcmp(part->name, "image[]") != 0)
            continue;
        if (n_images >= HD_SEQ_MAX_REFS) {
            parsed_fields_free(&f);
            multipart_form_free(&form);
            server_job_free(j);
            http_error_full(fd, g_cfg.cors, 400, "invalid_request_error",
                            "too many input images", "image",
                            "too_many_images");
            return;
        }
        if (part->len > O1_SERVER_MAX_IMAGE_BYTES) {
            parsed_fields_free(&f);
            multipart_form_free(&form);
            server_job_free(j);
            http_error_full(fd, g_cfg.cors, 413, "invalid_request_error",
                            "input image is too large", "image",
                            "invalid_image");
            return;
        }
        char *path = materialize_upload(part->data, part->len);
        if (!path) {
            parsed_fields_free(&f);
            multipart_form_free(&form);
            server_job_free(j);
            http_error(fd, g_cfg.cors, 500, "could not store the uploaded image");
            return;
        }
        /* Reject undecodable uploads here (400 invalid_image) rather than
         * letting the generation worker fail with a 500. */
        hd_image probe;
        if (hd_image_load(path, &probe) != HD_OK) {
            unlink(path);
            free(path);
            parsed_fields_free(&f);
            multipart_form_free(&form);
            server_job_free(j);
            http_error_full(fd, g_cfg.cors, 400, "invalid_request_error",
                            "the uploaded image could not be decoded", "image",
                            "invalid_image");
            return;
        }
        hd_image_free(&probe);
        j->ref_paths[n_images] = path;
        j->refs[n_images].path = path;
        j->refs[n_images].role = HD_REF_SUBJECT;
        j->refs[n_images].alias = NULL;
        n_images++;
    }
    for (int i = 0; i < f.n_aliases && i < n_images; i++) {
        j->ref_aliases[i] = xstrdup(f.aliases[i]);
        j->refs[i].alias = j->ref_aliases[i];
    }

    multipart_form_free(&form);

    if (!job_submit(j, &f, n_images, true)) {
        parsed_fields_free(&f);
        http_error_full(fd, g_cfg.cors, j->status, j->error_type, j->error_msg,
                        j->error_param, j->error_code);
        server_job_free(j);
        return;
    }
    parsed_fields_free(&f);
    dispatch_job(fd, j);
}

static bool authorized(http_request *r) {
    if (!g_cfg.api_key) return true;
    const char *h = r->authorization;
    if (!h) return false;
    const char *prefix = "Bearer ";
    if (strncmp(h, prefix, strlen(prefix)) != 0) return false;
    const char *tok = h + strlen(prefix);
    const char *want = g_cfg.api_key;
    size_t n = strlen(tok), m = strlen(want);
    unsigned char diff = (unsigned char)(n ^ m);
    for (size_t i = 0; i < n && i < m; i++)
        diff |= (unsigned char)(tok[i] ^ want[i]);
    return diff == 0;
}

static void handle_request(int fd, http_request *r) {
    if (strcmp(r->method, "OPTIONS") == 0) {
        http_response(fd, g_cfg.cors, 204, NULL, NULL);
        return;
    }
    /* The operational health probe is intentionally unauthenticated: it is
     * used by supervisors and must report liveness even when an API key is
     * configured. It exposes no request or model data. */
    if (strcmp(r->method, "GET") == 0 && strcmp(r->path, "/healthz") == 0) {
        send_health(fd);
        return;
    }
    if (!authorized(r)) {
        http_error_full(fd, g_cfg.cors, 401, "invalid_request_error",
                        "missing or invalid API key", NULL, "invalid_api_key");
        return;
    }
    if (strcmp(r->method, "GET") == 0 && strcmp(r->path, "/v1/models") == 0) {
        send_models(fd);
        return;
    }
    if (strcmp(r->method, "POST") == 0 &&
        strcmp(r->path, "/v1/images/generations") == 0) {
        handle_generations(fd, r);
        return;
    }
    if (strcmp(r->method, "POST") == 0 &&
        strcmp(r->path, "/v1/images/edits") == 0) {
        if (r->content_type &&
            strncasecmp(r->content_type, "multipart/form-data", 19) == 0) {
            handle_edits_multipart(fd, r);
        } else if (r->content_type &&
                   strncasecmp(r->content_type, "application/json", 16) == 0) {
            handle_edits_json(fd, r);
        } else {
            http_error_full(fd, g_cfg.cors, 415, "invalid_request_error",
                            "Content-Type must be multipart/form-data",
                            NULL, "unsupported_media_type");
        }
        return;
    }
    http_error(fd, g_cfg.cors, 404, "unknown endpoint");
}

/* ------------------------------------------------------------------ */
/* CLI                                                                 */
/* ------------------------------------------------------------------ */

static void usage(FILE *f) {
    fprintf(f,
        "hidream-server — OpenAI-compatible Images API for HiDream-O1\n"
        "\n"
        "Usage: hidream-server [options]\n"
        "\n"
        "  --model-path PATH       production GGUF model file (required)\n"
        "  --device N              CUDA device index (default 0)\n"
        "  --host HOST             listen address (default 127.0.0.1)\n"
        "  --port PORT             listen port (default 8000)\n"
        "  --cors                  enable permissive CORS headers\n"
        "  --max-body-mb N         maximum request body size (default 64)\n"
        "  --queue-depth N         pending generation jobs (default 8)\n"
        "  --api-key KEY           require Authorization: Bearer KEY\n"
        "  --lora FILE[:MULT]      merge a LoRA adapter at startup (repeatable)\n"
        "  -h, --help              show this help\n"
        "\n"
        "Internal / debug:\n"
        "  --model dev|base        profile name (internal; prefer --model-path)\n"
        "  --model-dir PATH        safetensors directory (internal)\n"
        "  --config-dir DIR        profile config directory (default config)\n");
}

static bool parse_lora_spec(const char *arg, hd_lora_spec *out) {
    const char *colon = strrchr(arg, ':');
    if (colon && colon != arg) {
        char *end = NULL;
        float m = strtof(colon + 1, &end);
        if (end && *end == '\0' && end != colon + 1) {
            out->path = xstrndup(arg, (size_t)(colon - arg));
            out->multiplier = m;
            return true;
        }
    }
    out->path = xstrdup(arg);
    out->multiplier = 1.0f;
    return true;
}

#ifndef O1_SERVER_TEST
int main(int argc, char **argv) {
    const char *profile = "dev";
    const char *model_dir = NULL;
    const char *model_path = NULL;
    const char *config_dir = "config";
    int device_id = 0;
    const char *host = "127.0.0.1";
    int port = 8000;
    bool cors = false;
    size_t max_body = 64u * 1024u * 1024u;
    int queue_depth = 8;
    const char *api_key = NULL;
    hd_lora_spec loras[O1_SERVER_MAX_LORAS];
    int lora_count = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(stdout);
            return 0;
        } else if (!strcmp(a, "--model-path") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (!strcmp(a, "--model") && i + 1 < argc) {
            profile = argv[++i];
        } else if (!strcmp(a, "--model-dir") && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (!strcmp(a, "--config-dir") && i + 1 < argc) {
            config_dir = argv[++i];
        } else if (!strcmp(a, "--device") && i + 1 < argc) {
            device_id = atoi(argv[++i]);
        } else if (!strcmp(a, "--host") && i + 1 < argc) {
            host = argv[++i];
        } else if (!strcmp(a, "--port") && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (!strcmp(a, "--cors")) {
            cors = true;
        } else if (!strcmp(a, "--max-body-mb") && i + 1 < argc) {
            long mb = atol(argv[++i]);
            if (mb <= 0) { fprintf(stderr, "invalid --max-body-mb\n"); return 2; }
            max_body = (size_t)mb * 1024u * 1024u;
        } else if (!strcmp(a, "--queue-depth") && i + 1 < argc) {
            queue_depth = atoi(argv[++i]);
        } else if (!strcmp(a, "--api-key") && i + 1 < argc) {
            api_key = argv[++i];
        } else if (!strcmp(a, "--lora") && i + 1 < argc) {
            if (lora_count >= O1_SERVER_MAX_LORAS) {
                fprintf(stderr, "too many --lora options\n");
                return 2;
            }
            parse_lora_spec(argv[++i], &loras[lora_count++]);
        } else {
            fprintf(stderr, "hidream-server: unknown option '%s'\n", a);
            usage(stderr);
            return 2;
        }
    }

    if (strcmp(profile, "dev") != 0 && strcmp(profile, "base") != 0) {
        fprintf(stderr, "hidream-server: --model must be dev or base\n");
        return 2;
    }
    if (port <= 0 || port > 65535) {
        fprintf(stderr, "hidream-server: invalid --port\n");
        return 2;
    }
    if (queue_depth <= 0) queue_depth = 1;

    /* Resolve the runtime profile. The public interface is a single
     * --model-path pointing at a production GGUF; the profile is inferred
     * from its metadata. --model/--model-dir remain for internal debug. */
    hd_profile resolved;
    memset(&resolved, 0, sizeof(resolved));
    int have_resolved = 0;
    if (model_path) {
        if (hd_profile_from_gguf(model_path, &resolved) != HD_OK) {
            fprintf(stderr, "hidream-server: %s\n", hd_last_error());
            return 2;
        }
        have_resolved = 1;
        profile = resolved.profile;
        model_dir = model_path;
    } else if (!model_dir) {
        model_dir = strcmp(profile, "base") == 0
                        ? "artifacts/models/hidream-o1-base-bf16.gguf"
                        : "artifacts/models/hidream-o1-dev-bf16.gguf";
    }

    memset(&g_cfg, 0, sizeof(g_cfg));
    g_cfg.profile = profile;
    g_cfg.config_dir = config_dir;
    g_cfg.model_path = model_dir;
    g_cfg.device_id = device_id;
    g_cfg.host = host;
    g_cfg.port = port;
    g_cfg.cors = cors;
    g_cfg.max_body = max_body;
    g_cfg.queue_depth = queue_depth;
    g_cfg.api_key = api_key;
    g_cfg.lora_count = lora_count;
    for (int i = 0; i < lora_count; i++) g_cfg.loras[i] = loras[i];

    g_queue_cap = queue_depth;
    g_queue = xmalloc((size_t)g_queue_cap * sizeof(*g_queue));

    signal(SIGINT, stop_signal_handler);
    signal(SIGTERM, stop_signal_handler);
    signal(SIGPIPE, SIG_IGN);

    /* The model is loaded BEFORE the socket is opened: a client must never
     * observe a listening port that cannot serve a request. */
    fprintf(stderr, "hidream-server: loading %s model from %s ...\n", profile,
            model_dir);
    hd_lora_config lora_cfg;
    lora_cfg.items = loras;
    lora_cfg.count = (size_t)lora_count;
    hd_generation_engine_options opts;
    memset(&opts, 0, sizeof(opts));
    opts.profile = profile;
    opts.config_dir = config_dir;
    opts.model_path = model_dir;
    opts.device_id = device_id;
    opts.lora = lora_count > 0 ? &lora_cfg : NULL;
    g_engine = hd_generation_engine_open(&opts);
    if (!g_engine) {
        fprintf(stderr, "hidream-server: model load failed: %s\n",
                hd_last_error());
        return 1;
    }

    int listen_fd = listen_on(host, port);
    if (listen_fd < 0) {
        fprintf(stderr, "hidream-server: cannot listen on %s:%d: %s\n", host,
                port, strerror(errno));
        hd_generation_engine_close(g_engine);
        return 1;
    }
    g_listen_fd = listen_fd;

    pthread_t worker;
    if (pthread_create(&worker, NULL, worker_main, NULL) != 0) {
        fprintf(stderr, "hidream-server: cannot start worker thread\n");
        close(listen_fd);
        hd_generation_engine_close(g_engine);
        return 1;
    }

    fprintf(stderr, "hidream-server: listening on http://%s:%d (model %s)\n",
            host, port, canonical_model_id(profile));

    while (!g_stop_requested) {
        /* A signal may be delivered to any thread, so closing the listen fd
         * from the handler does not reliably wake a blocked accept(). Poll
         * with a short timeout so the stop flag is observed promptly. */
        struct pollfd pfd;
        pfd.fd = listen_fd;
        pfd.events = POLLIN;
        pfd.revents = 0;
        int pr = poll(&pfd, 1, 200);
        if (pr < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (pr == 0) continue;
        if (g_stop_requested) break;

        struct sockaddr_in sa;
        socklen_t slen = sizeof(sa);
        int cfd = accept(listen_fd, (struct sockaddr *)&sa, &slen);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            if (g_stop_requested) break;
            if (errno == EBADF || errno == EINVAL) break;
            continue;
        }
        pthread_t th;
        if (pthread_create(&th, NULL, client_main, (void *)(intptr_t)cfd) != 0) {
            close(cfd);
            continue;
        }
        pthread_detach(th);
    }

    fprintf(stderr, "hidream-server: shutting down\n");
    if (g_listen_fd >= 0) {
        close((int)g_listen_fd);
        g_listen_fd = -1;
    }
    queue_stop();
    pthread_join(worker, NULL);
    /* Preload lifecycle release gate (spec 32.2): the resident engine must
     * have loaded the weights and resolved both bindings exactly once, and
     * must have served every accepted request. */
    if (g_engine) {
        fprintf(stderr,
                "hidream-server: lifecycle open=1 weight_load=%d "
                "forward_resolve=%d vision_resolve=%d requests=%d close=1\n",
                hd_generation_engine_load_count(g_engine),
                hd_generation_engine_forward_resolve_count(g_engine),
                hd_generation_engine_vision_resolve_count(g_engine),
                hd_generation_engine_request_count(g_engine));
    }
    hd_generation_engine_close(g_engine);
    g_engine = NULL;
    free(g_queue);
    if (have_resolved) hd_profile_free(&resolved);
    return 0;
}
#endif /* O1_SERVER_TEST */
