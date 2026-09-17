/*
 * refiner.c — native prompt-refiner clients (contract section 7).
 *
 * Minimal HTTP/1.1 POST over POSIX sockets; no libcurl, no Python.
 * JSON built/parsed with src/io/json.h.
 *
 * The refiner is an OPTIONAL companion: no engine code calls these
 * functions implicitly, so the engine remains offline-capable.
 */

#define _POSIX_C_SOURCE 200809L /* strdup, getaddrinfo */

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include "json.h"
#include "refiner.h"

/* ── System prompts (frozen audit F_PROMPT_STORYBOARD_AUDIT.md) ─────────── */

/*
 * Legacy agent (prompt_agent.py:6-64): reasoning-driven rewrite into an
 * explicit, detailed, directly image-generatable English prompt using the
 * SCALIST framework; output is a JSON object {"prompt", "reasoning",
 * "resolved_knowledge"}.
 */
static const char kLegacySystemPrompt[] =
    "You are a professional AI image-generation prompt engineer and creative "
    "director. Analyze the user's raw image request, reason out implicit "
    "knowledge and the best visual plan, and rewrite it into an explicit, "
    "detailed English prompt that can be used directly for image generation. "
    "Use the SCALIST framework (Subject, Composition, Action, Location, Image "
    "style, Specs, Text rendering). Resolve implicit knowledge (poems, quotes, "
    "formulas, historical figures, landmarks, cultural symbols) into concrete "
    "visible details; anchor spatial relations explicitly; keep requested text "
    "verbatim in double quotes with font, color, material and position. Output "
    "a single coherent English paragraph of 80-220 words, self-contained, with "
    "the most important subject and intent first. Output only JSON with no "
    "other text: {\"prompt\": \"English single-paragraph prompt\", "
    "\"reasoning\": \"your reasoning in Chinese\", \"resolved_knowledge\": "
    "\"resolved implicit knowledge in Chinese, or 'none'\"}";

/*
 * Dev-2604 (prompt_agent_v2.py:7-43): English rewrite only, no JSON/reasoning
 * output.
 */
static const char kV2SystemPrompt[] =
    "You are a professional AI image-generation prompt engineer and creative "
    "director. Analyze the user's raw image request, reason out implicit "
    "knowledge and the best visual plan, and rewrite it into an explicit, "
    "detailed English prompt that can be used directly for image generation. "
    "Use the SCALIST framework (Subject, Composition, Action, Location, Image "
    "style, Specs, Text rendering). Resolve implicit knowledge into concrete "
    "visible details; anchor spatial relations explicitly; keep requested text "
    "verbatim in double quotes. Output only the rewritten English prompt, "
    "nothing else.";

/* ── JSON escaping ──────────────────────────────────────────────────────── */

static char *json_escape(const char *s) {
    size_t n = 0;
    for (const char *p = s; *p; p++) {
        unsigned char c = (unsigned char)*p;
        if (c == '"' || c == '\\') n += 2;
        else if (c == '\n' || c == '\r' || c == '\t') n += 2;
        else if (c < 0x20) n += 6; /* \u00XX */
        else n += 1;
    }
    char *out = (char *)malloc(n + 1);
    if (!out) return NULL;
    char *q = out;
    for (const char *p = s; *p; p++) {
        unsigned char c = (unsigned char)*p;
        switch (c) {
            case '"':  *q++ = '\\'; *q++ = '"';  break;
            case '\\': *q++ = '\\'; *q++ = '\\'; break;
            case '\n': *q++ = '\\'; *q++ = 'n';  break;
            case '\r': *q++ = '\\'; *q++ = 'r';  break;
            case '\t': *q++ = '\\'; *q++ = 't';  break;
            default:
                if (c < 0x20) {
                    static const char kHex[] = "0123456789abcdef";
                    *q++ = '\\'; *q++ = 'u'; *q++ = '0'; *q++ = '0';
                    *q++ = kHex[(c >> 4) & 0xF]; *q++ = kHex[c & 0xF];
                } else {
                    *q++ = (char)c;
                }
        }
    }
    *q = '\0';
    return out;
}

/* ── Endpoint parsing ───────────────────────────────────────────────────── */

typedef struct {
    char host[256];
    char port[16];
    char path[512];
} hd_refiner_endpoint;

static int parse_endpoint(const char *endpoint, hd_refiner_endpoint *ep) {
    memset(ep, 0, sizeof(*ep));
    strcpy(ep->port, "80");
    strcpy(ep->path, "/v1/chat/completions");

    const char *p = endpoint;
    if (strncmp(p, "http://", 7) == 0) {
        p += 7;
    } else if (strncmp(p, "https://", 8) == 0) {
        return -1; /* no TLS in the minimal client */
    }
    if (*p == '\0') return -1;

    const char *slash = strchr(p, '/');
    const char *colon = strchr(p, ':');
    size_t host_len;
    if (slash && (!colon || slash < colon)) {
        host_len = (size_t)(slash - p);
    } else if (colon) {
        host_len = (size_t)(colon - p);
    } else {
        host_len = strlen(p);
    }
    if (host_len == 0 || host_len >= sizeof(ep->host)) return -1;
    memcpy(ep->host, p, host_len);
    ep->host[host_len] = '\0';

    if (colon && (!slash || colon < slash)) {
        const char *port_start = colon + 1;
        const char *port_end = slash ? slash : port_start + strlen(port_start);
        size_t port_len = (size_t)(port_end - port_start);
        if (port_len == 0 || port_len >= sizeof(ep->port)) return -1;
        memcpy(ep->port, port_start, port_len);
        ep->port[port_len] = '\0';
    }

    if (slash) {
        size_t path_len = strlen(slash);
        if (path_len >= sizeof(ep->path)) return -1;
        memcpy(ep->path, slash, path_len + 1);
        /* strip trailing slashes, then append chat/completions */
        while (path_len > 1 && ep->path[path_len - 1] == '/') {
            ep->path[--path_len] = '\0';
        }
        if (strstr(ep->path, "/chat/completions") == NULL) {
            size_t need = path_len + strlen("/chat/completions");
            if (need >= sizeof(ep->path)) return -1;
            strcat(ep->path, "/chat/completions");
        }
    }
    return 0;
}

/* ── Minimal HTTP/1.1 POST ──────────────────────────────────────────────── */

static int http_post(const hd_refiner_endpoint *ep, const char *body,
                     char **out_response) {
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    struct addrinfo *res = NULL;
    int rc = getaddrinfo(ep->host, ep->port, &hints, &res);
    if (rc != 0) return -1;
    if (!res) return -1;

    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) return -1;

    size_t body_len = strlen(body);
    char head[1024];
    int hn = snprintf(head, sizeof(head),
                      "POST %s HTTP/1.1\r\n"
                      "Host: %s:%s\r\n"
                      "Content-Type: application/json\r\n"
                      "Content-Length: %zu\r\n"
                      "Connection: close\r\n"
                      "\r\n",
                      ep->path, ep->host, ep->port, body_len);
    if (hn < 0 || (size_t)hn >= sizeof(head)) {
        close(fd);
        return -1;
    }

    /* send head, then body (separate buffers) */
    size_t sent = 0;
    while (sent < (size_t)hn) {
        ssize_t n = send(fd, head + sent, (size_t)hn - sent, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            close(fd);
            return -1;
        }
        sent += (size_t)n;
    }
    sent = 0;
    while (sent < body_len) {
        ssize_t n = send(fd, body + sent, body_len - sent, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            close(fd);
            return -1;
        }
        sent += (size_t)n;
    }

    /* read response */
    size_t cap = 8192, len = 0;
    char *resp = (char *)malloc(cap);
    if (!resp) {
        close(fd);
        return -1;
    }
    for (;;) {
        if (len + 4096 + 1 > cap) {
            size_t ncap = cap * 2;
            char *nr = (char *)realloc(resp, ncap);
            if (!nr) {
                free(resp);
                close(fd);
                return -1;
            }
            resp = nr;
            cap = ncap;
        }
        ssize_t n = recv(fd, resp + len, cap - len - 1, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            free(resp);
            close(fd);
            return -1;
        }
        if (n == 0) break;
        len += (size_t)n;
    }
    close(fd);
    resp[len] = '\0';
    *out_response = resp;
    return 0;
}

/* ── Shared request/response handling ───────────────────────────────────── */

static hd_status refine_common(const char *endpoint, const char *api_key,
                               const char *system_prompt, const char *raw_prompt,
                               int with_max_tokens, char **out_refined) {
    (void)api_key; /* legacy agent sends no Authorization header */
    if (!endpoint || !raw_prompt || !out_refined) return HD_ERR_PARSE;
    *out_refined = NULL;

    hd_refiner_endpoint ep;
    if (parse_endpoint(endpoint, &ep) != 0) return HD_ERR_IO;

    char *sys_esc = json_escape(system_prompt);
    char *usr_esc = json_escape(raw_prompt);
    if (!sys_esc || !usr_esc) {
        free(sys_esc);
        free(usr_esc);
        return HD_ERR_OOM;
    }

    const char *model = "HiDream-ai/Prompt-Refine";
    char *model_esc = json_escape(model);
    if (!model_esc) {
        free(sys_esc);
        free(usr_esc);
        return HD_ERR_OOM;
    }

    char *body = NULL;
    if (with_max_tokens) {
        size_t need = strlen("{\"model\":\"\",\"messages\":[{\"role\":\"system\","
                             "\"content\":\"\"},{\"role\":\"user\",\"content\":\"\"}],"
                             "\"max_tokens\":2048}")
                      + strlen(model_esc) + strlen(sys_esc) + strlen(usr_esc) + 1;
        body = (char *)malloc(need);
        if (body) {
            snprintf(body, need,
                     "{\"model\":\"%s\",\"messages\":[{\"role\":\"system\","
                     "\"content\":\"%s\"},{\"role\":\"user\",\"content\":\"%s\"}],"
                     "\"max_tokens\":2048}",
                     model_esc, sys_esc, usr_esc);
        }
    } else {
        size_t need = strlen("{\"model\":\"\",\"messages\":[{\"role\":\"system\","
                             "\"content\":\"\"},{\"role\":\"user\",\"content\":\"\"}]}")
                      + strlen(model_esc) + strlen(sys_esc) + strlen(usr_esc) + 1;
        body = (char *)malloc(need);
        if (body) {
            snprintf(body, need,
                     "{\"model\":\"%s\",\"messages\":[{\"role\":\"system\","
                     "\"content\":\"%s\"},{\"role\":\"user\",\"content\":\"%s\"}]}",
                     model_esc, sys_esc, usr_esc);
        }
    }
    free(model_esc);
    free(sys_esc);
    free(usr_esc);
    if (!body) return HD_ERR_OOM;

    char *response = NULL;
    if (http_post(&ep, body, &response) != 0) {
        free(body);
        return HD_ERR_IO;
    }
    free(body);

    /* split headers / body */
    char *hdr_end = strstr(response, "\r\n\r\n");
    const char *json_text = response;
    if (hdr_end) json_text = hdr_end + 4;

    const char *err = NULL;
    hd_json *root = hd_json_parse(json_text, &err);
    if (!root) {
        free(response);
        return HD_ERR_PARSE;
    }

    const hd_json *choices = hd_json_get(root, "choices");
    const hd_json *first = (choices && hd_json_array_len(choices) > 0)
                               ? hd_json_array_at(choices, 0)
                               : NULL;
    const hd_json *message = first ? hd_json_get(first, "message") : NULL;
    const hd_json *content = message ? hd_json_get(message, "content") : NULL;
    const char *text = content ? hd_json_string(content) : NULL;

    hd_status status = HD_ERR_PARSE;
    if (text) {
        char *dup = strdup(text);
        if (dup) {
            *out_refined = dup;
            status = HD_OK;
        } else {
            status = HD_ERR_OOM;
        }
    }
    hd_json_free(root);
    free(response);
    return status;
}

hd_status hd_refiner_refine_legacy(const char *endpoint, const char *api_key,
                                   const char *raw_prompt, char **out_refined) {
    (void)api_key; /* legacy agent sends no auth header */
    return refine_common(endpoint, api_key, kLegacySystemPrompt, raw_prompt, 0,
                         out_refined);
}

hd_status hd_refiner_refine_v2(const char *endpoint, const char *raw_prompt,
                               char **out_refined) {
    return refine_common(endpoint, NULL, kV2SystemPrompt, raw_prompt, 1,
                         out_refined);
}