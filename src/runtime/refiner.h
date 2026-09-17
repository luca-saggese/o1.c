#ifndef HD_REFINER_H
#define HD_REFINER_H

/*
 * M1-post native prompt-refiner clients (contract section 7).
 *
 * Parity targets:
 *   - legacy: python/prompt_agent.py rewrite_prompt_api() — OpenAI-compatible
 *     chat.completions, system prompt at prompt_agent.py:6-64, user = raw input,
 *     fields model + messages (NO temperature/top_p).
 *   - Dev-2604: python/prompt_agent_v2.py — English rewrite only, targets
 *     HiDream-ai/Prompt-Refine at http://localhost:8000/v1, fields model,
 *     system/user messages, max_tokens (NO temperature/top_p).
 *
 * The refiner is an OPTIONAL companion: nothing in the engine calls these
 * functions implicitly. The engine stays Python-free and offline-capable.
 *
 * Transport: minimal HTTP/1.1 POST over POSIX sockets (no libcurl, no Python).
 * Endpoint format: "http://host:port" or "http://host:port/v1" — the
 * chat/completions path is appended automatically.
 */

#include "hidream.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Legacy prompt-agent refinement (OpenAI-compatible chat.completions).
 * `endpoint` is the base URL (e.g. "http://localhost:8000/v1").
 * `api_key` may be NULL/empty (sent as "Bearer " then).
 * On success returns HD_OK and sets *out_refined to a malloc'd string
 * (caller frees). Returns HD_ERR_IO on network failure and HD_ERR_PARSE
 * on malformed JSON / missing choices[0].message.content.
 */
hd_status hd_refiner_refine_legacy(const char *endpoint, const char *api_key,
                                   const char *raw_prompt, char **out_refined);

/*
 * Dev-2604 Prompt-Refine client (English rewrite only, max_tokens).
 * `endpoint` is the base URL (default "http://localhost:8000/v1").
 * Same return/ownership contract as hd_refiner_refine_legacy.
 */
hd_status hd_refiner_refine_v2(const char *endpoint, const char *raw_prompt,
                               char **out_refined);

#ifdef __cplusplus
}
#endif

#endif /* HD_REFINER_H */