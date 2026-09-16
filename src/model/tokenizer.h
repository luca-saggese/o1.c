#ifndef HD_TOKENIZER_H
#define HD_TOKENIZER_H

/*
 * M1.6 native prompt tokenizer / input path.
 *
 * Reproduces the frozen oracle tokenization deterministically in pure C,
 * without loading the Python tokenizer or any model weights. The tokenizer
 * identity is Qwen2Tokenizer (a GPT-2 byte-level BPE) applied to the im-chat
 * template built from a single user prompt, exactly as the M1.0 freeze path
 * did (processor.apply_chat_template(..., tokenize=False,
 * add_generation_prompt=True) followed by tokenizer.encode(..., False)).
 *
 * All data (byte->piece mapping, BPE merge table, vocab subset and special
 * token ids) is embedded at build time from the frozen manifests — see
 * tokenizer_tables.h. No model load, no network, no /python dependency.
 *
 * All functions return hd_status and set an error string via hd_last_error().
 */

#include <stddef.h>
#include <stdint.h>

#include "hidream.h"

/* Embedded tables for the frozen tokenizer vocab / merges. */
#include "tokenizer_tables.h"

/* Frozen tokenizer identity. */
#define HD_TOK_IDENTITY "Qwen2Tokenizer (GPT-2 byte-level BPE)"

/* Known special-token ids from the frozen tokenizer config. */
#define HD_TOK_IM_START     151644
#define HD_TOK_IM_END       151645
#define HD_TOK_VISION_START 151652
#define HD_TOK_VISION_END   151653
#define HD_TOK_IMAGE_PAD    151655
#define HD_TOK_VIDEO_PAD    151656

/*
 * Encodes `prompt` by first building the canonical im-chat template for a
 * single user message with add_generation_prompt=True, then running the
 * frozen byte-level BPE with add_special_tokens=False.
 *
 * On success *out_ids (caller-freed with hd_tokenizer_free_ids) receives the
 * malloc'd token id array and *out_count its length. Returns HD_OK only when
 * every piece resolves to a vocab id (fails closed, never silently drops).
 */
hd_status hd_tokenizer_encode_prompt(const char *prompt,
                                     int **out_ids, size_t *out_count);

/*
 * Builds only the im-chat template string for a single user prompt with
 * add_generation_prompt=True (the same string the oracle fed to encode).
 * Caller frees *out with free().
 */
hd_status hd_tokenizer_build_template(const char *prompt, char **out);

/*
 * Builds the ref-mode im-chat template: K <|vision_start|><|image_pad|>
 * <|vision_end|> placeholders followed by the caption, then the assistant
 * generation prompt (oracle apply_chat_template for
 * content=[{"type":"image"}]*K + [{"type":"text","text":caption}]).
 * Caller frees *out with free().
 */
hd_status hd_tokenizer_build_ref_template(const char *caption, int k,
                                          char **out);

/*
 * Encodes an arbitrary pre-built template/string with the frozen byte-level
 * BPE and special-token handling (add_special_tokens=False semantics).
 * Caller frees *out_ids with hd_tokenizer_free_ids.
 */
hd_status hd_tokenizer_encode(const char *text,
                              int **out_ids, size_t *out_count);

/* Returns the token id for a special-token content string, or -1 if unknown. */
int hd_tokenizer_special_id(const char *content);

/* Frees an id array returned by any hd_tokenizer_encode* function. */
void hd_tokenizer_free_ids(int *ids);

#endif /* HD_TOKENIZER_H */
