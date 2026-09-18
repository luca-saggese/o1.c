#ifndef HD_REF_ALIAS_H
#define HD_REF_ALIAS_H

/*
 * Named reference aliases (frontend-only prompt expansion).
 *
 * A reference image may carry an optional semantic name ("alias") supplied on
 * the command line or via the C API as `NAME=PATH`. The alias is pure
 * frontend metadata: it never reaches the model, the tokenizer, the CUDA
 * kernels or the reference tensor ordering. It only lets a prompt refer to a
 * reference by name using the `@name` syntax, which is expanded into an
 * explicit "reference image N" phrase before tokenization.
 *
 * This module is deliberately free of any model/CUDA dependency.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum references tracked (matches HD_SEQ_MAX_REFS). */
#define HD_REF_ALIAS_MAX 16

/* One effective reference (user or internal) plus its aliases. */
typedef struct {
    const char *path;        /* image path */
    const char *user_alias;  /* explicit alias, or NULL */
    int is_internal;         /* 1 for runtime-generated refs (__*) */
    int user_index;          /* index among USER references (0-based) */
    int effective_index;     /* index in the effective (final) list */
} hd_ref_alias_entry;

typedef struct {
    hd_ref_alias_entry entries[HD_REF_ALIAS_MAX];
    int count;
} hd_ref_alias_table;

/*
 * Parse a `--ref-image` argument into an explicit alias and a path.
 *
 * The substring before the first '=' is treated as an alias ONLY if it is a
 * valid alias token ([A-Za-z_][A-Za-z0-9_-]*) and non-empty; otherwise the
 * whole argument is the path (so paths containing '=' are preserved).
 *
 * Returns HD_OK and sets *alias (NULL if none) / *path. The returned
 * pointers alias the input string (no allocation). Returns HD_ERR_MISSING
 * for a NULL/empty argument.
 */
int hd_ref_alias_split(const char *arg, const char **alias, const char **path);

/*
 * Validate an alias token: [A-Za-z_][A-Za-z0-9_-]*, non-empty, must not
 * start with "__" (reserved for internal refs), no whitespace. Returns 1 if
 * valid, 0 otherwise.
 */
int hd_ref_alias_valid(const char *alias);

/*
 * Build the alias table from user references plus internal references.
 *
 * Automatic aliases `refN` (N = 1-based effective index) are always assigned.
 * A duplicate explicit alias (or a collision with a reserved internal alias)
 * is a hard error. Internal refs are appended after the user refs and are
 * marked is_internal; they never displace user aliases.
 *
 * On success returns HD_OK and fills `tbl`. On failure returns
 * HD_ERR_MISMATCH and writes a human-readable message via hd_ref_alias_error.
 */
int hd_ref_alias_build(hd_ref_alias_table *tbl,
                       const char *const *user_paths,
                       const char *const *user_aliases,
                       int n_user,
                       const char *const *internal_paths,
                       int n_internal);

/* Last error message from hd_ref_alias_build / hd_ref_alias_expand. */
const char *hd_ref_alias_error(void);

/*
 * Resolve a single alias token (without the leading '@') to the 1-based
 * reference image number used in the expanded prompt. Returns >0 on success,
 * 0 if unknown. Both explicit and automatic (`refN`) aliases are searched.
 */
int hd_ref_alias_lookup(const hd_ref_alias_table *tbl, const char *name);

/*
 * Expand `@alias` occurrences in `prompt` into `reference image N ("alias")`.
 *
 * Behaviour:
 *   - If the prompt contains no KNOWN '@alias' at all, *out_expanded is left
 *     NULL and the caller must use the original prompt unchanged (this
 *     preserves byte-identical behaviour for prompts without aliases).
 *   - Otherwise *out_expanded is a malloc'd string starting with a short
 *     mapping header followed by the rewritten body. Caller frees it.
 *   - An unknown `@token` is left verbatim in the prompt (never an error and
 *     never dropped); only known aliases are rewritten.
 *
 * Tokenization is lexical (a leading '@' followed by a valid alias token), so
 * `@ref1` and `@ref10` do not collide.
 */
int hd_ref_alias_expand(const hd_ref_alias_table *tbl, const char *prompt,
                        char **out_expanded);

/* Write a human-readable mapping table to `f` (verbose/debug only). */
void hd_ref_alias_dump(const hd_ref_alias_table *tbl, void *f);

#ifdef __cplusplus
}
#endif

#endif /* HD_REF_ALIAS_H */
