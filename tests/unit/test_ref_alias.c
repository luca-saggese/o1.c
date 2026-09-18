/*
 * Named reference alias gate (frontend-only prompt expansion).
 *
 * Verifies the alias parser, table construction, lookup and prompt expansion
 * contract (REFIMAGES.md). No model/CUDA involved.
 */

#include "hidream.h"
#include "ref_alias.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
    else printf("ok: %s\n", msg); \
} while (0)

static int build(hd_ref_alias_table *t, const char *const *paths,
                 const char *const *aliases, int n) {
    return hd_ref_alias_build(t, paths, aliases, n, NULL, 0);
}

int main(void) {
    /* ---- 1: plain path -> automatic @ref1 ---- */
    {
        const char *a = NULL, *p = NULL;
        CHECK(hd_ref_alias_split("foo.png", &a, &p) == HD_OK &&
              a == NULL && strcmp(p, "foo.png") == 0,
              "plain path keeps path, no alias");
        const char *paths[] = { "foo.png" };
        const char *als[] = { NULL };
        hd_ref_alias_table t;
        CHECK(build(&t, paths, als, 1) == HD_OK, "build 1 plain ref");
        CHECK(hd_ref_alias_lookup(&t, "ref1") == 1, "@ref1 -> 1");
        CHECK(hd_ref_alias_lookup(&t, "foo") == 0, "no implicit name");
    }

    /* ---- 2: NAME=PATH -> @NAME and @ref1 ---- */
    {
        const char *a = NULL, *p = NULL;
        CHECK(hd_ref_alias_split("person=foo.png", &a, &p) == HD_OK &&
              a && strncmp(a, "person=", 7) == 0 && strcmp(p, "foo.png") == 0,
              "named ref splits into alias+path");
        const char *paths[] = { "foo.png" };
        const char *als[] = { "person" };
        hd_ref_alias_table t;
        CHECK(build(&t, paths, als, 1) == HD_OK, "build named ref");
        CHECK(hd_ref_alias_lookup(&t, "person") == 1, "@person -> 1");
        CHECK(hd_ref_alias_lookup(&t, "ref1") == 1, "@ref1 still -> 1");
    }

    /* ---- 3: path containing '=' but invalid alias prefix ---- */
    {
        const char *a = NULL, *p = NULL;
        CHECK(hd_ref_alias_split("a/b=c/d.png", &a, &p) == HD_OK &&
              a == NULL && strcmp(p, "a/b=c/d.png") == 0,
              "ambiguous '=' kept as path when prefix invalid");
    }

    /* ---- 4: two named refs, stable mapping ---- */
    {
        const char *paths[] = { "a.png", "b.png" };
        const char *als[] = { "person", "dress" };
        hd_ref_alias_table t;
        CHECK(build(&t, paths, als, 2) == HD_OK, "build 2 named refs");
        CHECK(hd_ref_alias_lookup(&t, "person") == 1, "person -> 1");
        CHECK(hd_ref_alias_lookup(&t, "dress") == 2, "dress -> 2");
        CHECK(hd_ref_alias_lookup(&t, "ref2") == 2, "@ref2 -> 2");
    }

    /* ---- 5: duplicate alias fails ---- */
    {
        const char *paths[] = { "a.png", "b.png" };
        const char *als[] = { "person", "person" };
        hd_ref_alias_table t;
        CHECK(build(&t, paths, als, 2) == HD_ERR_MISMATCH,
              "duplicate alias fails closed");
        CHECK(strstr(hd_ref_alias_error(), "duplicate") != NULL,
              "duplicate error message");
    }

    /* ---- 6: reserved __alias fails ---- */
    {
        const char *paths[] = { "a.png" };
        const char *als[] = { "__layout" };
        hd_ref_alias_table t;
        CHECK(build(&t, paths, als, 1) == HD_ERR_MISMATCH,
              "reserved __ alias fails");
    }

    /* ---- 7: unknown @alias in prompt is left literal (no failure) ---- */
    {
        const char *paths[] = { "a.png" };
        const char *als[] = { "person" };
        hd_ref_alias_table t;
        build(&t, paths, als, 1);
        char *exp = NULL;
        CHECK(hd_ref_alias_expand(&t, "a photo of @foo", &exp) == HD_OK,
              "unknown @alias does not fail");
        CHECK(exp == NULL, "prompt with only unknown @alias is un-expanded");
        /* mixed: known + unknown -> expanded, unknown kept verbatim */
        char *exp2 = NULL;
        CHECK(hd_ref_alias_expand(&t, "@person and @foo", &exp2) == HD_OK &&
              exp2 != NULL, "mixed known/unknown expands");
        if (exp2) {
            CHECK(strstr(exp2, "reference image 1 (\"person\")") != NULL,
                  "known alias rewritten");
            CHECK(strstr(exp2, "@foo") != NULL, "unknown @foo left verbatim");
            free(exp2);
        }
    }

    /* ---- 8: prompt without alias is untouched ---- */
    {
        const char *paths[] = { "a.png" };
        const char *als[] = { "person" };
        hd_ref_alias_table t;
        build(&t, paths, als, 1);
        char *exp = (char *)0x1;
        CHECK(hd_ref_alias_expand(&t, "a plain prompt", &exp) == HD_OK &&
              exp == NULL,
              "prompt without alias left un-expanded (NULL)");
    }

    /* ---- 9: expansion rewrites alias + adds mapping header ---- */
    {
        const char *paths[] = { "a.png", "b.png" };
        const char *als[] = { "person", "dress" };
        hd_ref_alias_table t;
        build(&t, paths, als, 2);
        char *exp = NULL;
        CHECK(hd_ref_alias_expand(&t, "@person wearing @dress", &exp) == HD_OK &&
              exp != NULL, "expansion returns a string");
        if (exp) {
            CHECK(strstr(exp, "Reference image mapping:") != NULL,
                  "mapping header present");
            CHECK(strstr(exp, "- \"person\" = reference image 1") != NULL,
                  "person mapped to 1");
            CHECK(strstr(exp, "reference image 1 (\"person\")") != NULL,
                  "@person expanded in body");
            CHECK(strstr(exp, "reference image 2 (\"dress\")") != NULL,
                  "@dress expanded in body");
            free(exp);
        }
    }

    /* ---- 10: @ref1 vs @ref10 no collision ---- */
    {
        const char *paths[10];
        const char *als[10];
        for (int i = 0; i < 10; i++) { paths[i] = "x.png"; als[i] = NULL; }
        hd_ref_alias_table t;
        CHECK(build(&t, paths, als, 10) == HD_OK, "build 10 auto refs");
        CHECK(hd_ref_alias_lookup(&t, "ref1") == 1, "@ref1 -> 1");
        CHECK(hd_ref_alias_lookup(&t, "ref10") == 10, "@ref10 -> 10");
        char *exp = NULL;
        CHECK(hd_ref_alias_expand(&t, "use @ref1 and @ref10", &exp) == HD_OK,
              "@ref1 and @ref10 both expand");
        if (exp) {
            CHECK(strstr(exp, "reference image 1 (\"ref1\")") != NULL,
                  "ref1 -> image 1");
            CHECK(strstr(exp, "reference image 10 (\"ref10\")") != NULL,
                  "ref10 -> image 10");
            free(exp);
        }
    }

    /* ---- 11: internal ref does not change user aliases ---- */
    {
        const char *up[] = { "u0.png", "u1.png" };
        const char *ua[] = { "person", "dress" };
        const char *ip[] = { "layout.png" };
        hd_ref_alias_table t;
        CHECK(hd_ref_alias_build(&t, up, ua, 2, ip, 1) == HD_OK,
              "build with internal ref");
        CHECK(hd_ref_alias_lookup(&t, "person") == 1, "person still -> 1");
        CHECK(hd_ref_alias_lookup(&t, "dress") == 2, "dress still -> 2");
        CHECK(t.count == 3 && t.entries[2].is_internal == 1,
              "internal ref appended at effective index 2");
    }

    printf("\n%d assertions passed, %d failed\n", 0, failures);
    if (failures) return 1;
    return 0;
}
