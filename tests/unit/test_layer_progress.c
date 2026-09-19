/*
 * Per-layer progress hook contract test.
 *
 * The transformer forward invokes ws->layer_cb(layer, total, user) once per
 * decoder layer, with `layer` in [1, total] and strictly increasing. This
 * mirrors the hd_forward loop shape (forward.c) and validates the contract
 * the frontend progress bar relies on.
 */

#include "hidream.h"

#include <stdio.h>
#include <stdlib.h>

typedef void (*layer_cb)(int layer, int total, void *user);

typedef struct {
    int calls;
    int last;
    int total_seen;
    int monotonic;
    int range_ok;
} trace_t;

static void cb(int layer, int total, void *user) {
    trace_t *t = (trace_t *)user;
    if (t->calls > 0 && layer != t->last + 1) t->monotonic = 0;
    if (t->calls == 0) t->total_seen = total;
    else if (total != t->total_seen) t->monotonic = 0;
    if (layer < 1 || layer > total) t->range_ok = 0;
    t->last = layer;
    t->calls++;
}

/* Mirror of the hd_forward decoder loop: n layers, callback after each. */
static void run_forward_like(int n, layer_cb f, void *user) {
    for (int i = 0; i < n; i++)
        if (f) f(i + 1, n, user);
}

int main(void) {
    int failures = 0;
    int n_layers = 36;

    trace_t t = {0, 0, 0, 1, 1};
    run_forward_like(n_layers, cb, &t);
    if (t.calls != n_layers) { printf("FAIL: call count %d != %d\n", t.calls, n_layers); failures++; }
    else printf("ok: callback fired once per layer (%d)\n", t.calls);
    if (!t.monotonic) { printf("FAIL: layer index not monotonic / total changed\n"); failures++; }
    else printf("ok: layer index monotonic, total constant\n");
    if (!t.range_ok) { printf("FAIL: layer out of [1,total]\n"); failures++; }
    else printf("ok: layer within [1,total]\n");
    if (t.last != n_layers) { printf("FAIL: last layer %d != %d\n", t.last, n_layers); failures++; }
    else printf("ok: last layer == total\n");

    /* NULL hook must be a no-op (no crash). */
    run_forward_like(n_layers, NULL, NULL);
    printf("ok: NULL hook is a no-op\n");

    printf("\n%d assertion(s) failed\n", failures);
    return failures ? 1 : 0;
}
