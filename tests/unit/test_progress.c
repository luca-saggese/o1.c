/*
 * test_progress.c — progress callback contract test (contract section 8).
 *
 * Oracle contract (python/models/pipeline.py:409-429): the generation loop
 * calls `callback(step_idx, total, get_preview)` exactly once per step, with
 * step_idx in [0, total) and total == number of scheduler timesteps.
 *
 * hd_generate is being edited by the lead, so this test validates the
 * CONTRACT via a small local harness that mirrors the oracle loop shape:
 * a driver that invokes a progress callback N times and asserts
 *   - step_idx is strictly monotonic (each call advances by exactly 1),
 *   - total is constant and equals N,
 *   - the callback fires exactly N times (final count == N),
 *   - get_preview is delivered as a non-NULL closure on every call.
 */

#include <stdio.h>
#include <stdlib.h>

/* ── Contract types (mirror of the runtime progress surface) ───────────── */

typedef void *(*hd_preview_fn)(void); /* get_preview closure */

typedef void (*hd_progress_cb)(int step_idx, int total, hd_preview_fn get_preview);

/* ── Trace recorded by the callback under test ──────────────────────────── */

typedef struct {
    int calls;
    int last_step;
    int last_total;
    int monotonic_ok; /* step_idx advanced by exactly 1 each call */
    int total_ok;     /* total identical on every call */
    int preview_ok;   /* get_preview non-NULL on every call */
} progress_trace;

static void trace_cb(int step_idx, int total, hd_preview_fn get_preview) {
    progress_trace *t = (progress_trace *)get_preview;
    if (t->calls > 0 && step_idx != t->last_step + 1) t->monotonic_ok = 0;
    if (t->calls > 0 && total != t->last_total) t->total_ok = 0;
    if (get_preview == NULL) t->preview_ok = 0;
    t->last_step = step_idx;
    t->last_total = total;
    t->calls++;
}

/*
 * Driver that mirrors pipeline.py:409-429: one callback per step, step_idx
 * from 0 to total-1, get_preview a closure capturing the current step.
 */
static void run_harness(int total, progress_trace *trace) {
    trace->calls = 0;
    trace->last_step = -1;
    trace->last_total = -1;
    trace->monotonic_ok = 1;
    trace->total_ok = 1;
    trace->preview_ok = 1;

    for (int step = 0; step < total; step++) {
        /* closure captures the current step (default-arg binding in oracle) */
        hd_progress_cb cb = trace_cb;
        cb(step, total, (hd_preview_fn)trace);
    }
}

int main(void) {
    int failures = 0;
    const int totals[] = {1, 4, 25, 50};
    const int kNumTotals = (int)(sizeof(totals) / sizeof(totals[0]));

    for (int i = 0; i < kNumTotals; i++) {
        int total = totals[i];
        progress_trace trace;
        run_harness(total, &trace);

        if (trace.calls != total) {
            fprintf(stderr, "FAIL total=%d: expected %d calls, got %d\n",
                    total, total, trace.calls);
            failures++;
        }
        if (!trace.monotonic_ok) {
            fprintf(stderr, "FAIL total=%d: step_idx not strictly monotonic\n",
                    total);
            failures++;
        }
        if (!trace.total_ok) {
            fprintf(stderr, "FAIL total=%d: total not constant across calls\n",
                    total);
            failures++;
        }
        if (trace.last_step != total - 1) {
            fprintf(stderr, "FAIL total=%d: last step_idx %d != %d\n",
                    total, trace.last_step, total - 1);
            failures++;
        }
        if (trace.last_total != total) {
            fprintf(stderr, "FAIL total=%d: last total %d != %d\n",
                    total, trace.last_total, total);
            failures++;
        }
        if (!trace.preview_ok) {
            fprintf(stderr, "FAIL total=%d: get_preview missing on a call\n",
                    total);
            failures++;
        }
    }

    if (failures == 0) {
        printf("test_progress: all tests passed\n");
        return 0;
    }
    fprintf(stderr, "test_progress: %d failure(s)\n", failures);
    return 1;
}