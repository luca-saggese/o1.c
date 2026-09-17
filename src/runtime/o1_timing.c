/*
 * M2 pre-baseline timing instrumentation (compile-time gated).
 *
 * Backing implementation for o1_timing.h. Compiled only when
 * -DO1_DEBUG_TIMING is set; the header turns every macro into a no-op
 * otherwise, so production builds never link this file's symbols.
 *
 * Design:
 *   - CPU regions: clock_gettime(CLOCK_MONOTONIC) wall-clock pairs.
 *   - GPU regions: cudaEvent pairs recorded on the current stream; the
 *     elapsed time is folded into the accumulator at o1_timing_report()
 *     after a cudaDeviceSynchronize().
 *   - o1_timing_add() lets callers fold externally measured values (e.g.
 *     a CPU-side decode loop) into the same accumulator.
 *
 * The report prints a compact per-region summary to stdout and writes a
 * machine-readable JSON dump to the requested path.
 */
#include "o1_timing.h"

#ifdef O1_DEBUG_TIMING

#define _POSIX_C_SOURCE 200809L
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define O1_TIMING_MAX_REGIONS 64
#define O1_TIMING_MAX_EVENTS 4096

typedef struct {
    const char *name;
    double cpu_seconds;
    double gpu_seconds;
    long long cpu_count;
    long long gpu_count;
} o1_timing_region;

typedef struct {
    const char *name;
    cudaEvent_t start;
    cudaEvent_t end;
} o1_timing_event;

static o1_timing_region g_regions[O1_TIMING_MAX_REGIONS];
static int g_region_count = 0;
static o1_timing_event g_events[O1_TIMING_MAX_EVENTS];
static int g_event_count = 0;
static int g_events_overflow = 0;

/* Stack of open CPU regions (name + start time). */
typedef struct {
    const char *name;
    double start;
} o1_timing_open;
static o1_timing_open g_open[O1_TIMING_MAX_REGIONS];
static int g_open_count = 0;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static o1_timing_region *find_or_create(const char *name) {
    int i;
    for (i = 0; i < g_region_count; i++) {
        if (strcmp(g_regions[i].name, name) == 0) {
            return &g_regions[i];
        }
    }
    if (g_region_count >= O1_TIMING_MAX_REGIONS) {
        return NULL;
    }
    g_regions[g_region_count].name = name;
    g_regions[g_region_count].cpu_seconds = 0.0;
    g_regions[g_region_count].gpu_seconds = 0.0;
    g_regions[g_region_count].cpu_count = 0;
    g_regions[g_region_count].gpu_count = 0;
    return &g_regions[g_region_count++];
}

void o1_timing_reset(void) {
    g_region_count = 0;
    g_event_count = 0;
    g_events_overflow = 0;
    g_open_count = 0;
}

void o1_timing_begin_cpu(const char *name) {
    if (g_open_count >= O1_TIMING_MAX_REGIONS) {
        return;
    }
    g_open[g_open_count].name = name;
    g_open[g_open_count].start = now_seconds();
    g_open_count++;
}

void o1_timing_end_cpu(const char *name) {
    double end = now_seconds();
    int i;
    for (i = g_open_count - 1; i >= 0; i--) {
        if (strcmp(g_open[i].name, name) == 0) {
            o1_timing_region *r = find_or_create(name);
            if (r != NULL) {
                r->cpu_seconds += end - g_open[i].start;
                r->cpu_count++;
            }
            /* Remove this entry by shifting the tail down. */
            memmove(&g_open[i], &g_open[i + 1],
                    (size_t)(g_open_count - i - 1) * sizeof(o1_timing_open));
            g_open_count--;
            return;
        }
    }
}

void o1_timing_begin_gpu(const char *name) {
    if (g_event_count >= O1_TIMING_MAX_EVENTS) {
        g_events_overflow = 1;
        return;
    }
    o1_timing_event *e = &g_events[g_event_count++];
    e->name = name;
    cudaEventCreate(&e->start);
    cudaEventCreate(&e->end);
    cudaEventRecord(e->start, 0);
}

void o1_timing_end_gpu(const char *name) {
    if (g_event_count == 0) {
        return;
    }
    /* Match the most recent open event with this name. */
    int i;
    for (i = g_event_count - 1; i >= 0; i--) {
        if (g_events[i].end == NULL) {
            continue;
        }
        if (strcmp(g_events[i].name, name) == 0) {
            cudaEventRecord(g_events[i].end, 0);
            return;
        }
    }
}

void o1_timing_add(const char *name, double seconds) {
    o1_timing_region *r = find_or_create(name);
    if (r == NULL) {
        return;
    }
    r->cpu_seconds += seconds;
    r->cpu_count++;
}

void o1_timing_add_gpu(const char *name, double seconds) {
    o1_timing_region *r = find_or_create(name);
    if (r == NULL) {
        return;
    }
    r->gpu_seconds += seconds;
    r->gpu_count++;
}

static void fold_gpu_events(void) {
    int i;
    for (i = 0; i < g_event_count; i++) {
        o1_timing_event *e = &g_events[i];
        float ms = 0.0f;
        cudaEventSynchronize(e->end);
        cudaEventElapsedTime(&ms, e->start, e->end);
        o1_timing_region *r = find_or_create(e->name);
        if (r != NULL) {
            r->gpu_seconds += (double)ms * 1e-3;
            r->gpu_count++;
        }
        cudaEventDestroy(e->start);
        cudaEventDestroy(e->end);
    }
    g_event_count = 0;
}

void o1_timing_report(const char *json_path) {
    int i;
    cudaDeviceSynchronize();
    fold_gpu_events();

    printf("\n=== O1 TIMING REPORT ===\n");
    printf("%-28s %12s %12s %8s %8s\n", "region", "cpu(s)", "gpu(s)", "cpu#", "gpu#");
    for (i = 0; i < g_region_count; i++) {
        o1_timing_region *r = &g_regions[i];
        printf("%-28s %12.4f %12.4f %8lld %8lld\n",
               r->name, r->cpu_seconds, r->gpu_seconds, r->cpu_count, r->gpu_count);
    }
    if (g_events_overflow) {
        printf("WARNING: GPU event buffer overflowed (%d events)\n", O1_TIMING_MAX_EVENTS);
    }
    printf("=== END O1 TIMING REPORT ===\n");

    if (json_path != NULL) {
        FILE *f = fopen(json_path, "w");
        if (f == NULL) {
            fprintf(stderr, "o1_timing: cannot write %s\n", json_path);
            return;
        }
        fprintf(f, "{\n  \"regions\": [\n");
        for (i = 0; i < g_region_count; i++) {
            o1_timing_region *r = &g_regions[i];
            fprintf(f,
                    "    {\"name\": \"%s\", \"cpu_seconds\": %.6f, \"gpu_seconds\": %.6f, "
                    "\"cpu_count\": %lld, \"gpu_count\": %lld}%s\n",
                    r->name, r->cpu_seconds, r->gpu_seconds, r->cpu_count, r->gpu_count,
                    (i + 1 < g_region_count) ? "," : "");
        }
        fprintf(f, "  ]\n}\n");
        fclose(f);
    }
}

#endif /* O1_DEBUG_TIMING */