/*
 * M1-post scheduler matrix CPU-only unit test.
 *
 * Validates the native sigma derivations, noise_scale_schedule, and step math
 * (Euler for flash/flow_match, UniPC multistep for default) against the frozen
 * Python oracle dump (tests/unit/oracle_dump_matrix.txt, produced by
 * tests/unit/tools/dump_scheduler_matrix_oracle.py with an explicit MT19937
 * CPU generator, seed 1234).
 *
 * The test reimplements the step arithmetic in plain C (fp32, torch operation
 * order) so it needs no CUDA at build or link time. RNG draws are reproduced
 * bit-exactly from src/runtime/torch_rng.c (a port of torch's CPU MT19937).
 *
 * Tolerances (per the frozen M1_POST_SCHEDULER_MATRIX doc):
 *   - sigma schedules            <= 1e-6
 *   - noise_scale_schedule       <= 1e-6
 *   - flash/flow_match Euler step <= 1e-5
 *   - default UniPC (3 steps)     <= 1e-4
 *
 * Build (single gcc command, no Makefile changes):
 *   gcc -O2 -o scheduler_matrix_test tests/unit/scheduler_matrix.c \
 *       src/model/scheduler.c src/runtime/torch_rng.c \
 *       -Isrc/model -Iinclude -Isrc/runtime -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "scheduler.h"
#include "torch_rng.h"

#define N 64            /* synthetic feature dim used by the dump tool */
#define MAX_RECIPES 8
#define MAX_STEP_N 200

/* hd_set_error is defined in src/model/model.c; stub it here (variadic). */
void hd_set_error(const char *fmt, ...) {
    (void)fmt;
}

/*
 * Stubs for the CUDA scheduler kernels that scheduler.c's hd_scheduler_step
 * references at link time. The unit test does not exercise hd_scheduler_step
 * (it reimplements the step math below), so these are never called.
 */
void hd_sched_bf16_upcast(const void *in, float *out, int n) { (void)in; (void)out; (void)n; }
void hd_sched_denoised(const float *z, const float *mo, float sigma,
                       float *denoised, int n) { (void)z; (void)mo; (void)sigma; (void)denoised; (void)n; }
void hd_sched_z_next(const float *noise, const float *denoised, float sigma_next,
                     float s_noise, float *out, int n) { (void)noise; (void)denoised; (void)sigma_next; (void)s_noise; (void)out; (void)n; }
void hd_f32_convert_bf16(const float *in, void *out, int n) { (void)in; (void)out; (void)n; }
void hd_sched_vcond(const void *z, const void *xp, float sigma, float *mo, int n) {
    (void)z; (void)xp; (void)sigma; (void)mo; (void)n;
}

/* ------------------------------------------------------------------ */
/* Dump parsing                                                        */
/* ------------------------------------------------------------------ */

typedef struct {
    char name[64];
    int steps;
    float shift;
    char sched[32];
    int has_tlist;
    float sigmas[HD_SCHED_MAX_STEPS];
    int n_sigmas;
    float noisescale[HD_SCHED_MAX_STEPS];
    int n_noisescale;
    float step[MAX_STEP_N];
    int n_step;
} recipe_t;

static int parse_recipe(FILE *f, recipe_t *r) {
    char line[16384];
    /* RECIPE line */
    if (fgets(line, sizeof(line), f) == NULL) return 0;
    if (strncmp(line, "RECIPE", 6) != 0 || line[6] != ' ') return 0;
    {
        char tlist_s[8];
        if (sscanf(line, "RECIPE %63s %d %f %31s %7s",
                   r->name, &r->steps, &r->shift, r->sched, tlist_s) != 5)
            return 0;
        r->has_tlist = (strcmp(tlist_s, "True") == 0);
    }
    r->n_sigmas = r->n_noisescale = r->n_step = 0;

    /* SIGMAS line */
    if (fgets(line, sizeof(line), f) == NULL) return 0;
    if (strncmp(line, "SIGMAS", 6) != 0 || line[6] != ' ') return 0;
    {
        char *p = line + 7;
        while (*p && r->n_sigmas < HD_SCHED_MAX_STEPS) {
            char *end;
            float v = strtof(p, &end);
            if (end == p) break;
            r->sigmas[r->n_sigmas++] = v;
            p = end;
        }
    }

    /* NOISESCALE line */
    if (fgets(line, sizeof(line), f) == NULL) return 0;
    if (strncmp(line, "NOISESCALE", 10) != 0 || line[10] != ' ') return 0;
    {
        char *p = line + 11;
        while (*p && r->n_noisescale < HD_SCHED_MAX_STEPS) {
            char *end;
            float v = strtof(p, &end);
            if (end == p) break;
            r->noisescale[r->n_noisescale++] = v;
            p = end;
        }
    }

    /* STEP line */
    if (fgets(line, sizeof(line), f) == NULL) return 0;
    if (strncmp(line, "STEP", 4) != 0 || line[4] != ' ') return 0;
    {
        char *p = line + 5;
        while (*p && r->n_step < MAX_STEP_N) {
            char *end;
            float v = strtof(p, &end);
            if (end == p) break;
            r->step[r->n_step++] = v;
            p = end;
        }
    }
    return 1;
}

static int load_dump(const char *path, recipe_t *recipes, int maxn) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    int n = 0;
    char line[16384];
    while (n < maxn) {
        /* Skip comment (#) and blank lines between recipes. */
        long pos = ftell(f);
        if (!fgets(line, sizeof(line), f)) break;
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        fseek(f, pos, SEEK_SET);
        if (!parse_recipe(f, &recipes[n])) break;
        n++;
    }
    fclose(f);
    return n;
}

/* ------------------------------------------------------------------ */
/* Reimplemented step math (plain C, fp32, torch order)                */
/* ------------------------------------------------------------------ */

/* unbiased std over out[0..n) in fp32 semantics (fp64 accumulation),
   matching torch noise.std() (default unbiased, divisor n-1) within ~1e-7. */
static float pop_std(const float *x, int n) {
    double m = 0.0;
    for (int i = 0; i < n; i++) m += (double)x[i];
    m /= (double)n;
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)x[i] - m;
        s += d * d;
    }
    if (n <= 1) return 0.0f;
    return (float)sqrt(s / (double)(n - 1));
}

/* flash step: denoised = z - mo*sigma; noise clip; z_next. */
static void flash_step(const float *z, const float *mo, const float *noise,
                       float sigma, float sigma_next, float s_noise,
                       float noise_clip_std, float *out, int n) {
    float denoised[MAX_STEP_N];
    float nz[MAX_STEP_N];
    for (int i = 0; i < n; i++) {
        denoised[i] = z[i] - mo[i] * sigma; /* fp32, torch order */
    }
    if (noise_clip_std > 0.0f) {
        float std = pop_std(noise, n);
        float clip = noise_clip_std * std;
        for (int i = 0; i < n; i++) {
            float v = noise[i];
            if (v > clip) v = clip;
            else if (v < -clip) v = -clip;
            nz[i] = v;
        }
    } else {
        memcpy(nz, noise, (size_t)n * sizeof(float));
    }
    for (int i = 0; i < n; i++) {
        /* (sigma_next*noise)*s_noise + (1-sigma_next)*denoised, torch order */
        out[i] = (sigma_next * nz[i]) * s_noise +
                 (1.0f - sigma_next) * denoised[i];
    }
}

/* flow_match step: dt = sigma_next - sigma; prev = sample + dt*mo. */
static void flow_match_step(const float *z, const float *mo,
                            float sigma, float sigma_next, float *out, int n) {
    float dt = sigma_next - sigma; /* fp32 */
    for (int i = 0; i < n; i++) {
        out[i] = z[i] + dt * mo[i]; /* fp32 */
    }
}

/* ------------------------------------------------------------------ */

static float maxabs_diff(const float *a, const float *b, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(int argc, char **argv) {
    const char *path = (argc > 1)
        ? argv[1]
        : "tests/unit/oracle_dump_matrix.txt";
    recipe_t recipes[MAX_RECIPES];
    int nrec = load_dump(path, recipes, MAX_RECIPES);
    if (nrec <= 0) {
        fprintf(stderr, "failed to parse dump %s\n", path);
        return 1;
    }

    int pass = 1;
    float max_sigma_diff = 0.0f;
    float max_ns_diff = 0.0f;
    float max_flash_diff = 0.0f;
    float max_fm_diff = 0.0f;
    float max_unipc_diff = 0.0f;

    /* Single MT19937 generator seeded once, matching the dump tool. */
    hd_torch_rng rng;
    hd_torch_rng_seed(&rng, 1234);

    for (int r = 0; r < nrec; r++) {
        recipe_t *rec = &recipes[r];
        hd_scheduler s;
        int wrote;

        printf("== %s (%d steps, shift %.1f, %s%s) ==\n",
               rec->name, rec->steps, rec->shift, rec->sched,
               rec->has_tlist ? ", tlist" : "");

        /* ---- 1. sigma derivation ----
           tlist recipes (Dev/Dev-edit frozen) use the explicit DEFAULT_TIMESTEPS
           via hd_scheduler_derive_dev; generic recipes use the class derivation. */
        if (rec->has_tlist)
            wrote = hd_scheduler_derive_dev(&s, 8.0f);
        else if (strcmp(rec->sched, "flash") == 0)
            wrote = hd_scheduler_derive_flash(&s, rec->steps, rec->shift, 8.0f);
        else if (strcmp(rec->sched, "flow_match") == 0)
            wrote = hd_scheduler_derive_flow_match(&s, rec->steps, rec->shift, 8.0f);
        else
            wrote = hd_scheduler_derive_default(&s, rec->steps, rec->shift, 8.0f);

        if (wrote != rec->n_sigmas) {
            printf("  FAIL: expected %d sigmas, got %d\n", rec->n_sigmas, wrote);
            pass = 0;
            continue;
        }
        float sd = maxabs_diff(s.sigmas, rec->sigmas, rec->n_sigmas);
        if (sd > max_sigma_diff) max_sigma_diff = sd;
        printf("  sigma  maxdiff = %.3e %s\n", sd,
               (sd <= 1e-6f) ? "OK" : "FAIL");
        if (sd > 1e-6f) pass = 0;

        /* ---- 2. noise_scale_schedule ---- */
        float nd = 0.0f;
        for (int i = 0; i < rec->n_noisescale; i++) {
            float v = hd_scheduler_noise_scale(&s, i, 8.0f, 8.0f);
            float d = fabsf(v - rec->noisescale[i]);
            if (d > nd) nd = d;
        }
        if (nd > max_ns_diff) max_ns_diff = nd;
        printf("  noise   maxdiff = %.3e %s\n", nd,
               (nd <= 1e-6f) ? "OK" : "FAIL");
        if (nd > 1e-6f) pass = 0;

        /* ---- RNG draws, replicating dump tool draw order ----
           make_example() always draws z, mo, noise; the mo is scaled by 0.3.
           flash then draws an extra noise inside step(). */
        float z[N], mo[N], unused[N];
        hd_torch_randn_f32(&rng, z, N);
        hd_torch_randn_f32(&rng, mo, N);
        hd_torch_randn_f32(&rng, unused, N);
        for (int i = 0; i < N; i++) mo[i] = mo[i] * 0.3f;

        float sigma0 = s.sigmas[0];
        float sigma1 = s.sigmas[1];
        float out[MAX_STEP_N];

        if (strcmp(rec->sched, "flash") == 0) {
            float noise[N];
            hd_torch_randn_f32(&rng, noise, N);
            flash_step(z, mo, noise, sigma0, sigma1, 8.0f, 8.0f, out, N);
            float d = maxabs_diff(out, rec->step, N);
            if (d > max_flash_diff) max_flash_diff = d;
            printf("  flash step maxdiff = %.3e %s\n", d,
                   (d <= 1e-5f) ? "OK" : "FAIL");
            if (d > 1e-5f) pass = 0;
        } else if (strcmp(rec->sched, "flow_match") == 0) {
            flow_match_step(z, mo, sigma0, sigma1, out, N);
            float d = maxabs_diff(out, rec->step, N);
            if (d > max_fm_diff) max_fm_diff = d;
            printf("  flow_match step maxdiff = %.3e %s\n", d,
                   (d <= 1e-5f) ? "OK" : "FAIL");
            if (d > 1e-5f) pass = 0;
        } else { /* default -> UniPC, 3 steps with constant mo */
            float *hist0 = malloc(N * sizeof(float));
            float *hist1 = malloc(N * sizeof(float));
            float *lasts = malloc(N * sizeof(float));
            float *scratch = malloc(6 * N * sizeof(float));
            float *sample = malloc(N * sizeof(float));
            float *prev = malloc(N * sizeof(float));
            if (!hist0 || !hist1 || !lasts || !scratch || !sample || !prev) {
                fprintf(stderr, "OOM\n");
                return 1;
            }
            memcpy(sample, z, N * sizeof(float));

            hd_scheduler_unipc u;
            memset(&u, 0, sizeof(u));
            u.model_outputs[0] = hist0;
            u.model_outputs[1] = hist1;
            u.last_sample = lasts;

            int nsteps = rec->n_step ? 3 : 0;
            for (int i = 0; i < nsteps; i++) {
                hd_scheduler_unipc_step(&s, &u, mo, sample, prev, N, scratch);
                memcpy(sample, prev, N * sizeof(float));
            }
            float d = maxabs_diff(sample, rec->step, N);
            if (d > max_unipc_diff) max_unipc_diff = d;
            printf("  unipc 3-step maxdiff = %.3e %s\n", d,
                   (d <= 1e-4f) ? "OK" : "FAIL");
            if (d > 1e-4f) pass = 0;

            free(hist0); free(hist1); free(lasts);
            free(scratch); free(sample); free(prev);
        }
        printf("\n");
    }

    printf("===========================================\n");
    printf("sigma maxdiff        : %.3e (tol 1e-6)\n", max_sigma_diff);
    printf("noise_scale maxdiff  : %.3e (tol 1e-6)\n", max_ns_diff);
    printf("flash step maxdiff   : %.3e (tol 1e-5)\n", max_flash_diff);
    printf("flow_match step diff : %.3e (tol 1e-5)\n", max_fm_diff);
    printf("unipc 3-step maxdiff : %.3e (tol 1e-4)\n", max_unipc_diff);
    printf("RESULT: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
