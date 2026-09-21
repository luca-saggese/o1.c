#define _POSIX_C_SOURCE 200809L
#include "hidream.h"
#include "generate.h"
#include "json.h"
#include "o1_timing.h"
#include "png_wrap.h"
#include "request.h"
#include "ref_alias.h"
#include "sequence.h"
#include "safetensors.h"
#include "weights.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* Progress bar                                                        */
/* ------------------------------------------------------------------ */

/*
 * Minimal in-place progress bar for the denoise pipeline. Prints
 * "step i/N" plus elapsed and estimated remaining time on one line and
 * refreshes with a carriage return; disabled when stderr is not a TTY or
 * --no-progress is given. It is a pure frontend: it never touches device
 * state.
 */
typedef struct {
    int steps;         /* total denoise steps */
    int cur_step;      /* 1-based current step */
    int enabled;
    int started;       /* clock started */
    int last_len;      /* length of the last printed line */
    struct timespec t0;
} progress_state;

static double progress_now(const progress_state *p) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)(t.tv_sec - p->t0.tv_sec) +
           (double)(t.tv_nsec - p->t0.tv_nsec) / 1e9;
}

/* mm:ss (or h:mm:ss past an hour) */
static void progress_fmt_time(double secs, char *out, size_t n) {
    if (secs < 0.0) secs = 0.0;
    int s = (int)(secs + 0.5);
    if (s >= 3600)
        snprintf(out, n, "%d:%02d:%02d", s / 3600, (s / 60) % 60, s % 60);
    else
        snprintf(out, n, "%02d:%02d", s / 60, s % 60);
}

static void progress_render(progress_state *p) {
    if (!p->enabled) return;
    int width = 0;
    struct winsize wsz;
    if (ioctl(STDERR_FILENO, TIOCGWINSZ, &wsz) == 0 && wsz.ws_col > 0)
        width = wsz.ws_col;
    else
        width = 80;

    long total = p->steps > 0 ? p->steps : 1;
    long done = p->cur_step;
    if (done < 0) done = 0;
    if (done > total) done = total;
    double frac = (double)done / (double)total;
    if (frac < 0.0) frac = 0.0;
    if (frac > 1.0) frac = 1.0;

    /* Elapsed is measured; remaining extrapolates from the overall rate. */
    double elapsed = p->started ? progress_now(p) : 0.0;
    double remaining = (frac > 0.0) ? elapsed * (1.0 - frac) / frac : 0.0;
    char es[16], rs[16];
    progress_fmt_time(elapsed, es, sizeof(es));
    progress_fmt_time(remaining, rs, sizeof(rs));

    char tail[96];
    int tn = snprintf(tail, sizeof(tail), " step %d/%d  %s elapsed  %s left",
                      p->cur_step, p->steps, es, rs);
    int bar_w = width - tn - 8; /* "[...] NN%" */
    if (bar_w < 10) bar_w = 10;
    if (bar_w > 60) bar_w = 60;

    char line[256];
    int off = snprintf(line, sizeof(line), "\r[");
    int filled = (int)(frac * bar_w + 0.5);
    for (int i = 0; i < bar_w; i++) {
        if (off < (int)sizeof(line) - 1)
            line[off++] = (i < filled) ? '#' : '-';
    }
    off += snprintf(line + off, sizeof(line) - (size_t)off, "] %3d%%%s",
                    (int)(frac * 100.0 + 0.5), tail);
    /* pad to clear the previous longer line */
    while (off < p->last_len && off < (int)sizeof(line) - 1) line[off++] = ' ';
    line[off] = '\0';
    fputs(line, stderr);
    fflush(stderr);
    p->last_len = off;
}

static void progress_step_cb(int step, int total, void *user) {
    progress_state *p = (progress_state *)user;
    if (!p->enabled) return;
    if (!p->started) {
        clock_gettime(CLOCK_MONOTONIC, &p->t0);
        p->started = 1;
    }
    p->cur_step = step;
    p->steps = total;
    progress_render(p);
}

static void usage(const char *argv0) {
    printf("usage: %s [options]\n", argv0);
    printf("\n");
    printf("HiDream-O1 native image generation:\n");
    printf("  --model-path PATH       production GGUF model file (public interface;\n");
    printf("                          profile is inferred from the GGUF metadata)\n");
    printf("  --prompt TEXT           user prompt\n");
    printf("  --mode t2i|edit|personalize|...   generation mode (default: t2i)\n");
    printf("  --ref-image PATH        reference image (repeatable, max %d)\n",
           HD_SEQ_MAX_REFS);
    printf("  --ref-image NAME=PATH   named reference; use @NAME in --prompt\n");
    printf("  --keep-original-aspect  single ref: derive output dims from ref\n");
    printf("  --layout-bboxes JSON    layout bboxes for personalize+layout\n");
    printf("  --verbose               print reference alias mapping / expanded prompt\n");
    printf("  --no-progress           disable the generation progress bar\n");
    printf("  --width N               output width (default: 1024)\n");
    printf("  --height N              output height (default: 1024)\n");
    printf("  --steps N               inference steps (default per model profile)\n");
    printf("  --seed N                RNG seed (default: 123456)\n");
    printf("  --scheduler flash|default|flow_match   (default per model profile)\n");
    printf("  --guidance F            CFG scale (default per model profile)\n");
    printf("  --shift F               scheduler shift (default per model profile)\n");
    printf("  --output PATH           output PNG path (default: output.png)\n");
    printf("  --device N              CUDA device index (default: 0)\n");
    printf("  --noise-start F         noise_scale_start (default 8.0)\n");
    printf("  --noise-end F           noise_scale_end (default 8.0)\n");
    printf("  --noise-clip F          noise_clip_std (default 8.0)\n");
    printf("  --lora FILE[:MULT]      apply LoRA adapter (repeatable)\n");
    printf("\n");
    printf("Internal / debug (safetensors development checkouts):\n");
    printf("  --model dev|base        profile name (internal; prefer --model-path)\n");
    printf("  --model-dir DIR         safetensors directory (internal)\n");
    printf("  --config-dir DIR        config directory (default: config)\n");
    printf("  --inventory             cross-check shards against frozen manifest\n");
    printf("  --probe                 fingerprint representative tensors\n");
    printf("  --to-device             load weights into deterministic CUDA buffers\n");
    printf("\n");
    printf("  -h, --help              show this help\n");
}

/* JSON metadata writer for the generation artifact. */
static void json_escape(FILE *f, const char *s) {
    fputc('"', f);
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        switch (c) {
        case '"': fputs("\\\"", f); break;
        case '\\': fputs("\\\\", f); break;
        case '\n': fputs("\\n", f); break;
        case '\r': fputs("\\r", f); break;
        case '\t': fputs("\\t", f); break;
        default:
            if (c < 0x20) fprintf(f, "\\u%04x", c);
            else fputc(c, f);
        }
    }
    fputc('"', f);
}

static int write_metadata(const char *png_path, const hd_generation_request *req,
                          const char *variant, const char *engine_commit) {
    char meta_path[1024];
    size_t n = strlen(png_path);
    if (n < 4 || strcmp(png_path + n - 4, ".png")) return -1;
    snprintf(meta_path, sizeof(meta_path), "%.*s.json", (int)(n - 4), png_path);

    FILE *f = fopen(meta_path, "w");
    if (!f) return -1;
    fprintf(f, "{\n");
    fprintf(f, "  \"model\": \"%s\",\n", variant ? variant : req->profile);
    fprintf(f, "  \"mode\": \"%s\",\n", hd_mode_name(req->mode));
    fprintf(f, "  \"width\": %d,\n", req->width);
    fprintf(f, "  \"height\": %d,\n", req->height);
    fprintf(f, "  \"steps\": %d,\n", req->steps);
    fprintf(f, "  \"seed\": %llu,\n", (unsigned long long)req->seed);
    fprintf(f, "  \"precision\": \"bf16\",\n");
    fprintf(f, "  \"scheduler\": \"%s\",\n", hd_scheduler_name(req->scheduler));
    fprintf(f, "  \"guidance_scale\": %.4f,\n", req->guidance_scale);
    fprintf(f, "  \"shift\": %.4f,\n", req->shift);
    fprintf(f, "  \"engine_commit\": \"%s\",\n", engine_commit ? engine_commit : "");
    fprintf(f, "  \"prompt\": ");
    json_escape(f, req->prompt);
    fprintf(f, "\n}\n");
    fclose(f);
    return 0;
}

int main(int argc, char **argv) {
    O1_TIMING_BEGIN("TOTAL_PROCESS");
    const char *profile = "dev";
    const char *config_dir = "config";
    const char *model_dir = NULL;
    const char *model_path = NULL;
    int device_id = 0;
    int do_inventory = 0, do_probe = 0, do_to_device = 0;

    const char *prompt = NULL;
    const char *output = NULL;
    int width = 0, height = 0, steps = 0;
    uint64_t seed = 123456;
    hd_mode mode = HD_MODE_T2I;
    hd_scheduler_kind sched = HD_SCHED_DEFAULT; /* sentinel: "not set" */
    float guidance = -1.0f, shift = -1.0f;
    float noise_start = 0.0f, noise_end = 0.0f, noise_clip = 0.0f;
    hd_lora_config lora_cfg = {0};
    char lora_paths[8][512];
    float lora_mults[8];
    int lora_count = 0;
    hd_reference_image refs[HD_SEQ_MAX_REFS];
    int ref_count = 0;
    char ref_aliases[HD_SEQ_MAX_REFS][128];
    int verbose = 0;
    int no_progress = 0;
    int keep_original_aspect = 0;
    const char *layout_bboxes = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model-path") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            profile = argv[++i];
        } else if (strcmp(argv[i], "--config-dir") == 0 && i + 1 < argc) {
            config_dir = argv[++i];
        } else if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output = argv[++i];
        } else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            width = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            height = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = hd_mode_from_name(argv[++i]);
        } else if (strcmp(argv[i], "--scheduler") == 0 && i + 1 < argc) {
            sched = hd_scheduler_from_name(argv[++i]);
        } else if (strcmp(argv[i], "--guidance") == 0 && i + 1 < argc) {
            guidance = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--shift") == 0 && i + 1 < argc) {
            shift = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--noise-start") == 0 && i + 1 < argc) {
            noise_start = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--noise-end") == 0 && i + 1 < argc) {
            noise_end = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--noise-clip") == 0 && i + 1 < argc) {
            noise_clip = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--lora") == 0 && i + 1 < argc) {
            if (lora_count >= 8) {
                fprintf(stderr, "too many --lora (max 8)\n");
                return 2;
            }
            const char *arg = argv[++i];
            const char *colon = strchr(arg, ':');
            float mult = 1.0f;
            size_t plen = strlen(arg);
            if (colon) {
                plen = (size_t)(colon - arg);
                mult = (float)atof(colon + 1);
            }
            if (plen == 0 || plen >= sizeof(lora_paths[0])) {
                fprintf(stderr, "invalid --lora: %s\n", arg);
                return 2;
            }
            memcpy(lora_paths[lora_count], arg, plen);
            lora_paths[lora_count][plen] = '\0';
            lora_mults[lora_count] = mult;
            lora_count++;
        } else if (strcmp(argv[i], "--ref-image") == 0 && i + 1 < argc) {
            if (ref_count >= HD_SEQ_MAX_REFS) {
                fprintf(stderr, "too many --ref-image (max %d)\n",
                        HD_SEQ_MAX_REFS);
                return 2;
            }
            const char *alias = NULL, *rpath = NULL;
            if (hd_ref_alias_split(argv[++i], &alias, &rpath) != HD_OK) {
                fprintf(stderr, "invalid --ref-image: %s\n", argv[i]);
                return 2;
            }
            if (alias) {
                /* alias points at the argument prefix; materialize it. */
                size_t an = (size_t)(strchr(alias, '=') - alias);
                if (an >= sizeof(ref_aliases[ref_count])) {
                    fprintf(stderr, "reference alias too long\n");
                    return 2;
                }
                memcpy(ref_aliases[ref_count], alias, an);
                ref_aliases[ref_count][an] = '\0';
                refs[ref_count].alias = ref_aliases[ref_count];
            } else {
                ref_aliases[ref_count][0] = '\0';
                refs[ref_count].alias = NULL;
            }
            refs[ref_count].path = rpath;
            refs[ref_count].role = HD_REF_SUBJECT;
            ref_count++;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            verbose = 1;
        } else if (strcmp(argv[i], "--no-progress") == 0) {
            no_progress = 1;
        } else if (strcmp(argv[i], "--keep-original-aspect") == 0) {
            keep_original_aspect = 1;
        } else if (strcmp(argv[i], "--layout-bboxes") == 0 && i + 1 < argc) {
            layout_bboxes = argv[++i];
        } else if (strcmp(argv[i], "--inventory") == 0) {
            do_inventory = 1;
        } else if (strcmp(argv[i], "--probe") == 0) {
            do_probe = 1;
        } else if (strcmp(argv[i], "--to-device") == 0) {
            do_to_device = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            usage(argv[0]);
            return 2;
        }
    }

    /* ---- M1.8 production generation path ---- */
    if (prompt) {
        O1_TIMING_BEGIN("MODEL_STARTUP");
        hd_generation_request req;
        memset(&req, 0, sizeof(req));
        req.prompt = prompt;
        req.mode = mode;
        req.width = width;
        req.height = height;
        req.seed = seed;
        req.steps = steps;
        req.guidance_scale = guidance;
        req.shift = shift;
        req.scheduler = sched;
        req.noise_scale_start = noise_start;
        req.noise_scale_end = noise_end;
        req.noise_clip_std = noise_clip;
        req.progress_cb = NULL;
        /* In-place progress bar (stderr, TTY only unless --no-progress). */
        progress_state prog;
        memset(&prog, 0, sizeof(prog));
        prog.steps = req.steps > 0 ? req.steps : 1;
        prog.enabled = !no_progress && isatty(STDERR_FILENO);
        if (prog.enabled) {
            req.progress_cb = progress_step_cb;
            req.progress_user = &prog;
        }
        req.references = ref_count > 0 ? refs : NULL;
        req.reference_count = (size_t)ref_count;
        req.keep_original_aspect = keep_original_aspect;
        if (verbose) setenv("O1_VERBOSE_REF", "1", 1);

        if (layout_bboxes) {
            hd_layout_condition *layout_conds = NULL;
            size_t n_layout = 0;
            hd_status lst = hd_layout_parse(layout_bboxes, &layout_conds,
                                            &n_layout);
            if (lst != HD_OK) {
                fprintf(stderr, "FAIL: layout bboxes: %s\n", hd_last_error());
                return 1;
            }
            req.layout = layout_conds;
        }

        if (lora_count > 0) {
            static hd_lora_spec lora_specs[8];
            for (int k = 0; k < lora_count; k++) {
                lora_specs[k].path = lora_paths[k];
                lora_specs[k].multiplier = lora_mults[k];
            }
            lora_cfg.items = lora_specs;
            lora_cfg.count = (size_t)lora_count;
            req.lora = &lora_cfg;
        }

        hd_profile p;
        hd_status st;
        if (model_path) {
            st = hd_profile_from_gguf(model_path, &p);
        } else {
            st = hd_profile_load(profile, config_dir, &p);
        }
        if (st != HD_OK) {
            fprintf(stderr, "FAIL: %s\n", hd_last_error());
            return 1;
        }
        req.profile = p.profile;
        const char *dir = model_path ? model_path
                                     : (model_dir ? model_dir : p.local_path);

        hd_request_defaults(&req);
        if (req.width <= 0) req.width = 1024;
        if (req.height <= 0) req.height = 1024;
        if (!output) output = "output.png";
        st = hd_request_validate(&req);
        if (st != HD_OK) {
            fprintf(stderr, "FAIL: %s\n", hd_last_error());
            hd_profile_free(&p);
            return 1;
        }

        setvbuf(stdout, NULL, _IONBF, 0);
        printf("[hidream] generating %dx%d %d-step %s (seed %llu, scheduler %s)\n",
               req.width, req.height, req.steps, p.variant ? p.variant : req.profile,
               (unsigned long long)req.seed, hd_scheduler_name(req.scheduler));

        O1_TIMING_BEGIN("REQUEST_TOTAL");
        unsigned char *rgb = NULL;
        int ow = 0, oh = 0;
        st = hd_generate(&req, dir, device_id, &rgb, &ow, &oh);
        O1_TIMING_END("REQUEST_TOTAL");
        if (prog.enabled) {
            /* Finish the bar at 100% and move off the progress line. */
            prog.cur_step = prog.steps;
            progress_render(&prog);
            fputc('\n', stderr);
        }
        if (st != HD_OK) {
            fprintf(stderr, "FAIL: generation: %s\n", hd_last_error());
            hd_profile_free(&p);
            return 1;
        }

        O1_TIMING_BEGIN("IMAGE_ENCODE_WRITE");
        int rc = hd_png_write_rgb(output, ow, oh, rgb, "hidream",
                                  hd_mode_name(req.mode));
        O1_TIMING_END("IMAGE_ENCODE_WRITE");
        if (rc != 0) {
            fprintf(stderr, "FAIL: PNG write %s\n", output);
            free(rgb);
            hd_profile_free(&p);
            return 1;
        }
        free(rgb);

        /* engine commit for metadata */
        char engine_commit[64] = "";
        FILE *headf = fopen(".git/HEAD", "r");
        if (headf) {
            char ref[256] = "";
            if (fscanf(headf, "ref: %255s", ref) == 1) {
                char refpath[320];
                snprintf(refpath, sizeof(refpath), ".git/%s", ref);
                FILE *rf = fopen(refpath, "r");
                if (rf) {
                    if (fscanf(rf, "%63s", engine_commit) == 1) {
                        /* ok */
                    }
                    fclose(rf);
                }
            }
            fclose(headf);
        }
        if (write_metadata(output, &req, p.variant, engine_commit[0] ? engine_commit : NULL) != 0)
            fprintf(stderr, "warn: metadata write failed\n");

        O1_TIMING_END("MODEL_STARTUP");
        O1_TIMING_END("TOTAL_PROCESS");
#ifdef O1_DEBUG_TIMING
        {
            const char *tp = getenv("O1_TIMING_JSON");
            o1_timing_report(tp ? tp : "artifacts/m2/prebaseline/native_timing.json");
        }
#endif

        printf("PASS: generation %s (%dx%d)\n", output, ow, oh);
        hd_profile_free(&p);
        return 0;
    }

    if (model_path) {
        fprintf(stderr, "FAIL: --model-path requires --prompt\n");
        return 2;
    }

    hd_status st = hd_validate_profile(config_dir, profile);
    if (st != HD_OK) {
        fprintf(stderr, "FAIL: %s\n", hd_last_error());
        return 1;
    }

    if (!do_inventory && !do_probe && !do_to_device) {
        printf("PASS: %s profile valid\n", profile);
        return 0;
    }

    hd_profile p;
    st = hd_profile_load(profile, config_dir, &p);
    if (st != HD_OK) {
        fprintf(stderr, "FAIL: %s\n", hd_last_error());
        return 1;
    }
    const char *dir = model_dir ? model_dir : p.local_path;

    char manifest_path[1024];
    snprintf(manifest_path, sizeof(manifest_path), "%s/tensor_manifest_%s.json",
             config_dir, profile);
    hd_tensor_manifest manifest;
    st = hd_tensor_manifest_load(manifest_path, &manifest);
    if (st != HD_OK) {
        fprintf(stderr, "FAIL: %s\n", hd_last_error());
        hd_profile_free(&p);
        return 1;
    }

    int rc = 0;

    if (do_inventory || do_probe) {
        hd_weight_inventory inv;
        st = hd_weights_inventory(dir, &manifest, &inv);
        if (st != HD_OK) {
            fprintf(stderr, "FAIL: %s\n", hd_weights_last_error());
            rc = 1;
            goto done;
        }
        printf("[hd] inventory: expected=%lld found=%lld missing=%lld unexpected=%lld\n",
               (long long)inv.n_expected, (long long)inv.n_found,
               (long long)inv.missing, (long long)inv.unexpected);
        printf("[hd] inventory: shape_mismatch=%lld dtype_mismatch=%lld numel_mismatch=%lld\n",
               (long long)inv.shape_mismatch, (long long)inv.dtype_mismatch,
               (long long)inv.numel_mismatch);
        printf("[hd] inventory: total_bytes=%lld total_numel=%lld largest=%s (%lld)\n",
               (long long)inv.total_bytes, (long long)inv.total_numel,
               inv.largest_name, (long long)inv.largest_numel);

        if (do_probe) {
            hd_st_index idx;
            st = hd_st_index_load(dir, &idx);
            if (st != HD_OK) {
                fprintf(stderr, "FAIL: %s\n", hd_st_last_error());
                rc = 1;
                goto done;
            }
            st = hd_weights_probe_representatives(&idx, &inv);
            if (st != HD_OK) {
                fprintf(stderr, "FAIL: %s\n", hd_weights_last_error());
                hd_st_index_free(&idx);
                rc = 1;
                goto done;
            }
            printf("[hd] probes: %d representative tensors\n", inv.n_probes);
            for (int i = 0; i < inv.n_probes; i++) {
                const hd_weight_probe *pr = &inv.probes[i];
                printf("  %-60s %s\n", pr->name, pr->sha256);
                printf("    shard=%s numel=%lld bytes=%lld first=[%s]\n",
                       pr->shard, (long long)pr->numel, (long long)pr->nbytes,
                       pr->first_values);
            }
            hd_st_index_free(&idx);
        }
    }

    if (do_to_device) {
        hd_st_index idx;
        st = hd_st_index_load(dir, &idx);
        if (st != HD_OK) {
            fprintf(stderr, "FAIL: %s\n", hd_st_last_error());
            rc = 1;
            goto done;
        }
        hd_weight_store store = {0};
        st = hd_weights_to_device(dir, &idx, device_id, &store);
        if (st != HD_OK) {
            fprintf(stderr, "FAIL: %s\n", hd_weights_last_error());
            hd_st_index_free(&idx);
            rc = 1;
            goto done;
        }
        printf("[hd] device: id=%d name=%s cc=%d.%d mem=%.1f GiB\n",
               store.device_id, store.device_name, store.compute_major,
               store.compute_minor,
               (double)store.device_mem_bytes / (1024.0 * 1024 * 1024));
        printf("[hd] device: host_bytes_loaded=%lld device_bytes_allocated=%lld\n",
               (long long)store.host_bytes_loaded,
               (long long)store.device_bytes_allocated);
        printf("[hd] device: n_allocations=%lld largest=%s (%lld)\n",
               (long long)store.n_allocs, store.largest_name,
               (long long)store.largest_numel);
        hd_weight_store_free(&store);
        printf("[hd] device: all buffers freed\n");
        hd_st_index_free(&idx);
    }

done:
    hd_tensor_manifest_free(&manifest);
    hd_profile_free(&p);
    if (rc == 0) printf("PASS: %s weight ingestion\n", profile);
    return rc;
}