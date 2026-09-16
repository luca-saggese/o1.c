#include "hidream.h"
#include "generate.h"
#include "json.h"
#include "png_wrap.h"
#include "request.h"
#include "safetensors.h"
#include "weights.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0) {
    printf("usage: %s [options]\n", argv0);
    printf("\n");
    printf("M1.8 production prompt-to-image generation:\n");
    printf("  --model dev|base        profile to use (default: dev)\n");
    printf("  --prompt TEXT           user prompt\n");
    printf("  --mode t2i|edit|personalize|...   generation mode (default: t2i)\n");
    printf("  --ref-image PATH        reference image (repeatable, edit/personalize)\n");
    printf("  --width N               output width (default: 1024)\n");
    printf("  --height N              output height (default: 1024)\n");
    printf("  --steps N               inference steps (default per profile)\n");
    printf("  --seed N                RNG seed (default: 123456)\n");
    printf("  --scheduler flash|default|flow_match   (default per profile)\n");
    printf("  --guidance F            CFG scale (default per profile)\n");
    printf("  --shift F               scheduler shift (default per profile)\n");
    printf("  --output PATH           output PNG path (default: output.png)\n");
    printf("  --model-dir DIR         override profile local_path\n");
    printf("  --device N              CUDA device index (default: 0)\n");
    printf("  --noise-start F         noise_scale_start (default 8.0)\n");
    printf("  --noise-end F           noise_scale_end (default 8.0)\n");
    printf("  --noise-clip F          noise_clip_std (default 8.0)\n");
    printf("\n");
    printf("M1.0 profile validation (default):\n");
    printf("  --config-dir DIR        config directory (default: config)\n");
    printf("\n");
    printf("M1.1 weight ingestion (V0, no model forward):\n");
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
                          const char *engine_commit) {
    char meta_path[1024];
    size_t n = strlen(png_path);
    if (n < 4 || strcmp(png_path + n - 4, ".png")) return -1;
    snprintf(meta_path, sizeof(meta_path), "%.*s.json", (int)(n - 4), png_path);

    FILE *f = fopen(meta_path, "w");
    if (!f) return -1;
    fprintf(f, "{\n");
    fprintf(f, "  \"model\": \"%s\",\n", req->profile);
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
    const char *profile = "dev";
    const char *config_dir = "config";
    const char *model_dir = NULL;
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

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
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
        hd_generation_request req;
        memset(&req, 0, sizeof(req));
        req.prompt = prompt;
        req.profile = profile;
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

        hd_request_defaults(&req);
        if (req.width <= 0) req.width = 1024;
        if (req.height <= 0) req.height = 1024;
        if (!output) output = "output.png";
        hd_status st = hd_request_validate(&req);
        if (st != HD_OK) {
            fprintf(stderr, "FAIL: %s\n", hd_last_error());
            return 1;
        }

        hd_profile p;
        st = hd_profile_load(profile, config_dir, &p);
        if (st != HD_OK) {
            fprintf(stderr, "FAIL: %s\n", hd_last_error());
            return 1;
        }
        const char *dir = model_dir ? model_dir : p.local_path;

        setvbuf(stdout, NULL, _IONBF, 0);
        printf("[hidream] generating %dx%d %d-step %s (seed %llu, scheduler %s)\n",
               req.width, req.height, req.steps, req.profile,
               (unsigned long long)req.seed, hd_scheduler_name(req.scheduler));

        unsigned char *rgb = NULL;
        int ow = 0, oh = 0;
        st = hd_generate(&req, dir, device_id, &rgb, &ow, &oh);
        if (st != HD_OK) {
            fprintf(stderr, "FAIL: generation: %s\n", hd_last_error());
            hd_profile_free(&p);
            return 1;
        }

        int rc = hd_png_write_rgb(output, ow, oh, rgb, "hidream",
                                  hd_mode_name(req.mode));
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
        if (write_metadata(output, &req, engine_commit[0] ? engine_commit : NULL) != 0)
            fprintf(stderr, "warn: metadata write failed\n");

        printf("PASS: generation %s (%dx%d)\n", output, ow, oh);
        hd_profile_free(&p);
        return 0;
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
        hd_weight_store store;
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