#include "hidream.h"
#include "safetensors.h"
#include "weights.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0) {
    printf("usage: %s [options]\n", argv0);
    printf("\n");
    printf("M1.0 profile validation (default):\n");
    printf("  --model dev|base        profile to validate (default: dev)\n");
    printf("  --config-dir DIR        config directory (default: config)\n");
    printf("\n");
    printf("M1.1 weight ingestion (V0, no model forward):\n");
    printf("  --inventory             cross-check shards against frozen manifest\n");
    printf("  --probe                 fingerprint representative tensors\n");
    printf("  --to-device             load weights into deterministic CUDA buffers\n");
    printf("  --device N              CUDA device index (default: 0)\n");
    printf("  --model-dir DIR         override profile local_path\n");
    printf("\n");
    printf("  -h, --help              show this help\n");
}

int main(int argc, char **argv) {
    const char *profile = "dev";
    const char *config_dir = "config";
    const char *model_dir = NULL;
    int device_id = 0;
    int do_inventory = 0, do_probe = 0, do_to_device = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            profile = argv[++i];
        } else if (strcmp(argv[i], "--config-dir") == 0 && i + 1 < argc) {
            config_dir = argv[++i];
        } else if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            device_id = atoi(argv[++i]);
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