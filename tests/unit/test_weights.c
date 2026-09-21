#include "hidream.h"
#include "safetensors.h"
#include "weights.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s (at %s:%d)\n", msg, __FILE__, __LINE__); \
        failures++; \
    } else { \
        printf("ok: %s\n", msg); \
    } \
} while (0)

#define DEV_DIR "models/dev"

static void test_index_load(void) {
    hd_st_index idx;
    hd_status st = hd_st_index_load(DEV_DIR, &idx);
    CHECK(st == HD_OK, "load dev safetensors index");
    if (st != HD_OK) { printf("  err: %s\n", hd_st_last_error()); return; }

    CHECK(idx.n_tensors == 759, "index exposes 759 tensors");
    CHECK(idx.total_bytes == 35219551168LL, "index total payload bytes");

    /* The table must be name-sorted so binary search is valid. */
    int sorted = 1;
    for (int64_t i = 1; i < idx.n_tensors; i++) {
        if (strcmp(idx.tensors[i - 1].name, idx.tensors[i].name) > 0) { sorted = 0; break; }
    }
    CHECK(sorted, "tensor table is name-sorted");

    const hd_st_tensor *t = hd_st_index_find(&idx, "lm_head.weight");
    CHECK(t != NULL, "find lm_head.weight");
    if (t) {
        CHECK(t->rank == 2, "lm_head rank 2");
        CHECK(t->shape[0] == 151936 && t->shape[1] == 4096, "lm_head shape");
        CHECK(t->dtype == HD_DTYPE_F32, "lm_head dtype F32");
        CHECK(t->numel == 622329856LL, "lm_head numel");
        CHECK(t->nbytes == 2489319424LL, "lm_head bytes");
    }

    CHECK(hd_st_index_find(&idx, "does.not.exist") == NULL, "unknown tensor not found");

    hd_st_index_free(&idx);
}

static void test_inventory(void) {
    hd_tensor_manifest m;
    hd_status st = hd_tensor_manifest_load("config/tensor_manifest_dev.json", &m);
    CHECK(st == HD_OK, "load dev manifest for inventory");
    if (st != HD_OK) return;

    hd_weight_inventory inv;
    st = hd_weights_inventory(DEV_DIR, &m, &inv);
    CHECK(st == HD_OK, "inventory cross-check passes");
    if (st != HD_OK) { printf("  err: %s\n", hd_weights_last_error()); hd_tensor_manifest_free(&m); return; }

    CHECK(inv.n_expected == 759, "inventory expected 759");
    CHECK(inv.n_found == 759, "inventory found 759");
    CHECK(inv.missing == 0, "no missing tensors");
    CHECK(inv.unexpected == 0, "no unexpected tensors");
    CHECK(inv.shape_mismatch == 0, "no shape mismatches");
    CHECK(inv.dtype_mismatch == 0, "no dtype mismatches");
    CHECK(inv.numel_mismatch == 0, "no numel mismatches");
    CHECK(inv.total_numel == 8804887792LL, "inventory total numel");
    CHECK(inv.total_bytes == 35219551168LL, "inventory total bytes");
    CHECK(strcmp(inv.largest_name, "lm_head.weight") == 0, "largest tensor is lm_head");

    hd_tensor_manifest_free(&m);
}

static void test_probes(void) {
    hd_st_index idx;
    hd_status st = hd_st_index_load(DEV_DIR, &idx);
    CHECK(st == HD_OK, "load index for probes");
    if (st != HD_OK) return;

    hd_weight_inventory inv;
    memset(&inv, 0, sizeof(inv));
    st = hd_weights_probe_representatives(&idx, &inv);
    CHECK(st == HD_OK, "representative probes resolve");
    if (st != HD_OK) { printf("  err: %s\n", hd_weights_last_error()); hd_st_index_free(&idx); return; }

    CHECK(inv.n_probes == 17, "17 representative roles probed");

    /* Fingerprints captured from the frozen Python oracle (safetensors). */
    struct { const char *name; const char *sha; } expect[] = {
        {"model.visual.patch_embed.proj.weight",
         "9c29763ca3d6a45177c142879996380a8a30480cf96bd2a8af1bec1a775bb749"},
        {"model.language_model.layers.0.self_attn.q_proj.weight",
         "8cc6b011fa1149efc9731ad8ab798d34fffeb6bf30d1e11ad23703f35a6b35a2"},
        {"model.language_model.layers.0.self_attn.k_proj.weight",
         "e20c1114f05ecd7bebd4f17555735516a4a038bea7f46f626157bbae2c86aec5"},
        {"model.language_model.layers.0.self_attn.v_proj.weight",
         "2c22739d10f14b1df0c5eb91bdab4da1f9644f4147724c1aeeb3842e13a90293"},
        {"model.language_model.layers.18.mlp.gate_proj.weight",
         "68c16a075c441ff1f5cb190730e5b9ff3d5e45f4daeef7e468bc394edbaccaf1"},
        {"model.language_model.layers.18.mlp.down_proj.weight",
         "27ce0713471f25f7a5d8f61314ed4fdeb4c4607d4b4c982d34f85115c8ff51bd"},
        {"model.language_model.layers.35.self_attn.o_proj.weight",
         "9f3d7b98c06183398cc68a7548de21ef4296201f89108e75f7694aa8e2dab1ab"},
        {"model.language_model.layers.35.mlp.up_proj.weight",
         "01ecc5cfe807ad096b12182c90dd5d619a09188327702b570545d6276fd10fa9"},
        {"model.language_model.norm.weight",
         "2de9314e4f0a8b191b9716d94facd41e1c31dee79332d5e3284e51d7d901a3fb"},
        {"lm_head.weight",
         "ac88aa0b85ea30c8d6dc7070e599748e755c8a9996c74823820a0583348cfa1b"},
        {"model.t_embedder1.mlp.0.weight",
         "2d5b05fc16dcf4eafff6e62c51df9a2f9d98da011c676e29a13e294d01c15332"},
        {"model.x_embedder.proj1.weight",
         "3fc610d0323459723c6c906f93dbf3757dff8d695a984a8f23178b4227c7d081"},
        {"model.final_layer2.linear.weight",
         "df07d06e0cb79d82209c35b5bedd9d8e777fe96757d2256bceb91a6de616e6fb"},
        {"model.visual.blocks.0.attn.qkv.weight",
         "d60073472a9ca2de339b2485a87c629ad233703fdad4e1cf3f27895bca4e9329"},
        {"model.visual.blocks.26.mlp.linear_fc1.weight",
         "14cf11816f9bf8d16431870332138b8d8f74730ef5aaf0ea23146904eed1c778"},
        {"model.language_model.embed_tokens.weight",
         "8154913afd92e81cc680ac4a025493f6b37b5479800aafb266b39c4ae24542d2"},
        {"model.visual.deepstack_merger_list.0.linear_fc1.weight",
         "94b51931c08ba93b0e9b287465c673964b265ba2521efaa7c2f3de2d7924fe19"},
    };
    int n_expect = (int)(sizeof(expect) / sizeof(expect[0]));
    CHECK(inv.n_probes == n_expect, "probe count matches oracle table");

    for (int i = 0; i < n_expect && i < inv.n_probes; i++) {
        char msg[320];
        snprintf(msg, sizeof(msg), "probe fingerprint %s", expect[i].name);
        CHECK(strcmp(inv.probes[i].name, expect[i].name) == 0, msg);
        snprintf(msg, sizeof(msg), "probe sha256 %s", expect[i].name);
        CHECK(strcmp(inv.probes[i].sha256, expect[i].sha) == 0, msg);
    }

    hd_st_index_free(&idx);
}

static void test_device_placement(void) {
    hd_st_index idx;
    hd_status st = hd_st_index_load(DEV_DIR, &idx);
    CHECK(st == HD_OK, "load index for device placement");
    if (st != HD_OK) return;

    hd_weight_store store;
    st = hd_weights_to_device(DEV_DIR, &idx, 0, &store);
    CHECK(st == HD_OK, "place all dev weights on device");
    if (st != HD_OK) { printf("  err: %s\n", hd_weights_last_error()); hd_st_index_free(&idx); return; }

    CHECK(store.n_allocs == 759, "one tracked allocation per tensor");
    CHECK(store.device_bytes_allocated == 35219551168LL, "device bytes == payload bytes");
    CHECK(store.host_bytes_loaded == store.device_bytes_allocated, "host bytes == device bytes");
    CHECK(strcmp(store.largest_name, "lm_head.weight") == 0, "largest allocation is lm_head");
    CHECK(store.compute_major == 12 && store.compute_minor == 1, "GB10 compute capability 12.1");

    int all_nonnull = 1;
    for (int64_t i = 0; i < store.n_allocs; i++) {
        if (!store.allocs[i].dev_ptr) { all_nonnull = 0; break; }
    }
    CHECK(all_nonnull, "every tracked buffer has a device pointer");

    hd_weight_store_free(&store);
    CHECK(store.n_allocs == 0 && store.allocs == NULL, "cleanup releases the allocation table");

    /* A second placement must reproduce the same accounting, proving the first
     * run freed everything it owned. */
    hd_weight_store again;
    st = hd_weights_to_device(DEV_DIR, &idx, 0, &again);
    CHECK(st == HD_OK, "second placement succeeds after cleanup");
    if (st == HD_OK) {
        CHECK(again.device_bytes_allocated == 35219551168LL, "second placement same device bytes");
        CHECK(again.n_allocs == 759, "second placement same allocation count");
        hd_weight_store_free(&again);
    }

    hd_st_index_free(&idx);
}

static void test_base_profile_supported(void) {
    hd_profile p;
    hd_status st = hd_profile_load("base", "config", &p);
    CHECK(st == HD_OK, "base profile still loads");
    if (st == HD_OK) {
        CHECK(strcmp(p.local_path, "models/base") == 0, "base local path preserved");
        CHECK(p.immutable_revision && strlen(p.immutable_revision) == 40,
              "base revision is an immutable 40-char sha");
        CHECK(p.variant && strcmp(p.variant, "base") == 0, "base variant set");
        hd_profile_free(&p);
    }
}

/* R1: the public interface resolves the runtime profile from GGUF metadata. */
static void test_profile_from_gguf(void) {
    const char *path = getenv("O1_TEST_GGUF");
    if (!path || !*path) {
        printf("skip: O1_TEST_GGUF not set (profile-from-gguf)\n");
        return;
    }
    hd_profile p;
    hd_status st = hd_profile_from_gguf(path, &p);
    CHECK(st == HD_OK, "profile resolves from GGUF metadata");
    if (st != HD_OK) { printf("  err: %s\n", hd_last_error()); return; }
    CHECK(p.profile && (!strcmp(p.profile, "dev") || !strcmp(p.profile, "base")),
          "resolved profile is dev or base");
    CHECK(p.variant && p.variant[0], "resolved variant is set");
    CHECK(p.immutable_revision && strlen(p.immutable_revision) >= 40,
          "resolved revision is a full upstream sha");
    CHECK(p.dtype && p.dtype[0], "resolved dtype is set");
    CHECK(p.layout_version > 0, "resolved layout_version is set");
    CHECK(p.num_inference_steps == (!strcmp(p.profile, "base") ? 50 : 28),
          "resolved steps match the execution profile");
    CHECK(p.local_path && strcmp(p.local_path, path) == 0,
          "resolved local_path is the GGUF path");
    hd_profile_free(&p);

    hd_profile bad;
    st = hd_profile_from_gguf("/nonexistent-model.gguf", &bad);
    CHECK(st != HD_OK, "missing GGUF fails closed");
}

int main(void) {
    test_index_load();
    test_inventory();
    test_probes();
    test_device_placement();
    test_base_profile_supported();
    test_profile_from_gguf();

    if (failures) {
        printf("\n%d assertion(s) failed\n", failures);
        return 1;
    }
    printf("\nall weight ingestion assertions passed\n");
    return 0;
}