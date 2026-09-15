#include "hidream.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        printf("FAIL: %s (at %s:%d)\n", msg, __FILE__, __LINE__); \
        failures++; \
    } else { \
        printf("ok: %s\n", msg); \
    } \
} while (0)

static void test_dtype(void) {
    CHECK(hd_dtype_from_string("float32") == HD_DTYPE_F32, "float32 dtype");
    CHECK(hd_dtype_from_string("torch.float32") == HD_DTYPE_F32, "torch.float32 dtype");
    CHECK(hd_dtype_from_string("bfloat16") == HD_DTYPE_BF16, "bfloat16 dtype");
    CHECK(hd_dtype_from_string("torch.bfloat16") == HD_DTYPE_BF16, "torch.bfloat16 dtype");
    CHECK(hd_dtype_from_string("nonsense") == HD_DTYPE_UNKNOWN, "unknown dtype");
    CHECK(strcmp(hd_dtype_to_string(HD_DTYPE_F32), "float32") == 0, "dtype to string");
}

static void test_numel(void) {
    int64_t shape[] = {2, 3, 4};
    CHECK(hd_numel(shape, 3) == 24, "numel product");
}

static void test_manifest_load(void) {
    hd_tensor_manifest m;
    hd_status st = hd_tensor_manifest_load("config/tensor_manifest_dev.json", &m);
    CHECK(st == HD_OK, "load dev tensor manifest");
    CHECK(m.n_tensors == 759, "dev tensor count == 759");
    CHECK(m.parameter_count == 8804887792LL, "dev parameter count == 8,804,887,792");
    CHECK(strcmp(m.profile, "dev") == 0, "manifest profile == dev");

    const char *first = "model.visual.patch_embed.proj.weight";
    const char *last = "lm_head.weight";
    CHECK(strcmp(m.tensors[0].name, first) == 0, "first tensor name");
    CHECK(m.tensors[0].rank == 5, "first tensor rank 5");
    CHECK(strcmp(m.tensors[m.n_tensors - 1].name, last) == 0, "last tensor name");
    CHECK(m.tensors[m.n_tensors - 1].rank == 2, "last tensor rank 2");
    hd_tensor_manifest_free(&m);
}

static void test_manifest_compare_identical(void) {
    hd_tensor_manifest a, b;
    CHECK(hd_tensor_manifest_load("config/tensor_manifest_dev.json", &a) == HD_OK, "load A");
    CHECK(hd_tensor_manifest_load("config/tensor_manifest_dev.json", &b) == HD_OK, "load B");
    hd_manifest_diff diff;
    CHECK(hd_tensor_manifest_compare(&a, &b, &diff) == HD_OK, "identical compare");
    CHECK(diff.missing == 0 && diff.unexpected == 0 && diff.shape_mismatch == 0 &&
          diff.dtype_mismatch == 0 && diff.numel_mismatch == 0, "zero diff");
    hd_tensor_manifest_free(&a);
    hd_tensor_manifest_free(&b);
}

static void test_manifest_compare_mismatch(void) {
    hd_tensor_manifest a, b;
    CHECK(hd_tensor_manifest_load("config/tensor_manifest_dev.json", &a) == HD_OK, "load A2");
    CHECK(hd_tensor_manifest_load("config/tensor_manifest_dev.json", &b) == HD_OK, "load B2");
    b.tensors[1].shape[0] += 1;
    b.tensors[1].numel += 1;
    hd_manifest_diff diff;
    CHECK(hd_tensor_manifest_compare(&a, &b, &diff) == HD_ERR_MISMATCH, "mismatch detected");
    CHECK(diff.shape_mismatch == 1, "one shape mismatch");
    hd_tensor_manifest_free(&a);
    hd_tensor_manifest_free(&b);
}

static void test_unknown_profile(void) {
    CHECK(hd_validate_profile("config", "nope") == HD_ERR_PROFILE, "unknown profile fails closed");
}

static void test_profile_load_traversal(void) {
    hd_profile p;
    CHECK(hd_profile_load("../python", "config", &p) == HD_ERR_PROFILE, "path traversal rejected");
}

int main(void) {
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd))) printf("cwd: %s\n", cwd);

    test_dtype();
    test_numel();
    test_manifest_load();
    test_manifest_compare_identical();
    test_manifest_compare_mismatch();
    test_unknown_profile();
    test_profile_load_traversal();

    if (failures) {
        printf("\n%d failure(s)\n", failures);
        return 1;
    }
    printf("\nall tests passed\n");
    return 0;
}
