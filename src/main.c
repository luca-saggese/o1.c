#include "hidream.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *profile = "dev";
    const char *config_dir = "config";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            profile = argv[++i];
        } else if (strcmp(argv[i], "--config-dir") == 0 && i + 1 < argc) {
            config_dir = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("usage: %s [--model dev|base] [--config-dir DIR]\n", argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 2;
        }
    }

    hd_status st = hd_validate_profile(config_dir, profile);
    if (st != HD_OK) {
        fprintf(stderr, "FAIL: %s\n", hd_last_error());
        return 1;
    }
    printf("PASS: %s profile valid\n", profile);
    return 0;
}
