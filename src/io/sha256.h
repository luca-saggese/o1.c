#ifndef HD_SHA256_H
#define HD_SHA256_H

/* Self-contained SHA-256 used for weight inventory fingerprints.
 * No external dependency so the production runtime stays pure C. */

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t buf[64];
    size_t buflen;
} hd_sha256_ctx;

void hd_sha256_init(hd_sha256_ctx *c);
void hd_sha256_update(hd_sha256_ctx *c, const void *data, size_t len);
void hd_sha256_final(hd_sha256_ctx *c, uint8_t out[32]);

/* Writes a lowercase hex digest plus NUL terminator into out[65]. */
void hd_sha256_hex(const uint8_t digest[32], char out[65]);

#endif /* HD_SHA256_H */
