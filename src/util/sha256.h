/*
 * src/util/sha256.h — minimal zero-dependency SHA-256.
 *
 * Used by the .eosm v1 reader/writer for the mandatory file-trailer
 * hash (see docs/file_format.md §10). Not part of the public ABI.
 */
#ifndef EOSI_UTIL_SHA256_H
#define EOSI_UTIL_SHA256_H

#include <stddef.h>
#include <stdint.h>

#define EOSI_SHA256_DIGEST_BYTES 32

typedef struct eosi_sha256_ctx {
    uint32_t state[8];
    uint64_t bit_count;
    uint32_t buf_len;
    uint8_t  buf[64];
} eosi_sha256_ctx_t;

void eosi_sha256_init  (eosi_sha256_ctx_t *ctx);
void eosi_sha256_update(eosi_sha256_ctx_t *ctx, const void *data, size_t len);
void eosi_sha256_final (eosi_sha256_ctx_t *ctx, uint8_t out[EOSI_SHA256_DIGEST_BYTES]);

/* One-shot convenience. */
void eosi_sha256(const void *data, size_t len,
                 uint8_t out[EOSI_SHA256_DIGEST_BYTES]);

#endif /* EOSI_UTIL_SHA256_H */
