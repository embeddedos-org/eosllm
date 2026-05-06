/*
 * src/kernels/scalar/matmul_q8.c
 *
 * Reference matmul against a GGUF-compatible q8_0 weight matrix.
 *
 * GGUF q8_0 layout (canonical):
 *   one block = struct { eosi_f16_t scale; int8_t w[32]; }  (34 bytes)
 *   K must be a multiple of 32.
 *
 * Note: GGUF stores blocks packed end-to-end; alignment is 2 bytes
 * (only the f16 scale is naturally aligned). Reading via memcpy is
 * safe regardless of host alignment.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "scalar.h"
#include "../../util/f16.h"

eos_status_t eosi_scalar_matmul_q8_0(const float *a, const void *b_q8,
                                     float *c,
                                     uint32_t m, uint32_t n, uint32_t k) {
    uint32_t i, j;
    uint32_t blocks_per_row;
    const uint8_t *b_bytes = (const uint8_t *)b_q8;

    if (a == NULL || b_q8 == NULL || c == NULL) return EOS_E_INVALID_ARG;
    if ((k % EOSI_Q8_0_BLOCK_ELEMS) != 0) return EOS_E_INVALID_ARG;

    blocks_per_row = k / EOSI_Q8_0_BLOCK_ELEMS;

    for (i = 0; i < m; ++i) {
        const float *ar = a + (size_t)i * k;
        for (j = 0; j < n; ++j) {
            const uint8_t *br = b_bytes
                                + (size_t)j * blocks_per_row * EOSI_Q8_0_BLOCK_BYTES;
            float acc = 0.0f;
            uint32_t b;
            for (b = 0; b < blocks_per_row; ++b) {
                eosi_f16_t scale_h;
                int8_t     w[32];
                float      scale;
                const float *aa = ar + (size_t)b * EOSI_Q8_0_BLOCK_ELEMS;
                int p;
                memcpy(&scale_h, br, 2);
                memcpy(w, br + 2, 32);
                br += EOSI_Q8_0_BLOCK_BYTES;
                scale = eosi_f16_to_f32(scale_h);
                for (p = 0; p < EOSI_Q8_0_BLOCK_ELEMS; ++p) {
                    acc += aa[p] * ((float)w[p] * scale);
                }
            }
            c[(size_t)i * n + j] = acc;
        }
    }
    return EOS_OK;
}
