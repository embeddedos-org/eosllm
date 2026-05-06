/*
 * src/kernels/scalar/matmul_q4_k.c
 *
 * Reference matmul against a GGUF-compatible q4_k weight matrix.
 *
 * Layout per super-block (256 elements, 144 bytes):
 *   eosi_f16_t  d;            // outer scale for sub-block scales
 *   eosi_f16_t  dmin;         // outer scale for sub-block mins
 *   uint8_t     scales[12];   // 8 sub-block scales + 8 mins, 6 bits each
 *   uint8_t     qs[128];      // 256 4-bit quants (low nibble first sub-block,
 *                             //                   high nibble next)
 *
 * Sub-block layout: 8 sub-blocks of 32 elements. Sub-block j has
 * 6-bit scale sc_j and 6-bit min m_j, decoded by the same packing
 * scheme as ggml-quants:
 *
 *   if (j < 4):  sc = scales[j]   & 0x3F
 *                m  = scales[j+4] & 0x3F
 *   else:        sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
 *                m  = (scales[j+4] >>  4) | ((scales[j  ] >> 6) << 4)
 *
 * y = d * sc * (q & 0xF) - dmin * m         for the low-nibble half
 * y = d * sc * (q >> 4)  - dmin * m         for the high-nibble half
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "scalar.h"
#include "../../util/f16.h"

static inline void get_scale_min_q4_k(int j, const uint8_t *q,
                                      uint8_t *out_d, uint8_t *out_m) {
    if (j < 4) {
        *out_d = q[j]     & 0x3Fu;
        *out_m = q[j + 4] & 0x3Fu;
    } else {
        *out_d = (q[j + 4] & 0x0Fu) | (uint8_t)((q[j - 4] >> 6) << 4);
        *out_m = (q[j + 4] >> 4)    | (uint8_t)((q[j    ] >> 6) << 4);
    }
}

eos_status_t eosi_scalar_matmul_q4_k(const float *a, const void *b_q4,
                                     float *c,
                                     uint32_t m, uint32_t n, uint32_t k) {
    uint32_t i, j;
    uint32_t blocks_per_row;
    const uint8_t *b_bytes = (const uint8_t *)b_q4;

    if (a == NULL || b_q4 == NULL || c == NULL) return EOS_E_INVALID_ARG;
    if ((k % EOSI_Q4_K_BLOCK_ELEMS) != 0) return EOS_E_INVALID_ARG;

    blocks_per_row = k / EOSI_Q4_K_BLOCK_ELEMS;

    for (i = 0; i < m; ++i) {
        const float *ar = a + (size_t)i * k;
        for (j = 0; j < n; ++j) {
            const uint8_t *br = b_bytes
                                + (size_t)j * blocks_per_row * EOSI_Q4_K_BLOCK_BYTES;
            float acc = 0.0f;
            uint32_t b;
            for (b = 0; b < blocks_per_row; ++b) {
                eosi_f16_t   d_h, dmin_h;
                uint8_t      scales[12];
                const uint8_t *qs;
                float        d, dmin;
                const float *aa = ar + (size_t)b * EOSI_Q4_K_BLOCK_ELEMS;
                int sub;
                memcpy(&d_h,    br,     2);
                memcpy(&dmin_h, br + 2, 2);
                memcpy(scales,  br + 4, 12);
                qs = br + 16;
                br += EOSI_Q4_K_BLOCK_BYTES;
                d    = eosi_f16_to_f32(d_h);
                dmin = eosi_f16_to_f32(dmin_h);

                /* Process 8 sub-blocks of 32 elements; iterate them in
                 * pairs (low-nibble then high-nibble of the same 32 qs). */
                for (sub = 0; sub < 8; sub += 2) {
                    uint8_t sc_lo, m_lo, sc_hi, m_hi;
                    float   d1, m1, d2, m2;
                    int l;
                    get_scale_min_q4_k(sub,     scales, &sc_lo, &m_lo);
                    get_scale_min_q4_k(sub + 1, scales, &sc_hi, &m_hi);
                    d1 = d * (float)sc_lo;  m1 = dmin * (float)m_lo;
                    d2 = d * (float)sc_hi;  m2 = dmin * (float)m_hi;
                    /* low nibbles */
                    for (l = 0; l < 32; ++l) {
                        const float w = d1 * (float)(qs[l] & 0x0Fu) - m1;
                        acc += aa[(sub) * 32 + l] * w;
                    }
                    /* high nibbles */
                    for (l = 0; l < 32; ++l) {
                        const float w = d2 * (float)(qs[l] >> 4) - m2;
                        acc += aa[(sub + 1) * 32 + l] * w;
                    }
                    qs += 32;
                }
            }
            c[(size_t)i * n + j] = acc;
        }
    }
    return EOS_OK;
}
