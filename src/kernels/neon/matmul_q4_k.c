/*
 * src/kernels/neon/matmul_q4_k.c — ARM NEON fused dequant + dot for
 * GGUF-compatible Q4_K super-blocks. Structural mirror of
 * src/kernels/avx2/matmul_q4_k.c — see that file for the per-block
 * layout and overall strategy.
 *
 * NEON is 128-bit (4 x f32 per vector), so we process 8 nibbles per
 * inner step using two 4-wide accumulator updates. The dequant
 * formula y = d * nibble - m is computed as
 *     vsubq_f32(vmulq_f32(d_v, x_v), m_v)
 * which is intentionally NOT fused — `vfmsq_f32(acc, a, b)` computes
 * `acc - a*b`, the wrong sign for our case. The FMA on the outer
 * accumulator (`acc += a*w`) does use vmlaq_f32.
 *
 * Build only when EOSLLM_HAVE_KERNEL_NEON=1. Tested via the parity
 * harness in tests/unit/test_runner.c (gated on the same flag).
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "eosllm/eosllm.h"
#include "../scalar/scalar.h"
#include "../../util/f16.h"
#include "neon_internal.h"

#if EOSLLM_HAVE_KERNEL_NEON

#include <arm_neon.h>

#define BLOCK_ELEMS 256
#define BLOCK_BYTES 144

static inline void get_scale_min(int j, const uint8_t *q,
                                 uint8_t *out_d, uint8_t *out_m) {
    if (j < 4) {
        *out_d = q[j]     & 0x3Fu;
        *out_m = q[j + 4] & 0x3Fu;
    } else {
        *out_d = (q[j + 4] & 0x0Fu) | (uint8_t)((q[j - 4] >> 6) << 4);
        *out_m = (q[j + 4] >> 4)    | (uint8_t)((q[j    ] >> 6) << 4);
    }
}

/* Dequant 8 nibbles (already extracted into 8 x u8) into two
 * 4-lane f32 vectors via uint16x8 → split → uint32x4 → f32. */
static inline void dequant8(uint8x8_t nibbles, float32x4_t d_v, float32x4_t m_v,
                            float32x4_t *out_lo, float32x4_t *out_hi) {
    uint16x8_t u16 = vmovl_u8(nibbles);
    uint32x4_t lo  = vmovl_u16(vget_low_u16(u16));
    uint32x4_t hi  = vmovl_u16(vget_high_u16(u16));
    float32x4_t fl = vcvtq_f32_u32(lo);
    float32x4_t fh = vcvtq_f32_u32(hi);
    *out_lo = vsubq_f32(vmulq_f32(d_v, fl), m_v);
    *out_hi = vsubq_f32(vmulq_f32(d_v, fh), m_v);
}

eos_status_t eosi_neon_matmul_q4_k(const float *a, const void *b_q4,
                                   float *c,
                                   uint32_t m, uint32_t n, uint32_t k) {
    const uint8_t *b_bytes = (const uint8_t *)b_q4;
    const uint8x8_t mask_low = vdup_n_u8(0x0Fu);
    uint32_t blocks_per_row;
    uint32_t i, j;

    if (a == NULL || b_q4 == NULL || c == NULL) return EOS_E_INVALID_ARG;
    if ((k % BLOCK_ELEMS) != 0) return EOS_E_INVALID_ARG;
    blocks_per_row = k / BLOCK_ELEMS;

    for (i = 0; i < m; ++i) {
        const float *ar = a + (size_t)i * k;
        for (j = 0; j < n; ++j) {
            const uint8_t *br = b_bytes
                              + (size_t)j * blocks_per_row * BLOCK_BYTES;
            float32x4_t acc = vdupq_n_f32(0.0f);
            uint32_t b;

            for (b = 0; b < blocks_per_row; ++b) {
                eosi_f16_t d_h, dmin_h;
                uint8_t scales[12];
                const uint8_t *qs;
                float d, dmin;
                const float *aa = ar + (size_t)b * BLOCK_ELEMS;
                int sub;

                memcpy(&d_h,    br,     2);
                memcpy(&dmin_h, br + 2, 2);
                memcpy(scales,  br + 4, 12);
                qs = br + 16;
                br += BLOCK_BYTES;
                d    = eosi_f16_to_f32(d_h);
                dmin = eosi_f16_to_f32(dmin_h);

                for (sub = 0; sub < 8; sub += 2) {
                    uint8_t sc_lo, m_lo, sc_hi, m_hi;
                    float32x4_t d1_v, m1_v, d2_v, m2_v;
                    int l;

                    get_scale_min(sub,     scales, &sc_lo, &m_lo);
                    get_scale_min(sub + 1, scales, &sc_hi, &m_hi);

                    d1_v = vdupq_n_f32(d * (float)sc_lo);
                    m1_v = vdupq_n_f32(dmin * (float)m_lo);
                    d2_v = vdupq_n_f32(d * (float)sc_hi);
                    m2_v = vdupq_n_f32(dmin * (float)m_hi);

                    for (l = 0; l < 32; l += 8) {
                        uint8x8_t qs_v = vld1_u8(qs + l);
                        uint8x8_t lo_n = vand_u8 (qs_v, mask_low);
                        uint8x8_t hi_n = vshr_n_u8(qs_v, 4);
                        float32x4_t w_lo, w_hi, av_lo, av_hi;

                        /* low nibbles -> sub_block (sub) */
                        dequant8(lo_n, d1_v, m1_v, &w_lo, &w_hi);
                        av_lo = vld1q_f32(aa + sub * 32 + l);
                        av_hi = vld1q_f32(aa + sub * 32 + l + 4);
                        acc   = vmlaq_f32(acc, av_lo, w_lo);
                        acc   = vmlaq_f32(acc, av_hi, w_hi);

                        /* high nibbles -> sub_block (sub + 1) */
                        dequant8(hi_n, d2_v, m2_v, &w_lo, &w_hi);
                        av_lo = vld1q_f32(aa + (sub + 1) * 32 + l);
                        av_hi = vld1q_f32(aa + (sub + 1) * 32 + l + 4);
                        acc   = vmlaq_f32(acc, av_lo, w_lo);
                        acc   = vmlaq_f32(acc, av_hi, w_hi);
                    }
                    qs += 32;
                }
            }

            /* Horizontal reduce. aarch64 has vaddvq_f32; on 32-bit
             * NEON we do a manual pairwise reduction. */
#ifdef __aarch64__
            c[(size_t)i * n + j] = vaddvq_f32(acc);
#else
            {
                float32x2_t s = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
                s = vpadd_f32(s, s);
                c[(size_t)i * n + j] = vget_lane_f32(s, 0);
            }
#endif
        }
    }
    return EOS_OK;
}

eos_status_t eosi_neon_matmul_q(const float *a, const void *b_q,
                                eos_dtype_t b_dtype,
                                float *c,
                                uint32_t m, uint32_t n, uint32_t k) {
    switch (b_dtype) {
        case EOS_DT_Q4_K:
            return eosi_neon_matmul_q4_k(a, b_q, c, m, n, k);
        case EOS_DT_Q8_0:
            /* No NEON q8_0 yet — scalar fallback. */
            return eosi_scalar_matmul_q8_0(a, b_q, c, m, n, k);
        default:
            return EOS_E_UNSUPPORTED;
    }
}

#endif /* EOSLLM_HAVE_KERNEL_NEON */
