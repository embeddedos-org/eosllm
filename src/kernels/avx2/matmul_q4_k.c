/*
 * src/kernels/avx2/matmul_q4_k.c — AVX2 + FMA fused dequant + dot for
 * GGUF-compatible Q4_K super-blocks.
 *
 * Per-super-block layout (256 elements, 144 bytes) — see
 * src/kernels/scalar/matmul_q4_k.c for the canonical reference.
 *
 * Strategy:
 *   For each super-block on a row, compute d * sc_j and dmin * m_j
 *   per sub-block in scalar registers. Inside each 32-element sub-
 *   block, process 8 elements per AVX2 lane:
 *     1. Load 8 nibble bytes into __m128i.
 *     2. Mask to low (or shift+mask to high) 4 bits per byte.
 *     3. _mm256_cvtepu8_epi32 expands to 8 x i32.
 *     4. _mm256_cvtepi32_ps to f32.
 *     5. w = d_v * nibble_v - m_v   (via _mm256_fmsub_ps).
 *     6. acc += a_v * w             (via _mm256_fmadd_ps).
 *
 * Accumulation order differs from scalar (8-lane parallel reduce, then
 * horizontal sum at the end), so results match within ~1 ULP per output
 * element times K — well inside the 1e-4 tolerance the q4_k tests use.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "eosllm/eosllm.h"
#include "../scalar/scalar.h"
#include "../../util/f16.h"
#include "avx2_internal.h"

#if EOSLLM_HAVE_KERNEL_AVX2

#include <immintrin.h>

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

static inline __m256 dequant8_low(uint64_t qs_8, __m128i mask_low,
                                  __m256 d_v, __m256 m_v) {
    __m128i v   = _mm_and_si128(_mm_set1_epi64x((long long)qs_8), mask_low);
    __m256i ix  = _mm256_cvtepu8_epi32(v);
    __m256  fv  = _mm256_cvtepi32_ps(ix);
    return _mm256_fmsub_ps(d_v, fv, m_v);
}

static inline __m256 dequant8_high(uint64_t qs_8, __m128i mask_low,
                                   __m256 d_v, __m256 m_v) {
    /* Shift each byte right by 4 via srli_epi16 then mask 0x0F. The
     * srli_epi16-by-4 trick correctly extracts the high nibble of each
     * byte (the mask cleans up bits that flowed across a u16 boundary). */
    __m128i v0 = _mm_set1_epi64x((long long)qs_8);
    __m128i v  = _mm_and_si128(_mm_srli_epi16(v0, 4), mask_low);
    __m256i ix = _mm256_cvtepu8_epi32(v);
    __m256  fv = _mm256_cvtepi32_ps(ix);
    return _mm256_fmsub_ps(d_v, fv, m_v);
}

eos_status_t eosi_avx2_matmul_q4_k(const float *a, const void *b_q4,
                                   float *c,
                                   uint32_t m, uint32_t n, uint32_t k) {
    const uint8_t *b_bytes = (const uint8_t *)b_q4;
    const __m128i  mask_low = _mm_set1_epi8(0x0F);
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
            __m256 acc = _mm256_setzero_ps();
            uint32_t b;

            for (b = 0; b < blocks_per_row; ++b) {
                eosi_f16_t   d_h, dmin_h;
                uint8_t      scales[12];
                const uint8_t *qs;
                float        d, dmin;
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
                    __m256 d1_v, m1_v, d2_v, m2_v;
                    int l;

                    get_scale_min(sub,     scales, &sc_lo, &m_lo);
                    get_scale_min(sub + 1, scales, &sc_hi, &m_hi);

                    d1_v = _mm256_set1_ps(d * (float)sc_lo);
                    m1_v = _mm256_set1_ps(dmin * (float)m_lo);
                    d2_v = _mm256_set1_ps(d * (float)sc_hi);
                    m2_v = _mm256_set1_ps(dmin * (float)m_hi);

                    for (l = 0; l < 32; l += 8) {
                        uint64_t qs_8;
                        __m256   w, av;

                        memcpy(&qs_8, qs + l, 8);

                        /* low nibbles */
                        w  = dequant8_low(qs_8, mask_low, d1_v, m1_v);
                        av = _mm256_loadu_ps(aa + sub * 32 + l);
                        acc = _mm256_fmadd_ps(av, w, acc);

                        /* high nibbles */
                        w  = dequant8_high(qs_8, mask_low, d2_v, m2_v);
                        av = _mm256_loadu_ps(aa + (sub + 1) * 32 + l);
                        acc = _mm256_fmadd_ps(av, w, acc);
                    }
                    qs += 32;
                }
            }

            /* Horizontal reduce acc -> scalar (SSE2-only). */
            {
                __m128 lo = _mm256_castps256_ps128(acc);
                __m128 hi = _mm256_extractf128_ps(acc, 1);
                __m128 s  = _mm_add_ps(lo, hi);
                __m128 sh = _mm_movehl_ps(s, s);
                s  = _mm_add_ps(s, sh);
                sh = _mm_shuffle_ps(s, s, _MM_SHUFFLE(0,0,0,1));
                s  = _mm_add_ss(s, sh);
                c[(size_t)i * n + j] = _mm_cvtss_f32(s);
            }
        }
    }
    return EOS_OK;
}

eos_status_t eosi_avx2_matmul_q(const float *a, const void *b_q,
                                eos_dtype_t b_dtype,
                                float *c,
                                uint32_t m, uint32_t n, uint32_t k) {
    switch (b_dtype) {
        case EOS_DT_Q4_K:
            return eosi_avx2_matmul_q4_k(a, b_q, c, m, n, k);
        case EOS_DT_Q8_0:
            /* No AVX2 q8_0 yet — fall back to scalar reference. */
            return eosi_scalar_matmul_q8_0(a, b_q, c, m, n, k);
        default:
            return EOS_E_UNSUPPORTED;
    }
}

#endif /* EOSLLM_HAVE_KERNEL_AVX2 */
