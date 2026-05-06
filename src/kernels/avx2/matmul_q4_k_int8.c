/*
 * src/kernels/avx2/matmul_q4_k_int8.c — AVX2 fused-int8 matmul against
 * GGUF-compatible Q4_K super-blocks.
 *
 * The point of this code path vs. the f32-dequant path in
 * matmul_q4_k.c is to halve the memory traffic on the weight buffer
 * AND amortize the activation block-quantize cost across all output
 * columns:
 *
 *   - The 4-bit weights stay packed-then-u8 through the dot product
 *     (no f32 expansion inside the inner loop).
 *   - The activation is block-quantized to int8 once per super-block
 *     (8 sub-blocks of 32 elements), cached on the stack, and re-used
 *     across all N output columns. Quantization cost is N× cheaper
 *     than the naive "quantize per (col, super-block)" structure.
 *   - The per-column dot product runs through _mm256_maddubs_epi16 +
 *     _mm256_madd_epi16 over u4 × i8 pairs and is rescaled to f32 once
 *     per sub-block.
 *
 * Per-sub-block identity (32 elements):
 *   sum(w_f32 * a_f32)
 *     = sum( (d*sc*q - dmin*m) * a )
 *     = d*sc * sum(q*a) - dmin*m * sum(a)
 *     ~= d*sc * a_scale * dot_int(q, a_q)
 *        - dmin*m * a_scale * sum_int(a_q)
 *
 * where:
 *   q     in [0, 15]      -- the Q4_K weight nibble
 *   a_q   in [-127, +127] -- the block-quantized activation
 *   a_scale = max_abs(a_f32) / 127       (a_q = round(a / a_scale))
 *
 * Activation block-quantization adds a per-sub-block rounding error of
 * ~1 i8 ULP / 127 ~= 0.4%. The companion parity test in
 * tests/unit/test_runner.c uses a 5e-3 relative tolerance to account
 * for this; the bound is documented in docs/quant_schemes.md.
 *
 * Compile only when EOSLLM_HAVE_KERNEL_AVX2=1.
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
#define SUB_PER_BLOCK 8
#define ELEMS_PER_SUB 32

/* Same 6-bit scale/min unpack as the scalar reference + the f32 path. */
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

/* Horizontal sum of 8 i32 lanes -> scalar i32. SSE2-only after the
 * cross-lane extract (matches the SSE2-only reduce convention used in
 * the f32 AVX2 paths). */
static inline int32_t hadd_epi32_8(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    __m128i sh = _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2));
    s  = _mm_add_epi32(s, sh);
    sh = _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1));
    s  = _mm_add_epi32(s, sh);
    return _mm_cvtsi128_si32(s);
}

/* Horizontal max of 8 f32 lanes -> scalar f32. SSE2-only after extract. */
static inline float hmax_ps_8(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 m  = _mm_max_ps(lo, hi);
    __m128 sh = _mm_movehl_ps(m, m);
    m  = _mm_max_ps(m, sh);
    sh = _mm_shuffle_ps(m, m, _MM_SHUFFLE(0, 0, 0, 1));
    m  = _mm_max_ss(m, sh);
    return _mm_cvtss_f32(m);
}

/* Block-quantize 32 f32 activations -> int8 lanes packed into a single
 * __m256i, plus the f32 scale and the i32 sum of the int8 lanes (used
 * to subtract off the per-sub-block min term). */
static inline void quantize_sub32(const float *aa,
                                  __m256i *out_aq,
                                  float   *out_scale,
                                  int32_t *out_sum_aq) {
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 va0 = _mm256_loadu_ps(aa +  0);
    __m256 va1 = _mm256_loadu_ps(aa +  8);
    __m256 va2 = _mm256_loadu_ps(aa + 16);
    __m256 va3 = _mm256_loadu_ps(aa + 24);

    __m256 ma0 = _mm256_and_ps(va0, abs_mask);
    __m256 ma1 = _mm256_and_ps(va1, abs_mask);
    __m256 ma2 = _mm256_and_ps(va2, abs_mask);
    __m256 ma3 = _mm256_and_ps(va3, abs_mask);
    __m256 mx  = _mm256_max_ps(_mm256_max_ps(ma0, ma1),
                               _mm256_max_ps(ma2, ma3));
    float max_abs = hmax_ps_8(mx);

    float scale, inv_scale;
    if (max_abs > 0.0f) {
        scale     = max_abs / 127.0f;
        inv_scale = 127.0f / max_abs;
    } else {
        scale     = 0.0f;
        inv_scale = 0.0f;
    }

    __m256 inv = _mm256_set1_ps(inv_scale);
    /* _mm256_cvtps_epi32 uses MXCSR (default round-to-nearest-even). */
    __m256i q0 = _mm256_cvtps_epi32(_mm256_mul_ps(va0, inv));
    __m256i q1 = _mm256_cvtps_epi32(_mm256_mul_ps(va1, inv));
    __m256i q2 = _mm256_cvtps_epi32(_mm256_mul_ps(va2, inv));
    __m256i q3 = _mm256_cvtps_epi32(_mm256_mul_ps(va3, inv));

    /* sum_aq: scalar i32 horizontal sum of the 32 int8 lanes. The i32
     * lanes are guaranteed to be in [-127, +127] after the inv_scale
     * rounding (we scale by 127/max_abs), so summing pre-saturation
     * matches summing post-pack. */
    __m256i s_int = _mm256_add_epi32(_mm256_add_epi32(q0, q1),
                                     _mm256_add_epi32(q2, q3));
    int32_t sum_aq = hadd_epi32_8(s_int);

    /* Pack 4 x i32 vectors -> 32 i8 lanes. _mm256_packs_epi{32,16}
     * saturate and produce lane-interleaved output across the two
     * 128-bit halves; _mm256_permutevar8x32_epi32 with index
     * [0,4,1,5,2,6,3,7] undoes the cross-lane interleave so the final
     * 32 i8 lanes are in source order [aa[0]..aa[31]]. */
    __m256i p16_01 = _mm256_packs_epi32(q0, q1);
    __m256i p16_23 = _mm256_packs_epi32(q2, q3);
    __m256i p8     = _mm256_packs_epi16(p16_01, p16_23);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    __m256i aq = _mm256_permutevar8x32_epi32(p8, perm);

    *out_aq      = aq;
    *out_scale   = scale;
    *out_sum_aq  = sum_aq;
}

/* Dot product of 32 u8 weight lanes x 32 i8 activation lanes -> scalar
 * i32. Uses the maddubs+madd ML primitive pair. */
static inline int32_t dot_u8_i8(__m256i wq, __m256i aq, __m256i ones16) {
    __m256i p16 = _mm256_maddubs_epi16(wq, aq);
    __m256i p32 = _mm256_madd_epi16(p16, ones16);
    return hadd_epi32_8(p32);
}

eos_status_t eosi_avx2_matmul_q4_k_int8(const float *a, const void *b_q4,
                                        float *c,
                                        uint32_t m, uint32_t n, uint32_t k) {
    const uint8_t *b_bytes = (const uint8_t *)b_q4;
    const __m256i  mask4   = _mm256_set1_epi8(0x0F);
    const __m256i  ones16  = _mm256_set1_epi16(1);
    uint32_t blocks_per_row;
    uint32_t i, j, b;

    if (a == NULL || b_q4 == NULL || c == NULL) return EOS_E_INVALID_ARG;
    if ((k % BLOCK_ELEMS) != 0) return EOS_E_INVALID_ARG;

    /* Small-n fallback. The activation block-quantize + cache strategy
     * needs at least a few output columns to amortize. Empirically
     * (microbench on Threadripper-class x86), the int8 path becomes
     * strictly faster than the f32-dequant path around n>=4 and the
     * speedup ramps up to ~1.9x for n>=64. For n in {1,2,3} the
     * f32-dequant path is faster; route there to avoid a regression. */
    if (n < 4) {
        return eosi_avx2_matmul_q4_k(a, b_q4, c, m, n, k);
    }

    blocks_per_row = k / BLOCK_ELEMS;

    for (i = 0; i < m; ++i) {
        const float *ar = a + (size_t)i * k;
        float       *cr = c + (size_t)i * n;

        /* Zero the row of c so we can accumulate across super-blocks
         * inside the b loop. */
        for (j = 0; j < n; ++j) cr[j] = 0.0f;

        for (b = 0; b < blocks_per_row; ++b) {
            const float *aa = ar + (size_t)b * BLOCK_ELEMS;

            /* Cache the quantized activation for this super-block. The
             * 256 int8 lanes fit in 8 __m256i registers / 256 bytes of
             * stack; the per-sub-block scales and sums are tiny. The
             * compiler keeps the small arrays L1-resident. */
            __m256i aq[SUB_PER_BLOCK];
            float   a_scales[SUB_PER_BLOCK];
            int32_t sum_aqs [SUB_PER_BLOCK];
            int     sub;

            for (sub = 0; sub < SUB_PER_BLOCK; ++sub) {
                quantize_sub32(aa + (size_t)sub * ELEMS_PER_SUB,
                               &aq[sub], &a_scales[sub], &sum_aqs[sub]);
            }

            /* Now stream the weight buffer for this super-block across
             * all N output columns, re-using the cached activation. */
            for (j = 0; j < n; ++j) {
                const uint8_t *br = b_bytes
                                  + (size_t)j * blocks_per_row * BLOCK_BYTES
                                  + (size_t)b * BLOCK_BYTES;
                eosi_f16_t   d_h, dmin_h;
                uint8_t      scales[12];
                const uint8_t *qs;
                float        d, dmin;
                float        partial = 0.0f;

                memcpy(&d_h,    br,     2);
                memcpy(&dmin_h, br + 2, 2);
                memcpy(scales,  br + 4, 12);
                qs = br + 16;
                d    = eosi_f16_to_f32(d_h);
                dmin = eosi_f16_to_f32(dmin_h);

                /* 8 sub-blocks, processed in pairs that share 32 qs
                 * bytes (sub = low nibbles, sub+1 = high nibbles). */
                for (sub = 0; sub < SUB_PER_BLOCK; sub += 2) {
                    uint8_t sc_lo, m_lo, sc_hi, m_hi;
                    __m256i qs_v, wq_lo, wq_hi;
                    int32_t dot_lo, dot_hi;

                    get_scale_min(sub,     scales, &sc_lo, &m_lo);
                    get_scale_min(sub + 1, scales, &sc_hi, &m_hi);

                    qs_v  = _mm256_loadu_si256((const __m256i *)qs);
                    wq_lo = _mm256_and_si256(qs_v, mask4);
                    wq_hi = _mm256_and_si256(_mm256_srli_epi16(qs_v, 4),
                                             mask4);
                    qs += 32;

                    dot_lo = dot_u8_i8(wq_lo, aq[sub],     ones16);
                    dot_hi = dot_u8_i8(wq_hi, aq[sub + 1], ones16);

                    /* Rescale and accumulate. Two terms come from
                     *   sum(w*a) = d*sc * sum(q*a) - dmin*m * sum(a)
                     * with a ~= a_scale * a_q. */
                    partial += d * (float)sc_lo * a_scales[sub]
                               * (float)dot_lo
                             - dmin * (float)m_lo * a_scales[sub]
                               * (float)sum_aqs[sub];
                    partial += d * (float)sc_hi * a_scales[sub + 1]
                               * (float)dot_hi
                             - dmin * (float)m_hi * a_scales[sub + 1]
                               * (float)sum_aqs[sub + 1];
                }

                cr[j] += partial;
            }
        }
    }
    return EOS_OK;
}

#endif /* EOSLLM_HAVE_KERNEL_AVX2 */
