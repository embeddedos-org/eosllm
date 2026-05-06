/*
 * src/kernels/avx2/avx2.c — x86 AVX2 backend.
 *
 * Phase 4 implements:
 *   matmul_f32   — fused multiply-add over 8-lane f32 vectors.
 * All other ops fall back to the scalar backend (the dispatcher
 * picks the highest-priority non-NULL entry per op).
 *
 * Compile only when EOSLLM_HAVE_KERNEL_AVX2=1 and the host compiler
 * supports `-mavx2 -mfma`. Probe checks at runtime via cpuid.
 */
#include <stddef.h>
#include <stdint.h>

#include "eosllm/eosllm.h"
#include "eosllm/backend.h"

#include "../kernels_internal.h"
#include "avx2_internal.h"

#if EOSLLM_HAVE_KERNEL_AVX2

#include <immintrin.h>

static eos_status_t avx2_matmul_f32(const float *a, const float *b, float *c,
                                    uint32_t m, uint32_t n, uint32_t k) {
    uint32_t i, j, p;
    if (a == NULL || b == NULL || c == NULL) return EOS_E_INVALID_ARG;
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            const float *ar = a + (size_t)i * k;
            const float *br = b + (size_t)j * k;
            __m256 acc = _mm256_setzero_ps();
            uint32_t k8 = k & ~7u;
            float tail = 0.0f;
            for (p = 0; p < k8; p += 8) {
                __m256 va = _mm256_loadu_ps(ar + p);
                __m256 vb = _mm256_loadu_ps(br + p);
                acc = _mm256_fmadd_ps(va, vb, acc);
            }
            for (p = k8; p < k; ++p) tail += ar[p] * br[p];
            {
                /* horizontal-sum acc into one float (SSE2-only path
                 * to avoid relying on -msse3 for _mm_hadd_ps). */
                __m128 lo = _mm256_castps256_ps128(acc);
                __m128 hi = _mm256_extractf128_ps(acc, 1);
                __m128 s  = _mm_add_ps(lo, hi);
                __m128 sh = _mm_movehl_ps(s, s);                  /* lanes 2,3,2,3 */
                s  = _mm_add_ps(s, sh);                            /* sums of {0+2, 1+3} in low 2 */
                sh = _mm_shuffle_ps(s, s, _MM_SHUFFLE(0,0,0,1));   /* lane 1 in lane 0 */
                s  = _mm_add_ss(s, sh);
                c[(size_t)i * n + j] = _mm_cvtss_f32(s) + tail;
            }
        }
    }
    return EOS_OK;
}

#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
static int has_avx2_fma(void) {
    unsigned a, b, c, d;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return 0;
    if (!(b & (1u << 5))) return 0;          /* AVX2 */
    if (!__get_cpuid(1, &a, &b, &c, &d)) return 0;
    if (!(c & (1u << 12))) return 0;         /* FMA */
    return 1;
}
#else
static int has_avx2_fma(void) { return 1; }
#endif

static eos_status_t avx2_probe(void) {
    return has_avx2_fma() ? EOS_OK : EOS_E_UNSUPPORTED;
}

static const eos_backend_vt_t k_avx2 = {
    "x86_avx2", 50, avx2_probe,
    {
        avx2_matmul_f32,
        eosi_avx2_matmul_q,   /* Q4_K fast-path; Q8_0 -> scalar; others -> unsupported */
        NULL,           /* rmsnorm    — falls back to scalar */
        NULL, NULL, NULL, NULL
    }
};

eos_status_t eosi_backend_avx2_register(void) {
    return eos_backend_register(&k_avx2);
}

#else
eos_status_t eosi_backend_avx2_register(void) { return EOS_E_UNSUPPORTED; }
#endif
