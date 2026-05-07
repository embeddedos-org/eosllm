/*
 * src/kernels/neon/neon.c — ARM NEON backend (matmul_f32 only for now).
 */
#include <stddef.h>
#include <stdint.h>

#include "eosllm/eosllm.h"
#include "eosllm/backend.h"

#include "../kernels_internal.h"
#include "neon_internal.h"

#if EOSLLM_HAVE_KERNEL_NEON

#include <arm_neon.h>

static eos_status_t neon_matmul_f32(const float *a, const float *b, float *c,
                                    uint32_t m, uint32_t n, uint32_t k) {
    uint32_t i, j, p;
    if (a == NULL || b == NULL || c == NULL) return EOS_E_INVALID_ARG;
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            const float *ar = a + (size_t)i * k;
            const float *br = b + (size_t)j * k;
            float32x4_t acc = vdupq_n_f32(0.0f);
            uint32_t k4 = k & ~3u;
            float tail = 0.0f;
            for (p = 0; p < k4; p += 4) {
                float32x4_t va = vld1q_f32(ar + p);
                float32x4_t vb = vld1q_f32(br + p);
                acc = vmlaq_f32(acc, va, vb);
            }
            for (p = k4; p < k; ++p) tail += ar[p] * br[p];
#ifdef __aarch64__
            c[(size_t)i * n + j] = vaddvq_f32(acc) + tail;
#elif defined(__arm64__) || defined(_M_ARM64)
            /* Apple/MSVC arm64 also have vaddvq_f32 — the macro name
             * just isn't __aarch64__ on every toolchain. */
            c[(size_t)i * n + j] = vaddvq_f32(acc) + tail;
#else
            {
                float32x2_t s = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
                s = vpadd_f32(s, s);
                c[(size_t)i * n + j] = vget_lane_f32(s, 0) + tail;
            }
#endif
        }
    }
    return EOS_OK;
}

static eos_status_t neon_probe(void) { return EOS_OK; }   /* always present where compiled */

static const eos_backend_vt_t k_neon = {
    "arm_neon", 40, neon_probe,
    {
        neon_matmul_f32,
        eosi_neon_matmul_q,    /* Q4_K fused; Q8_0 -> scalar; others -> unsupported */
        NULL, NULL, NULL, NULL, NULL
    }
};

eos_status_t eosi_backend_neon_register(void) {
    return eos_backend_register(&k_neon);
}

#else
eos_status_t eosi_backend_neon_register(void) { return EOS_E_UNSUPPORTED; }
#endif
