/*
 * src/kernels/scalar/register.c
 *
 * Wraps the scalar kernels into a backend vtable and registers it
 * via eos_backend_register at engine init.
 *
 * Priority is 0 — the lowest. The scalar backend is always present
 * and acts as the fallback for any op an accelerated backend has not
 * implemented for this build.
 */
#include <stddef.h>
#include <stdint.h>

#include "eosllm/backend.h"
#include "eosllm/eosllm.h"
#include "scalar.h"

#if EOSLLM_HAVE_KERNEL_SCALAR

static eos_status_t scalar_matmul_q(const float *a, const void *b_q,
                                    eos_dtype_t b_dtype,
                                    float *c,
                                    uint32_t m, uint32_t n, uint32_t k) {
    switch (b_dtype) {
        case EOS_DT_Q8_0:
            return eosi_scalar_matmul_q8_0(a, b_q, c, m, n, k);
        case EOS_DT_Q4_K:
            return eosi_scalar_matmul_q4_k(a, b_q, c, m, n, k);
        default:
            /* Phase 2 wires up Q2_K / Q1_58 etc. */
            return EOS_E_UNSUPPORTED;
    }
}

static const eos_backend_vt_t k_scalar_backend = {
    "scalar",                       /* name     */
    0,                              /* priority */
    NULL,                           /* probe    — always available */
    {
        eosi_scalar_matmul_f32,
        scalar_matmul_q,
        eosi_scalar_rmsnorm,
        eosi_scalar_softmax,
        eosi_scalar_rope,
        eosi_scalar_silu,
        eosi_scalar_gelu
    }
};

eos_status_t eosi_scalar_backend_register(void) {
    return eos_backend_register(&k_scalar_backend);
}

#else

eos_status_t eosi_scalar_backend_register(void) {
    return EOS_E_UNSUPPORTED;
}

#endif
