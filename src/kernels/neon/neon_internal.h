/*
 * src/kernels/neon/neon_internal.h — internal entry points for the
 * NEON-accelerated kernels. Mirrors src/kernels/avx2/avx2_internal.h.
 */
#ifndef EOSI_KERNELS_NEON_INTERNAL_H
#define EOSI_KERNELS_NEON_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "eosllm/eosllm.h"
#include "eosllm/tensor.h"

#if EOSLLM_HAVE_KERNEL_NEON

eos_status_t eosi_neon_matmul_q4_k(const float *a, const void *b_q4,
                                   float *c,
                                   uint32_t m, uint32_t n, uint32_t k);

eos_status_t eosi_neon_matmul_q(const float *a, const void *b_q,
                                eos_dtype_t b_dtype,
                                float *c,
                                uint32_t m, uint32_t n, uint32_t k);

#endif /* EOSLLM_HAVE_KERNEL_NEON */

#endif /* EOSI_KERNELS_NEON_INTERNAL_H */
