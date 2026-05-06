/*
 * src/kernels/avx2/avx2_internal.h — declarations for the
 * accelerated AVX2 ops, callable both from src/kernels/avx2/avx2.c
 * (which wires them into the backend vtable) and from the unit
 * tests (which exercise them directly for parity vs. scalar).
 *
 * Only declared when EOSLLM_HAVE_KERNEL_AVX2=1 in the build config;
 * test code must guard call sites with the same flag.
 */
#ifndef EOSI_KERNELS_AVX2_INTERNAL_H
#define EOSI_KERNELS_AVX2_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "eosllm/eosllm.h"
#include "eosllm/tensor.h"

#if EOSLLM_HAVE_KERNEL_AVX2

eos_status_t eosi_avx2_matmul_q4_k(const float *a, const void *b_q4,
                                   float *c,
                                   uint32_t m, uint32_t n, uint32_t k);

/* Fused-int8 Q4_K matmul. Same contract as eosi_avx2_matmul_q4_k but
 * keeps weights packed-then-u8 through the inner loop and quantizes
 * the activation to int8 once per super-block, reusing the cached int8
 * lanes across all N output columns. ~2x faster than the f32-dequant
 * path on AVX2 hosts. Activation block-quantization adds a known
 * per-sub-block rounding error of ~5e-3 relative; documented in
 * docs/quant_schemes.md. */
eos_status_t eosi_avx2_matmul_q4_k_int8(const float *a, const void *b_q4,
                                        float *c,
                                        uint32_t m, uint32_t n, uint32_t k);

/* Dtype-dispatching matmul_q for the AVX2 backend's vtable slot.
 * Q4_K → AVX2 int8 fused path; Q8_0 → scalar (no AVX2 impl yet);
 * other → unsupported. */
eos_status_t eosi_avx2_matmul_q(const float *a, const void *b_q,
                                eos_dtype_t b_dtype,
                                float *c,
                                uint32_t m, uint32_t n, uint32_t k);

#endif /* EOSLLM_HAVE_KERNEL_AVX2 */

#endif /* EOSI_KERNELS_AVX2_INTERNAL_H */
