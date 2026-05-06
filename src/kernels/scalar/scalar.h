/*
 * src/kernels/scalar/scalar.h — internal declarations for the scalar
 * reference kernels.
 *
 * These functions are the oracle every accelerated backend is tested
 * against. They are slow but easy to read and trivially correct, and
 * they have no dependency on any quant scheme being registered: each
 * file pins down its own block layout in a comment at the top.
 */
#ifndef EOSI_KERNELS_SCALAR_H
#define EOSI_KERNELS_SCALAR_H

#include <stddef.h>
#include <stdint.h>

#include "eosllm/eosllm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* C[m,n] = sum_k A[m,k] * B[n,k]   (i.e. C = A · B^T). Row-major. */
eos_status_t eosi_scalar_matmul_f32(const float *a, const float *b, float *c,
                                    uint32_t m, uint32_t n, uint32_t k);

/*
 * Q8 matmul against the GGUF-compatible q8_0 layout:
 *   one block = 32 int8 weights + 1 f16 scale, packed as
 *     struct { eosi_f16_t scale; int8_t w[32]; } — 34 bytes.
 *   K must be a multiple of 32.
 *
 * This is the layout the real q8_0 quant scheme uses (see
 * src/quant/q8_0.c). The kernel here is the always-available oracle
 * the accelerated backends are tested against.
 */
eos_status_t eosi_scalar_matmul_q8_0(const float *a, const void *b_q8,
                                     float *c,
                                     uint32_t m, uint32_t n, uint32_t k);

/*
 * Q4_K matmul against the GGUF-compatible q4_k super-block layout:
 *   one super-block = 256 elements split into 8 sub-blocks of 32,
 *   layout matches GGUF (see src/quant/q4_k.c). 144 bytes per 256.
 *   K must be a multiple of 256.
 */
eos_status_t eosi_scalar_matmul_q4_k(const float *a, const void *b_q4,
                                     float *c,
                                     uint32_t m, uint32_t n, uint32_t k);

/* y = x * rsqrt(mean(x^2) + eps) * w, applied row-by-row. */
eos_status_t eosi_scalar_rmsnorm(const float *x, const float *w, float eps,
                                 float *y, uint32_t rows, uint32_t cols);

/* In-place row-wise softmax. mask_to_col[i] (if non-NULL) is the first
 * column at and after which row i is masked out (set to -inf before
 * the exp). Pass mask_to_col=NULL for unmasked. */
eos_status_t eosi_scalar_softmax(float *x, uint32_t rows, uint32_t cols,
                                 const int32_t *mask_to_col);

/* Rotary positional embedding applied in place to a single (q or k)
 * tensor of shape [n_heads, head_dim]. head_dim must be even. */
eos_status_t eosi_scalar_rope(float *x, uint32_t n_heads, uint32_t head_dim,
                              uint32_t pos, float base);

/* y = x * sigmoid(x). */
eos_status_t eosi_scalar_silu(const float *x, float *y, size_t n);

/* y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
 * (Tanh approximation, the same one used by Llama / Qwen.) */
eos_status_t eosi_scalar_gelu(const float *x, float *y, size_t n);

/* Registers the scalar backend with the registry. Returns
 * EOS_E_UNSUPPORTED when the build excludes EOSLLM_HAVE_KERNEL_SCALAR. */
eos_status_t eosi_scalar_backend_register(void);

/* Block layouts (see comments above each kernel signature). */
#define EOSI_Q8_0_BLOCK_ELEMS 32
#define EOSI_Q8_0_BLOCK_BYTES 34

#define EOSI_Q4_K_BLOCK_ELEMS 256
#define EOSI_Q4_K_BLOCK_BYTES 144

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* EOSI_KERNELS_SCALAR_H */
