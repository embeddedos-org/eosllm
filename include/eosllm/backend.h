/*
 * backend.h — kernel backend registration.
 *
 * A backend exposes a vtable of accelerated implementations of the
 * small set of hot ops. Any op left as NULL falls back to the scalar
 * oracle in src/kernels/scalar/. The session's dispatch table is built
 * once at eos_session_open by walking registered backends in priority
 * order and picking the highest-priority non-NULL entry per op.
 */
#ifndef EOSLLM_BACKEND_H
#define EOSLLM_BACKEND_H
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "eosllm.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Op vtable. All shapes are "logical": rows × inner × cols, with strides
 * in elements (not bytes). The runtime is responsible for blocking and
 * for choosing the right kernel; the kernel is responsible for the math.
 *
 * Adding a new op is an additive ABI change: append it at the end of
 * this struct and bump EOSLLM_ABI_VERSION.
 */
typedef struct eos_backend_ops {
    /* C = A · B^T, all f32, row-major. */
    eos_status_t (*matmul_f32)(const float *a, const float *b, float *c,
                               uint32_t m, uint32_t n, uint32_t k);

    /*
     * C = A · dequantize(B)^T. A is row-major f32 (m×k). B is a
     * quantized weight matrix; the backend looks up its layout via
     * eos_quant_find(). The scalar reference path simply dequantizes B
     * to a scratch buffer and calls matmul_f32; ISA backends fuse.
     */
    eos_status_t (*matmul_q)(const float *a, const void *b_quant,
                             eos_dtype_t b_dtype,
                             float *c,
                             uint32_t m, uint32_t n, uint32_t k);

    /* y = x * rsqrt(mean(x^2) + eps) * w, row-wise. */
    eos_status_t (*rmsnorm)(const float *x, const float *w, float eps,
                            float *y, uint32_t rows, uint32_t cols);

    /* In-place softmax along the last axis, masked above mask_to_col. */
    eos_status_t (*softmax)(float *x, uint32_t rows, uint32_t cols,
                            const int32_t *mask_to_col /* may be NULL */);

    /* Rotary positional embedding, applied to query/key in-place. */
    eos_status_t (*rope)(float *x, uint32_t n_heads, uint32_t head_dim,
                         uint32_t pos, float base);

    /* y = x * sigmoid(x). */
    eos_status_t (*silu)(const float *x, float *y, size_t n);

    /* y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))). */
    eos_status_t (*gelu)(const float *x, float *y, size_t n);
} eos_backend_ops_t;

typedef struct eos_backend_vt {
    const char *name;     /* e.g. "scalar", "x86_avx2", "arm_neon" */
    /*
     * Higher-priority backends override lower ones for any op they
     * implement. The scalar backend registers at priority 0 and acts
     * as the always-present fallback.
     */
    int32_t     priority;
    /*
     * Probes the runtime CPU and returns EOS_OK if this backend is
     * usable on this machine. May be NULL (always available).
     */
    eos_status_t (*probe)(void);

    eos_backend_ops_t ops;
} eos_backend_vt_t;

eos_status_t eos_backend_register(const eos_backend_vt_t *vt);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* EOSLLM_BACKEND_H */
