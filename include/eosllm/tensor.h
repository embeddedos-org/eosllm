/*
 * tensor.h — opaque tensor handle and dtype enum.
 */
#ifndef EOSLLM_TENSOR_H
#define EOSLLM_TENSOR_H
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "eosllm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum tensor rank exposed across the ABI. Bumping this is a
 * breaking change. Most ops use rank ≤ 4. */
#define EOS_MAX_RANK 6

typedef enum eos_dtype {
    EOS_DT_F32     = 0,
    EOS_DT_F16     = 1,
    EOS_DT_BF16    = 2,
    EOS_DT_I32     = 3,
    EOS_DT_I8      = 4,
    EOS_DT_U8      = 5,

    /* Quantized dtypes are a single enum value; the per-block scales /
     * zero-points / packed bits are described by the quant scheme
     * registered against this dtype. */
    EOS_DT_Q8_0    = 32,
    EOS_DT_Q4_K    = 33,
    EOS_DT_Q2_K    = 34,
    EOS_DT_Q1_58   = 35,   /* BitNet-style ternary */
    EOS_DT_MIXED   = 36,   /* per-channel / per-layer mixing */
    EOS_DT_CALIBRTD = 37   /* AWQ/GPTQ-style calibrated */
} eos_dtype_t;

/* Returns the size in bytes of one element of dt for non-quantized
 * dtypes, or 0 for quantized dtypes (their footprint is per-block and
 * computed by the quant scheme). */
size_t eos_dtype_elem_size(eos_dtype_t dt);

/* Reads the shape of a tensor. Writes up to EOS_MAX_RANK dims into
 * out_dims and the actual rank into *out_rank. */
eos_status_t eos_tensor_shape(const eos_tensor_t *t,
                              uint32_t            out_dims[EOS_MAX_RANK],
                              uint32_t           *out_rank);

eos_dtype_t  eos_tensor_dtype(const eos_tensor_t *t);
const char  *eos_tensor_name (const eos_tensor_t *t);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* EOSLLM_TENSOR_H */
