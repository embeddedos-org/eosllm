/*
 * quant.h — quant scheme registration vtable.
 *
 * A quant scheme owns:
 *   - the on-disk packed layout of a tensor block
 *   - dequantization to f32 (used by the scalar oracle and any backend
 *     that hasn't implemented a fused matmul for that dtype yet)
 *   - optional fast paths registered via the backend vtable
 */
#ifndef EOSLLM_QUANT_H
#define EOSLLM_QUANT_H
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "eosllm.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Block descriptor for a quantized dtype. The library treats a
 * quantized tensor as a flat array of blocks.
 */
typedef struct eos_quant_layout {
    uint32_t elements_per_block;  /* logical elements packed per block */
    uint32_t bytes_per_block;     /* size of one block on disk and in RAM */
    uint8_t  alignment;           /* required block alignment, in bytes */
    uint8_t  _reserved[7];
} eos_quant_layout_t;

typedef struct eos_quant_vt {
    const char  *name;            /* e.g. "q8_0", "q4_k", "q1_58" */
    eos_dtype_t  dtype;

    /* Static layout; no allocation. */
    void (*layout)(eos_quant_layout_t *out);

    /*
     * Dequantize n_elements (must be a multiple of elements_per_block)
     * from src into f32 dst. This is the oracle path; every backend's
     * fused matmul is tested bit-exactly against
     *   matmul_f32(dequantize(W), x).
     */
    eos_status_t (*dequantize)(const void *src, float *dst, size_t n_elements);

    /*
     * Quantize from f32 to packed bytes. Used by tools/eosllm-convert
     * and tools/eosllm-quant-lab. The runtime never quantizes.
     */
    eos_status_t (*quantize)(const float *src, void *dst, size_t n_elements);
} eos_quant_vt_t;

eos_status_t eos_quant_register(const eos_quant_vt_t *vt);

/* Looks up a registered scheme by name; returns NULL if not present. */
const eos_quant_vt_t *eos_quant_find(const char *name);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* EOSLLM_QUANT_H */
