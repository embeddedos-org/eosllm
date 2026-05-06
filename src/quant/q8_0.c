/*
 * src/quant/q8_0.c — GGUF-compatible q8_0 quant scheme.
 *
 * Block layout (34 bytes per 32 elements):
 *   eosi_f16_t scale;
 *   int8_t     qs[32];
 *
 * Compiled in only when EOSLLM_HAVE_QUANT_Q8_0=1.
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "eosllm/eosllm.h"
#include "eosllm/quant.h"

#include "../util/f16.h"
#include "quant_internal.h"

#if EOSLLM_HAVE_QUANT_Q8_0

#define BLOCK_ELEMS 32
#define BLOCK_BYTES 34

static void q8_0_layout(eos_quant_layout_t *out) {
    out->elements_per_block = BLOCK_ELEMS;
    out->bytes_per_block    = BLOCK_BYTES;
    out->alignment          = 2;
    memset(out->_reserved, 0, sizeof(out->_reserved));
}

static eos_status_t q8_0_dequantize(const void *src, float *dst,
                                    size_t n_elements) {
    size_t b, blocks;
    const uint8_t *p = (const uint8_t *)src;

    if (src == NULL || dst == NULL) return EOS_E_INVALID_ARG;
    if ((n_elements % BLOCK_ELEMS) != 0) return EOS_E_INVALID_ARG;
    blocks = n_elements / BLOCK_ELEMS;

    for (b = 0; b < blocks; ++b) {
        eosi_f16_t sh;
        int8_t     w[BLOCK_ELEMS];
        float      scale;
        int        i;
        memcpy(&sh, p, 2);
        memcpy(w,   p + 2, BLOCK_ELEMS);
        p += BLOCK_BYTES;
        scale = eosi_f16_to_f32(sh);
        for (i = 0; i < BLOCK_ELEMS; ++i) {
            dst[b * BLOCK_ELEMS + i] = (float)w[i] * scale;
        }
    }
    return EOS_OK;
}

static eos_status_t q8_0_quantize(const float *src, void *dst,
                                  size_t n_elements) {
    size_t b, blocks;
    uint8_t *p = (uint8_t *)dst;

    if (src == NULL || dst == NULL) return EOS_E_INVALID_ARG;
    if ((n_elements % BLOCK_ELEMS) != 0) return EOS_E_INVALID_ARG;
    blocks = n_elements / BLOCK_ELEMS;

    for (b = 0; b < blocks; ++b) {
        const float *bs = src + b * BLOCK_ELEMS;
        float amax = 0.0f;
        float scale, inv_scale;
        eosi_f16_t sh;
        int8_t w[BLOCK_ELEMS];
        int i;
        for (i = 0; i < BLOCK_ELEMS; ++i) {
            float a = bs[i] < 0.0f ? -bs[i] : bs[i];
            if (a > amax) amax = a;
        }
        scale = amax / 127.0f;
        inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        for (i = 0; i < BLOCK_ELEMS; ++i) {
            float v = bs[i] * inv_scale;
            int   q = (int)(v < 0.0f ? v - 0.5f : v + 0.5f);
            if (q >  127) q =  127;
            if (q < -128) q = -128;
            w[i] = (int8_t)q;
        }
        sh = eosi_f32_to_f16(scale);
        memcpy(p,     &sh, 2);
        memcpy(p + 2, w,   BLOCK_ELEMS);
        p += BLOCK_BYTES;
    }
    return EOS_OK;
}

static const eos_quant_vt_t k_q8_0 = {
    "q8_0", EOS_DT_Q8_0,
    q8_0_layout, q8_0_dequantize, q8_0_quantize
};

eos_status_t eosi_quant_q8_0_register(void) {
    return eos_quant_register(&k_q8_0);
}

#else
eos_status_t eosi_quant_q8_0_register(void) { return EOS_E_UNSUPPORTED; }
#endif
