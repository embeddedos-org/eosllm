/*
 * src/quant/q4_k.c — GGUF-compatible q4_k quant scheme.
 *
 * Block layout (144 bytes per 256 elements). See
 * src/kernels/scalar/matmul_q4_k.c for the on-disk packing details.
 *
 * Compiled in only when EOSLLM_HAVE_QUANT_Q4_K=1.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "eosllm/eosllm.h"
#include "eosllm/quant.h"

#include "../util/f16.h"
#include "quant_internal.h"

#if EOSLLM_HAVE_QUANT_Q4_K

#define BLOCK_ELEMS 256
#define BLOCK_BYTES 144

static void q4_k_layout(eos_quant_layout_t *out) {
    out->elements_per_block = BLOCK_ELEMS;
    out->bytes_per_block    = BLOCK_BYTES;
    out->alignment          = 2;
    memset(out->_reserved, 0, sizeof(out->_reserved));
}

static inline void get_scale_min(int j, const uint8_t *q,
                                 uint8_t *out_d, uint8_t *out_m) {
    if (j < 4) {
        *out_d = q[j]     & 0x3Fu;
        *out_m = q[j + 4] & 0x3Fu;
    } else {
        *out_d = (q[j + 4] & 0x0Fu) | (uint8_t)((q[j - 4] >> 6) << 4);
        *out_m = (q[j + 4] >> 4)    | (uint8_t)((q[j    ] >> 6) << 4);
    }
}

static eos_status_t q4_k_dequantize(const void *src, float *dst,
                                    size_t n_elements) {
    size_t b, blocks;
    const uint8_t *p = (const uint8_t *)src;

    if (src == NULL || dst == NULL) return EOS_E_INVALID_ARG;
    if ((n_elements % BLOCK_ELEMS) != 0) return EOS_E_INVALID_ARG;
    blocks = n_elements / BLOCK_ELEMS;

    for (b = 0; b < blocks; ++b) {
        eosi_f16_t d_h, dmin_h;
        uint8_t    scales[12];
        const uint8_t *qs;
        float d, dmin;
        int sub;
        memcpy(&d_h,    p,     2);
        memcpy(&dmin_h, p + 2, 2);
        memcpy(scales,  p + 4, 12);
        qs = p + 16;
        p += BLOCK_BYTES;
        d    = eosi_f16_to_f32(d_h);
        dmin = eosi_f16_to_f32(dmin_h);

        for (sub = 0; sub < 8; sub += 2) {
            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            float d1, m1, d2, m2;
            int l;
            get_scale_min(sub,     scales, &sc_lo, &m_lo);
            get_scale_min(sub + 1, scales, &sc_hi, &m_hi);
            d1 = d * (float)sc_lo;  m1 = dmin * (float)m_lo;
            d2 = d * (float)sc_hi;  m2 = dmin * (float)m_hi;
            for (l = 0; l < 32; ++l) {
                dst[b * BLOCK_ELEMS + sub       * 32 + l] = d1 * (float)(qs[l] & 0x0Fu) - m1;
                dst[b * BLOCK_ELEMS + (sub + 1) * 32 + l] = d2 * (float)(qs[l] >> 4)    - m2;
            }
            qs += 32;
        }
    }
    return EOS_OK;
}

static eos_status_t q4_k_quantize(const float *src, void *dst,
                                  size_t n_elements) {
    /*
     * High-quality q4_k quantization needs per-super-block search
     * over scales/mins for each sub-block. That logic lives in
     * tools/eosllm-convert (Python, with PyTorch). The runtime
     * never quantizes q4_k.
     */
    (void)src; (void)dst; (void)n_elements;
    return EOS_E_UNSUPPORTED;
}

static const eos_quant_vt_t k_q4_k = {
    "q4_k", EOS_DT_Q4_K,
    q4_k_layout, q4_k_dequantize, q4_k_quantize
};

eos_status_t eosi_quant_q4_k_register(void) {
    return eos_quant_register(&k_q4_k);
}

#else
eos_status_t eosi_quant_q4_k_register(void) { return EOS_E_UNSUPPORTED; }
#endif
