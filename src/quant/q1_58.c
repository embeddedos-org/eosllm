/*
 * src/quant/q1_58.c — BitNet-style 1.58-bit ternary quant.
 *
 * Each weight is one of {-1, 0, +1}. Five ternary digits pack into
 * one byte (3^5 = 243 < 256), giving log2(3) ≈ 1.585 bits/weight.
 *
 * Block layout (38 bytes per 160 elements):
 *   eosi_f16_t  scale;      // shared block scale
 *   uint16_t    n_zeros;    // optional sparsity hint (count of zeros in qs)
 *   uint8_t     qs[32];     // 32 bytes × 5 trits/byte = 160 trits total
 *
 * Compiled in only when EOSLLM_HAVE_QUANT_Q1_58=1.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "eosllm/eosllm.h"
#include "eosllm/quant.h"

#include "../util/f16.h"
#include "quant_internal.h"

#if EOSLLM_HAVE_QUANT_Q1_58

#define BLOCK_ELEMS 160
#define BLOCK_BYTES 36     /* 2 (scale) + 2 (n_zeros) + 32 (qs) */

/* Decode one byte (5 ternary digits) into 5 signed values in {-1,0,+1}. */
static inline void decode_byte(uint8_t b, int8_t out[5]) {
    int i;
    for (i = 0; i < 5; ++i) {
        uint8_t t = (uint8_t)(b % 3u);
        b = (uint8_t)(b / 3u);
        out[i] = (int8_t)((int)t - 1);   /* 0→-1, 1→0, 2→+1 */
    }
}

/* Encode 5 signed values in {-1,0,+1} into one byte. */
static inline uint8_t encode_byte(const int8_t in[5]) {
    int i;
    uint8_t b = 0;
    /* base-3 little-digit-first */
    for (i = 4; i >= 0; --i) {
        uint8_t t = (uint8_t)(in[i] + 1);   /* -1→0, 0→1, +1→2 */
        if (t > 2) t = 2;
        b = (uint8_t)(b * 3u + t);
    }
    return b;
}

static void q1_58_layout(eos_quant_layout_t *out) {
    out->elements_per_block = BLOCK_ELEMS;
    out->bytes_per_block    = BLOCK_BYTES;
    out->alignment          = 2;
    memset(out->_reserved, 0, sizeof(out->_reserved));
}

static eos_status_t q1_58_dequantize(const void *src, float *dst,
                                     size_t n_elements) {
    size_t b, blocks;
    const uint8_t *p = (const uint8_t *)src;

    if (src == NULL || dst == NULL) return EOS_E_INVALID_ARG;
    if ((n_elements % BLOCK_ELEMS) != 0) return EOS_E_INVALID_ARG;
    blocks = n_elements / BLOCK_ELEMS;

    for (b = 0; b < blocks; ++b) {
        eosi_f16_t sh;
        float scale;
        int byte;
        memcpy(&sh, p, 2);
        scale = eosi_f16_to_f32(sh);
        for (byte = 0; byte < 32; ++byte) {
            int8_t trits[5];
            int t;
            decode_byte(p[4 + byte], trits);
            for (t = 0; t < 5; ++t) {
                dst[b * BLOCK_ELEMS + (size_t)byte * 5 + t]
                    = (float)trits[t] * scale;
            }
        }
        p += BLOCK_BYTES;
    }
    return EOS_OK;
}

static eos_status_t q1_58_quantize(const float *src, void *dst,
                                   size_t n_elements) {
    size_t b, blocks;
    uint8_t *p = (uint8_t *)dst;

    if (src == NULL || dst == NULL) return EOS_E_INVALID_ARG;
    if ((n_elements % BLOCK_ELEMS) != 0) return EOS_E_INVALID_ARG;
    blocks = n_elements / BLOCK_ELEMS;

    for (b = 0; b < blocks; ++b) {
        const float *bs = src + b * BLOCK_ELEMS;
        float amean = 0.0f;
        float scale, threshold, inv;
        int i, byte, n_zeros = 0;
        eosi_f16_t sh;
        /*
         * Standard BitNet 1.58 absmean quant:
         *   scale     = mean(|w|)
         *   threshold = 0.5 * scale
         *   q[i] = sign(w[i]) if |w[i]| > threshold else 0
         */
        for (i = 0; i < BLOCK_ELEMS; ++i) {
            float a = bs[i] < 0.0f ? -bs[i] : bs[i];
            amean += a;
        }
        amean /= (float)BLOCK_ELEMS;
        scale = amean;
        threshold = 0.5f * scale;
        inv = (scale > 0.0f) ? 1.0f / scale : 0.0f;
        (void)inv;
        sh = eosi_f32_to_f16(scale);
        memcpy(p, &sh, 2);

        for (byte = 0; byte < 32; ++byte) {
            int8_t trits[5];
            int t;
            for (t = 0; t < 5; ++t) {
                float v = bs[(size_t)byte * 5 + t];
                if (v >  threshold)      trits[t] =  1;
                else if (v < -threshold) trits[t] = -1;
                else { trits[t] = 0; n_zeros++; }
            }
            p[4 + byte] = encode_byte(trits);
        }
        {
            uint16_t nz = (uint16_t)(n_zeros < 0xFFFF ? n_zeros : 0xFFFF);
            memcpy(p + 2, &nz, 2);
        }
        p += BLOCK_BYTES;
    }
    return EOS_OK;
}

static const eos_quant_vt_t k_q1_58 = {
    "q1_58", EOS_DT_Q1_58,
    q1_58_layout, q1_58_dequantize, q1_58_quantize
};

eos_status_t eosi_quant_q1_58_register(void) {
    return eos_quant_register(&k_q1_58);
}

#else
eos_status_t eosi_quant_q1_58_register(void) { return EOS_E_UNSUPPORTED; }
#endif
