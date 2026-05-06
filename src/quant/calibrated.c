/*
 * src/quant/calibrated.c — AWQ/GPTQ-style calibrated quant.
 *
 * The runtime side is a thin layer on top of an underlying scheme
 * (typically q4_k or q1_58) plus per-group activation-aware scales
 * stored alongside the weights in the .eosm file.
 *
 * Calibration itself runs in tools/eosllm-quant-lab (Python). The
 * runtime never calibrates.
 */
#include <stddef.h>
#include <string.h>

#include "eosllm/eosllm.h"
#include "eosllm/quant.h"
#include "quant_internal.h"

#if EOSLLM_HAVE_QUANT_CALIBRATED

static void calibrated_layout(eos_quant_layout_t *out) {
    /* Per-tensor; described in the .eosm manifest. */
    out->elements_per_block = 0;
    out->bytes_per_block    = 0;
    out->alignment          = 0;
    memset(out->_reserved, 0, sizeof(out->_reserved));
}

static eos_status_t calibrated_dequantize(const void *src, float *dst,
                                          size_t n_elements) {
    (void)src; (void)dst; (void)n_elements;
    /* TODO(phase-2): apply per-group scale ⊙ underlying.dequantize(). */
    return EOS_E_UNSUPPORTED;
}

static eos_status_t calibrated_quantize(const float *src, void *dst,
                                        size_t n_elements) {
    (void)src; (void)dst; (void)n_elements;
    return EOS_E_UNSUPPORTED;
}

static const eos_quant_vt_t k_calibrated = {
    "calibrated", EOS_DT_CALIBRTD,
    calibrated_layout, calibrated_dequantize, calibrated_quantize
};

eos_status_t eosi_quant_calibrated_register(void) {
    return eos_quant_register(&k_calibrated);
}

#else
eos_status_t eosi_quant_calibrated_register(void) { return EOS_E_UNSUPPORTED; }
#endif
