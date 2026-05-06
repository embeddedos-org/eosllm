/*
 * src/quant/mixed.c — per-tensor / per-channel mixed-precision quant.
 *
 * Mixed is a *meta-scheme*: each tensor (or each row/channel within
 * a tensor) carries a manifest naming the underlying scheme to use.
 * The runtime delegates dequantize and matmul to the named scheme.
 *
 * The per-tensor manifest is part of the .eosm file format (Phase 2);
 * for now this module just registers the meta-scheme name.
 */
#include <stddef.h>
#include <string.h>

#include "eosllm/eosllm.h"
#include "eosllm/quant.h"
#include "quant_internal.h"

#if EOSLLM_HAVE_QUANT_MIXED

static void mixed_layout(eos_quant_layout_t *out) {
    /* Layout is per-tensor in the .eosm manifest, not a fixed block. */
    out->elements_per_block = 0;
    out->bytes_per_block    = 0;
    out->alignment          = 0;
    memset(out->_reserved, 0, sizeof(out->_reserved));
}

static eos_status_t mixed_dequantize(const void *src, float *dst,
                                     size_t n_elements) {
    (void)src; (void)dst; (void)n_elements;
    /* TODO(phase-2): consult the per-tensor manifest in the session. */
    return EOS_E_UNSUPPORTED;
}

static eos_status_t mixed_quantize(const float *src, void *dst,
                                   size_t n_elements) {
    (void)src; (void)dst; (void)n_elements;
    return EOS_E_UNSUPPORTED;
}

static const eos_quant_vt_t k_mixed = {
    "mixed", EOS_DT_MIXED,
    mixed_layout, mixed_dequantize, mixed_quantize
};

eos_status_t eosi_quant_mixed_register(void) {
    return eos_quant_register(&k_mixed);
}

#else
eos_status_t eosi_quant_mixed_register(void) { return EOS_E_UNSUPPORTED; }
#endif
