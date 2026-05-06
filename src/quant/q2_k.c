/*
 * src/quant/q2_k.c — Phase 2 q2_k (placeholder).
 *
 * Real GGUF q2_k implementation is forthcoming. This file makes the
 * symbol available so the registry helper links in any build.
 */
#include "eosllm/eosllm.h"
#include "quant_internal.h"

#if EOSLLM_HAVE_QUANT_Q2_K
eos_status_t eosi_quant_q2_k_register(void) {
    /* TODO(phase-2): wire up real q2_k. */
    return EOS_E_UNSUPPORTED;
}
#else
eos_status_t eosi_quant_q2_k_register(void) { return EOS_E_UNSUPPORTED; }
#endif
