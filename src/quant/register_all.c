/*
 * src/quant/register_all.c — registers every quant scheme compiled in.
 *
 * Schemes excluded at compile time return EOS_E_UNSUPPORTED from their
 * register fn; we ignore that — the corresponding capability bit in
 * eos_caps() reflects what was actually compiled in.
 */
#include "quant_internal.h"

eos_status_t eosi_quant_register_all(void) {
    eos_status_t s;
    s = eosi_quant_q8_0_register();        if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    s = eosi_quant_q4_k_register();        if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    s = eosi_quant_q2_k_register();        if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    s = eosi_quant_q1_58_register();       if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    s = eosi_quant_mixed_register();       if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    s = eosi_quant_calibrated_register();  if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    return EOS_OK;
}
