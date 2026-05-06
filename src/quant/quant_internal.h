/*
 * src/quant/quant_internal.h — internal registration entry points for
 * each quant scheme. The engine init code (or a test) calls these to
 * populate the registry.
 *
 * Each function returns EOS_E_UNSUPPORTED if its module was excluded
 * at compile time, never failing the build.
 */
#ifndef EOSI_QUANT_INTERNAL_H
#define EOSI_QUANT_INTERNAL_H

#include "eosllm/eosllm.h"

eos_status_t eosi_quant_q8_0_register   (void);
eos_status_t eosi_quant_q4_k_register   (void);
eos_status_t eosi_quant_q2_k_register   (void);
eos_status_t eosi_quant_q1_58_register  (void);
eos_status_t eosi_quant_mixed_register  (void);
eos_status_t eosi_quant_calibrated_register(void);

/* Convenience: register every quant scheme compiled into this build.
 * Defined in src/quant/register_all.c. */
eos_status_t eosi_quant_register_all(void);

#endif /* EOSI_QUANT_INTERNAL_H */
