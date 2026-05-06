/*
 * src/sched/sched_internal.h
 */
#ifndef EOSI_SCHED_INTERNAL_H
#define EOSI_SCHED_INTERNAL_H

#include "eosllm/eosllm.h"

eos_status_t eosi_sched_greedy_register  (void);
eos_status_t eosi_sched_deadline_register(void);
eos_status_t eosi_sched_batched_register (void);

eos_status_t eosi_sched_register_all(void);

/* Phase 5 helpers (currently stubs). */
eos_status_t eosi_paged_kv_init   (void);
eos_status_t eosi_speculative_init(void);

/* Greedy scheduler attach helpers (used by session.c). The "text"
 * pointer is an opaque eosi_text_t* (the scheduler treats it as a
 * void* and forwards it back to the modality on each step). */
void eosi_greedy_attach(void *state, void *text, float *logits);
void eosi_greedy_seed  (void *state, uint32_t token);

#endif
