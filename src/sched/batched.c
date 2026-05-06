/*
 * src/sched/batched.c — Phase 5 continuous-batched scheduler.
 *
 * Multiplexes many sessions sharing one set of weights, with a paged
 * KV cache (see paged_kv.c). Phase-1 stub.
 */
#include <stddef.h>
#include "eosllm/eosllm.h"
#include "eosllm/scheduler.h"
#include "sched_internal.h"

#if EOSLLM_HAVE_SCHED_BATCHED

static eos_status_t open_(eos_session_t *s, void **st)
    { (void)s; (void)st; return EOS_E_UNSUPPORTED; }
static eos_status_t step_(void *st, uint32_t *t)
    { (void)st; (void)t; return EOS_E_UNSUPPORTED; }
static void         close_(void *st) { (void)st; }

static const eos_sched_vt_t k_batched = { "batched", open_, step_, close_ };
eos_status_t eosi_sched_batched_register(void) { return eos_sched_register(&k_batched); }

#else
eos_status_t eosi_sched_batched_register(void) { return EOS_E_UNSUPPORTED; }
#endif
