/*
 * src/sched/deadline.c — Phase 4 deadline-driven scheduler.
 *
 * Yields between layers when the session's deadline is exceeded so a
 * caller can interleave the engine with hard-real-time work. The
 * Phase-1 file is a registration stub; the actual yield logic lands
 * in Phase 4 once the matmul kernels expose preemption points.
 */
#include <stddef.h>
#include "eosllm/eosllm.h"
#include "eosllm/scheduler.h"
#include "sched_internal.h"

#if EOSLLM_HAVE_SCHED_DEADLINE

static eos_status_t open_(eos_session_t *s, void **st)
    { (void)s; (void)st; return EOS_E_UNSUPPORTED; }
static eos_status_t step_(void *st, uint32_t *t)
    { (void)st; (void)t; return EOS_E_UNSUPPORTED; }
static void         close_(void *st) { (void)st; }

static const eos_sched_vt_t k_deadline = { "deadline", open_, step_, close_ };
eos_status_t eosi_sched_deadline_register(void) { return eos_sched_register(&k_deadline); }

#else
eos_status_t eosi_sched_deadline_register(void) { return EOS_E_UNSUPPORTED; }
#endif
