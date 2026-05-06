/*
 * scheduler.h — scheduler policy registration and the deadline API.
 *
 * A scheduler decides what work runs next: a single decode step, a
 * batched step across multiple sessions, a speculative draft, or a
 * deadline-bounded step that may yield mid-layer.
 */
#ifndef EOSLLM_SCHEDULER_H
#define EOSLLM_SCHEDULER_H
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "eosllm.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * A scheduler implements this vtable. open() builds per-session state
 * inside the session arena (no separate allocation). step() advances
 * one decoder step under whatever policy this scheduler implements;
 * it must respect any deadline set via eos_session_set_deadline().
 */
typedef struct eos_sched_vt {
    const char *name;            /* e.g. "greedy", "deadline", "batched" */
    eos_status_t (*open) (eos_session_t *session, void **out_state);
    eos_status_t (*step) (void *state, uint32_t *out_token);
    void         (*close)(void *state);
} eos_sched_vt_t;

eos_status_t eos_sched_register(const eos_sched_vt_t *vt);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* EOSLLM_SCHEDULER_H */
