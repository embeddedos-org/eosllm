/*
 * src/sched/paged_kv.c — Phase 5 paged KV cache helper.
 *
 * Allocates KV-cache pages of fixed size and lets multiple sessions
 * share them, evicting LRU pages under pressure. Phase-1 stub.
 */
#include <stddef.h>
#include "eosllm/eosllm.h"
#include "sched_internal.h"

eos_status_t eosi_paged_kv_init(void) { return EOS_E_UNSUPPORTED; }
