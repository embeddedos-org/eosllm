#include "sched_internal.h"

eos_status_t eosi_sched_register_all(void) {
    eos_status_t s;
    s = eosi_sched_greedy_register();   if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    s = eosi_sched_deadline_register(); if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    s = eosi_sched_batched_register();  if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    return EOS_OK;
}
