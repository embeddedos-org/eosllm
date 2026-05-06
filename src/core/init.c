/*
 * src/core/init.c — eos_init_defaults: one-call setup for hosts that
 * just want everything compiled in to be ready.
 */
#include "eosllm/eosllm.h"
#include "eosllm/os.h"

#include "internal.h"
#include "../kernels/kernels_internal.h"
#include "../quant/quant_internal.h"
#include "../modality/modality_internal.h"
#include "../tokenizer/tokenizer_internal.h"
#include "../format/format_internal.h"
#include "../sched/sched_internal.h"

eos_status_t eos_init_defaults(void) {
    eosi_registry_t *r = eosi_registry();
    eos_status_t s;

    /* AP contract: clear thread-local last-error so EOS_OK leaves the
     * slot empty (matches every other public entry point). */
    eosi_set_error(NULL);

    /* Idempotent: each register_all pushes into a fixed table. Without
     * this guard a second call would silently double-register every
     * built-in module. */
    if (r->defaults_initialized) return EOS_OK;

    if (!r->os_provided) {
        s = eos_os_use_posix();
        if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    }

    s = eosi_backend_register_all();    if (s != EOS_OK) return s;
    s = eosi_quant_register_all();      if (s != EOS_OK) return s;
    s = eosi_modality_register_all();   if (s != EOS_OK) return s;
    s = eosi_tokenizer_register_all();  if (s != EOS_OK) return s;
    s = eosi_format_register_all();     if (s != EOS_OK) return s;
    s = eosi_sched_register_all();      if (s != EOS_OK) return s;
    r->defaults_initialized = 1;
    return EOS_OK;
}
