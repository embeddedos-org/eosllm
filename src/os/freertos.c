/*
 * src/os/freertos.c — FreeRTOS OS shim (Phase 4 skeleton).
 */
#include "eosllm/eosllm.h"
#include "eosllm/os.h"
#include "os_internal.h"

#if EOSLLM_HAVE_FREERTOS
eos_status_t eosi_os_use_freertos(void) { return EOS_E_UNSUPPORTED; }
#else
eos_status_t eosi_os_use_freertos(void) { return EOS_E_UNSUPPORTED; }
#endif
