/*
 * src/os/os_internal.h — internal OS shim entry points for non-POSIX
 * targets (zephyr, freertos, baremetal). The default POSIX shim has
 * its public eos_os_use_posix() in include/eosllm/os.h.
 */
#ifndef EOSI_OS_INTERNAL_H
#define EOSI_OS_INTERNAL_H

#include "eosllm/eosllm.h"

eos_status_t eosi_os_use_zephyr   (void);
eos_status_t eosi_os_use_freertos (void);
eos_status_t eosi_os_use_baremetal(void);
eos_status_t eosi_os_use_win32    (void);
eos_status_t eosi_os_use_wasm     (void);

#endif
