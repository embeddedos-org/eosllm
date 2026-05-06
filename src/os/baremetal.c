/*
 * src/os/baremetal.c — bare-metal OS shim (Phase 4 skeleton).
 *
 * Bare-metal targets typically expose:
 *   - a static arena for `alloc`,
 *   - a hardware cycle counter for `now_ns`,
 *   - no file I/O (host streams a `.eosm` blob via a custom
 *     eos_stream_t backed by SPI flash or similar).
 */
#include "eosllm/eosllm.h"
#include "eosllm/os.h"
#include "os_internal.h"

#if EOSLLM_HAVE_BAREMETAL
eos_status_t eosi_os_use_baremetal(void) { return EOS_E_UNSUPPORTED; }
#else
eos_status_t eosi_os_use_baremetal(void) { return EOS_E_UNSUPPORTED; }
#endif
