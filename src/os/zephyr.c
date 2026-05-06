/*
 * src/os/zephyr.c — Zephyr OS shim (Phase 4 skeleton).
 *
 * Real implementation needs k_malloc/k_free, k_uptime_get_ns,
 * fs_open/fs_read/fs_close from <zephyr/...>. That comes in Phase 4
 * once a Zephyr build target lands in CI.
 */
#include "eosllm/eosllm.h"
#include "eosllm/os.h"
#include "os_internal.h"

#if EOSLLM_HAVE_ZEPHYR
eos_status_t eosi_os_use_zephyr(void) {
    /* TODO(phase-4): provide a shim backed by Zephyr's kernel APIs. */
    return EOS_E_UNSUPPORTED;
}
#else
eos_status_t eosi_os_use_zephyr(void) { return EOS_E_UNSUPPORTED; }
#endif
