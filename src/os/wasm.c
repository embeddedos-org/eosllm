/*
 * src/os/wasm.c — Emscripten / WASM OS shim (Phase F skeleton).
 *
 * The real implementation needs:
 *   - emscripten_get_now() (via <emscripten/emscripten.h>) for timing
 *   - in-memory file streams only (no fopen against the host FS)
 *   - host-supplied async loader for the GGUF blob
 *
 * Phase F reservation: this file exists so the build system can flip
 * EOSLLM_HAVE_OS_WASM=1 without touching the SRC list. Mirrors the
 * existing zephyr.c / freertos.c / baremetal.c skeleton pattern.
 */
#include "eosllm/eosllm.h"
#include "eosllm/os.h"
#include "os_internal.h"

#if EOSLLM_HAVE_OS_WASM
eos_status_t eosi_os_use_wasm(void) {
    /* TODO(phase-F): provide a shim backed by emscripten_get_now and
     * a host-supplied in-memory file map. */
    return EOS_E_UNSUPPORTED;
}
#else
eos_status_t eosi_os_use_wasm(void) { return EOS_E_UNSUPPORTED; }
#endif
