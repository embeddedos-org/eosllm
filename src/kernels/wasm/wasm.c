/*
 * src/kernels/wasm/wasm.c — WASM SIMD128 backend skeleton (Phase F).
 *
 * Reserves the registry slot. The real implementation will provide a
 * matmul_q4_k path using <wasm_simd128.h> intrinsics; deferred per
 * the v0.2 release plan.
 */
#include "eosllm/eosllm.h"
#include "eosllm/backend.h"
#include "../kernels_internal.h"

#if EOSLLM_HAVE_KERNEL_WASM_SIMD
static eos_status_t probe_(void) { return EOS_E_UNSUPPORTED; }
static const eos_backend_vt_t k_vt = { "wasm_simd128", 35, probe_,
                                       { NULL, NULL, NULL, NULL, NULL, NULL, NULL } };
eos_status_t eosi_backend_wasm_register(void) { return eos_backend_register(&k_vt); }
#else
eos_status_t eosi_backend_wasm_register(void) { return EOS_E_UNSUPPORTED; }
#endif
