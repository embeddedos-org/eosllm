/*
 * src/kernels/avx512/avx512.c — Phase 4+ AVX-512 backend (skeleton).
 */
#include "eosllm/eosllm.h"
#include "eosllm/backend.h"
#include "../kernels_internal.h"
#if EOSLLM_HAVE_KERNEL_AVX512
static eos_status_t probe_(void) { return EOS_E_UNSUPPORTED; }
static const eos_backend_vt_t k_vt = { "x86_avx512", 60, probe_,
                                       { NULL, NULL, NULL, NULL, NULL, NULL, NULL } };
eos_status_t eosi_backend_avx512_register(void) { return eos_backend_register(&k_vt); }
#else
eos_status_t eosi_backend_avx512_register(void) { return EOS_E_UNSUPPORTED; }
#endif
