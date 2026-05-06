#include "eosllm/eosllm.h"
#include "eosllm/backend.h"
#include "../kernels_internal.h"
#if EOSLLM_HAVE_KERNEL_RVV
static eos_status_t probe_(void) { return EOS_E_UNSUPPORTED; }
static const eos_backend_vt_t k_vt = { "riscv_v", 45, probe_,
                                       { NULL, NULL, NULL, NULL, NULL, NULL, NULL } };
eos_status_t eosi_backend_rvv_register(void) { return eos_backend_register(&k_vt); }
#else
eos_status_t eosi_backend_rvv_register(void) { return EOS_E_UNSUPPORTED; }
#endif
