/*
 * src/kernels/npu/npu.c — NPU delegate (Phase 4 skeleton).
 *
 * The "backend" here is a thin trampoline that, when configured for
 * a specific NPU (CoreML / NNAPI / QNN / EdgeTPU), forwards a whole
 * sub-graph instead of individual ops. Phase 1 stub.
 */
#include "eosllm/eosllm.h"
#include "eosllm/backend.h"
#include "../kernels_internal.h"
#if EOSLLM_HAVE_KERNEL_NPU
static eos_status_t probe_(void) { return EOS_E_UNSUPPORTED; }
static const eos_backend_vt_t k_vt = { "npu", 80, probe_,
                                       { NULL, NULL, NULL, NULL, NULL, NULL, NULL } };
eos_status_t eosi_backend_npu_register(void) { return eos_backend_register(&k_vt); }
#else
eos_status_t eosi_backend_npu_register(void) { return EOS_E_UNSUPPORTED; }
#endif
