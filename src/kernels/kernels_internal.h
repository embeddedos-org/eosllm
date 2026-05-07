/*
 * src/kernels/kernels_internal.h — internal kernel-backend register fns.
 */
#ifndef EOSI_KERNELS_INTERNAL_H
#define EOSI_KERNELS_INTERNAL_H

#include "eosllm/eosllm.h"

eos_status_t eosi_scalar_backend_register(void);
eos_status_t eosi_backend_avx2_register  (void);
eos_status_t eosi_backend_avx512_register(void);
eos_status_t eosi_backend_neon_register  (void);
eos_status_t eosi_backend_sve_register   (void);
eos_status_t eosi_backend_rvv_register   (void);
eos_status_t eosi_backend_hvx_register   (void);
eos_status_t eosi_backend_npu_register   (void);
eos_status_t eosi_backend_wasm_register  (void);

eos_status_t eosi_backend_register_all(void);

#endif
