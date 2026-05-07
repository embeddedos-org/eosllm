#include "kernels_internal.h"

eos_status_t eosi_backend_register_all(void) {
    eos_status_t s;
    s = eosi_scalar_backend_register();   if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    s = eosi_backend_avx2_register();     if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    s = eosi_backend_avx512_register();   if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    s = eosi_backend_neon_register();     if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    s = eosi_backend_sve_register();      if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    s = eosi_backend_rvv_register();      if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    s = eosi_backend_hvx_register();      if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    s = eosi_backend_npu_register();      if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    s = eosi_backend_wasm_register();     if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    return EOS_OK;
}
