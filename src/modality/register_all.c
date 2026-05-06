#include "modality_internal.h"

eos_status_t eosi_modality_register_all(void) {
    eos_status_t s;
    s = eosi_modality_text_register();   if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    s = eosi_modality_vision_register(); if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    s = eosi_modality_audio_register();  if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    return EOS_OK;
}
