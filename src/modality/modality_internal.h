/*
 * src/modality/modality_internal.h
 */
#ifndef EOSI_MODALITY_INTERNAL_H
#define EOSI_MODALITY_INTERNAL_H

#include "eosllm/eosllm.h"

eos_status_t eosi_modality_text_register  (void);
eos_status_t eosi_modality_vision_register(void);
eos_status_t eosi_modality_audio_register (void);

eos_status_t eosi_fusion_splice(void *session_state,
                                const float *modality_emb, size_t n_tokens);

eos_status_t eosi_modality_register_all(void);

#endif
