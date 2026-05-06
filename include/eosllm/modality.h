/*
 * modality.h — modality (text/vision/audio) registration.
 *
 * A modality module knows how to turn raw input bytes into the
 * sequence of embedding vectors that the decoder consumes.
 */
#ifndef EOSLLM_MODALITY_H
#define EOSLLM_MODALITY_H
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "eosllm.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * A modality module's vtable. open() builds per-session encoder state
 * (mostly pointers into the session arena). encode() consumes one
 * chunk of raw input and writes embedding vectors into the slots that
 * fusion.c will splice into the decoder's input sequence.
 */
typedef struct eos_modality_vt {
    const char         *name;            /* e.g. "text", "vision-vit", "audio-whisper" */
    eos_modality_kind_t kind;

    eos_status_t (*open) (eos_session_t *session, eos_model_t *model,
                          void **out_state);
    eos_status_t (*encode)(void *state, const void *data, size_t len);
    void         (*close)(void *state);
} eos_modality_vt_t;

eos_status_t eos_modality_register(const eos_modality_vt_t *vt);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* EOSLLM_MODALITY_H */
