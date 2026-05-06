/*
 * src/modality/audio/audio.c — Phase 3 Whisper-class audio encoder skeleton.
 *
 * Pipeline (planned):
 *   PCM s16/f32 mono 16 kHz → log-mel spectrogram → conv1d × 2 →
 *   transformer encoder × N → embeddings.
 */
#include <stddef.h>
#include "eosllm/eosllm.h"
#include "eosllm/modality.h"
#include "../modality_internal.h"

#if EOSLLM_HAVE_MODALITY_AUDIO

static eos_status_t open_  (eos_session_t *s, eos_model_t *m, void **st)
    { (void)s; (void)m; (void)st; return EOS_E_UNSUPPORTED; }
static eos_status_t encode_(void *st, const void *d, size_t n)
    { (void)st; (void)d; (void)n; return EOS_E_UNSUPPORTED; }
static void         close_ (void *st) { (void)st; }

static const eos_modality_vt_t k_audio = {
    "audio-whisper", EOS_MODALITY_AUDIO, open_, encode_, close_
};

eos_status_t eosi_modality_audio_register(void) {
    return eos_modality_register(&k_audio);
}

#else
eos_status_t eosi_modality_audio_register(void) { return EOS_E_UNSUPPORTED; }
#endif
