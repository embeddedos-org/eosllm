/*
 * src/modality/fusion.c — Phase 3 cross-modal embedding splice.
 *
 * Once vision/audio encoders produce per-token embedding vectors,
 * fusion replaces the corresponding placeholder tokens (e.g. <image>,
 * <audio>) in the decoder's input sequence with those embeddings.
 *
 * Phase 1 ships the file (so the directory shape matches the plan)
 * but the function body comes in Phase 3.
 */
#include <stddef.h>
#include "eosllm/eosllm.h"
#include "modality_internal.h"

eos_status_t eosi_fusion_splice(void *session_state,
                                const float *modality_emb, size_t n_tokens) {
    (void)session_state; (void)modality_emb; (void)n_tokens;
    return EOS_E_UNSUPPORTED;
}
