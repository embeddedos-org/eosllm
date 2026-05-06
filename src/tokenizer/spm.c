/*
 * src/tokenizer/spm.c — SentencePiece-compatible tokenizer (skeleton).
 *
 * Phase 1 ships byte-level BPE only. SentencePiece support comes in
 * Phase 2 alongside the .eosm format.
 */
#include "eosllm/eosllm.h"
#include "tokenizer_internal.h"

#if EOSLLM_HAVE_TOKENIZER_SPM
eos_status_t eosi_tokenizer_spm_register(void) {
    /* TODO(phase-2): unigram LM tokenizer + protobuf-free .model loader. */
    return EOS_E_UNSUPPORTED;
}
#else
eos_status_t eosi_tokenizer_spm_register(void) { return EOS_E_UNSUPPORTED; }
#endif
