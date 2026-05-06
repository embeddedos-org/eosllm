/*
 * src/tokenizer/register_all.c
 */
#include "tokenizer_internal.h"

eos_status_t eosi_tokenizer_register_all(void) {
    eos_status_t s;
    s = eosi_tokenizer_bpe_register(); if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    s = eosi_tokenizer_spm_register(); if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    return EOS_OK;
}
