/*
 * src/tokenizer/tokenizer_internal.h
 */
#ifndef EOSI_TOKENIZER_INTERNAL_H
#define EOSI_TOKENIZER_INTERNAL_H

#include "eosllm/eosllm.h"

eos_status_t eosi_tokenizer_bpe_register(void);
eos_status_t eosi_tokenizer_spm_register(void);

eos_status_t eosi_tokenizer_register_all(void);

#endif
