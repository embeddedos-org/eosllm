/*
 * src/format/format_internal.h — internal entry points for each
 * file-format module's registration function.
 */
#ifndef EOSI_FORMAT_INTERNAL_H
#define EOSI_FORMAT_INTERNAL_H

#include "eosllm/eosllm.h"

eos_status_t eosi_format_gguf_register(void);
eos_status_t eosi_format_eosm_register(void);

eos_status_t eosi_format_register_all(void);

#endif
