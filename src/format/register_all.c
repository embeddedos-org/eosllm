/*
 * src/format/register_all.c — registers every file-format reader
 * compiled into this build.
 */
#include "format_internal.h"

eos_status_t eosi_format_register_all(void) {
    eos_status_t s;
    s = eosi_format_gguf_register(); if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    s = eosi_format_eosm_register(); if (s != EOS_OK && s != EOS_E_UNSUPPORTED) return s;
    return EOS_OK;
}
