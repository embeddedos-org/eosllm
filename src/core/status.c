/*
 * src/core/status.c — eos_status_str.
 */
#include "eosllm/eosllm.h"

const char *eos_status_str(eos_status_t s) {
    switch (s) {
        case EOS_OK:                 return "OK";
        case EOS_E_INVALID_ARG:      return "invalid argument";
        case EOS_E_OUT_OF_MEMORY:    return "out of memory";
        case EOS_E_UNSUPPORTED:      return "unsupported (not compiled in)";
        case EOS_E_FORMAT:           return "malformed model file";
        case EOS_E_VERSION_MISMATCH: return "version mismatch";
        case EOS_E_NOT_FOUND:        return "not found";
        case EOS_E_DEADLINE:         return "deadline exceeded";
        case EOS_E_IO:               return "I/O error";
        case EOS_E_INTERNAL:         return "internal error";
    }
    return "unknown error";
}
