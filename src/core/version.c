/*
 * src/core/version.c — version macros exposed at runtime.
 */
#include "eosllm/eosllm.h"

#define EOSI_STR_(x) #x
#define EOSI_STR(x)  EOSI_STR_(x)

static const char k_version_string[] =
    EOSI_STR(EOSLLM_VERSION_MAJOR) "."
    EOSI_STR(EOSLLM_VERSION_MINOR) "."
    EOSI_STR(EOSLLM_VERSION_PATCH);

int eos_abi_version(void) {
    return EOSLLM_ABI_VERSION;
}

const char *eos_version_string(void) {
    return k_version_string;
}
