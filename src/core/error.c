/*
 * src/core/error.c — thread-local last-error string.
 *
 * EOSI_LOG_ERROR() routes both to the OS shim's log callback (when
 * provided) and to a thread-local slot retrievable via the public
 * eos_last_error(). This lets hosts that don't install a log shim
 * still surface "what went wrong" without rebuilding the engine.
 *
 * The buffer is fixed-size (256 bytes including null terminator);
 * messages longer than that are truncated. Set to NULL/empty to
 * clear. The contents are valid only on the calling thread.
 */
#include <stddef.h>
#include <string.h>

#include "internal.h"
#include "eosllm/eosllm.h"

/* __thread is gcc/clang; portable "_Thread_local" is C11.  Most of our
 * targets have at least one. The fallback (static, not thread-local)
 * keeps small embedded targets compiling and is documented in
 * include/eosllm/eosllm.h. */
#if defined(__GNUC__) || defined(__clang__)
#  define EOSI_TLS __thread
#elif __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_THREADS__)
#  define EOSI_TLS _Thread_local
#else
#  define EOSI_TLS    /* not thread-local */
#endif

#define EOSI_ERROR_BUF_SIZE 256u

static EOSI_TLS char g_last_error[EOSI_ERROR_BUF_SIZE];

void eosi_set_error(const char *msg) {
    size_t n;
    if (msg == NULL || msg[0] == '\0') {
        g_last_error[0] = '\0';
        return;
    }
    n = strlen(msg);
    if (n >= EOSI_ERROR_BUF_SIZE) n = EOSI_ERROR_BUF_SIZE - 1u;
    memcpy(g_last_error, msg, n);
    g_last_error[n] = '\0';
}

const char *eos_last_error(void) {
    return (g_last_error[0] != '\0') ? g_last_error : NULL;
}
