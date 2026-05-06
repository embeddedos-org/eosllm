/*
 * src/core/internal.h — engine-internal declarations.
 *
 * Visible only inside src/. Never installed, never part of the ABI.
 */
#ifndef EOSI_INTERNAL_H
#define EOSI_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "eosllm/eosllm.h"
#include "eosllm/backend.h"
#include "eosllm/modality.h"
#include "eosllm/os.h"
#include "eosllm/quant.h"
#include "eosllm/scheduler.h"

/* Compile-time table sizes. Sized for the realistic upper bound of
 * what a single build can have linked in; cheap. */
#define EOSI_MAX_BACKENDS    16
#define EOSI_MAX_QUANTS      16
#define EOSI_MAX_MODALITIES   8
#define EOSI_MAX_TOKENIZERS   4
#define EOSI_MAX_SCHEDULERS   8
#define EOSI_MAX_FORMATS      4

typedef struct eosi_registry {
    const eos_backend_vt_t   *backends   [EOSI_MAX_BACKENDS];
    size_t                    n_backends;

    const eos_quant_vt_t     *quants     [EOSI_MAX_QUANTS];
    size_t                    n_quants;

    const eos_modality_vt_t  *modalities [EOSI_MAX_MODALITIES];
    size_t                    n_modalities;

    const eos_tokenizer_vt_t *tokenizers [EOSI_MAX_TOKENIZERS];
    size_t                    n_tokenizers;

    const eos_sched_vt_t     *schedulers [EOSI_MAX_SCHEDULERS];
    size_t                    n_schedulers;

    /* Forward-declared in eosllm/model.h; not pulled in here to keep
     * this header light. */
    const struct eos_format_vt *formats  [EOSI_MAX_FORMATS];
    size_t                    n_formats;

    /* OS shim — exactly one. */
    eos_os_shim_t             os;
    int                       os_provided;

    /*
     * Set by eos_init_defaults() on first successful invocation.
     * Subsequent calls return EOS_OK without re-registering anything.
     * (Without this, every call would push every built-in module again,
     * silently growing the registry tables.)
     */
    int                       defaults_initialized;

    /*
     * Thread-safety policy: the registry is mutated only before the
     * first eos_session_open(). After the first session is opened, the
     * registry is "frozen": all eos_*_register entry points (and
     * eos_os_provide) reject with EOS_E_INVALID_ARG. This lets sessions
     * read the registry tables without locks and lets module code rely
     * on dispatch tables not changing under it.
     *
     * The flag is one-way and never reset. Documented in docs/abi.md.
     */
    int                       frozen;
} eosi_registry_t;

/* The single process-wide registry. Defined in src/core/registry.c. */
eosi_registry_t *eosi_registry(void);

/* Convenience wrappers for the shim, used everywhere in src/. */
void    *eosi_alloc(size_t bytes, size_t alignment);
void     eosi_free (void *ptr);
uint64_t eosi_now_ns(void);
void     eosi_log  (int level, const char *msg);

/* Internal log helpers. */
#define EOSI_LOG_DEBUG(msg) eosi_log(0, (msg))
#define EOSI_LOG_INFO(msg)  eosi_log(1, (msg))
#define EOSI_LOG_WARN(msg)  eosi_log(2, (msg))
/* EOSI_LOG_ERROR also stashes the message into the thread-local
 * eos_last_error() slot so hosts that don't install a log shim can
 * still surface "what went wrong" via the public API. */
#define EOSI_LOG_ERROR(msg) do { eosi_set_error((msg)); eosi_log(3, (msg)); } while (0)

/* Sets the thread-local last-error string (NULL clears it). */
void eosi_set_error(const char *msg);

#endif /* EOSI_INTERNAL_H */
