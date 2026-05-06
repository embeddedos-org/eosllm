/*
 * os.h — OS shim ABI.
 *
 * The eosllm core never calls malloc / pthread_* / fopen / clock_*
 * directly. Everything goes through the eos_os_* shim, which the host
 * provides at startup via eos_os_provide(&shim).
 *
 * A default POSIX implementation is built into the library when
 * EOSLLM_HAVE_POSIX is on; calling eos_os_use_posix() installs it.
 */
#ifndef EOSLLM_OS_H
#define EOSLLM_OS_H
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "eosllm.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * The host-supplied OS shim. Any field left NULL falls back to a
 * conservative default if one is available, otherwise the engine
 * returns EOS_E_UNSUPPORTED for any operation that needs it.
 *
 * All function pointers must be thread-safe iff EOSLLM_HAVE_THREADS
 * was on at compile time.
 */
typedef struct eos_os_shim {
    /* Memory. The runtime allocates at most a handful of times per
     * session lifetime; these are not on the hot path. */
    void *(*alloc)(size_t bytes, size_t alignment);
    void  (*free) (void *ptr);

    /* Monotonic nanosecond clock. */
    uint64_t (*now_ns)(void);

    /* File I/O. May be NULL on MCU targets that load models from a
     * fixed memory region; in that case the host supplies models via
     * a custom eos_stream_t instead of eos_model_open(). */
    int     (*file_open) (const char *path, void **out_handle);
    int64_t (*file_read) (void *handle, void *buf, size_t len);
    int     (*file_seek) (void *handle, int64_t offset);
    int64_t (*file_size) (void *handle);
    void    (*file_close)(void *handle);

    /* Logging. May be NULL; if NULL, log lines are dropped. level is
     * 0=debug, 1=info, 2=warn, 3=error. */
    void (*log)(int level, const char *msg);
} eos_os_shim_t;

/* Installs the host's shim. Must be called before eos_session_open. */
eos_status_t eos_os_provide(const eos_os_shim_t *shim);

/* Convenience: install a default POSIX shim. Returns EOS_E_UNSUPPORTED
 * if the library was built without EOSLLM_HAVE_POSIX. */
eos_status_t eos_os_use_posix(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* EOSLLM_OS_H */
