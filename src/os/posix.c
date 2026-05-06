/*
 * src/os/posix.c — POSIX implementation of the eos_os_* shim.
 *
 * Compiled in only when EOSLLM_HAVE_POSIX=1. Calling
 * eos_os_use_posix() installs this shim into the registry.
 */
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "eosllm/eosllm.h"
#include "eosllm/os.h"

#if EOSLLM_HAVE_POSIX

static void *posix_alloc(size_t bytes, size_t alignment) {
    void *p = NULL;
    if (alignment < sizeof(void *)) alignment = sizeof(void *);
    /* posix_memalign requires alignment to be a power of two and a
     * multiple of sizeof(void*). We don't validate here; the runtime
     * always passes a sane alignment. */
    if (posix_memalign(&p, alignment, bytes) != 0) return NULL;
    return p;
}

static void posix_free(void *ptr) {
    free(ptr);
}

static uint64_t posix_now_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int posix_file_open(const char *path, void **out_handle) {
    FILE *f;
    if (path == NULL || out_handle == NULL) return -1;
    f = fopen(path, "rb");
    if (f == NULL) return -1;
    *out_handle = f;
    return 0;
}

static int64_t posix_file_read(void *handle, void *buf, size_t len) {
    size_t n;
    if (handle == NULL || buf == NULL) return -1;
    n = fread(buf, 1, len, (FILE *)handle);
    if (n == 0 && ferror((FILE *)handle)) return -1;
    return (int64_t)n;
}

static int posix_file_seek(void *handle, int64_t offset) {
    if (handle == NULL) return -1;
#if defined(_WIN32) || defined(_WIN64)
    return _fseeki64((FILE *)handle, offset, SEEK_SET) == 0 ? 0 : -1;
#else
    if (fseek((FILE *)handle, (long)offset, SEEK_SET) != 0) return -1;
    return 0;
#endif
}

static int64_t posix_file_size(void *handle) {
    long here, end;
    if (handle == NULL) return -1;
    here = ftell((FILE *)handle);
    if (here < 0) return -1;
    if (fseek((FILE *)handle, 0, SEEK_END) != 0) return -1;
    end = ftell((FILE *)handle);
    if (fseek((FILE *)handle, here, SEEK_SET) != 0) return -1;
    return (int64_t)end;
}

static void posix_file_close(void *handle) {
    if (handle != NULL) fclose((FILE *)handle);
}

static void posix_log(int level, const char *msg) {
    static const char *prefix[4] = { "[debug] ", "[info]  ", "[warn]  ", "[error] " };
    const char *p = (level >= 0 && level <= 3) ? prefix[level] : "[?]     ";
    fputs(p, stderr);
    fputs(msg ? msg : "", stderr);
    fputc('\n', stderr);
}

static const eos_os_shim_t k_posix_shim = {
    posix_alloc,
    posix_free,
    posix_now_ns,
    posix_file_open,
    posix_file_read,
    posix_file_seek,
    posix_file_size,
    posix_file_close,
    posix_log
};

eos_status_t eos_os_use_posix(void) {
    return eos_os_provide(&k_posix_shim);
}

#else /* !EOSLLM_HAVE_POSIX */

eos_status_t eos_os_use_posix(void) {
    return EOS_E_UNSUPPORTED;
}

#endif
