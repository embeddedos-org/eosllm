/*
 * src/os/win32.c — Windows implementation of the eos_os_* shim.
 *
 * Compiled in only when EOSLLM_HAVE_WIN32=1. Calling
 * eosi_os_use_win32() installs this shim into the registry. The
 * vtable shape matches src/os/posix.c exactly; only the underlying
 * primitives differ (_aligned_malloc/_aligned_free,
 * QueryPerformanceCounter, _fseeki64).
 *
 * On non-Windows builds the symbol resolves to EOS_E_UNSUPPORTED so
 * the rest of the engine can reference it unconditionally without
 * link errors.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eosllm/eosllm.h"
#include "eosllm/os.h"
#include "os_internal.h"

#if EOSLLM_HAVE_WIN32 && (defined(_WIN32) || defined(_WIN64))

#include <malloc.h>     /* _aligned_malloc / _aligned_free */
#include <windows.h>    /* QueryPerformanceCounter / Frequency */

static void *win32_alloc(size_t bytes, size_t alignment) {
    if (alignment < sizeof(void *)) alignment = sizeof(void *);
    /* _aligned_malloc requires alignment to be a power of two; the
     * runtime always passes a sane value (16 or 64). */
    return _aligned_malloc(bytes, alignment);
}

static void win32_free(void *ptr) {
    /* _aligned_free is the matching deallocator for _aligned_malloc;
     * passing one to the other's deallocator is undefined on MSVCRT. */
    if (ptr != NULL) _aligned_free(ptr);
}

static uint64_t win32_now_ns(void) {
    /* QueryPerformanceCounter returns ticks at QueryPerformanceFrequency
     * Hz. Convert to nanoseconds via 1e9 / freq. The frequency is
     * effectively constant for the process lifetime, so we cache it. */
    static LARGE_INTEGER freq = { { 0, 0 } };
    LARGE_INTEGER now;
    if (freq.QuadPart == 0) {
        if (!QueryPerformanceFrequency(&freq) || freq.QuadPart == 0) {
            return 0;
        }
    }
    if (!QueryPerformanceCounter(&now)) return 0;
    /* (now * 1e9) / freq, computed without overflow for the realistic
     * uptime range (1e9 ticks ≈ ~3 days at 10 MHz QPC frequency). */
    return (uint64_t)((double)now.QuadPart * 1.0e9 / (double)freq.QuadPart);
}

static int win32_file_open(const char *path, void **out_handle) {
    FILE *f;
    if (path == NULL || out_handle == NULL) return -1;
    f = fopen(path, "rb");
    if (f == NULL) return -1;
    *out_handle = f;
    return 0;
}

static int64_t win32_file_read(void *handle, void *buf, size_t len) {
    size_t n;
    if (handle == NULL || buf == NULL) return -1;
    n = fread(buf, 1, len, (FILE *)handle);
    if (n == 0 && ferror((FILE *)handle)) return -1;
    return (int64_t)n;
}

static int win32_file_seek(void *handle, int64_t offset) {
    if (handle == NULL) return -1;
    /* _fseeki64 supports >2 GiB models, unlike fseek(long). */
    return _fseeki64((FILE *)handle, offset, SEEK_SET) == 0 ? 0 : -1;
}

static int64_t win32_file_size(void *handle) {
    int64_t here, end;
    if (handle == NULL) return -1;
    here = _ftelli64((FILE *)handle);
    if (here < 0) return -1;
    if (_fseeki64((FILE *)handle, 0, SEEK_END) != 0) return -1;
    end = _ftelli64((FILE *)handle);
    if (_fseeki64((FILE *)handle, here, SEEK_SET) != 0) return -1;
    return end;
}

static void win32_file_close(void *handle) {
    if (handle != NULL) fclose((FILE *)handle);
}

static void win32_log(int level, const char *msg) {
    static const char *prefix[4] = { "[debug] ", "[info]  ", "[warn]  ", "[error] " };
    const char *p = (level >= 0 && level <= 3) ? prefix[level] : "[?]     ";
    fputs(p, stderr);
    fputs(msg ? msg : "", stderr);
    fputc('\n', stderr);
}

static const eos_os_shim_t k_win32_shim = {
    win32_alloc,
    win32_free,
    win32_now_ns,
    win32_file_open,
    win32_file_read,
    win32_file_seek,
    win32_file_size,
    win32_file_close,
    win32_log
};

eos_status_t eosi_os_use_win32(void) {
    return eos_os_provide(&k_win32_shim);
}

#else /* !EOSLLM_HAVE_WIN32 || not on Windows */

eos_status_t eosi_os_use_win32(void) {
    return EOS_E_UNSUPPORTED;
}

#endif
