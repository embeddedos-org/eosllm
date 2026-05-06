/*
 * eosllm.h — top-level public ABI for the eosllm inference engine.
 *
 * Conformance: ISO C99. No compiler-specific extensions in this header.
 * Include this header (and only this header) from host code:
 *
 *     #include <eosllm/eosllm.h>
 *
 * Other headers in eosllm/ are pulled in transitively as needed; you
 * may also include them directly when registering a module.
 */
#ifndef EOSLLM_EOSLLM_H
#define EOSLLM_EOSLLM_H
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Version & ABI                                                       */
/* ------------------------------------------------------------------ */

#define EOSLLM_VERSION_MAJOR 0
#define EOSLLM_VERSION_MINOR 1
#define EOSLLM_VERSION_PATCH 0

/* Bumped on every breaking ABI change while VERSION_MAJOR == 0. */
#define EOSLLM_ABI_VERSION   1

/* Returns EOSLLM_ABI_VERSION compiled into the library. */
int eos_abi_version(void);

/* Returns a stable "M.m.p" string compiled into the library. */
const char *eos_version_string(void);

/* ------------------------------------------------------------------ */
/* Compiler hints (always defined; expand to nothing on unknown compilers) */
/* ------------------------------------------------------------------ */

#if defined(__GNUC__) || defined(__clang__)
#  define EOS_LIKELY(x)   __builtin_expect(!!(x), 1)
#  define EOS_UNLIKELY(x) __builtin_expect(!!(x), 0)
#  define EOS_INLINE      static inline __attribute__((always_inline))
#  define EOS_NORETURN    __attribute__((noreturn))
#else
#  define EOS_LIKELY(x)   (x)
#  define EOS_UNLIKELY(x) (x)
#  define EOS_INLINE      static inline
#  define EOS_NORETURN
#endif

/* ------------------------------------------------------------------ */
/* Error codes                                                         */
/* ------------------------------------------------------------------ */

/*
 * Return value from every entry point. The library never aborts, never
 * calls exit(), and never longjmps. Out-parameters are written only on
 * EOS_OK. New codes are appended; hosts must treat unknown codes as
 * EOS_E_INTERNAL.
 */
typedef enum eos_status {
    EOS_OK                  = 0,
    EOS_E_INVALID_ARG       = 1,
    EOS_E_OUT_OF_MEMORY     = 2,
    EOS_E_UNSUPPORTED       = 3,  /* capability not compiled in */
    EOS_E_FORMAT            = 4,  /* malformed model file */
    EOS_E_VERSION_MISMATCH  = 5,
    EOS_E_NOT_FOUND         = 6,
    EOS_E_DEADLINE          = 7,  /* eos_session_step ran out of time */
    EOS_E_IO                = 8,
    EOS_E_INTERNAL          = 9   /* invariant violation; report a bug */
} eos_status_t;

const char *eos_status_str(eos_status_t s);

/* ------------------------------------------------------------------ */
/* Capability bits                                                     */
/* ------------------------------------------------------------------ */

/*
 * eos_caps() lets a host inspect what the library was compiled with.
 * Each field is either 0 (absent) or 1 (present). New fields are
 * appended; hosts must zero-initialize the struct before calling.
 */
typedef struct eos_caps {
    /* OS shims */
    unsigned have_posix      : 1;
    unsigned have_zephyr     : 1;
    unsigned have_freertos   : 1;
    unsigned have_baremetal  : 1;

    /* Threading */
    unsigned have_threads    : 1;

    /* Backends (kernels) */
    unsigned have_k_scalar   : 1;
    unsigned have_k_avx2     : 1;
    unsigned have_k_avx512   : 1;
    unsigned have_k_neon     : 1;
    unsigned have_k_sve      : 1;
    unsigned have_k_rvv      : 1;
    unsigned have_k_hvx      : 1;
    unsigned have_k_npu      : 1;

    /* Quant schemes */
    unsigned have_q_q8_0     : 1;
    unsigned have_q_q4_k     : 1;
    unsigned have_q_q2_k     : 1;
    unsigned have_q_q1_58    : 1;
    unsigned have_q_mixed    : 1;
    unsigned have_q_calibrtd : 1;

    /* Modalities */
    unsigned have_m_text     : 1;
    unsigned have_m_vision   : 1;
    unsigned have_m_audio    : 1;

    /* Tokenizers */
    unsigned have_t_bpe      : 1;
    unsigned have_t_spm      : 1;

    /* File formats */
    unsigned have_f_eosm     : 1;
    unsigned have_f_gguf     : 1;

    /* Schedulers */
    unsigned have_s_greedy   : 1;
    unsigned have_s_deadline : 1;
    unsigned have_s_batched  : 1;

    /* On-disk format max version supported. 0 if no format compiled in. */
    uint16_t eosm_max_version;

    /*
     * Runtime-detected backend availability bits. The corresponding
     * EOSLLM_HAVE_KERNEL_* macro tells you what was COMPILED IN; this
     * byte tells you what the host CPU's runtime probe ALSO accepts
     * (e.g. AVX2 was compiled in but the actual silicon doesn't have
     * it). Populated by eos_caps() via eos_backend_available(). Use
     * the EOS_CAPS_RT_* masks below to test individual bits.
     */
    uint8_t  runtime_bits;

    /* Reserved for future capabilities; always zero today. */
    uint8_t  _reserved[5];
} eos_caps_t;

/* Bit masks for eos_caps_t::runtime_bits. */
#define EOS_CAPS_RT_AVX2    (1u << 0)
#define EOS_CAPS_RT_AVX512  (1u << 1)
#define EOS_CAPS_RT_NEON    (1u << 2)
#define EOS_CAPS_RT_SVE     (1u << 3)
#define EOS_CAPS_RT_RVV     (1u << 4)
#define EOS_CAPS_RT_HVX     (1u << 5)
#define EOS_CAPS_RT_NPU     (1u << 6)

/* Fills *out with capability bits for the linked library build. */
eos_status_t eos_caps(eos_caps_t *out);

/*
 * Returns 1 if a registered backend with the given name probes
 * successfully on the current CPU (i.e. it is both compiled in AND
 * the runtime ISA features it needs are present). Returns 0 if not
 * registered, probe failed, or name is NULL. Use to decide
 * "should I rely on AVX2 here" rather than only "was AVX2 compiled in".
 *
 * Names: "scalar" (always 1), "x86_avx2", "x86_avx512", "arm_neon",
 * "arm_sve", "rvv", "hvx", "npu". Adding a backend extends this set.
 */
int eos_backend_available(const char *name);

/*
 * Returns the most recently recorded human-readable error string for
 * the calling thread, or NULL if no error has been recorded since the
 * thread was created (or since the last successful eos_* call that
 * cleared the slot — readers may clear opportunistically; do not rely
 * on the buffer surviving an EOS_OK return).
 *
 * The returned pointer's contents are valid until the next eos_*
 * call on the same thread; copy the bytes if you need to keep them.
 *
 * Lifetime / thread-safety: the buffer is thread-local where the
 * compiler supports `__thread` (gcc / clang) and falls back to a
 * single static buffer otherwise. New compilers that need a
 * different storage class can be added to src/core/error.c.
 *
 * The string carries the same information as `EOSI_LOG_ERROR(...)`
 * messages (e.g. "gguf: tensor has rank 0 or > EOS_MAX_RANK"), so
 * hosts that don't install a log shim can still surface why a load
 * failed.
 */
const char *eos_last_error(void);

/* ------------------------------------------------------------------ */
/* Modalities (also referenced by modality.h)                          */
/* ------------------------------------------------------------------ */

typedef enum eos_modality_kind {
    EOS_MODALITY_TEXT   = 0,
    EOS_MODALITY_VISION = 1,
    EOS_MODALITY_AUDIO  = 2
} eos_modality_kind_t;

/* ------------------------------------------------------------------ */
/* Opaque handles                                                      */
/* ------------------------------------------------------------------ */

typedef struct eos_model     eos_model_t;
typedef struct eos_session   eos_session_t;
typedef struct eos_tensor    eos_tensor_t;
typedef struct eos_tokenizer eos_tokenizer_t;

/* ------------------------------------------------------------------ */
/* Session lifecycle                                                   */
/* ------------------------------------------------------------------ */

/*
 * Sampling and runtime parameters supplied at session_open. The host
 * must zero-initialize this struct (e.g. = {0}) before populating; new
 * fields are appended in future versions and zero must mean "default".
 */
typedef struct eos_session_params {
    /* Maximum input + generated context, in tokens. 0 ⇒ model default. */
    uint32_t max_context;

    /* Bytes the engine may use for its arena. 0 ⇒ heuristic. */
    size_t   arena_bytes;

    /* Sampling. */
    float    temperature;   /* 0.0 ⇒ greedy */
    uint32_t top_k;         /* 0 ⇒ disabled */
    float    top_p;         /* 0.0 ⇒ disabled */
    uint64_t rng_seed;

    /* Name of the scheduler policy to use; NULL ⇒ "greedy". */
    const char *scheduler;

    /* Reserved. */
    uint8_t  _reserved[16];
} eos_session_params_t;

/* Opens a session against a model. Single allocation site for the arena. */
eos_status_t eos_session_open(eos_model_t                *model,
                              const eos_session_params_t *params,
                              eos_session_t             **out_session);

/*
 * Feeds one chunk of input of the given modality. For TEXT, data is
 * UTF-8 bytes and len is byte count; the session's tokenizer encodes it.
 * For VISION/AUDIO, the modality module defines the layout.
 */
eos_status_t eos_session_feed(eos_session_t      *session,
                              eos_modality_kind_t modality,
                              const void         *data,
                              size_t              len);

/*
 * Advances one decoder step. Writes the next token id into *out_token.
 * If a deadline is set and elapses mid-step, returns EOS_E_DEADLINE
 * without writing *out_token; the host may call again to resume.
 */
eos_status_t eos_session_step(eos_session_t *session, uint32_t *out_token);

/* Sets a deadline relative to "now", in nanoseconds. 0 ⇒ no deadline. */
eos_status_t eos_session_set_deadline(eos_session_t *session,
                                      uint64_t       ns_from_now);

/* Convenience: install POSIX OS shim (where available) and register
 * every built-in backend, quant scheme, modality, tokenizer, file
 * format, and scheduler compiled into this library. Hosts that need
 * fine-grained control should call the individual eosi_*_register
 * entry points instead — but those are internal symbols not declared
 * here. Idempotent. */
eos_status_t eos_init_defaults(void);

/* Frees the session's arena and all derived resources. Idempotent on NULL. */
void eos_session_close(eos_session_t *session);

/*
 * Releases a buffer that the engine allocated (e.g. the .eosm bytes
 * returned by tools/eosllm-convert via eosi_eosm_from_model). The
 * pointer must have been returned by an engine call documented to
 * transfer ownership to the host. Calling on a host-allocated buffer,
 * a buffer the engine still owns, or a buffer already freed is
 * undefined behavior. Idempotent on NULL.
 *
 * Why a dedicated free: hosts cannot use plain libc free because the
 * engine routes allocations through the OS shim (eos_os_provide), so
 * the matching deallocator may not be free().
 */
void eos_free(void *ptr);

/* Decode a sequence of token ids back to UTF-8 bytes via the session's
 * tokenizer. Pass out_text=NULL to query just the required size. */
eos_status_t eos_session_decode(eos_session_t *session,
                                const uint32_t *ids, size_t n_ids,
                                char *out_text, size_t *io_len);

/* ------------------------------------------------------------------ */
/* Tokenizer registration                                              */
/* ------------------------------------------------------------------ */

/*
 * A tokenizer module implements this vtable and registers it with
 * eos_tokenizer_register(name, &vt). Encode writes up to *io_len ids
 * and updates *io_len to the count actually written; if the buffer is
 * too small, returns EOS_E_OUT_OF_MEMORY and writes the required size.
 */
typedef struct eos_tokenizer_vt {
    const char *name;
    eos_status_t (*open) (const void *blob, size_t blob_len, void **out_state);
    eos_status_t (*encode)(void *state, const char *text, size_t text_len,
                           uint32_t *out_ids, size_t *io_len);
    eos_status_t (*decode)(void *state, const uint32_t *ids, size_t n_ids,
                           char *out_text, size_t *io_len);
    void         (*close) (void *state);
} eos_tokenizer_vt_t;

eos_status_t eos_tokenizer_register(const eos_tokenizer_vt_t *vt);

#ifdef __cplusplus
} /* extern "C" */
#endif

/* Convenience: pull in the rest of the public ABI for users who just
 * want to register modules without remembering each header path. */
#include "tensor.h"
#include "model.h"

#endif /* EOSLLM_EOSLLM_H */
