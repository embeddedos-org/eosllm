/*
 * model.h — model file open / close / introspect, plus file-format
 * registration for adding readers (eosm native, gguf for bring-up).
 */
#ifndef EOSLLM_MODEL_H
#define EOSLLM_MODEL_H
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "eosllm.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Model lifecycle                                                     */
/* ------------------------------------------------------------------ */

/*
 * Opens a model file by path. Detection is by magic bytes; the first
 * registered file format whose probe accepts the file wins.
 */
eos_status_t eos_model_open (const char *path, eos_model_t **out_model);
void         eos_model_close(eos_model_t *model);

/* Number of tensors in the model. */
size_t       eos_model_num_tensors(const eos_model_t *model);

/* Returns the i-th tensor or NULL if i is out of range. The handle's
 * lifetime is bound to the model. */
const eos_tensor_t *eos_model_tensor(const eos_model_t *model, size_t i);

/* Returns NULL if the key is unknown. The string's lifetime is bound
 * to the model. Used for arch / vocab / metadata lookups. */
const char *eos_model_meta_str(const eos_model_t *model, const char *key);
eos_status_t eos_model_meta_u64(const eos_model_t *model, const char *key,
                                uint64_t *out_value);
eos_status_t eos_model_meta_f32(const eos_model_t *model, const char *key,
                                float *out_value);

/* Array metadata. Pass out=NULL to query just the element count. */
eos_status_t eos_model_meta_arr_str_count(const eos_model_t *model,
                                          const char *key, size_t *out_count);
eos_status_t eos_model_meta_arr_str_load (const eos_model_t *model,
                                          const char *key,
                                          const char **out_strs, size_t *io_n);
eos_status_t eos_model_meta_arr_u32_count(const eos_model_t *model,
                                          const char *key, size_t *out_count);
eos_status_t eos_model_meta_arr_u32_load (const eos_model_t *model,
                                          const char *key,
                                          uint32_t *out_vals, size_t *io_n);

/* Lookup a tensor by name (e.g. "blk.0.attn_q.weight"). Returns NULL
 * if not present. */
const eos_tensor_t *eos_model_tensor_by_name(const eos_model_t *model,
                                             const char *name);

/* ------------------------------------------------------------------ */
/* File-format registration                                            */
/* ------------------------------------------------------------------ */

/*
 * Stream the runtime reads model bytes through. Backed by eos_os_*
 * file ops in the default POSIX build, but the host may supply a
 * custom stream (e.g. mmap from flash on an MCU).
 */
typedef struct eos_stream {
    void *opaque;
    /* Read up to len bytes; returns count actually read or -1 on error. */
    int64_t (*read) (void *opaque, void *buf, size_t len);
    /* Seek to absolute offset; returns 0 or -1. */
    int     (*seek) (void *opaque, int64_t offset);
    /* Returns total length in bytes or -1 if unknown. */
    int64_t (*size) (void *opaque);
    /* Closes the stream. */
    void    (*close)(void *opaque);
} eos_stream_t;

/*
 * A file-format module implements this vtable. probe() reads up to a
 * few hundred bytes from the stream and returns EOS_OK if it can read
 * the file, EOS_E_FORMAT otherwise. open() parses metadata and tensor
 * descriptors; the actual tensor bytes are loaded lazily.
 */
typedef struct eos_format_vt {
    const char *name;
    eos_status_t (*probe)(eos_stream_t *stream);
    eos_status_t (*open) (eos_stream_t *stream, eos_model_t **out_model);
    void         (*close)(eos_model_t *model);
} eos_format_vt_t;

eos_status_t eos_format_register(const eos_format_vt_t *vt);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* EOSLLM_MODEL_H */
