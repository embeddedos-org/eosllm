/*
 * src/core/model_internal.h — internal model dispatch.
 *
 * Each file-format reader (gguf, eosm) returns an eos_model_t whose
 * `ops` field points at the format's dispatch table and `impl` field
 * points at the format-specific state. The public eos_model_*
 * accessors in src/core/model.c forward through ops.
 */
#ifndef EOSI_MODEL_INTERNAL_H
#define EOSI_MODEL_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "eosllm/eosllm.h"
#include "eosllm/model.h"
#include "eosllm/tensor.h"

typedef struct eosi_model_ops {
    void         (*close)         (void *impl);
    size_t       (*num_tensors)   (const void *impl);
    const eos_tensor_t *(*tensor) (const void *impl, size_t i);
    const eos_tensor_t *(*tensor_by_name)(const void *impl, const char *name);
    const char  *(*meta_str)      (const void *impl, const char *key);
    eos_status_t (*meta_u64)      (const void *impl, const char *key,
                                   uint64_t *out_value);
    eos_status_t (*meta_f32)      (const void *impl, const char *key,
                                   float *out_value);
    /*
     * Returns the count of elements in an array-typed metadata value.
     * For string arrays, copies pointers to up to *io_n entries into
     * out_strs. For numeric arrays, copies up to *io_n values.
     */
    eos_status_t (*meta_arr_str)  (const void *impl, const char *key,
                                   const char **out_strs, size_t *io_n);
    eos_status_t (*meta_arr_u32)  (const void *impl, const char *key,
                                   uint32_t *out_vals, size_t *io_n);
} eosi_model_ops_t;

struct eos_model {
    const eosi_model_ops_t *ops;
    void                   *impl;
};

/*
 * Internal tensor accessor structure shared by all formats. A tensor
 * is identified by its dtype, shape, name, and a pointer to its raw
 * bytes within the format-owned data blob.
 */
struct eos_tensor {
    const char *name;
    eos_dtype_t dtype;
    uint32_t    rank;
    uint32_t    dims[EOS_MAX_RANK];
    const void *data;       /* points into the model's owned buffer */
    size_t      data_bytes;
};

/* Internal: raw data pointer accessor used by the graph executor. */
const void *eosi_tensor_data(const eos_tensor_t *t);

#endif /* EOSI_MODEL_INTERNAL_H */
