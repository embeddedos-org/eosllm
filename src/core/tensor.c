/*
 * src/core/tensor.c — dtype size table and tensor accessors that read
 * from the internal eos_tensor struct populated by format readers.
 */
#include <stddef.h>

#include "eosllm/tensor.h"
#include "eosllm/eosllm.h"

#include "model_internal.h"

size_t eos_dtype_elem_size(eos_dtype_t dt) {
    switch (dt) {
        case EOS_DT_F32:  return 4;
        case EOS_DT_F16:  return 2;
        case EOS_DT_BF16: return 2;
        case EOS_DT_I32:  return 4;
        case EOS_DT_I8:   return 1;
        case EOS_DT_U8:   return 1;
        /* Quantized dtypes have no fixed per-element size; the quant
         * scheme owns the layout (elements_per_block / bytes_per_block). */
        default:          return 0;
    }
}

eos_status_t eos_tensor_shape(const eos_tensor_t *t,
                              uint32_t            out_dims[EOS_MAX_RANK],
                              uint32_t           *out_rank) {
    uint32_t i;
    if (t == NULL || out_dims == NULL || out_rank == NULL)
        return EOS_E_INVALID_ARG;
    for (i = 0; i < EOS_MAX_RANK; ++i) out_dims[i] = (i < t->rank) ? t->dims[i] : 0;
    *out_rank = t->rank;
    return EOS_OK;
}

eos_dtype_t eos_tensor_dtype(const eos_tensor_t *t) {
    return (t != NULL) ? t->dtype : EOS_DT_F32;
}

const char *eos_tensor_name(const eos_tensor_t *t) {
    return (t != NULL) ? t->name : NULL;
}

/* Internal accessor: raw data pointer (used by the graph executor). */
const void *eosi_tensor_data(const eos_tensor_t *t) {
    return (t != NULL) ? t->data : NULL;
}
