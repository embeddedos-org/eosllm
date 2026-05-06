/*
 * src/format/eosm/eosm_internal.h — internal entry points for the
 * .eosm v1 reader and writer (see docs/file_format.md).
 *
 * Visible only inside src/. Not part of the public ABI.
 */
#ifndef EOSI_FORMAT_EOSM_INTERNAL_H
#define EOSI_FORMAT_EOSM_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "eosllm/eosllm.h"
#include "eosllm/tensor.h"

/* Magic bytes "EOSM" little-endian. */
#define EOSI_EOSM_MAGIC   0x4D534F45u
#define EOSI_EOSM_VERSION 1u

/* KV value-type codes (mirror docs/file_format.md §5). */
#define EOSI_EOSM_KV_U8        0u
#define EOSI_EOSM_KV_I8        1u
#define EOSI_EOSM_KV_U16       2u
#define EOSI_EOSM_KV_I16       3u
#define EOSI_EOSM_KV_U32       4u
#define EOSI_EOSM_KV_I32       5u
#define EOSI_EOSM_KV_F32       6u
#define EOSI_EOSM_KV_BOOL      7u
#define EOSI_EOSM_KV_STRING    8u
#define EOSI_EOSM_KV_ARRAY     9u
#define EOSI_EOSM_KV_U64       10u
#define EOSI_EOSM_KV_I64       11u
#define EOSI_EOSM_KV_F64       12u
#define EOSI_EOSM_KV_BLOB      13u

/* Capability bits (mirror docs/file_format.md §4). */
#define EOSI_EOSM_CAP_QUANT_Q8_0           (1ull <<  0)
#define EOSI_EOSM_CAP_QUANT_Q4_K           (1ull <<  1)
#define EOSI_EOSM_CAP_QUANT_Q2_K           (1ull <<  2)
#define EOSI_EOSM_CAP_QUANT_Q1_58          (1ull <<  3)
#define EOSI_EOSM_CAP_QUANT_MIXED          (1ull <<  4)
#define EOSI_EOSM_CAP_QUANT_CALIBRATED     (1ull <<  5)
#define EOSI_EOSM_CAP_MODALITY_TEXT        (1ull <<  8)
#define EOSI_EOSM_CAP_MODALITY_VISION      (1ull <<  9)
#define EOSI_EOSM_CAP_MODALITY_AUDIO       (1ull << 10)
#define EOSI_EOSM_CAP_FUSION_ADAPTERS      (1ull << 16)
#define EOSI_EOSM_CAP_LOAD_GROUPS_GT_1     (1ull << 24)
#define EOSI_EOSM_CAP_HASH_SHA256_TRAILER  (1ull << 32)

/* Calibration-table type codes. */
#define EOSI_EOSM_CALIB_NONE                  0u
#define EOSI_EOSM_CALIB_AWQ_PER_GROUP_F16     1u
#define EOSI_EOSM_CALIB_GPTQ_RESIDUAL_F16     2u
#define EOSI_EOSM_CALIB_BITNET_TERNARY_MASK   3u

#define EOSI_EOSM_HEADER_BYTES 64

/* ----------- Writer input layout (caller fills, writer copies). ----- */

typedef struct {
    const char *key;
    uint64_t    key_len;
    uint32_t    value_type;          /* EOSI_EOSM_KV_* */
    /* Discriminated by value_type; caller sets the appropriate field. */
    uint8_t     u8;  int8_t  i8;
    uint16_t    u16; int16_t i16;
    uint32_t    u32; int32_t i32;
    uint64_t    u64; int64_t i64;
    float       f32; double  f64;
    uint8_t     boolean;
    /* For STRING / BLOB: bytes + len. For ARRAY: elem_type + count + raw element bytes. */
    const void *bytes;
    uint64_t    bytes_len;
    uint32_t    arr_elem_type;
    uint64_t    arr_count;
} eosi_eosm_kv_in_t;

typedef struct {
    const char *name;
    uint64_t    name_len;
    uint32_t    dtype;                 /* eos_dtype_t value */
    uint32_t    quant_scheme_id;       /* 0 = none */
    uint32_t    calibration_table_idx; /* 0xFFFFFFFF = none */
    uint8_t     rank;                  /* 1..EOS_MAX_RANK */
    uint8_t     load_group_id;
    uint64_t    dims[EOS_MAX_RANK];
    const void *data;                  /* tensor body bytes */
    uint64_t    data_bytes;
} eosi_eosm_tensor_in_t;

typedef struct {
    uint32_t    table_type;            /* EOSI_EOSM_CALIB_* */
    const void *data;
    uint64_t    data_bytes;
} eosi_eosm_calib_in_t;

typedef struct {
    uint8_t     id;
    const char *name;
    uint64_t    name_len;
} eosi_eosm_group_in_t;

typedef struct {
    uint64_t                       capability_bits;
    uint32_t                       alignment;        /* power of two, 32..65536 */
    const eosi_eosm_kv_in_t       *kvs;
    uint64_t                       n_kvs;
    const eosi_eosm_tensor_in_t   *tensors;
    uint64_t                       n_tensors;
    const eosi_eosm_calib_in_t    *calib;
    uint64_t                       n_calib;
    const eosi_eosm_group_in_t    *groups;
    uint64_t                       n_groups;
} eosi_eosm_writer_in_t;

/*
 * Compute the layout, allocate a buffer via eosi_alloc, and emit a
 * v1-conformant .eosm file with SHA-256 trailer. Caller frees via
 * eosi_free. Tensor data within each load group is aligned to
 * `in->alignment`. The order of tensors in `in->tensors` is preserved
 * within each load group (caller is responsible for grouping by id).
 */
eos_status_t eosi_eosm_write(const eosi_eosm_writer_in_t *in,
                             void **out_buf, size_t *out_size);

/* ----- Converter: any eos_model_t -> v1 .eosm -------------------- */

/*
 * Caller supplies a list of metadata keys to copy (with their KV
 * types so the converter knows which model accessor to call). Tensors
 * are enumerated automatically via the model's num_tensors / tensor
 * accessors and copied byte-for-byte into a single load group named
 * "default" (id = 0). Quant scheme id and calibration table idx are
 * left at zero / 0xFFFFFFFF respectively (lossless pass-through of
 * raw / GGUF-style quants; no calibration repacking).
 *
 * Currently supported KV types in this converter:
 *   STRING, U64, F32. Other types can be added with an additional
 *   accessor; today's GGUF reader only exposes those three through
 *   the public model API.
 *
 * Returns the same buffer/size convention as eosi_eosm_write.
 */
typedef struct {
    const char *key;
    uint32_t    type;            /* EOSI_EOSM_KV_* */
    uint32_t    arr_elem_type;   /* only meaningful when type == ARRAY */
} eosi_eosm_key_spec_t;

eos_status_t eosi_eosm_from_model(eos_model_t                *src,
                                  const eosi_eosm_key_spec_t *specs,
                                  size_t                       n_specs,
                                  uint64_t                     extra_capability_bits,
                                  uint32_t                     alignment,
                                  void                       **out_buf,
                                  size_t                      *out_size);

#endif /* EOSI_FORMAT_EOSM_INTERNAL_H */
