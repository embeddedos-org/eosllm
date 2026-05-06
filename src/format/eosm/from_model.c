/*
 * src/format/eosm/from_model.c — convert any eos_model_t into a v1
 * .eosm buffer.
 *
 * The converter is intentionally format-agnostic. It uses only the
 * public model accessors (eos_model_meta_str / _u64 / _f32 and
 * eos_model_num_tensors / eos_model_tensor) plus the internal tensor
 * struct (for the data pointer + data_bytes). This means it converts
 * a GGUF model, a previously-written .eosm, or any in-memory mock
 * that satisfies eosi_model_ops_t.
 *
 * Caller passes a list of (key, kv_type) specs telling the converter
 * which metadata keys to copy. The list is the place to encode model-
 * format knowledge (e.g. which keys a Llama-3 GGUF carries) so the
 * converter itself stays format-blind.
 *
 * All tensors land in load group 0 ("default"). Multi-modal sharding
 * across load groups is a follow-up (would take an additional
 * "tensor name -> group_id" mapping callback).
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "eosllm/eosllm.h"
#include "eosllm/model.h"

#include "../../core/internal.h"
#include "../../core/model_internal.h"
#include "eosm_internal.h"

#if EOSLLM_HAVE_FORMAT_EOSM

eos_status_t eosi_eosm_from_model(eos_model_t                *src,
                                  const eosi_eosm_key_spec_t *specs,
                                  size_t                       n_specs,
                                  uint64_t                     extra_capability_bits,
                                  uint32_t                     alignment,
                                  void                       **out_buf,
                                  size_t                      *out_size) {
    static const char k_default_group_name[] = "default";

    eosi_eosm_kv_in_t     *kvs     = NULL;
    eosi_eosm_tensor_in_t *tensors = NULL;
    eosi_eosm_group_in_t   group;
    eosi_eosm_writer_in_t  in;
    /* Per-spec scratch buffer for ARRAY entries (must outlive the
     * eosi_eosm_write call). NULL for non-ARRAY specs. Freed in
     * cleanup. */
    void                 **arr_scratch = NULL;
    size_t                 n_tensors;
    size_t                 i;
    eos_status_t           rc = EOS_E_INVALID_ARG;

    if (src == NULL || out_buf == NULL || out_size == NULL)
        return EOS_E_INVALID_ARG;
    if (n_specs > 0 && specs == NULL)
        return EOS_E_INVALID_ARG;
    *out_buf = NULL; *out_size = 0;

    /* ----- Build kvs[] from specs ----- */
    if (n_specs > 0) {
        kvs = (eosi_eosm_kv_in_t *)eosi_alloc(
            n_specs * sizeof(eosi_eosm_kv_in_t), 8);
        arr_scratch = (void **)eosi_alloc(n_specs * sizeof(void *), 8);
        if (kvs == NULL || arr_scratch == NULL) {
            rc = EOS_E_OUT_OF_MEMORY; goto cleanup;
        }
        memset(kvs, 0, n_specs * sizeof(eosi_eosm_kv_in_t));
        memset(arr_scratch, 0, n_specs * sizeof(void *));
        for (i = 0; i < n_specs; ++i) {
            eosi_eosm_kv_in_t *kv = &kvs[i];
            if (specs[i].key == NULL) { rc = EOS_E_INVALID_ARG; goto cleanup; }
            kv->key        = specs[i].key;
            kv->key_len    = (uint64_t)strlen(specs[i].key);
            kv->value_type = specs[i].type;
            switch (specs[i].type) {
                case EOSI_EOSM_KV_STRING: {
                    const char *s = eos_model_meta_str(src, specs[i].key);
                    if (s == NULL) { rc = EOS_E_NOT_FOUND; goto cleanup; }
                    kv->bytes     = s;
                    kv->bytes_len = (uint64_t)strlen(s);
                } break;
                case EOSI_EOSM_KV_U64: {
                    uint64_t v;
                    rc = eos_model_meta_u64(src, specs[i].key, &v);
                    if (rc != EOS_OK) goto cleanup;
                    kv->u64 = v;
                } break;
                case EOSI_EOSM_KV_U32: {
                    uint64_t v;
                    rc = eos_model_meta_u64(src, specs[i].key, &v);
                    if (rc != EOS_OK) goto cleanup;
                    if (v > 0xFFFFFFFFu) { rc = EOS_E_INVALID_ARG; goto cleanup; }
                    kv->u32 = (uint32_t)v;
                } break;
                case EOSI_EOSM_KV_F32: {
                    float v;
                    rc = eos_model_meta_f32(src, specs[i].key, &v);
                    if (rc != EOS_OK) goto cleanup;
                    kv->f32 = v;
                } break;
                case EOSI_EOSM_KV_ARRAY: {
                    /* STRING arrays: serialize each (u64 len + bytes)
                     * into a scratch buffer. U32 arrays: fetch into a
                     * count*4 scratch buffer. Both buffers outlive
                     * the writer call (freed in cleanup). */
                    if (specs[i].arr_elem_type == EOSI_EOSM_KV_STRING) {
                        size_t n = 0, total = 0, j;
                        const char **strs = NULL;
                        uint8_t *out = NULL, *p;
                        rc = eos_model_meta_arr_str_count(src, specs[i].key, &n);
                        if (rc != EOS_OK) goto cleanup;
                        strs = (const char **)eosi_alloc(n * sizeof(char *), 8);
                        if (strs == NULL) { rc = EOS_E_OUT_OF_MEMORY; goto cleanup; }
                        {
                            size_t got = n;
                            rc = eos_model_meta_arr_str_load(
                                src, specs[i].key, strs, &got);
                            if (rc != EOS_OK) { eosi_free(strs); goto cleanup; }
                        }
                        for (j = 0; j < n; ++j) {
                            total += 8u + (strs[j] != NULL ? strlen(strs[j]) : 0u);
                        }
                        out = (uint8_t *)eosi_alloc(total > 0 ? total : 1, 8);
                        if (out == NULL) { eosi_free(strs); rc = EOS_E_OUT_OF_MEMORY; goto cleanup; }
                        p = out;
                        for (j = 0; j < n; ++j) {
                            uint64_t l = (strs[j] != NULL) ? (uint64_t)strlen(strs[j]) : 0u;
                            memcpy(p, &l, 8); p += 8;
                            if (l > 0) { memcpy(p, strs[j], (size_t)l); p += l; }
                        }
                        eosi_free(strs);
                        arr_scratch[i] = out;
                        kv->arr_elem_type = EOSI_EOSM_KV_STRING;
                        kv->arr_count     = (uint64_t)n;
                        kv->bytes         = out;
                        kv->bytes_len     = (uint64_t)total;
                    } else if (specs[i].arr_elem_type == EOSI_EOSM_KV_U32) {
                        size_t n = 0;
                        uint32_t *out = NULL;
                        rc = eos_model_meta_arr_u32_count(src, specs[i].key, &n);
                        if (rc != EOS_OK) goto cleanup;
                        out = (uint32_t *)eosi_alloc(
                            (n > 0 ? n : 1) * sizeof(uint32_t), 4);
                        if (out == NULL) { rc = EOS_E_OUT_OF_MEMORY; goto cleanup; }
                        {
                            size_t got = n;
                            rc = eos_model_meta_arr_u32_load(
                                src, specs[i].key, out, &got);
                            if (rc != EOS_OK) { eosi_free(out); goto cleanup; }
                        }
                        arr_scratch[i] = out;
                        kv->arr_elem_type = EOSI_EOSM_KV_U32;
                        kv->arr_count     = (uint64_t)n;
                        kv->bytes         = out;
                        kv->bytes_len     = (uint64_t)n * 4u;
                    } else {
                        /* Other element types: extend as needed. */
                        rc = EOS_E_UNSUPPORTED; goto cleanup;
                    }
                } break;
                default:
                    /* Other KV types (BOOL, signed ints, F64, ARRAY,
                     * BLOB) need additional accessors that the public
                     * model API does not yet expose. Add them as the
                     * need arises (e.g. arrays for
                     * tokenizer.ggml.tokens). */
                    rc = EOS_E_UNSUPPORTED; goto cleanup;
            }
        }
    }

    /* ----- Build tensors[] from src ----- */
    n_tensors = eos_model_num_tensors(src);
    if (n_tensors > 0) {
        tensors = (eosi_eosm_tensor_in_t *)eosi_alloc(
            n_tensors * sizeof(eosi_eosm_tensor_in_t), 8);
        if (tensors == NULL) { rc = EOS_E_OUT_OF_MEMORY; goto cleanup; }
        memset(tensors, 0, n_tensors * sizeof(eosi_eosm_tensor_in_t));
        for (i = 0; i < n_tensors; ++i) {
            const eos_tensor_t *t = eos_model_tensor(src, i);
            uint32_t r;
            if (t == NULL || t->name == NULL || t->data == NULL) {
                rc = EOS_E_FORMAT; goto cleanup;
            }
            tensors[i].name                  = t->name;
            tensors[i].name_len              = (uint64_t)strlen(t->name);
            tensors[i].dtype                 = (uint32_t)t->dtype;
            tensors[i].quant_scheme_id       = 0;
            tensors[i].calibration_table_idx = 0xFFFFFFFFu;
            tensors[i].rank                  = (uint8_t)t->rank;
            tensors[i].load_group_id         = 0;
            for (r = 0; r < t->rank && r < EOS_MAX_RANK; ++r) {
                tensors[i].dims[r] = t->dims[r];
            }
            tensors[i].data       = t->data;
            tensors[i].data_bytes = (uint64_t)t->data_bytes;
        }
    }

    /* ----- Single default load group ----- */
    memset(&group, 0, sizeof(group));
    group.id       = 0;
    group.name     = k_default_group_name;
    group.name_len = sizeof(k_default_group_name) - 1u;

    /* ----- Wire writer input ----- */
    memset(&in, 0, sizeof(in));
    in.capability_bits = EOSI_EOSM_CAP_HASH_SHA256_TRAILER
                       | extra_capability_bits;
    in.alignment       = alignment != 0 ? alignment : 64u;
    in.kvs             = kvs;     in.n_kvs     = n_specs;
    in.tensors         = tensors; in.n_tensors = n_tensors;
    in.calib           = NULL;    in.n_calib   = 0;
    in.groups          = &group;  in.n_groups  = 1;

    rc = eosi_eosm_write(&in, out_buf, out_size);

cleanup:
    if (arr_scratch != NULL) {
        size_t k;
        for (k = 0; k < n_specs; ++k) eosi_free(arr_scratch[k]);
        eosi_free(arr_scratch);
    }
    eosi_free(kvs);
    eosi_free(tensors);
    return rc;
}

#else
eos_status_t eosi_eosm_from_model(eos_model_t                *src,
                                  const eosi_eosm_key_spec_t *specs,
                                  size_t                       n_specs,
                                  uint64_t                     extra_capability_bits,
                                  uint32_t                     alignment,
                                  void                       **out_buf,
                                  size_t                      *out_size) {
    (void)src; (void)specs; (void)n_specs; (void)extra_capability_bits;
    (void)alignment; (void)out_buf; (void)out_size;
    return EOS_E_UNSUPPORTED;
}
#endif
