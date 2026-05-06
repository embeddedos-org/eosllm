/*
 * src/format/eosm/writer.c — .eosm v1 writer.
 *
 * Two-pass:
 *   pass 1: walk the input, accumulate total file size, compute
 *           per-tensor offset (within its load group) and per-group
 *           data_bytes + data_offset (in the file).
 *   pass 2: allocate one contiguous buffer, emit every section, then
 *           hash [0, file_end - 32) with SHA-256 and write the trailer.
 *
 * Layout matches docs/file_format.md §2 exactly.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "eosllm/eosllm.h"

#include "../../core/internal.h"
#include "../../util/sha256.h"
#include "eosm_internal.h"

#if EOSLLM_HAVE_FORMAT_EOSM

/* ------------------------------------------------------------------ */
/* Small write cursor                                                  */
/* ------------------------------------------------------------------ */

typedef struct {
    uint8_t *p;
    uint8_t *end;
    int      error;
} wcur_t;

static void wput(wcur_t *c, const void *src, size_t n) {
    if (c->error) return;
    if ((size_t)(c->end - c->p) < n) { c->error = 1; return; }
    memcpy(c->p, src, n);
    c->p += n;
}

static void wput_u8 (wcur_t *c, uint8_t  v) { wput(c, &v, 1); }
static void wput_u16(wcur_t *c, uint16_t v) { wput(c, &v, 2); }
static void wput_u32(wcur_t *c, uint32_t v) { wput(c, &v, 4); }
static void wput_u64(wcur_t *c, uint64_t v) { wput(c, &v, 8); }
static void wput_zero(wcur_t *c, size_t n) {
    if (c->error) return;
    if ((size_t)(c->end - c->p) < n) { c->error = 1; return; }
    memset(c->p, 0, n);
    c->p += n;
}

/* ------------------------------------------------------------------ */
/* Validation + sizing                                                 */
/* ------------------------------------------------------------------ */

static int is_pow2(uint32_t v) { return v != 0u && (v & (v - 1u)) == 0u; }

static size_t kv_value_bytes(const eosi_eosm_kv_in_t *kv) {
    switch (kv->value_type) {
        case EOSI_EOSM_KV_U8: case EOSI_EOSM_KV_I8:
        case EOSI_EOSM_KV_BOOL:                                  return 1;
        case EOSI_EOSM_KV_U16: case EOSI_EOSM_KV_I16:            return 2;
        case EOSI_EOSM_KV_U32: case EOSI_EOSM_KV_I32:
        case EOSI_EOSM_KV_F32:                                   return 4;
        case EOSI_EOSM_KV_U64: case EOSI_EOSM_KV_I64:
        case EOSI_EOSM_KV_F64:                                   return 8;
        case EOSI_EOSM_KV_STRING: case EOSI_EOSM_KV_BLOB:
            return 8 + (size_t)kv->bytes_len;
        case EOSI_EOSM_KV_ARRAY:
            /* 4 (elem_type) + 8 (count) + bytes_len (raw element data) */
            return 4 + 8 + (size_t)kv->bytes_len;
        default: return 0;
    }
}

/* Aligns `pos` up to a multiple of `align`. Caller must ensure align is pow2. */
static size_t align_up(size_t pos, size_t align) {
    return (pos + align - 1u) & ~(align - 1u);
}

/* ------------------------------------------------------------------ */
/* Writer entry                                                        */
/* ------------------------------------------------------------------ */

eos_status_t eosi_eosm_write(const eosi_eosm_writer_in_t *in,
                             void **out_buf, size_t *out_size) {
    size_t i, j;
    size_t cap_off, kv_off, manifest_off, calib_off, group_off;
    size_t cap_bytes, kv_bytes, manifest_bytes, calib_bytes, group_bytes;
    size_t data_off;
    size_t total;
    uint8_t *buf = NULL;
    /* Per-tensor offset (within its group) and per-group cumulative data bytes,
     * computed in pass 1. We allocate via eosi_alloc since these can be large. */
    uint64_t *t_offset = NULL;
    uint64_t *g_data_offset = NULL;
    uint64_t *g_data_bytes  = NULL;
    eos_status_t rc = EOS_E_INVALID_ARG;

    if (in == NULL || out_buf == NULL || out_size == NULL) return EOS_E_INVALID_ARG;
    *out_buf = NULL; *out_size = 0;

    if (!is_pow2(in->alignment) || in->alignment < 32u || in->alignment > 65536u)
        return EOS_E_INVALID_ARG;
    if (in->n_groups == 0) return EOS_E_INVALID_ARG;
    if (in->n_groups > 255u) return EOS_E_INVALID_ARG;
    /* Tensor sanity */
    for (i = 0; i < in->n_tensors; ++i) {
        const eosi_eosm_tensor_in_t *t = &in->tensors[i];
        if (t->name == NULL || t->name_len == 0) return EOS_E_INVALID_ARG;
        if (t->rank == 0 || t->rank > EOS_MAX_RANK) return EOS_E_INVALID_ARG;
        if (t->load_group_id >= in->n_groups) return EOS_E_INVALID_ARG;
    }

    /* ---- Pass 1: section sizes ---- */
    cap_off    = EOSI_EOSM_HEADER_BYTES;

    /* Capability section: number of bytes needed to fit the highest set
     * bit, rounded up to a byte. We always emit at least 8 bytes so the
     * common case (one u64) is a single fixed-size field. */
    cap_bytes = 8;

    /* Metadata KVs */
    kv_off    = cap_off + cap_bytes;
    kv_bytes  = 0;
    for (i = 0; i < in->n_kvs; ++i) {
        const eosi_eosm_kv_in_t *kv = &in->kvs[i];
        size_t vb;
        if (kv->key == NULL || kv->key_len == 0) goto bad_arg;
        kv_bytes += 8 + (size_t)kv->key_len + 4;     /* key_len + key + value_type */
        vb = kv_value_bytes(kv);
        if (vb == 0 && kv->value_type != EOSI_EOSM_KV_BOOL
                    && kv->value_type != EOSI_EOSM_KV_U8
                    && kv->value_type != EOSI_EOSM_KV_I8) goto bad_arg;
        kv_bytes += vb;
    }

    /* Tensor manifest */
    manifest_off = kv_off + kv_bytes;
    manifest_bytes = 0;
    for (i = 0; i < in->n_tensors; ++i) {
        const eosi_eosm_tensor_in_t *t = &in->tensors[i];
        manifest_bytes += 8 + (size_t)t->name_len  /* name_len + name */
                       + 4 + 4 + 4                 /* dtype + quant_id + calib_idx */
                       + 1 + 1 + 2                 /* rank + group_id + reserved[2] */
                       + (size_t)t->rank * 8u      /* dims[rank] u64 each */
                       + 8 + 8;                    /* offset + data_bytes */
    }

    /* Calibration tables */
    calib_off = manifest_off + manifest_bytes;
    calib_bytes = 0;
    for (i = 0; i < in->n_calib; ++i) {
        calib_bytes += 4 + 4 + 8 + (size_t)in->calib[i].data_bytes;
    }

    /* Load-group index */
    group_off = calib_off + calib_bytes;
    group_bytes = 0;
    for (i = 0; i < in->n_groups; ++i) {
        if (in->groups[i].name_len == 0 || in->groups[i].name == NULL) goto bad_arg;
        group_bytes += 1 + 7 + 8 + 8 + 8 + (size_t)in->groups[i].name_len;
    }

    /* Tensor data: per-group, aligned. */
    data_off = group_off + group_bytes;
    /* Align the start of every group's data block to alignment. */
    g_data_offset = (uint64_t *)eosi_alloc(in->n_groups * sizeof(uint64_t), 8);
    g_data_bytes  = (uint64_t *)eosi_alloc(in->n_groups * sizeof(uint64_t), 8);
    if (in->n_tensors > 0) {
        t_offset = (uint64_t *)eosi_alloc(in->n_tensors * sizeof(uint64_t), 8);
        if (t_offset == NULL) { rc = EOS_E_OUT_OF_MEMORY; goto cleanup; }
    }
    if (g_data_offset == NULL || g_data_bytes == NULL) {
        rc = EOS_E_OUT_OF_MEMORY; goto cleanup;
    }
    for (i = 0; i < in->n_groups; ++i) g_data_bytes[i] = 0;

    /* Walk tensors in caller order. For each tensor:
     *   - align local position within its group to alignment
     *   - record t_offset[i] = local position
     *   - bump local position by data_bytes
     * After the loop, each g_data_bytes[gid] holds the group's total. */
    for (i = 0; i < in->n_tensors; ++i) {
        const eosi_eosm_tensor_in_t *t = &in->tensors[i];
        uint8_t gid = t->load_group_id;
        uint64_t pos = g_data_bytes[gid];
        pos = (uint64_t)align_up((size_t)pos, in->alignment);
        t_offset[i] = pos;
        pos += t->data_bytes;
        g_data_bytes[gid] = pos;
    }

    /* Compute each group's absolute file offset. We also align each
     * group's data block to alignment in the file so per-tensor
     * absolute offsets land on aligned addresses. */
    {
        size_t cur = data_off;
        for (i = 0; i < in->n_groups; ++i) {
            cur = align_up(cur, in->alignment);
            g_data_offset[i] = cur;
            cur += g_data_bytes[i];
        }
        total = cur + 32;       /* trailing SHA-256 */
    }

    /* ---- Pass 2: emit ---- */
    buf = (uint8_t *)eosi_alloc(total, 16);
    if (buf == NULL) { rc = EOS_E_OUT_OF_MEMORY; goto cleanup; }
    memset(buf, 0, total);
    {
        wcur_t c = { buf, buf + total, 0 };

        /* Header */
        wput_u32 (&c, EOSI_EOSM_MAGIC);
        wput_u32 (&c, EOSI_EOSM_VERSION);
        wput_u64 (&c, (uint64_t)cap_bytes);
        wput_u64 (&c, in->n_kvs);
        wput_u64 (&c, in->n_tensors);
        wput_u64 (&c, in->n_calib);
        wput_u64 (&c, in->n_groups);
        wput_u32 (&c, in->alignment);
        wput_u32 (&c, 0);                  /* flags */
        wput_zero(&c, 8);                  /* reserved[8] */
        if ((size_t)(c.p - buf) != EOSI_EOSM_HEADER_BYTES) {
            rc = EOS_E_INTERNAL; goto cleanup;
        }

        /* Capability bits (8 bytes little-endian). */
        wput_u64(&c, in->capability_bits);

        /* Metadata KV */
        for (i = 0; i < in->n_kvs; ++i) {
            const eosi_eosm_kv_in_t *kv = &in->kvs[i];
            wput_u64(&c, kv->key_len);
            wput   (&c, kv->key, (size_t)kv->key_len);
            wput_u32(&c, kv->value_type);
            switch (kv->value_type) {
                case EOSI_EOSM_KV_U8:   wput_u8 (&c, kv->u8); break;
                case EOSI_EOSM_KV_I8:   wput    (&c, &kv->i8, 1); break;
                case EOSI_EOSM_KV_U16:  wput_u16(&c, kv->u16); break;
                case EOSI_EOSM_KV_I16:  wput    (&c, &kv->i16, 2); break;
                case EOSI_EOSM_KV_U32:  wput_u32(&c, kv->u32); break;
                case EOSI_EOSM_KV_I32:  wput    (&c, &kv->i32, 4); break;
                case EOSI_EOSM_KV_F32:  wput    (&c, &kv->f32, 4); break;
                case EOSI_EOSM_KV_BOOL: wput_u8 (&c, kv->boolean ? 1 : 0); break;
                case EOSI_EOSM_KV_U64:  wput_u64(&c, kv->u64); break;
                case EOSI_EOSM_KV_I64:  wput    (&c, &kv->i64, 8); break;
                case EOSI_EOSM_KV_F64:  wput    (&c, &kv->f64, 8); break;
                case EOSI_EOSM_KV_STRING:
                case EOSI_EOSM_KV_BLOB:
                    wput_u64(&c, kv->bytes_len);
                    wput   (&c, kv->bytes, (size_t)kv->bytes_len);
                    break;
                case EOSI_EOSM_KV_ARRAY:
                    wput_u32(&c, kv->arr_elem_type);
                    wput_u64(&c, kv->arr_count);
                    wput   (&c, kv->bytes, (size_t)kv->bytes_len);
                    break;
                default: rc = EOS_E_INVALID_ARG; goto cleanup;
            }
        }

        /* Tensor manifest */
        for (i = 0; i < in->n_tensors; ++i) {
            const eosi_eosm_tensor_in_t *t = &in->tensors[i];
            wput_u64(&c, t->name_len);
            wput   (&c, t->name, (size_t)t->name_len);
            wput_u32(&c, t->dtype);
            wput_u32(&c, t->quant_scheme_id);
            wput_u32(&c, t->calibration_table_idx);
            wput_u8 (&c, t->rank);
            wput_u8 (&c, t->load_group_id);
            wput_zero(&c, 2);
            for (j = 0; j < t->rank; ++j) wput_u64(&c, t->dims[j]);
            wput_u64(&c, t_offset[i]);
            wput_u64(&c, t->data_bytes);
        }

        /* Calibration tables */
        for (i = 0; i < in->n_calib; ++i) {
            const eosi_eosm_calib_in_t *cb = &in->calib[i];
            wput_u32(&c, cb->table_type);
            wput_u32(&c, 0);
            wput_u64(&c, cb->data_bytes);
            wput   (&c, cb->data, (size_t)cb->data_bytes);
        }

        /* Load-group index */
        for (i = 0; i < in->n_groups; ++i) {
            const eosi_eosm_group_in_t *g = &in->groups[i];
            wput_u8 (&c, g->id);
            wput_zero(&c, 7);
            wput_u64(&c, g_data_offset[i]);
            wput_u64(&c, g_data_bytes[i]);
            wput_u64(&c, g->name_len);
            wput   (&c, g->name, (size_t)g->name_len);
        }

        /* Pad up to first group's aligned data start. */
        if ((size_t)(c.p - buf) > g_data_offset[0]) {
            rc = EOS_E_INTERNAL; goto cleanup;
        }
        wput_zero(&c, g_data_offset[0] - (size_t)(c.p - buf));

        /* Tensor data per group, in tensor order. */
        for (i = 0; i < in->n_groups; ++i) {
            uint64_t group_start = g_data_offset[i];
            /* Align cursor up to group_start (writer wrote previous
             * group's data; pad to next group's start). */
            if ((size_t)(c.p - buf) < group_start) {
                wput_zero(&c, group_start - (size_t)(c.p - buf));
            }
            for (j = 0; j < in->n_tensors; ++j) {
                const eosi_eosm_tensor_in_t *t = &in->tensors[j];
                if (t->load_group_id != in->groups[i].id) continue;
                {
                    uint64_t target = group_start + t_offset[j];
                    if ((size_t)(c.p - buf) < target) {
                        wput_zero(&c, target - (size_t)(c.p - buf));
                    } else if ((size_t)(c.p - buf) > target) {
                        rc = EOS_E_INTERNAL; goto cleanup;
                    }
                }
                wput(&c, t->data, (size_t)t->data_bytes);
            }
        }

        /* Pad remaining bytes (if any) up to total - 32. */
        if ((size_t)(c.p - buf) > total - 32) {
            rc = EOS_E_INTERNAL; goto cleanup;
        }
        if ((size_t)(c.p - buf) < total - 32) {
            wput_zero(&c, (total - 32) - (size_t)(c.p - buf));
        }

        if (c.error) { rc = EOS_E_INTERNAL; goto cleanup; }
    }

    /* SHA-256 over [0, total - 32). */
    eosi_sha256(buf, total - 32, buf + total - 32);

    *out_buf  = buf;
    *out_size = total;
    rc = EOS_OK;
    buf = NULL;     /* hand off ownership */

cleanup:
    eosi_free(t_offset);
    eosi_free(g_data_offset);
    eosi_free(g_data_bytes);
    if (buf != NULL) eosi_free(buf);
    return rc;

bad_arg:
    rc = EOS_E_INVALID_ARG;
    goto cleanup;
}

#else
/* When EOSLLM_HAVE_FORMAT_EOSM=0, the writer compiles to a stub. */
eos_status_t eosi_eosm_write(const eosi_eosm_writer_in_t *in,
                             void **out_buf, size_t *out_size) {
    (void)in; (void)out_buf; (void)out_size;
    return EOS_E_UNSUPPORTED;
}
#endif
