/*
 * src/format/gguf/gguf.c — read-only GGUF v3 loader.
 *
 * Phase 1 scope: parse a Llama-3 / Qwen2-class GGUF file, expose
 *   - all metadata key/value pairs (string, scalar, and array values),
 *   - all tensor descriptors (name, dtype, shape, data pointer),
 *   - the raw tensor data blob (mapped or read into memory).
 *
 * No writing, no mmap (we read the whole file into one allocation).
 * Phase 4 will swap to mmap on POSIX targets that support it.
 *
 * Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "eosllm/eosllm.h"
#include "eosllm/model.h"
#include "eosllm/tensor.h"

#include "../../core/internal.h"
#include "../../core/model_internal.h"
#include "../format_internal.h"

#if EOSLLM_HAVE_FORMAT_GGUF

#define GGUF_MAGIC                0x46554747u   /* "GGUF" little-endian */
#define GGUF_VERSION              3u
#define GGUF_DEFAULT_ALIGNMENT    32u

/* GGUF metadata value types. */
typedef enum {
    GGUFV_U8 = 0, GGUFV_I8, GGUFV_U16, GGUFV_I16, GGUFV_U32, GGUFV_I32,
    GGUFV_F32, GGUFV_BOOL, GGUFV_STRING, GGUFV_ARRAY,
    GGUFV_U64, GGUFV_I64, GGUFV_F64
} gguf_value_type_t;

/* GGML tensor types that we understand. */
typedef enum {
    GGML_F32  = 0,
    GGML_F16  = 1,
    GGML_Q8_0 = 8,
    GGML_Q4_K = 12
} ggml_type_t;

typedef struct {
    const char *str;        /* points into the string pool, null-terminated */
    uint32_t    len;        /* original length excluding null */
} gguf_str_t;

typedef struct {
    gguf_str_t        key;
    gguf_value_type_t type;
    /* Discriminated payload. */
    union {
        uint8_t  u8; int8_t  i8;
        uint16_t u16; int16_t i16;
        uint32_t u32; int32_t i32;
        uint64_t u64; int64_t i64;
        float    f32; double  f64;
        gguf_str_t        s;
        struct {
            gguf_value_type_t elem_type;
            uint64_t          count;
            const uint8_t    *data;   /* pointer into file buffer */
            /* For string arrays we also resolve a pre-decoded list. */
            const gguf_str_t *strs;   /* NULL for non-string arrays */
        } a;
    } v;
} gguf_kv_t;

typedef struct {
    eos_tensor_t hdr;       /* embedded so &gguf_tensor->hdr is the public type */
    uint64_t     offset;    /* offset within the data blob */
} gguf_tensor_t;

typedef struct {
    /* Owned allocations. */
    void           *file_buf;       /* whole-file buffer */
    size_t          file_len;
    char           *str_pool;
    size_t          str_pool_len;
    gguf_kv_t      *kvs;
    size_t          n_kvs;
    gguf_tensor_t  *tensors;
    size_t          n_tensors;
    gguf_str_t     *str_arr_pool;   /* flat buffer of string-array elements */
    size_t          n_str_arr_pool;
    /* Slice of file_buf for tensor data (already aligned). */
    const uint8_t  *tensor_data;
} gguf_t;

/* ------------------------------------------------------------------ */
/* Cursor over the file buffer                                         */
/* ------------------------------------------------------------------ */

typedef struct {
    const uint8_t *p;
    const uint8_t *end;
    int            error;
} cursor_t;

static int cur_take(cursor_t *c, void *dst, size_t n) {
    if (c->error) return 0;
    if ((size_t)(c->end - c->p) < n) { c->error = 1; return 0; }
    memcpy(dst, c->p, n);
    c->p += n;
    return 1;
}

static int cur_skip(cursor_t *c, size_t n) {
    if (c->error) return 0;
    if ((size_t)(c->end - c->p) < n) { c->error = 1; return 0; }
    c->p += n;
    return 1;
}

static const uint8_t *cur_raw(cursor_t *c, size_t n) {
    const uint8_t *r;
    if (c->error || (size_t)(c->end - c->p) < n) { c->error = 1; return NULL; }
    r = c->p;
    c->p += n;
    return r;
}

static uint64_t cur_u64(cursor_t *c) { uint64_t v = 0; cur_take(c, &v, 8); return v; }
static uint32_t cur_u32(cursor_t *c) { uint32_t v = 0; cur_take(c, &v, 4); return v; }

/* ------------------------------------------------------------------ */
/* Value-type sizes                                                    */
/* ------------------------------------------------------------------ */

static size_t scalar_size(gguf_value_type_t t) {
    switch (t) {
        case GGUFV_U8: case GGUFV_I8: case GGUFV_BOOL:        return 1;
        case GGUFV_U16: case GGUFV_I16:                       return 2;
        case GGUFV_U32: case GGUFV_I32: case GGUFV_F32:       return 4;
        case GGUFV_U64: case GGUFV_I64: case GGUFV_F64:       return 8;
        default:                                              return 0;
    }
}

/* ------------------------------------------------------------------ */
/* Tensor sizing                                                       */
/* ------------------------------------------------------------------ */

static eos_dtype_t map_ggml_dtype(uint32_t t) {
    switch ((ggml_type_t)t) {
        case GGML_F32:  return EOS_DT_F32;
        case GGML_F16:  return EOS_DT_F16;
        case GGML_Q8_0: return EOS_DT_Q8_0;
        case GGML_Q4_K: return EOS_DT_Q4_K;
    }
    return (eos_dtype_t)0xFFFF;     /* sentinel for unsupported */
}

/*
 * Returns the on-disk byte size for a tensor of dtype `dt` with
 * `n_elements` elements. Returns 0 for unsupported dtypes OR when
 * `n_elements` is not a multiple of the dtype's block size (a malformed
 * GGUF condition). Caller must treat 0 as fatal when n_elements > 0.
 */
static size_t tensor_bytes(eos_dtype_t dt, uint64_t n_elements) {
    switch (dt) {
        case EOS_DT_F32:  return (size_t)(n_elements * 4u);
        case EOS_DT_F16:  return (size_t)(n_elements * 2u);
        case EOS_DT_Q8_0:
            if ((n_elements % 32u) != 0u) return 0;
            return (size_t)((n_elements / 32u) * 34u);
        case EOS_DT_Q4_K:
            if ((n_elements % 256u) != 0u) return 0;
            return (size_t)((n_elements / 256u) * 144u);
        default:          return 0;
    }
}

/* True iff v is a positive power of two. */
static int is_pow2(uint32_t v) {
    return v != 0u && (v & (v - 1u)) == 0u;
}

/* ------------------------------------------------------------------ */
/* String-pool allocator                                               */
/* ------------------------------------------------------------------ */

typedef struct {
    char  *base;
    size_t cap;
    size_t used;
} str_pool_t;

/* Copies len bytes + null terminator into the pool; returns a stable pointer. */
static const char *sp_intern(str_pool_t *sp, const uint8_t *src, size_t len) {
    char *out;
    if (sp->used + len + 1 > sp->cap) return NULL;
    out = sp->base + sp->used;
    memcpy(out, src, len);
    out[len] = '\0';
    sp->used += len + 1;
    return out;
}

/* ------------------------------------------------------------------ */
/* Probe + open                                                        */
/* ------------------------------------------------------------------ */

static eos_status_t gguf_probe(eos_stream_t *stream) {
    uint32_t magic = 0;
    uint8_t  buf[4];
    if (stream->read(stream->opaque, buf, 4) != 4) {
        EOSI_LOG_ERROR("gguf: stream read failed for the 4-byte magic");
        return EOS_E_FORMAT;
    }
    memcpy(&magic, buf, 4);
    if (magic != GGUF_MAGIC) {
        EOSI_LOG_ERROR("gguf: bad magic (expected \"GGUF\")");
        return EOS_E_FORMAT;
    }
    return EOS_OK;
}

/* Read a GGUF string (u64 length + bytes) into a pool-interned cstring. */
static int read_string(cursor_t *c, str_pool_t *sp, gguf_str_t *out) {
    uint64_t len = cur_u64(c);
    const uint8_t *bytes;
    if (c->error || len > 0xFFFFFFFFu) { c->error = 1; return 0; }
    bytes = cur_raw(c, (size_t)len);
    if (c->error) return 0;
    out->len = (uint32_t)len;
    out->str = sp_intern(sp, bytes, (size_t)len);
    return out->str != NULL;
}

/* Read a metadata value into kv->v, given the type already in kv->type. */
static int read_value(cursor_t *c, str_pool_t *sp, gguf_kv_t *kv,
                      gguf_str_t *str_arr_buf, size_t *str_arr_buf_used,
                      size_t str_arr_buf_cap) {
    switch (kv->type) {
        case GGUFV_U8:   return cur_take(c, &kv->v.u8, 1);
        case GGUFV_I8:   return cur_take(c, &kv->v.i8, 1);
        case GGUFV_U16:  return cur_take(c, &kv->v.u16, 2);
        case GGUFV_I16:  return cur_take(c, &kv->v.i16, 2);
        case GGUFV_U32:  return cur_take(c, &kv->v.u32, 4);
        case GGUFV_I32:  return cur_take(c, &kv->v.i32, 4);
        case GGUFV_F32:  return cur_take(c, &kv->v.f32, 4);
        case GGUFV_BOOL: { uint8_t b; if (!cur_take(c, &b, 1)) return 0;
                           kv->v.u8 = b; return 1; }
        case GGUFV_U64:  return cur_take(c, &kv->v.u64, 8);
        case GGUFV_I64:  return cur_take(c, &kv->v.i64, 8);
        case GGUFV_F64:  return cur_take(c, &kv->v.f64, 8);
        case GGUFV_STRING: return read_string(c, sp, &kv->v.s);
        case GGUFV_ARRAY: {
            uint32_t etype = cur_u32(c);
            uint64_t count = cur_u64(c);
            uint64_t i;
            if (c->error) return 0;
            kv->v.a.elem_type = (gguf_value_type_t)etype;
            kv->v.a.count     = count;
            kv->v.a.data      = c->p;
            kv->v.a.strs      = NULL;
            if (etype == GGUFV_STRING) {
                gguf_str_t *base;
                if (*str_arr_buf_used + (size_t)count > str_arr_buf_cap)
                    { c->error = 1; return 0; }
                base = str_arr_buf + *str_arr_buf_used;
                kv->v.a.strs = base;
                *str_arr_buf_used += (size_t)count;
                for (i = 0; i < count; ++i) {
                    if (!read_string(c, sp, &base[i])) return 0;
                }
            } else {
                size_t es = scalar_size((gguf_value_type_t)etype);
                if (es == 0) { c->error = 1; return 0; }
                if (!cur_skip(c, es * (size_t)count)) return 0;
            }
            return 1;
        }
    }
    c->error = 1; return 0;
}

/* ------------------------------------------------------------------ */
/* Operations vtable for eos_model_*                                   */
/* ------------------------------------------------------------------ */

static const gguf_kv_t *find_kv(const gguf_t *g, const char *key) {
    size_t i;
    for (i = 0; i < g->n_kvs; ++i) {
        if (g->kvs[i].key.str != NULL && strcmp(g->kvs[i].key.str, key) == 0) {
            return &g->kvs[i];
        }
    }
    return NULL;
}

static void op_close(void *impl) {
    gguf_t *g = (gguf_t *)impl;
    if (g == NULL) return;
    eosi_free(g->file_buf);
    eosi_free(g->str_pool);
    eosi_free(g->kvs);
    eosi_free(g->tensors);
    eosi_free(g->str_arr_pool);
    eosi_free(g);
}

static size_t op_num_tensors(const void *impl) {
    return ((const gguf_t *)impl)->n_tensors;
}

static const eos_tensor_t *op_tensor(const void *impl, size_t i) {
    const gguf_t *g = (const gguf_t *)impl;
    if (i >= g->n_tensors) return NULL;
    return &g->tensors[i].hdr;
}

static const eos_tensor_t *op_tensor_by_name(const void *impl, const char *name) {
    const gguf_t *g = (const gguf_t *)impl;
    size_t i;
    if (name == NULL) return NULL;
    for (i = 0; i < g->n_tensors; ++i) {
        if (g->tensors[i].hdr.name != NULL
            && strcmp(g->tensors[i].hdr.name, name) == 0) {
            return &g->tensors[i].hdr;
        }
    }
    return NULL;
}

static const char *op_meta_str(const void *impl, const char *key) {
    const gguf_kv_t *kv = find_kv((const gguf_t *)impl, key);
    if (kv == NULL || kv->type != GGUFV_STRING) return NULL;
    return kv->v.s.str;
}

static eos_status_t op_meta_u64(const void *impl, const char *key,
                                uint64_t *out) {
    const gguf_kv_t *kv = find_kv((const gguf_t *)impl, key);
    if (kv == NULL) return EOS_E_NOT_FOUND;
    switch (kv->type) {
        case GGUFV_U8:   *out = kv->v.u8;  return EOS_OK;
        case GGUFV_I8:   *out = (uint64_t)(int64_t)kv->v.i8;  return EOS_OK;
        case GGUFV_U16:  *out = kv->v.u16; return EOS_OK;
        case GGUFV_I16:  *out = (uint64_t)(int64_t)kv->v.i16; return EOS_OK;
        case GGUFV_U32:  *out = kv->v.u32; return EOS_OK;
        case GGUFV_I32:  *out = (uint64_t)(int64_t)kv->v.i32; return EOS_OK;
        case GGUFV_U64:  *out = kv->v.u64; return EOS_OK;
        case GGUFV_I64:  *out = (uint64_t)kv->v.i64; return EOS_OK;
        case GGUFV_BOOL: *out = kv->v.u8;  return EOS_OK;
        default:         return EOS_E_INVALID_ARG;
    }
}

static eos_status_t op_meta_f32(const void *impl, const char *key,
                                float *out) {
    const gguf_kv_t *kv = find_kv((const gguf_t *)impl, key);
    if (kv == NULL) return EOS_E_NOT_FOUND;
    if (kv->type != GGUFV_F32) return EOS_E_INVALID_ARG;
    *out = kv->v.f32;
    return EOS_OK;
}

static eos_status_t op_meta_arr_str(const void *impl, const char *key,
                                    const char **out, size_t *io_n) {
    const gguf_kv_t *kv = find_kv((const gguf_t *)impl, key);
    size_t i, n;
    if (kv == NULL) return EOS_E_NOT_FOUND;
    if (kv->type != GGUFV_ARRAY || kv->v.a.elem_type != GGUFV_STRING)
        return EOS_E_INVALID_ARG;
    n = (size_t)kv->v.a.count;
    if (out != NULL) {
        size_t cap = (*io_n < n) ? *io_n : n;
        for (i = 0; i < cap; ++i) out[i] = kv->v.a.strs[i].str;
    }
    *io_n = n;
    return EOS_OK;
}

static eos_status_t op_meta_arr_u32(const void *impl, const char *key,
                                    uint32_t *out, size_t *io_n) {
    const gguf_kv_t *kv = find_kv((const gguf_t *)impl, key);
    size_t i, n;
    size_t es;
    if (kv == NULL) return EOS_E_NOT_FOUND;
    if (kv->type != GGUFV_ARRAY) return EOS_E_INVALID_ARG;
    es = scalar_size(kv->v.a.elem_type);
    if (es == 0) return EOS_E_INVALID_ARG;
    n = (size_t)kv->v.a.count;
    if (out != NULL) {
        size_t cap = (*io_n < n) ? *io_n : n;
        const uint8_t *p = kv->v.a.data;
        for (i = 0; i < cap; ++i) {
            uint32_t v = 0;
            switch (kv->v.a.elem_type) {
                case GGUFV_U8:  v = p[i];  break;
                case GGUFV_I8:  v = (uint32_t)(int32_t)(int8_t)p[i]; break;
                case GGUFV_U16: { uint16_t x; memcpy(&x, p + i*2, 2); v = x; break; }
                case GGUFV_I16: { int16_t  x; memcpy(&x, p + i*2, 2); v = (uint32_t)(int32_t)x; break; }
                case GGUFV_U32: { uint32_t x; memcpy(&x, p + i*4, 4); v = x; break; }
                case GGUFV_I32: { int32_t  x; memcpy(&x, p + i*4, 4); v = (uint32_t)x; break; }
                default: return EOS_E_INVALID_ARG;
            }
            out[i] = v;
        }
    }
    *io_n = n;
    return EOS_OK;
}

static const eosi_model_ops_t k_gguf_ops = {
    op_close, op_num_tensors, op_tensor, op_tensor_by_name,
    op_meta_str, op_meta_u64, op_meta_f32,
    op_meta_arr_str, op_meta_arr_u32
};

/* ------------------------------------------------------------------ */
/* gguf_open                                                           */
/* ------------------------------------------------------------------ */

static eos_status_t gguf_open(eos_stream_t *stream, eos_model_t **out_model) {
    gguf_t      *g  = NULL;
    eos_model_t *m  = NULL;
    int64_t      sz;
    cursor_t     c;
    uint32_t     magic, version;
    uint64_t     tensor_count, kv_count;
    uint32_t     alignment = GGUF_DEFAULT_ALIGNMENT;
    size_t       i;
    size_t       n_str_arrays = 0;
    size_t       max_strings  = 0;
    size_t       str_arr_used = 0;
    /* Pass-1 exact accounting of bytes interned into the string pool.
     * Each interned string contributes (len + 1) bytes (null terminator). */
    size_t       pool_bytes_needed = 0;
    size_t       data_offset_in_file = 0;
    size_t       data_offset_aligned;

    *out_model = NULL;
    sz = stream->size(stream->opaque);
    if (sz <= 0) return EOS_E_FORMAT;

    g = (gguf_t *)eosi_alloc(sizeof(*g), 16);
    if (g == NULL) return EOS_E_OUT_OF_MEMORY;
    memset(g, 0, sizeof(*g));

    g->file_buf = eosi_alloc((size_t)sz, 32);
    g->file_len = (size_t)sz;
    if (g->file_buf == NULL) { op_close(g); return EOS_E_OUT_OF_MEMORY; }
    if (stream->seek(stream->opaque, 0) != 0)               { op_close(g); return EOS_E_IO; }
    if (stream->read(stream->opaque, g->file_buf, (size_t)sz) != sz)
        { op_close(g); return EOS_E_IO; }

    c.p = (const uint8_t *)g->file_buf;
    c.end = c.p + g->file_len;
    c.error = 0;

    magic   = cur_u32(&c);
    version = cur_u32(&c);
    tensor_count = cur_u64(&c);
    kv_count     = cur_u64(&c);
    if (c.error || magic != GGUF_MAGIC || version != GGUF_VERSION) {
        EOSI_LOG_ERROR("gguf_open: bad magic or unsupported version "
                       "(expected GGUF v3)");
        op_close(g); return EOS_E_FORMAT;
    }

    /*
     * Pass 1: scan kvs to size the string pool and the string-array buffer.
     * We can do this without saving anything because we'll re-walk in pass 2.
     */
    {
        cursor_t scan = c;
        for (i = 0; i < kv_count; ++i) {
            uint64_t key_len;
            uint32_t type;
            key_len = cur_u64(&scan);
            if (scan.error || key_len > 0xFFFFFFFFu) {
                EOSI_LOG_ERROR("gguf: pass-1 oversized kv key length");
                op_close(g); return EOS_E_FORMAT;
            }
            pool_bytes_needed += (size_t)key_len + 1u;
            cur_skip(&scan, (size_t)key_len);
            type = cur_u32(&scan);
            if (scan.error) { op_close(g); return EOS_E_FORMAT; }
            if (type == GGUFV_ARRAY) {
                uint32_t etype = cur_u32(&scan);
                uint64_t cnt   = cur_u64(&scan);
                if (scan.error) { op_close(g); return EOS_E_FORMAT; }
                if (etype == GGUFV_STRING) {
                    uint64_t j;
                    n_str_arrays++;
                    max_strings += (size_t)cnt;
                    for (j = 0; j < cnt; ++j) {
                        uint64_t l = cur_u64(&scan);
                        if (scan.error || l > 0xFFFFFFFFu) {
                            EOSI_LOG_ERROR("gguf: pass-1 oversized array string length");
                            op_close(g); return EOS_E_FORMAT;
                        }
                        pool_bytes_needed += (size_t)l + 1u;
                        cur_skip(&scan, (size_t)l);
                    }
                } else {
                    size_t es = scalar_size((gguf_value_type_t)etype);
                    if (es == 0) {
                        EOSI_LOG_ERROR("gguf: pass-1 unknown array element type");
                        op_close(g); return EOS_E_FORMAT;
                    }
                    cur_skip(&scan, es * (size_t)cnt);
                }
            } else if (type == GGUFV_STRING) {
                uint64_t l = cur_u64(&scan);
                if (scan.error || l > 0xFFFFFFFFu) {
                    EOSI_LOG_ERROR("gguf: pass-1 oversized string value length");
                    op_close(g); return EOS_E_FORMAT;
                }
                pool_bytes_needed += (size_t)l + 1u;
                cur_skip(&scan, (size_t)l);
            } else {
                size_t es = scalar_size((gguf_value_type_t)type);
                if (es == 0) {
                    EOSI_LOG_ERROR("gguf: pass-1 unknown scalar value type");
                    op_close(g); return EOS_E_FORMAT;
                }
                cur_skip(&scan, es);
            }
        }
        /* Tensor names also go into the string pool. */
        for (i = 0; i < tensor_count; ++i) {
            uint64_t name_len = cur_u64(&scan);
            uint32_t n_dims;
            uint32_t j;
            if (scan.error || name_len > 0xFFFFFFFFu) {
                EOSI_LOG_ERROR("gguf: pass-1 oversized tensor name length");
                op_close(g); return EOS_E_FORMAT;
            }
            max_strings += 1;       /* the name itself */
            pool_bytes_needed += (size_t)name_len + 1u;
            cur_skip(&scan, (size_t)name_len);
            n_dims = cur_u32(&scan);
            for (j = 0; j < n_dims; ++j) cur_u64(&scan);
            cur_skip(&scan, 4 + 8); /* type + offset */
        }
        if (scan.error) { op_close(g); return EOS_E_FORMAT; }
    }

    /* Allocate pools sized from pass 1 — exactly enough for every
     * interned string + a small 16-byte slack to absorb future tweaks. */
    g->str_pool_len = pool_bytes_needed + 16u;
    g->str_pool = (char *)eosi_alloc(g->str_pool_len, 8);
    if (g->str_pool == NULL) { op_close(g); return EOS_E_OUT_OF_MEMORY; }
    {
        str_pool_t spool = { g->str_pool, g->str_pool_len, 0 };
        gguf_str_t *str_arr_buf = NULL;
        size_t      str_arr_buf_cap = max_strings + 16;
        if (str_arr_buf_cap > 0) {
            g->str_arr_pool = (gguf_str_t *)eosi_alloc(
                str_arr_buf_cap * sizeof(gguf_str_t), 8);
            if (g->str_arr_pool == NULL) { op_close(g); return EOS_E_OUT_OF_MEMORY; }
            str_arr_buf = g->str_arr_pool;
            g->n_str_arr_pool = str_arr_buf_cap;
        }
        g->kvs = (gguf_kv_t *)eosi_alloc(kv_count * sizeof(gguf_kv_t), 8);
        g->n_kvs = kv_count;
        if (g->kvs == NULL && kv_count > 0) { op_close(g); return EOS_E_OUT_OF_MEMORY; }
        memset(g->kvs, 0, kv_count * sizeof(gguf_kv_t));

        /* Pass 2: actually parse kvs. */
        for (i = 0; i < kv_count; ++i) {
            gguf_kv_t *kv = &g->kvs[i];
            uint32_t   t;
            if (!read_string(&c, &spool, &kv->key)) { op_close(g); return EOS_E_FORMAT; }
            t = cur_u32(&c);
            kv->type = (gguf_value_type_t)t;
            if (!read_value(&c, &spool, kv,
                            str_arr_buf, &str_arr_used, str_arr_buf_cap)) {
                op_close(g); return EOS_E_FORMAT;
            }
            /* Honor general.alignment if present. Reject non-power-of-two
             * since we use it as a bitmask below. */
            if (kv->type == GGUFV_U32 && kv->key.str != NULL
                && strcmp(kv->key.str, "general.alignment") == 0) {
                uint32_t a = kv->v.u32;
                if (a == 0) a = GGUF_DEFAULT_ALIGNMENT;
                if (!is_pow2(a)) {
                    EOSI_LOG_ERROR("gguf: general.alignment is not power-of-two");
                    op_close(g); return EOS_E_FORMAT;
                }
                alignment = a;
            }
        }

        /* Tensor descriptors. */
        g->tensors = (gguf_tensor_t *)eosi_alloc(
            tensor_count * sizeof(gguf_tensor_t), 8);
        g->n_tensors = (size_t)tensor_count;
        if (g->tensors == NULL && tensor_count > 0) { op_close(g); return EOS_E_OUT_OF_MEMORY; }
        memset(g->tensors, 0, tensor_count * sizeof(gguf_tensor_t));

        for (i = 0; i < tensor_count; ++i) {
            gguf_tensor_t *t = &g->tensors[i];
            gguf_str_t name;
            uint32_t   n_dims, type;
            uint32_t   j;
            uint64_t   total_elems = 1;
            eos_dtype_t dt;
            if (!read_string(&c, &spool, &name)) {
                EOSI_LOG_ERROR("gguf: failed to read tensor name");
                op_close(g); return EOS_E_FORMAT;
            }
            t->hdr.name = name.str;
            n_dims = cur_u32(&c);
            if (n_dims == 0 || n_dims > EOS_MAX_RANK) {
                EOSI_LOG_ERROR("gguf: tensor has rank 0 or > EOS_MAX_RANK");
                op_close(g); return EOS_E_FORMAT;
            }
            t->hdr.rank = n_dims;
            for (j = 0; j < n_dims; ++j) {
                uint64_t d = cur_u64(&c);
                if (d == 0u || d > 0xFFFFFFFFu) {
                    EOSI_LOG_ERROR("gguf: tensor has zero or oversized dim");
                    op_close(g); return EOS_E_FORMAT;
                }
                /* Overflow-safe element-count multiplication. */
                if (total_elems > (uint64_t)0xFFFFFFFFFFFFFFFFu / d) {
                    EOSI_LOG_ERROR("gguf: tensor element count overflows u64");
                    op_close(g); return EOS_E_FORMAT;
                }
                t->hdr.dims[j] = (uint32_t)d;
                total_elems *= d;
            }
            for (j = n_dims; j < EOS_MAX_RANK; ++j) t->hdr.dims[j] = 0;
            type = cur_u32(&c);
            dt = map_ggml_dtype(type);
            if (dt == (eos_dtype_t)0xFFFF) {
                EOSI_LOG_ERROR("gguf: tensor uses unsupported dtype");
                op_close(g); return EOS_E_UNSUPPORTED;
            }
            t->hdr.dtype = dt;
            t->hdr.data_bytes = tensor_bytes(dt, total_elems);
            if (t->hdr.data_bytes == 0u && total_elems > 0u) {
                EOSI_LOG_ERROR("gguf: tensor element count not divisible by block size");
                op_close(g); return EOS_E_FORMAT;
            }
            t->offset = cur_u64(&c);
        }
        if (c.error) { op_close(g); return EOS_E_FORMAT; }
    }

    /* Compute aligned data offset (relative to start of file). Reject
     * if rounding-up overflows or pushes us past EOF. */
    data_offset_in_file = (size_t)(c.p - (const uint8_t *)g->file_buf);
    if (data_offset_in_file > (size_t)-1 - ((size_t)alignment - 1u)) {
        EOSI_LOG_ERROR("gguf: tensor data offset overflows when aligning");
        op_close(g); return EOS_E_FORMAT;
    }
    data_offset_aligned = (data_offset_in_file + (size_t)alignment - 1u)
                          & ~((size_t)alignment - 1u);
    if (data_offset_aligned > g->file_len) {
        EOSI_LOG_ERROR("gguf: aligned data offset is past EOF");
        op_close(g); return EOS_E_FORMAT;
    }
    g->tensor_data = (const uint8_t *)g->file_buf + data_offset_aligned;

    /* Resolve tensor data pointers. Bounds check is overflow-safe:
     * limit = file_len - data_offset_aligned, then check both
     * t->offset > limit and t->hdr.data_bytes > limit - t->offset. */
    {
        size_t limit = g->file_len - data_offset_aligned;
        for (i = 0; i < g->n_tensors; ++i) {
            gguf_tensor_t *t = &g->tensors[i];
            if ((size_t)t->offset > limit
                || t->hdr.data_bytes > limit - (size_t)t->offset) {
                EOSI_LOG_ERROR("gguf: tensor data extends past EOF");
                op_close(g); return EOS_E_FORMAT;
            }
            t->hdr.data = g->tensor_data + t->offset;
        }
    }

    /* Wrap into eos_model_t. */
    m = (eos_model_t *)eosi_alloc(sizeof(*m), 16);
    if (m == NULL) { op_close(g); return EOS_E_OUT_OF_MEMORY; }
    m->ops  = &k_gguf_ops;
    m->impl = g;
    *out_model = m;
    return EOS_OK;
}

/* ------------------------------------------------------------------ */
/* Format vtable + registration                                        */
/* ------------------------------------------------------------------ */

static const eos_format_vt_t k_gguf_format = {
    "gguf",
    gguf_probe,
    gguf_open,
    NULL    /* close handled by op_close via model_internal */
};

eos_status_t eosi_format_gguf_register(void) {
    return eos_format_register(&k_gguf_format);
}

#else /* !EOSLLM_HAVE_FORMAT_GGUF */

eos_status_t eosi_format_gguf_register(void) { return EOS_E_UNSUPPORTED; }

#endif
