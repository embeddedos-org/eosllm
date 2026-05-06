/*
 * src/format/eosm/eosm.c — native .eosm v1 file format reader.
 *
 * Spec: docs/file_format.md.
 *
 * Implementation choices for v1:
 *   - Read the whole file into one allocation. mmap is a follow-up.
 *   - Verify the SHA-256 trailer before parsing any sections.
 *   - Reject any capability bit not present in eos_caps() (with one
 *     exception: CAP_HASH_SHA256_TRAILER is always required and we
 *     simply require it to be set).
 *   - Tensor data pointers are resolved into the file buffer at open
 *     time; the model's lifetime owns them.
 *
 * The writer lives in src/format/eosm/writer.c. A roundtrip test
 * exercises both directions (tests/unit/test_runner.c).
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "eosllm/eosllm.h"
#include "eosllm/model.h"
#include "eosllm/tensor.h"

#include "../../core/internal.h"
#include "../../core/model_internal.h"
#include "../../util/sha256.h"
#include "../format_internal.h"
#include "eosm_internal.h"

#if EOSLLM_HAVE_FORMAT_EOSM

/* ------------------------------------------------------------------ */
/* On-heap structures                                                  */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *str;        /* points into the string pool, null-terminated */
    uint32_t    len;
} eosm_str_t;

typedef struct {
    eosm_str_t        key;
    uint32_t          type;
    /* Discriminated payload (mirrors writer input). */
    uint8_t  u8;  int8_t  i8;
    uint16_t u16; int16_t i16;
    uint32_t u32; int32_t i32;
    uint64_t u64; int64_t i64;
    float    f32; double  f64;
    uint8_t  boolean;
    eosm_str_t        s;            /* STRING (stored as null-terminated) */
    struct {
        const uint8_t *bytes;       /* BLOB raw bytes (in file buffer) */
        uint64_t       len;
    } blob;
    struct {
        uint32_t        elem_type;
        uint64_t        count;
        const uint8_t  *data;       /* raw element bytes (in file buffer) */
        const eosm_str_t *strs;     /* for STRING arrays, decoded list */
    } a;
} eosm_kv_t;

typedef struct {
    eos_tensor_t hdr;
    uint32_t     quant_scheme_id;
    uint32_t     calibration_table_idx;
    uint8_t      load_group_id;
    uint64_t     local_offset;       /* offset within group */
} eosm_tensor_t;

typedef struct {
    uint32_t        table_type;
    const uint8_t  *data;
    uint64_t        data_bytes;
} eosm_calib_t;

typedef struct {
    uint8_t      id;
    eosm_str_t   name;
    uint64_t     data_offset;
    uint64_t     data_bytes;
} eosm_group_t;

typedef struct {
    void          *file_buf;
    size_t         file_len;
    char          *str_pool;
    size_t         str_pool_len;

    uint64_t       capability_bits;
    uint32_t       alignment;

    eosm_kv_t     *kvs;
    size_t         n_kvs;
    eosm_tensor_t *tensors;
    size_t         n_tensors;
    eosm_calib_t  *calibs;
    size_t         n_calibs;
    eosm_group_t  *groups;
    size_t         n_groups;

    eosm_str_t    *str_arr_pool;
    size_t         n_str_arr_pool;
} eosm_t;

/* ------------------------------------------------------------------ */
/* Cursor                                                              */
/* ------------------------------------------------------------------ */

typedef struct {
    const uint8_t *p;
    const uint8_t *end;
    int            error;
} cur_t;

static int cur_take(cur_t *c, void *dst, size_t n) {
    if (c->error) return 0;
    if ((size_t)(c->end - c->p) < n) { c->error = 1; return 0; }
    memcpy(dst, c->p, n);
    c->p += n;
    return 1;
}

static const uint8_t *cur_raw(cur_t *c, size_t n) {
    const uint8_t *r;
    if (c->error || (size_t)(c->end - c->p) < n) { c->error = 1; return NULL; }
    r = c->p; c->p += n;
    return r;
}

static int cur_skip(cur_t *c, size_t n) {
    if (c->error) return 0;
    if ((size_t)(c->end - c->p) < n) { c->error = 1; return 0; }
    c->p += n;
    return 1;
}

static uint8_t  cur_u8 (cur_t *c) { uint8_t  v = 0; cur_take(c, &v, 1); return v; }
static uint16_t cur_u16(cur_t *c) { uint16_t v = 0; cur_take(c, &v, 2); return v; }
static uint32_t cur_u32(cur_t *c) { uint32_t v = 0; cur_take(c, &v, 4); return v; }
static uint64_t cur_u64(cur_t *c) { uint64_t v = 0; cur_take(c, &v, 8); return v; }

/* ------------------------------------------------------------------ */
/* Sizing helpers                                                      */
/* ------------------------------------------------------------------ */

static size_t kv_scalar_size(uint32_t t) {
    switch (t) {
        case EOSI_EOSM_KV_U8: case EOSI_EOSM_KV_I8:
        case EOSI_EOSM_KV_BOOL:                              return 1;
        case EOSI_EOSM_KV_U16: case EOSI_EOSM_KV_I16:        return 2;
        case EOSI_EOSM_KV_U32: case EOSI_EOSM_KV_I32:
        case EOSI_EOSM_KV_F32:                               return 4;
        case EOSI_EOSM_KV_U64: case EOSI_EOSM_KV_I64:
        case EOSI_EOSM_KV_F64:                               return 8;
        default:                                             return 0;
    }
}

static int is_pow2(uint32_t v) { return v != 0u && (v & (v - 1u)) == 0u; }

static eos_dtype_t map_dtype(uint32_t t) {
    /* Direct passthrough; .eosm uses the eos_dtype_t enum on disk. */
    switch (t) {
        case EOS_DT_F32: case EOS_DT_F16:
        case EOS_DT_Q8_0: case EOS_DT_Q4_K:
            return (eos_dtype_t)t;
        default: return (eos_dtype_t)0xFFFF;
    }
}

/* ------------------------------------------------------------------ */
/* String pool                                                         */
/* ------------------------------------------------------------------ */

typedef struct {
    char  *base;
    size_t cap;
    size_t used;
} sp_t;

static const char *sp_intern(sp_t *sp, const uint8_t *src, size_t len) {
    char *out;
    if (sp->used + len + 1 > sp->cap) return NULL;
    out = sp->base + sp->used;
    memcpy(out, src, len);
    out[len] = '\0';
    sp->used += len + 1;
    return out;
}

static int read_lstr(cur_t *c, sp_t *sp, eosm_str_t *out) {
    uint64_t len = cur_u64(c);
    const uint8_t *bytes;
    if (c->error || len > 0xFFFFFFFFu) { c->error = 1; return 0; }
    bytes = cur_raw(c, (size_t)len);
    if (c->error) return 0;
    out->len = (uint32_t)len;
    out->str = sp_intern(sp, bytes, (size_t)len);
    return out->str != NULL;
}

/* ------------------------------------------------------------------ */
/* Probe + open                                                        */
/* ------------------------------------------------------------------ */

static eos_status_t eosm_probe(eos_stream_t *stream) {
    uint8_t buf[4]; uint32_t magic;
    if (stream->read(stream->opaque, buf, 4) != 4) return EOS_E_FORMAT;
    memcpy(&magic, buf, 4);
    return (magic == EOSI_EOSM_MAGIC) ? EOS_OK : EOS_E_FORMAT;
}

/* Forward decl. */
static const eosi_model_ops_t k_eosm_ops;

static void eosm_op_close(void *impl) {
    eosm_t *e = (eosm_t *)impl;
    if (e == NULL) return;
    eosi_free(e->file_buf);
    eosi_free(e->str_pool);
    eosi_free(e->kvs);
    eosi_free(e->tensors);
    eosi_free(e->calibs);
    eosi_free(e->groups);
    eosi_free(e->str_arr_pool);
    eosi_free(e);
}

/* Pass-1: walk to compute string-pool size + array-of-strings count. */
static eos_status_t pass1_size(cur_t scan, uint64_t n_kvs, uint64_t n_tensors,
                               uint64_t n_calibs, uint64_t n_groups,
                               size_t *out_pool_bytes, size_t *out_str_arr_count) {
    size_t pool = 0, arr = 0;
    uint64_t i, j;
    for (i = 0; i < n_kvs; ++i) {
        uint64_t key_len = cur_u64(&scan);
        uint32_t t;
        if (scan.error || key_len > 0xFFFFFFFFu) return EOS_E_FORMAT;
        pool += (size_t)key_len + 1u;
        cur_skip(&scan, (size_t)key_len);
        t = cur_u32(&scan);
        switch (t) {
            case EOSI_EOSM_KV_STRING: {
                uint64_t l = cur_u64(&scan);
                if (scan.error || l > 0xFFFFFFFFu) return EOS_E_FORMAT;
                pool += (size_t)l + 1u;
                cur_skip(&scan, (size_t)l);
            } break;
            case EOSI_EOSM_KV_BLOB: {
                uint64_t l = cur_u64(&scan);
                if (scan.error || l > 0xFFFFFFFFu) return EOS_E_FORMAT;
                cur_skip(&scan, (size_t)l);
            } break;
            case EOSI_EOSM_KV_ARRAY: {
                uint32_t et = cur_u32(&scan);
                uint64_t cnt = cur_u64(&scan);
                if (scan.error) return EOS_E_FORMAT;
                if (et == EOSI_EOSM_KV_STRING) {
                    arr += (size_t)cnt;
                    for (j = 0; j < cnt; ++j) {
                        uint64_t l = cur_u64(&scan);
                        if (scan.error || l > 0xFFFFFFFFu) return EOS_E_FORMAT;
                        pool += (size_t)l + 1u;
                        cur_skip(&scan, (size_t)l);
                    }
                } else {
                    size_t es = kv_scalar_size(et);
                    if (es == 0) return EOS_E_FORMAT;
                    cur_skip(&scan, es * (size_t)cnt);
                }
            } break;
            default: {
                size_t es = kv_scalar_size(t);
                if (es == 0) return EOS_E_FORMAT;
                cur_skip(&scan, es);
            }
        }
    }
    /* Tensor names */
    for (i = 0; i < n_tensors; ++i) {
        uint64_t name_len = cur_u64(&scan);
        uint8_t  rank;
        if (scan.error || name_len > 0xFFFFFFFFu) return EOS_E_FORMAT;
        pool += (size_t)name_len + 1u;
        cur_skip(&scan, (size_t)name_len);
        cur_skip(&scan, 4 + 4 + 4);          /* dtype + quant_id + calib_idx */
        rank = cur_u8(&scan);
        cur_skip(&scan, 1 + 2);              /* group_id + reserved */
        cur_skip(&scan, (size_t)rank * 8u);  /* dims */
        cur_skip(&scan, 8 + 8);              /* offset + data_bytes */
    }
    /* Calibration tables: skip data */
    for (i = 0; i < n_calibs; ++i) {
        uint64_t db;
        cur_skip(&scan, 4 + 4);
        db = cur_u64(&scan);
        if (scan.error) return EOS_E_FORMAT;
        cur_skip(&scan, (size_t)db);
    }
    /* Group names */
    for (i = 0; i < n_groups; ++i) {
        uint64_t name_len;
        cur_skip(&scan, 1 + 7 + 8 + 8);
        name_len = cur_u64(&scan);
        if (scan.error || name_len > 0xFFFFFFFFu) return EOS_E_FORMAT;
        pool += (size_t)name_len + 1u;
        cur_skip(&scan, (size_t)name_len);
    }
    if (scan.error) return EOS_E_FORMAT;
    *out_pool_bytes = pool + 16u;
    *out_str_arr_count = arr;
    return EOS_OK;
}

/* ------------------------------------------------------------------ */
/* Open                                                                */
/* ------------------------------------------------------------------ */

static eos_status_t eosm_open(eos_stream_t *stream, eos_model_t **out_model) {
    eosm_t *e = NULL;
    eos_model_t *m = NULL;
    int64_t sz;
    cur_t c;
    uint32_t magic, version, flags, alignment;
    uint64_t cap_bytes, n_kvs, n_tensors, n_calibs, n_groups;
    size_t i, j;
    size_t pool_bytes = 0, str_arr_cap = 0, str_arr_used = 0;
    uint8_t hash_calc[32];

    *out_model = NULL;
    sz = stream->size(stream->opaque);
    if (sz < EOSI_EOSM_HEADER_BYTES + 32) return EOS_E_FORMAT;

    e = (eosm_t *)eosi_alloc(sizeof(*e), 16);
    if (e == NULL) return EOS_E_OUT_OF_MEMORY;
    memset(e, 0, sizeof(*e));

    e->file_buf = eosi_alloc((size_t)sz, 32);
    e->file_len = (size_t)sz;
    if (e->file_buf == NULL) { eosm_op_close(e); return EOS_E_OUT_OF_MEMORY; }
    if (stream->seek(stream->opaque, 0) != 0) { eosm_op_close(e); return EOS_E_IO; }
    if (stream->read(stream->opaque, e->file_buf, (size_t)sz) != sz)
        { eosm_op_close(e); return EOS_E_IO; }

    /* Verify SHA-256 trailer first. */
    eosi_sha256(e->file_buf, e->file_len - 32u, hash_calc);
    if (memcmp(hash_calc, (uint8_t *)e->file_buf + e->file_len - 32u, 32) != 0) {
        EOSI_LOG_ERROR("eosm: SHA-256 trailer mismatch");
        eosm_op_close(e); return EOS_E_FORMAT;
    }

    c.p = (const uint8_t *)e->file_buf;
    c.end = c.p + (e->file_len - 32u);     /* exclude trailer */
    c.error = 0;

    magic     = cur_u32(&c);
    version   = cur_u32(&c);
    cap_bytes = cur_u64(&c);
    n_kvs     = cur_u64(&c);
    n_tensors = cur_u64(&c);
    n_calibs  = cur_u64(&c);
    n_groups  = cur_u64(&c);
    alignment = cur_u32(&c);
    flags     = cur_u32(&c);
    cur_skip(&c, 8);   /* reserved */

    if (c.error || magic != EOSI_EOSM_MAGIC) {
        EOSI_LOG_ERROR("eosm: bad magic");
        eosm_op_close(e); return EOS_E_FORMAT;
    }
    if (version == 0 || version > EOSI_EOSM_VERSION) {
        EOSI_LOG_ERROR("eosm: unsupported version");
        eosm_op_close(e); return EOS_E_VERSION_MISMATCH;
    }
    if (flags != 0) {
        EOSI_LOG_ERROR("eosm: non-zero header flags (writer used an extension)");
        eosm_op_close(e); return EOS_E_UNSUPPORTED;
    }
    if (!is_pow2(alignment) || alignment < 32u || alignment > 65536u) {
        EOSI_LOG_ERROR("eosm: alignment is not a valid power of two in [32, 65536]");
        eosm_op_close(e); return EOS_E_FORMAT;
    }
    if (n_groups == 0 || n_groups > 255u) {
        EOSI_LOG_ERROR("eosm: bad n_groups");
        eosm_op_close(e); return EOS_E_FORMAT;
    }
    e->alignment = alignment;

    /* Capability bits. We only consume the first 8 bytes (a u64) for
     * v1 — extra bytes are reserved for v2 and ignored as long as
     * flags == 0 (already checked). */
    if (cap_bytes < 8) {
        EOSI_LOG_ERROR("eosm: capability bits section too small");
        eosm_op_close(e); return EOS_E_FORMAT;
    }
    e->capability_bits = cur_u64(&c);
    if (cap_bytes > 8) cur_skip(&c, (size_t)(cap_bytes - 8u));
    if (c.error) { eosm_op_close(e); return EOS_E_FORMAT; }

    if ((e->capability_bits & EOSI_EOSM_CAP_HASH_SHA256_TRAILER) == 0) {
        EOSI_LOG_ERROR("eosm: CAP_HASH_SHA256_TRAILER must be set in v1");
        eosm_op_close(e); return EOS_E_FORMAT;
    }

    /* Pass 1 (sizes only). Reuse cursor by copy. */
    {
        eos_status_t s = pass1_size(c, n_kvs, n_tensors, n_calibs, n_groups,
                                    &pool_bytes, &str_arr_cap);
        if (s != EOS_OK) { eosm_op_close(e); return s; }
    }

    e->str_pool_len = pool_bytes;
    e->str_pool = (char *)eosi_alloc(e->str_pool_len, 8);
    if (e->str_pool == NULL) { eosm_op_close(e); return EOS_E_OUT_OF_MEMORY; }

    if (str_arr_cap > 0) {
        e->n_str_arr_pool = str_arr_cap;
        e->str_arr_pool = (eosm_str_t *)eosi_alloc(
            str_arr_cap * sizeof(eosm_str_t), 8);
        if (e->str_arr_pool == NULL) { eosm_op_close(e); return EOS_E_OUT_OF_MEMORY; }
    }

    /* Allocate manifests. */
    if (n_kvs > 0) {
        e->kvs = (eosm_kv_t *)eosi_alloc(n_kvs * sizeof(eosm_kv_t), 8);
        if (e->kvs == NULL) { eosm_op_close(e); return EOS_E_OUT_OF_MEMORY; }
        memset(e->kvs, 0, n_kvs * sizeof(eosm_kv_t));
    }
    e->n_kvs = (size_t)n_kvs;
    if (n_tensors > 0) {
        e->tensors = (eosm_tensor_t *)eosi_alloc(
            n_tensors * sizeof(eosm_tensor_t), 8);
        if (e->tensors == NULL) { eosm_op_close(e); return EOS_E_OUT_OF_MEMORY; }
        memset(e->tensors, 0, n_tensors * sizeof(eosm_tensor_t));
    }
    e->n_tensors = (size_t)n_tensors;
    if (n_calibs > 0) {
        e->calibs = (eosm_calib_t *)eosi_alloc(
            n_calibs * sizeof(eosm_calib_t), 8);
        if (e->calibs == NULL) { eosm_op_close(e); return EOS_E_OUT_OF_MEMORY; }
        memset(e->calibs, 0, n_calibs * sizeof(eosm_calib_t));
    }
    e->n_calibs = (size_t)n_calibs;
    e->groups = (eosm_group_t *)eosi_alloc(
        n_groups * sizeof(eosm_group_t), 8);
    if (e->groups == NULL) { eosm_op_close(e); return EOS_E_OUT_OF_MEMORY; }
    memset(e->groups, 0, n_groups * sizeof(eosm_group_t));
    e->n_groups = (size_t)n_groups;

    /* Pass 2: actually parse. */
    {
        sp_t spool = { e->str_pool, e->str_pool_len, 0 };

        /* KVs */
        for (i = 0; i < (size_t)n_kvs; ++i) {
            eosm_kv_t *kv = &e->kvs[i];
            if (!read_lstr(&c, &spool, &kv->key)) { eosm_op_close(e); return EOS_E_FORMAT; }
            kv->type = cur_u32(&c);
            switch (kv->type) {
                case EOSI_EOSM_KV_U8:   kv->u8  = cur_u8 (&c); break;
                case EOSI_EOSM_KV_I8:   { uint8_t v = cur_u8(&c); kv->i8 = (int8_t)v; } break;
                case EOSI_EOSM_KV_U16:  kv->u16 = cur_u16(&c); break;
                case EOSI_EOSM_KV_I16:  { uint16_t v = cur_u16(&c); kv->i16 = (int16_t)v; } break;
                case EOSI_EOSM_KV_U32:  kv->u32 = cur_u32(&c); break;
                case EOSI_EOSM_KV_I32:  { uint32_t v = cur_u32(&c); kv->i32 = (int32_t)v; } break;
                case EOSI_EOSM_KV_F32:  cur_take(&c, &kv->f32, 4); break;
                case EOSI_EOSM_KV_BOOL: kv->boolean = cur_u8(&c) ? 1u : 0u; break;
                case EOSI_EOSM_KV_U64:  kv->u64 = cur_u64(&c); break;
                case EOSI_EOSM_KV_I64:  { uint64_t v = cur_u64(&c); kv->i64 = (int64_t)v; } break;
                case EOSI_EOSM_KV_F64:  cur_take(&c, &kv->f64, 8); break;
                case EOSI_EOSM_KV_STRING: {
                    if (!read_lstr(&c, &spool, &kv->s))
                        { eosm_op_close(e); return EOS_E_FORMAT; }
                } break;
                case EOSI_EOSM_KV_BLOB: {
                    uint64_t l = cur_u64(&c);
                    const uint8_t *b;
                    if (c.error || l > 0xFFFFFFFFu) { eosm_op_close(e); return EOS_E_FORMAT; }
                    b = cur_raw(&c, (size_t)l);
                    if (c.error) { eosm_op_close(e); return EOS_E_FORMAT; }
                    kv->blob.bytes = b;
                    kv->blob.len   = l;
                } break;
                case EOSI_EOSM_KV_ARRAY: {
                    uint32_t et = cur_u32(&c);
                    uint64_t cnt = cur_u64(&c);
                    if (c.error) { eosm_op_close(e); return EOS_E_FORMAT; }
                    kv->a.elem_type = et;
                    kv->a.count     = cnt;
                    kv->a.data      = c.p;
                    kv->a.strs      = NULL;
                    if (et == EOSI_EOSM_KV_STRING) {
                        eosm_str_t *base;
                        if (str_arr_used + (size_t)cnt > str_arr_cap)
                            { eosm_op_close(e); return EOS_E_FORMAT; }
                        base = e->str_arr_pool + str_arr_used;
                        kv->a.strs = base;
                        str_arr_used += (size_t)cnt;
                        for (j = 0; j < (size_t)cnt; ++j) {
                            if (!read_lstr(&c, &spool, &base[j]))
                                { eosm_op_close(e); return EOS_E_FORMAT; }
                        }
                    } else {
                        size_t es = kv_scalar_size(et);
                        if (es == 0) { eosm_op_close(e); return EOS_E_FORMAT; }
                        if (!cur_skip(&c, es * (size_t)cnt))
                            { eosm_op_close(e); return EOS_E_FORMAT; }
                    }
                } break;
                default:
                    EOSI_LOG_ERROR("eosm: unknown kv type");
                    eosm_op_close(e); return EOS_E_FORMAT;
            }
        }

        /* Tensor manifest */
        for (i = 0; i < (size_t)n_tensors; ++i) {
            eosm_tensor_t *t = &e->tensors[i];
            eosm_str_t name;
            uint32_t dtype, qid, cid;
            uint8_t  rank, gid;
            uint8_t  resvd2[2];
            uint64_t off, db;
            uint32_t k;
            eos_dtype_t dt;
            if (!read_lstr(&c, &spool, &name)) { eosm_op_close(e); return EOS_E_FORMAT; }
            t->hdr.name = name.str;
            dtype = cur_u32(&c);
            qid   = cur_u32(&c);
            cid   = cur_u32(&c);
            rank  = cur_u8 (&c);
            gid   = cur_u8 (&c);
            cur_take(&c, resvd2, 2);
            (void)resvd2;
            if (rank == 0 || rank > EOS_MAX_RANK) {
                EOSI_LOG_ERROR("eosm: tensor has rank 0 or > EOS_MAX_RANK");
                eosm_op_close(e); return EOS_E_FORMAT;
            }
            if (gid >= n_groups) {
                EOSI_LOG_ERROR("eosm: tensor references unknown load group");
                eosm_op_close(e); return EOS_E_FORMAT;
            }
            t->hdr.rank = rank;
            for (k = 0; k < rank; ++k) {
                uint64_t d = cur_u64(&c);
                if (d == 0u || d > 0xFFFFFFFFu) {
                    EOSI_LOG_ERROR("eosm: tensor has zero or oversized dim");
                    eosm_op_close(e); return EOS_E_FORMAT;
                }
                t->hdr.dims[k] = (uint32_t)d;
            }
            for (k = rank; k < EOS_MAX_RANK; ++k) t->hdr.dims[k] = 0;
            off = cur_u64(&c);
            db  = cur_u64(&c);
            dt = map_dtype(dtype);
            if (dt == (eos_dtype_t)0xFFFF) {
                EOSI_LOG_ERROR("eosm: tensor uses unsupported dtype");
                eosm_op_close(e); return EOS_E_UNSUPPORTED;
            }
            t->hdr.dtype           = dt;
            t->hdr.data_bytes      = (size_t)db;
            t->quant_scheme_id     = qid;
            t->calibration_table_idx = cid;
            t->load_group_id       = gid;
            t->local_offset        = off;
        }

        /* Calibration tables */
        for (i = 0; i < (size_t)n_calibs; ++i) {
            eosm_calib_t *cb = &e->calibs[i];
            uint64_t db;
            cb->table_type = cur_u32(&c);
            cur_skip(&c, 4);
            db = cur_u64(&c);
            cb->data_bytes = db;
            cb->data = cur_raw(&c, (size_t)db);
            if (c.error) { eosm_op_close(e); return EOS_E_FORMAT; }
        }

        /* Load-group index */
        for (i = 0; i < (size_t)n_groups; ++i) {
            eosm_group_t *g = &e->groups[i];
            g->id = cur_u8(&c);
            cur_skip(&c, 7);
            g->data_offset = cur_u64(&c);
            g->data_bytes  = cur_u64(&c);
            if (!read_lstr(&c, &spool, &g->name))
                { eosm_op_close(e); return EOS_E_FORMAT; }
            /* Validate group data block fits inside file (excluding trailer). */
            if (g->data_offset > e->file_len - 32u
                || g->data_bytes > e->file_len - 32u - g->data_offset) {
                EOSI_LOG_ERROR("eosm: load group data extends past file");
                eosm_op_close(e); return EOS_E_FORMAT;
            }
            /* Group data start must be aligned. */
            if ((g->data_offset & (uint64_t)(alignment - 1u)) != 0) {
                EOSI_LOG_ERROR("eosm: load group data_offset not aligned");
                eosm_op_close(e); return EOS_E_FORMAT;
            }
        }

        if (c.error) { eosm_op_close(e); return EOS_E_FORMAT; }
    }

    /* Resolve tensor data pointers using their group's data_offset. */
    for (i = 0; i < e->n_tensors; ++i) {
        eosm_tensor_t *t = &e->tensors[i];
        eosm_group_t  *g = NULL;
        uint64_t abs_off;
        for (j = 0; j < e->n_groups; ++j) {
            if (e->groups[j].id == t->load_group_id) { g = &e->groups[j]; break; }
        }
        if (g == NULL) {
            EOSI_LOG_ERROR("eosm: tensor's load_group_id missing from index");
            eosm_op_close(e); return EOS_E_FORMAT;
        }
        if (t->local_offset > g->data_bytes
            || t->hdr.data_bytes > g->data_bytes - t->local_offset) {
            EOSI_LOG_ERROR("eosm: tensor data extends past its load group");
            eosm_op_close(e); return EOS_E_FORMAT;
        }
        abs_off = g->data_offset + t->local_offset;
        t->hdr.data = (const uint8_t *)e->file_buf + abs_off;
    }

    /* Wrap into eos_model_t. */
    m = (eos_model_t *)eosi_alloc(sizeof(*m), 16);
    if (m == NULL) { eosm_op_close(e); return EOS_E_OUT_OF_MEMORY; }
    m->ops  = &k_eosm_ops;
    m->impl = e;
    *out_model = m;
    return EOS_OK;
}

/* ------------------------------------------------------------------ */
/* Model ops                                                           */
/* ------------------------------------------------------------------ */

static const eosm_kv_t *find_kv(const eosm_t *e, const char *key) {
    size_t i;
    for (i = 0; i < e->n_kvs; ++i) {
        if (e->kvs[i].key.str != NULL && strcmp(e->kvs[i].key.str, key) == 0)
            return &e->kvs[i];
    }
    return NULL;
}

static size_t op_num_tensors(const void *impl) {
    return ((const eosm_t *)impl)->n_tensors;
}
static const eos_tensor_t *op_tensor(const void *impl, size_t i) {
    const eosm_t *e = (const eosm_t *)impl;
    if (i >= e->n_tensors) return NULL;
    return &e->tensors[i].hdr;
}
static const eos_tensor_t *op_tensor_by_name(const void *impl, const char *name) {
    const eosm_t *e = (const eosm_t *)impl;
    size_t i;
    if (name == NULL) return NULL;
    for (i = 0; i < e->n_tensors; ++i) {
        if (e->tensors[i].hdr.name && strcmp(e->tensors[i].hdr.name, name) == 0)
            return &e->tensors[i].hdr;
    }
    return NULL;
}
static const char *op_meta_str(const void *impl, const char *key) {
    const eosm_kv_t *kv = find_kv((const eosm_t *)impl, key);
    if (kv == NULL || kv->type != EOSI_EOSM_KV_STRING) return NULL;
    return kv->s.str;
}
static eos_status_t op_meta_u64(const void *impl, const char *key, uint64_t *out) {
    const eosm_kv_t *kv = find_kv((const eosm_t *)impl, key);
    if (kv == NULL) return EOS_E_NOT_FOUND;
    switch (kv->type) {
        case EOSI_EOSM_KV_U8:   *out = kv->u8;  return EOS_OK;
        case EOSI_EOSM_KV_I8:   *out = (uint64_t)(int64_t)kv->i8; return EOS_OK;
        case EOSI_EOSM_KV_U16:  *out = kv->u16; return EOS_OK;
        case EOSI_EOSM_KV_I16:  *out = (uint64_t)(int64_t)kv->i16; return EOS_OK;
        case EOSI_EOSM_KV_U32:  *out = kv->u32; return EOS_OK;
        case EOSI_EOSM_KV_I32:  *out = (uint64_t)(int64_t)kv->i32; return EOS_OK;
        case EOSI_EOSM_KV_U64:  *out = kv->u64; return EOS_OK;
        case EOSI_EOSM_KV_I64:  *out = (uint64_t)kv->i64; return EOS_OK;
        case EOSI_EOSM_KV_BOOL: *out = kv->boolean; return EOS_OK;
        default: return EOS_E_INVALID_ARG;
    }
}
static eos_status_t op_meta_f32(const void *impl, const char *key, float *out) {
    const eosm_kv_t *kv = find_kv((const eosm_t *)impl, key);
    if (kv == NULL) return EOS_E_NOT_FOUND;
    if (kv->type != EOSI_EOSM_KV_F32) return EOS_E_INVALID_ARG;
    *out = kv->f32; return EOS_OK;
}
static eos_status_t op_meta_arr_str(const void *impl, const char *key,
                                    const char **out, size_t *io_n) {
    const eosm_kv_t *kv = find_kv((const eosm_t *)impl, key);
    size_t i, n;
    if (kv == NULL) return EOS_E_NOT_FOUND;
    if (kv->type != EOSI_EOSM_KV_ARRAY || kv->a.elem_type != EOSI_EOSM_KV_STRING)
        return EOS_E_INVALID_ARG;
    n = (size_t)kv->a.count;
    if (out != NULL) {
        size_t cap = (*io_n < n) ? *io_n : n;
        for (i = 0; i < cap; ++i) out[i] = kv->a.strs[i].str;
    }
    *io_n = n;
    return EOS_OK;
}
static eos_status_t op_meta_arr_u32(const void *impl, const char *key,
                                    uint32_t *out, size_t *io_n) {
    const eosm_kv_t *kv = find_kv((const eosm_t *)impl, key);
    size_t i, n;
    size_t es;
    if (kv == NULL) return EOS_E_NOT_FOUND;
    if (kv->type != EOSI_EOSM_KV_ARRAY) return EOS_E_INVALID_ARG;
    es = kv_scalar_size(kv->a.elem_type);
    if (es == 0) return EOS_E_INVALID_ARG;
    n = (size_t)kv->a.count;
    if (out != NULL) {
        size_t cap = (*io_n < n) ? *io_n : n;
        const uint8_t *p = kv->a.data;
        for (i = 0; i < cap; ++i) {
            uint32_t v = 0;
            switch (kv->a.elem_type) {
                case EOSI_EOSM_KV_U8:  v = p[i]; break;
                case EOSI_EOSM_KV_I8:  v = (uint32_t)(int32_t)(int8_t)p[i]; break;
                case EOSI_EOSM_KV_U16: { uint16_t x; memcpy(&x, p + i*2, 2); v = x; break; }
                case EOSI_EOSM_KV_I16: { int16_t  x; memcpy(&x, p + i*2, 2); v = (uint32_t)(int32_t)x; break; }
                case EOSI_EOSM_KV_U32: { uint32_t x; memcpy(&x, p + i*4, 4); v = x; break; }
                case EOSI_EOSM_KV_I32: { int32_t  x; memcpy(&x, p + i*4, 4); v = (uint32_t)x; break; }
                default: return EOS_E_INVALID_ARG;
            }
            out[i] = v;
        }
    }
    *io_n = n;
    return EOS_OK;
}

static const eosi_model_ops_t k_eosm_ops = {
    eosm_op_close, op_num_tensors, op_tensor, op_tensor_by_name,
    op_meta_str, op_meta_u64, op_meta_f32,
    op_meta_arr_str, op_meta_arr_u32
};

static const eos_format_vt_t k_eosm_format = {
    "eosm", eosm_probe, eosm_open, NULL
};

eos_status_t eosi_format_eosm_register(void) {
    return eos_format_register(&k_eosm_format);
}

#else /* !EOSLLM_HAVE_FORMAT_EOSM */
eos_status_t eosi_format_eosm_register(void) { return EOS_E_UNSUPPORTED; }
#endif
