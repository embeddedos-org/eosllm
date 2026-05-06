/*
 * src/modality/text/text.c — Llama-class transformer text decoder.
 *
 * Phase 1 implementation:
 *   - Loads architecture parameters from GGUF metadata under the
 *     `<arch>.*` namespace (llama.* / qwen2.*).
 *   - Looks up weight tensors by GGUF naming convention.
 *   - Forward pass: token-embed → N × (attn + FFN) → output norm →
 *     output projection. GQA supported (n_head, n_kv_head).
 *   - Weights may be F32, F16, Q8_0, or Q4_K. F16 is dequantized
 *     row-by-row into a scratch buffer; Q8_0/Q4_K go through the
 *     scalar backend's matmul_q (or any registered higher-priority
 *     fast path).
 *
 * Performance: this is the always-correct reference path. AVX2 / NEON
 * fast paths come in via the backend ops vtable (Phase 4).
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "eosllm/eosllm.h"
#include "eosllm/modality.h"
#include "eosllm/quant.h"

#include "text.h"
#include "../../core/internal.h"
#include "../../core/model_internal.h"
#include "../../util/f16.h"

#if EOSLLM_HAVE_MODALITY_TEXT

/* Internal accessor (defined in src/core/tensor.c). */
extern const void *eosi_tensor_data(const eos_tensor_t *t);

/* ------------------------------------------------------------------ */
/* Metadata loading                                                    */
/* ------------------------------------------------------------------ */

/*
 * GGUF metadata keys are namespaced by architecture: e.g.
 *   llama.embedding_length, llama.attention.head_count, ...
 * We query "general.architecture" first to get the namespace.
 */
static eos_status_t load_arch_u64(eos_model_t *m, const char *arch,
                                  const char *suffix, uint64_t *out) {
    char key[128];
    int n = snprintf(key, sizeof(key), "%s.%s", arch, suffix);
    if (n <= 0 || (size_t)n >= sizeof(key)) return EOS_E_INVALID_ARG;
    return eos_model_meta_u64(m, key, out);
}

static eos_status_t load_arch_f32(eos_model_t *m, const char *arch,
                                  const char *suffix, float *out) {
    char key[128];
    int n = snprintf(key, sizeof(key), "%s.%s", arch, suffix);
    if (n <= 0 || (size_t)n >= sizeof(key)) return EOS_E_INVALID_ARG;
    return eos_model_meta_f32(m, key, out);
}

/* Look up a per-layer weight: "blk.{i}.{name}". Returns NULL if absent. */
static const eos_tensor_t *find_layer(eos_model_t *m, uint32_t i, const char *name) {
    char key[128];
    int n = snprintf(key, sizeof(key), "blk.%u.%s", (unsigned)i, name);
    if (n <= 0 || (size_t)n >= sizeof(key)) return NULL;
    return eos_model_tensor_by_name(m, key);
}

/* ------------------------------------------------------------------ */
/* Matmul against a weight tensor of arbitrary dtype                   */
/* ------------------------------------------------------------------ */

static eos_status_t matmul_w(const eosi_text_t *st,
                             const float *x, const eos_tensor_t *w,
                             float *y, uint32_t out_dim, uint32_t in_dim,
                             float *scratch) {
    const void *wd = eosi_tensor_data(w);
    if (wd == NULL) return EOS_E_INVALID_ARG;
    switch (eos_tensor_dtype(w)) {
        case EOS_DT_F32:
            return st->ops->matmul_f32(x, (const float *)wd, y, 1, out_dim, in_dim);
        case EOS_DT_F16: {
            /* Dequantize entire weight to scratch then call matmul_f32. */
            const eosi_f16_t *src = (const eosi_f16_t *)wd;
            size_t i, total = (size_t)out_dim * in_dim;
            for (i = 0; i < total; ++i) scratch[i] = eosi_f16_to_f32(src[i]);
            return st->ops->matmul_f32(x, scratch, y, 1, out_dim, in_dim);
        }
        case EOS_DT_Q8_0:
        case EOS_DT_Q4_K:
            if (st->ops->matmul_q == NULL) return EOS_E_UNSUPPORTED;
            return st->ops->matmul_q(x, wd, eos_tensor_dtype(w),
                                     y, 1, out_dim, in_dim);
        default:
            return EOS_E_UNSUPPORTED;
    }
}

/* ------------------------------------------------------------------ */
/* eosi_text_open                                                      */
/* ------------------------------------------------------------------ */

eos_status_t eosi_text_open(eos_model_t *model,
                            const eos_backend_ops_t *ops,
                            uint32_t max_context,
                            eosi_text_t **out) {
    eosi_text_t *st;
    const char  *arch;
    uint64_t     v;
    uint32_t     i;
    size_t       largest_w = 0;
    size_t       kv_per_layer;
    eos_status_t s;

    if (model == NULL || ops == NULL || out == NULL) return EOS_E_INVALID_ARG;
    *out = NULL;

    arch = eos_model_meta_str(model, "general.architecture");
    if (arch == NULL) return EOS_E_FORMAT;

    st = (eosi_text_t *)eosi_alloc(sizeof(*st), 16);
    if (st == NULL) return EOS_E_OUT_OF_MEMORY;
    memset(st, 0, sizeof(*st));
    st->model = model;
    st->ops   = ops;

    /* Required parameters. */
    if ((s = load_arch_u64(model, arch, "embedding_length",         &v)) != EOS_OK) goto fail;
    st->n_embd = (uint32_t)v;
    if ((s = load_arch_u64(model, arch, "block_count",              &v)) != EOS_OK) goto fail;
    st->n_layer = (uint32_t)v;
    if ((s = load_arch_u64(model, arch, "attention.head_count",     &v)) != EOS_OK) goto fail;
    st->n_head  = (uint32_t)v;
    if (load_arch_u64(model, arch, "attention.head_count_kv", &v) == EOS_OK)
        st->n_kv_head = (uint32_t)v;
    else
        st->n_kv_head = st->n_head;
    if ((s = load_arch_u64(model, arch, "feed_forward_length",      &v)) != EOS_OK) goto fail;
    st->n_ff = (uint32_t)v;
    if ((s = load_arch_u64(model, arch, "context_length",           &v)) != EOS_OK) goto fail;
    st->max_context = (uint32_t)v;
    if (max_context > 0 && max_context < st->max_context) st->max_context = max_context;

    if (st->n_head == 0 || st->n_kv_head == 0) { s = EOS_E_FORMAT; goto fail; }
    st->head_dim = st->n_embd / st->n_head;

    if (load_arch_f32(model, arch, "rope.freq_base", &st->rope_base) != EOS_OK)
        st->rope_base = 10000.0f;
    if (load_arch_f32(model, arch, "attention.layer_norm_rms_epsilon",
                      &st->rms_eps) != EOS_OK)
        st->rms_eps = 1e-5f;

    /* Tensors. */
    st->tok_embd    = eos_model_tensor_by_name(model, "token_embd.weight");
    st->output_norm = eos_model_tensor_by_name(model, "output_norm.weight");
    st->output_w    = eos_model_tensor_by_name(model, "output.weight");
    if (st->output_w == NULL) st->output_w = st->tok_embd;   /* tied */
    if (st->tok_embd == NULL || st->output_norm == NULL) {
        s = EOS_E_FORMAT; goto fail;
    }
    {
        uint32_t vocab_dim_idx = 0;   /* tok_embd shape: [n_embd, n_vocab] in GGUF */
        st->n_vocab = st->tok_embd->dims[1];
        if (st->n_vocab == 0) st->n_vocab = st->tok_embd->dims[0];
        (void)vocab_dim_idx;
    }

    /* Per-layer pointers. */
#define ALLOC_LAYER_PTRS(field) \
    st->field = (const eos_tensor_t **)eosi_alloc( \
        st->n_layer * sizeof(const eos_tensor_t *), 8); \
    if (st->field == NULL) { s = EOS_E_OUT_OF_MEMORY; goto fail; }

    ALLOC_LAYER_PTRS(layer_attn_norm);
    ALLOC_LAYER_PTRS(layer_attn_q);
    ALLOC_LAYER_PTRS(layer_attn_k);
    ALLOC_LAYER_PTRS(layer_attn_v);
    ALLOC_LAYER_PTRS(layer_attn_o);
    ALLOC_LAYER_PTRS(layer_ffn_norm);
    ALLOC_LAYER_PTRS(layer_ffn_gate);
    ALLOC_LAYER_PTRS(layer_ffn_up);
    ALLOC_LAYER_PTRS(layer_ffn_down);
#undef ALLOC_LAYER_PTRS

    for (i = 0; i < st->n_layer; ++i) {
        st->layer_attn_norm[i] = find_layer(model, i, "attn_norm.weight");
        st->layer_attn_q   [i] = find_layer(model, i, "attn_q.weight");
        st->layer_attn_k   [i] = find_layer(model, i, "attn_k.weight");
        st->layer_attn_v   [i] = find_layer(model, i, "attn_v.weight");
        st->layer_attn_o   [i] = find_layer(model, i, "attn_output.weight");
        st->layer_ffn_norm [i] = find_layer(model, i, "ffn_norm.weight");
        st->layer_ffn_gate [i] = find_layer(model, i, "ffn_gate.weight");
        st->layer_ffn_up   [i] = find_layer(model, i, "ffn_up.weight");
        st->layer_ffn_down [i] = find_layer(model, i, "ffn_down.weight");
        if (!st->layer_attn_norm[i] || !st->layer_attn_q[i] || !st->layer_attn_k[i]
            || !st->layer_attn_v[i] || !st->layer_attn_o[i] || !st->layer_ffn_norm[i]
            || !st->layer_ffn_gate[i] || !st->layer_ffn_up[i] || !st->layer_ffn_down[i]) {
            s = EOS_E_FORMAT; goto fail;
        }
        /* Track the largest weight so we can size F16-dequant scratch. */
        {
            const eos_tensor_t *ws[] = { st->layer_attn_q[i], st->layer_attn_k[i],
                                         st->layer_attn_v[i], st->layer_attn_o[i],
                                         st->layer_ffn_gate[i], st->layer_ffn_up[i],
                                         st->layer_ffn_down[i] };
            unsigned k;
            for (k = 0; k < sizeof(ws)/sizeof(ws[0]); ++k) {
                size_t sz = (size_t)ws[k]->dims[0] * ws[k]->dims[1];
                if (sz > largest_w) largest_w = sz;
            }
        }
    }
    {
        size_t sz = (size_t)st->output_w->dims[0] * st->output_w->dims[1];
        if (sz > largest_w) largest_w = sz;
    }

    /* KV cache. */
    kv_per_layer = (size_t)st->max_context * st->n_kv_head * st->head_dim;
    st->k_cache = (float *)eosi_alloc(
        st->n_layer * kv_per_layer * sizeof(float), 16);
    st->v_cache = (float *)eosi_alloc(
        st->n_layer * kv_per_layer * sizeof(float), 16);
    if (!st->k_cache || !st->v_cache) { s = EOS_E_OUT_OF_MEMORY; goto fail; }

    /* Scratch: largest of {dequantized weight, sized work buffers}. */
    {
        size_t need_w = largest_w * sizeof(float);
        size_t need_qkv = (size_t)(st->n_embd + 2 * st->n_kv_head * st->head_dim
                                   + 2 * st->n_ff + st->n_vocab) * sizeof(float);
        st->scratch_bytes = (need_w > need_qkv) ? need_w : need_qkv;
        st->scratch = (float *)eosi_alloc(st->scratch_bytes, 16);
        if (st->scratch == NULL) { s = EOS_E_OUT_OF_MEMORY; goto fail; }
    }

    *out = st;
    return EOS_OK;

fail:
    eosi_text_close(st);
    return s;
}

void eosi_text_close(eosi_text_t *st) {
    if (st == NULL) return;
    eosi_free(st->layer_attn_norm);
    eosi_free(st->layer_attn_q);
    eosi_free(st->layer_attn_k);
    eosi_free(st->layer_attn_v);
    eosi_free(st->layer_attn_o);
    eosi_free(st->layer_ffn_norm);
    eosi_free(st->layer_ffn_gate);
    eosi_free(st->layer_ffn_up);
    eosi_free(st->layer_ffn_down);
    eosi_free(st->k_cache);
    eosi_free(st->v_cache);
    eosi_free(st->scratch);
    eosi_free(st);
}

/* ------------------------------------------------------------------ */
/* eosi_text_step                                                      */
/* ------------------------------------------------------------------ */

eos_status_t eosi_text_step(eosi_text_t *st, uint32_t in_token,
                            float *out_logits) {
    uint32_t n_embd, n_head, n_kvh, head_dim, n_ff;
    uint32_t pos;
    float   *x, *xn, *q, *k, *v, *attn_out, *ff_gate, *ff_up, *ff_tmp;
    uint8_t *cursor;
    uint32_t L, h, p, i;
    size_t   kv_stride;

    if (st == NULL || out_logits == NULL) return EOS_E_INVALID_ARG;
    if (in_token >= st->n_vocab)           return EOS_E_INVALID_ARG;
    if (st->pos >= st->max_context)        return EOS_E_OUT_OF_MEMORY;

    n_embd   = st->n_embd;
    n_head   = st->n_head;
    n_kvh    = st->n_kv_head;
    head_dim = st->head_dim;
    n_ff     = st->n_ff;
    pos      = st->pos;
    kv_stride = (size_t)st->max_context * n_kvh * head_dim;

    /* Lay out scratch. We need: x[n_embd], xn[n_embd], q[n_embd],
     * k[n_kvh*head_dim], v[n_kvh*head_dim], attn_out[n_embd],
     * ff_gate[n_ff], ff_up[n_ff], ff_tmp[n_ff], plus the largest
     * weight-dequant buffer for F16 fallback (used only inside matmul_w). */
    cursor = (uint8_t *)st->scratch;
    x        = (float *)cursor; cursor += n_embd * sizeof(float);
    xn       = (float *)cursor; cursor += n_embd * sizeof(float);
    q        = (float *)cursor; cursor += n_embd * sizeof(float);
    k        = (float *)cursor; cursor += n_kvh * head_dim * sizeof(float);
    v        = (float *)cursor; cursor += n_kvh * head_dim * sizeof(float);
    attn_out = (float *)cursor; cursor += n_embd * sizeof(float);
    ff_gate  = (float *)cursor; cursor += n_ff * sizeof(float);
    ff_up    = (float *)cursor; cursor += n_ff * sizeof(float);
    ff_tmp   = (float *)cursor; cursor += n_ff * sizeof(float);
    /* Remaining scratch is for matmul_w's F16 dequant. */
    {
        size_t used = (size_t)(cursor - (uint8_t *)st->scratch);
        if (used > st->scratch_bytes) return EOS_E_INTERNAL;
    }

    /* 1. Token embedding row → x. */
    {
        const eos_tensor_t *te = st->tok_embd;
        eos_dtype_t dt = eos_tensor_dtype(te);
        const void *td = eosi_tensor_data(te);
        if (td == NULL) return EOS_E_FORMAT;
        if (dt == EOS_DT_F32) {
            memcpy(x, (const float *)td + (size_t)in_token * n_embd,
                   n_embd * sizeof(float));
        } else if (dt == EOS_DT_F16) {
            const eosi_f16_t *src = (const eosi_f16_t *)td
                                    + (size_t)in_token * n_embd;
            for (i = 0; i < n_embd; ++i) x[i] = eosi_f16_to_f32(src[i]);
        } else {
            /*
             * Quantized embeddings: dequantize the row via the
             * registered quant scheme. For Phase 1 we materialize the
             * whole vocab row using its layout. (Slow; revisit later.)
             */
            return EOS_E_UNSUPPORTED;
        }
    }

    /* 2. Decoder layers. */
    for (L = 0; L < st->n_layer; ++L) {
        const eos_tensor_t *wq = st->layer_attn_q[L];
        const eos_tensor_t *wk = st->layer_attn_k[L];
        const eos_tensor_t *wv = st->layer_attn_v[L];
        const eos_tensor_t *wo = st->layer_attn_o[L];
        const eos_tensor_t *wg = st->layer_ffn_gate[L];
        const eos_tensor_t *wu = st->layer_ffn_up[L];
        const eos_tensor_t *wd = st->layer_ffn_down[L];
        const eos_tensor_t *an = st->layer_attn_norm[L];
        const eos_tensor_t *fn = st->layer_ffn_norm[L];
        eos_status_t r;

        /* 2a. xn = rmsnorm(x, an) */
        r = st->ops->rmsnorm(x, (const float *)eosi_tensor_data(an), st->rms_eps,
                             xn, 1, n_embd);
        if (r != EOS_OK) return r;

        /* 2b. q = xn · W_q^T, k = xn · W_k^T, v = xn · W_v^T */
        r = matmul_w(st, xn, wq, q, n_embd,           n_embd, ff_tmp); if (r) return r;
        r = matmul_w(st, xn, wk, k, n_kvh * head_dim, n_embd, ff_tmp); if (r) return r;
        r = matmul_w(st, xn, wv, v, n_kvh * head_dim, n_embd, ff_tmp); if (r) return r;

        /* 2c. RoPE on q (per head) and k (per kv-head). */
        r = st->ops->rope(q, n_head, head_dim, pos, st->rope_base); if (r) return r;
        r = st->ops->rope(k, n_kvh,  head_dim, pos, st->rope_base); if (r) return r;

        /* 2d. Append k, v to KV cache at position `pos`. */
        memcpy(&st->k_cache[L * kv_stride + (size_t)pos * n_kvh * head_dim],
               k, n_kvh * head_dim * sizeof(float));
        memcpy(&st->v_cache[L * kv_stride + (size_t)pos * n_kvh * head_dim],
               v, n_kvh * head_dim * sizeof(float));

        /* 2e. Scaled dot-product attention with GQA grouping. For each
         * query head h, the kv head is h * n_kvh / n_head. */
        {
            const float scale = 1.0f / sqrtf((float)head_dim);
            const float *Kc = &st->k_cache[L * kv_stride];
            const float *Vc = &st->v_cache[L * kv_stride];
            float *att = ff_gate;   /* reuse ff_gate as attn-score scratch */
            memset(attn_out, 0, n_embd * sizeof(float));
            for (h = 0; h < n_head; ++h) {
                uint32_t kvh = (h * n_kvh) / n_head;
                const float *qh = q + h * head_dim;
                float maxv = -1e30f, sum = 0.0f;
                for (p = 0; p <= pos; ++p) {
                    const float *kp = Kc + (size_t)p * n_kvh * head_dim
                                      + (size_t)kvh * head_dim;
                    float dot = 0.0f;
                    for (i = 0; i < head_dim; ++i) dot += qh[i] * kp[i];
                    att[p] = dot * scale;
                    if (att[p] > maxv) maxv = att[p];
                }
                for (p = 0; p <= pos; ++p) {
                    att[p] = expf(att[p] - maxv);
                    sum += att[p];
                }
                if (sum > 0.0f) {
                    float inv = 1.0f / sum;
                    for (p = 0; p <= pos; ++p) att[p] *= inv;
                }
                /* attn_out_h = sum_p att[p] * V[p, kvh, :] */
                for (p = 0; p <= pos; ++p) {
                    const float *vp = Vc + (size_t)p * n_kvh * head_dim
                                      + (size_t)kvh * head_dim;
                    float a = att[p];
                    for (i = 0; i < head_dim; ++i)
                        attn_out[h * head_dim + i] += a * vp[i];
                }
            }
        }

        /* 2f. attn_out = attn_out · W_o^T, add to residual. */
        r = matmul_w(st, attn_out, wo, ff_tmp /*n_embd-sized scratch*/, n_embd, n_embd, q);
        if (r) return r;
        for (i = 0; i < n_embd; ++i) x[i] += ff_tmp[i];

        /* 2g. xn = rmsnorm(x, fn) */
        r = st->ops->rmsnorm(x, (const float *)eosi_tensor_data(fn), st->rms_eps,
                             xn, 1, n_embd);
        if (r) return r;

        /* 2h. ff_gate = silu(xn · W_gate^T) ⊙ (xn · W_up^T) */
        r = matmul_w(st, xn, wg, ff_gate, n_ff, n_embd, q); if (r) return r;
        r = matmul_w(st, xn, wu, ff_up,   n_ff, n_embd, q); if (r) return r;
        r = st->ops->silu(ff_gate, ff_gate, n_ff);          if (r) return r;
        for (i = 0; i < n_ff; ++i) ff_tmp[i] = ff_gate[i] * ff_up[i];

        /* 2i. attn_out = ff_tmp · W_down^T, add to residual. */
        r = matmul_w(st, ff_tmp, wd, attn_out, n_embd, n_ff, q); if (r) return r;
        for (i = 0; i < n_embd; ++i) x[i] += attn_out[i];
    }

    /* 3. Final norm + output projection. */
    {
        eos_status_t r = st->ops->rmsnorm(
            x, (const float *)eosi_tensor_data(st->output_norm), st->rms_eps,
            xn, 1, n_embd);
        if (r) return r;
        r = matmul_w(st, xn, st->output_w, out_logits, st->n_vocab, n_embd, q);
        if (r) return r;
    }

    st->pos++;
    return EOS_OK;
}

/* ------------------------------------------------------------------ */
/* Public modality vtable                                              */
/* ------------------------------------------------------------------ */

/*
 * The public eos_modality_vt_t expects open(session, model, &state)
 * and encode(state, data, len). Phase 1 wiring of these into the
 * full session lifecycle lives in src/core/session.c; here we expose
 * the registration just so capability bits flip on.
 */
static eos_status_t modality_open(eos_session_t *s, eos_model_t *m, void **out) {
    (void)s; (void)m; (void)out;
    /* The session knows how to construct the text state directly via
     * eosi_text_open; this stub is here only to satisfy the registry. */
    return EOS_E_UNSUPPORTED;
}

static eos_status_t modality_encode(void *state, const void *data, size_t len) {
    (void)state; (void)data; (void)len;
    return EOS_E_UNSUPPORTED;
}

static void modality_close(void *state) { (void)state; }

static const eos_modality_vt_t k_text = {
    "text", EOS_MODALITY_TEXT,
    modality_open, modality_encode, modality_close
};

eos_status_t eosi_modality_text_register(void) {
    return eos_modality_register(&k_text);
}

#else /* !EOSLLM_HAVE_MODALITY_TEXT */

eos_status_t eosi_modality_text_register(void) { return EOS_E_UNSUPPORTED; }
eos_status_t eosi_text_open(eos_model_t *m, const eos_backend_ops_t *o,
                            uint32_t mc, eosi_text_t **out) {
    (void)m; (void)o; (void)mc; if (out) *out = NULL;
    return EOS_E_UNSUPPORTED;
}
eos_status_t eosi_text_step(eosi_text_t *s, uint32_t t, float *l) {
    (void)s; (void)t; (void)l; return EOS_E_UNSUPPORTED;
}
void eosi_text_close(eosi_text_t *s) { (void)s; }

#endif
