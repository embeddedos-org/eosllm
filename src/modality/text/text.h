/*
 * src/modality/text/text.h — internal interface for the text modality.
 */
#ifndef EOSI_MODALITY_TEXT_H
#define EOSI_MODALITY_TEXT_H

#include <stddef.h>
#include <stdint.h>

#include "eosllm/eosllm.h"
#include "eosllm/model.h"
#include "eosllm/tensor.h"
#include "eosllm/backend.h"

/*
 * Per-session text decoder state. Allocated by text_open, freed by
 * text_close. Lives inside the session's arena except for its
 * sub-allocations of weight scratch and the KV cache.
 */
typedef struct {
    eos_model_t            *model;
    const eos_backend_ops_t *ops;     /* dispatch built by session_open */

    /* Architecture parameters loaded from model metadata. */
    uint32_t n_vocab;
    uint32_t n_embd;
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_kv_head;
    uint32_t head_dim;
    uint32_t n_ff;
    uint32_t max_context;
    float    rope_base;
    float    rms_eps;

    /* Tensor pointers (looked up by name). */
    const eos_tensor_t  *tok_embd;       /* [n_vocab, n_embd] */
    const eos_tensor_t  *output_norm;    /* [n_embd] */
    const eos_tensor_t  *output_w;       /* [n_vocab, n_embd] (may alias tok_embd) */
    const eos_tensor_t **layer_attn_norm;   /* [n_layer] each [n_embd] */
    const eos_tensor_t **layer_attn_q;
    const eos_tensor_t **layer_attn_k;
    const eos_tensor_t **layer_attn_v;
    const eos_tensor_t **layer_attn_o;
    const eos_tensor_t **layer_ffn_norm;
    const eos_tensor_t **layer_ffn_gate;
    const eos_tensor_t **layer_ffn_up;
    const eos_tensor_t **layer_ffn_down;

    /*
     * KV cache: laid out as
     *   k_cache[layer][pos][n_kv_head * head_dim]
     *   v_cache[layer][pos][n_kv_head * head_dim]
     * One big f32 allocation per kind.
     */
    float   *k_cache;
    float   *v_cache;
    uint32_t pos;     /* number of positions already in the cache */

    /* Per-step scratch. Sized once at open. */
    float   *scratch;
    size_t   scratch_bytes;
} eosi_text_t;

/* Allocate state, look up tensors, allocate KV. */
eos_status_t eosi_text_open(eos_model_t *model,
                            const eos_backend_ops_t *ops,
                            uint32_t max_context,
                            eosi_text_t **out);

/* Append one token into the cache, return its logits in out_logits
 * (which the caller has sized to n_vocab). */
eos_status_t eosi_text_step(eosi_text_t *st, uint32_t in_token,
                            float *out_logits);

void eosi_text_close(eosi_text_t *st);

/* Module registration (registers a public eos_modality_vt_t named "text"). */
eos_status_t eosi_modality_text_register(void);

#endif
