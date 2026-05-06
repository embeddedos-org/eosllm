/*
 * src/core/session.c — session lifecycle and orchestration.
 *
 * The session ties together: the dispatch table built from registered
 * backends, the active text decoder state, the active tokenizer
 * state, and the active scheduler state. eos_session_step delegates
 * to the scheduler's step (Phase 1: greedy).
 */
#include <stddef.h>
#include <string.h>

#include "eosllm/eosllm.h"
#include "eosllm/backend.h"

#include "internal.h"
#include "../modality/text/text.h"
#include "../sched/sched_internal.h"

struct eos_session {
    eos_model_t                *model;
    eos_session_params_t        params;
    uint64_t                    deadline_abs_ns;

    eos_backend_ops_t           ops;        /* dispatched per-op table */

    eosi_text_t                *text;       /* active text decoder */
    void                       *tok_state;  /* tokenizer state */
    const eos_tokenizer_vt_t   *tok_vt;
    void                       *sched_state;
    const eos_sched_vt_t       *sched_vt;
};

/* ------------------------------------------------------------------ */
/* Build dispatch table from registered backends                       */
/* ------------------------------------------------------------------ */

static void merge_ops(eos_backend_ops_t *dst, const eos_backend_ops_t *src) {
    if (src->matmul_f32) dst->matmul_f32 = src->matmul_f32;
    if (src->matmul_q  ) dst->matmul_q   = src->matmul_q;
    if (src->rmsnorm   ) dst->rmsnorm    = src->rmsnorm;
    if (src->softmax   ) dst->softmax    = src->softmax;
    if (src->rope      ) dst->rope       = src->rope;
    if (src->silu      ) dst->silu       = src->silu;
    if (src->gelu      ) dst->gelu       = src->gelu;
}

static eos_status_t build_dispatch(eos_backend_ops_t *out) {
    eosi_registry_t *r = eosi_registry();
    size_t i, j;
    /* Pass 1: scalar (priority 0) is the baseline. */
    memset(out, 0, sizeof(*out));
    for (i = 0; i < r->n_backends; ++i) {
        if (r->backends[i]->priority == 0) {
            if (r->backends[i]->probe == NULL ||
                r->backends[i]->probe() == EOS_OK) {
                merge_ops(out, &r->backends[i]->ops);
            }
        }
    }
    /* Pass 2: walk in order of increasing priority and override.
     * For Phase 1 we just iterate twice (low → high) since the
     * registry is short. */
    for (j = 1; j < 256; ++j) {
        for (i = 0; i < r->n_backends; ++i) {
            if (r->backends[i]->priority == (int32_t)j) {
                if (r->backends[i]->probe == NULL ||
                    r->backends[i]->probe() == EOS_OK) {
                    merge_ops(out, &r->backends[i]->ops);
                }
            }
        }
    }
    if (out->matmul_f32 == NULL) return EOS_E_UNSUPPORTED;
    return EOS_OK;
}

/* ------------------------------------------------------------------ */
/* Lookup helpers                                                      */
/* ------------------------------------------------------------------ */

static const eos_tokenizer_vt_t *find_tokenizer(const char *name) {
    eosi_registry_t *r = eosi_registry();
    size_t i;
    for (i = 0; i < r->n_tokenizers; ++i) {
        if (r->tokenizers[i]->name && strcmp(r->tokenizers[i]->name, name) == 0)
            return r->tokenizers[i];
    }
    return NULL;
}

static const eos_sched_vt_t *find_scheduler(const char *name) {
    eosi_registry_t *r = eosi_registry();
    size_t i;
    for (i = 0; i < r->n_schedulers; ++i) {
        if (r->schedulers[i]->name && strcmp(r->schedulers[i]->name, name) == 0)
            return r->schedulers[i];
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Session lifecycle                                                   */
/* ------------------------------------------------------------------ */

eos_status_t eos_session_open(eos_model_t                *model,
                              const eos_session_params_t *params,
                              eos_session_t             **out_session) {
    eos_session_t *s;
    eos_status_t   rc;
    const char    *sched_name;

    if (out_session == NULL) return EOS_E_INVALID_ARG;
    *out_session = NULL;

    /* Clear thread-local last-error so EOS_OK leaves the slot empty;
     * any EOSI_LOG_ERROR inside the open path will set it for retrieval
     * via eos_last_error(). Documented in docs/abi.md. */
    eosi_set_error(NULL);

    /*
     * Freeze the registry. Once any session is opened the dispatch
     * tables must not change underneath in-flight sessions; subsequent
     * eos_*_register calls return EOS_E_INVALID_ARG. See docs/abi.md.
     * One-way; never reset.
     */
    eosi_registry()->frozen = 1;

    s = (eos_session_t *)eosi_alloc(sizeof(*s), 16);
    if (s == NULL) return EOS_E_OUT_OF_MEMORY;
    memset(s, 0, sizeof(*s));
    s->model = model;
    if (params != NULL) s->params = *params;

    /* Build dispatch table. */
    rc = build_dispatch(&s->ops);
    if (rc != EOS_OK) { eos_session_close(s); return rc; }

    /* If we have a model, try to open the text decoder. (Sessions
     * without a model are useful for the no-op smoke tests.) */
    if (model != NULL) {
        rc = eosi_text_open(model, &s->ops, s->params.max_context, &s->text);
        if (rc != EOS_OK && rc != EOS_E_UNSUPPORTED) {
            EOSI_LOG_ERROR("session_open: text decoder open failed "
                           "(check general.architecture metadata)");
            eos_session_close(s); return rc;
        }
        /* Tokenizer (BPE only in Phase 1). */
        s->tok_vt = find_tokenizer("bpe");
        if (s->tok_vt != NULL) {
            rc = s->tok_vt->open(model, 0, &s->tok_state);
            if (rc != EOS_OK) {
                EOSI_LOG_ERROR("session_open: tokenizer open failed "
                               "(check tokenizer.ggml.* metadata)");
                eos_session_close(s); return rc;
            }
        }
        /* Scheduler. */
        sched_name = (s->params.scheduler != NULL) ? s->params.scheduler : "greedy";
        s->sched_vt = find_scheduler(sched_name);
        if (s->sched_vt == NULL) {
            EOSI_LOG_ERROR("session_open: scheduler not found by name "
                           "(check params.scheduler against compiled-in set)");
            eos_session_close(s); return EOS_E_NOT_FOUND;
        }
        rc = s->sched_vt->open(s, &s->sched_state);
        if (rc != EOS_OK) {
            EOSI_LOG_ERROR("session_open: scheduler open failed");
            eos_session_close(s); return rc;
        }
        if (s->text != NULL && strcmp(s->sched_vt->name, "greedy") == 0) {
            float *logits = (float *)eosi_alloc(s->text->n_vocab * sizeof(float), 16);
            if (logits == NULL) {
                EOSI_LOG_ERROR("session_open: alloc failed for greedy logits buffer");
                eos_session_close(s); return EOS_E_OUT_OF_MEMORY;
            }
            eosi_greedy_attach(s->sched_state, s->text, logits);
        }
    }

    *out_session = s;
    return EOS_OK;
}

eos_status_t eos_session_feed(eos_session_t      *session,
                              eos_modality_kind_t modality,
                              const void         *data,
                              size_t              len) {
    eosi_set_error(NULL);
    if (session == NULL) return EOS_E_INVALID_ARG;
    if (modality != EOS_MODALITY_TEXT) return EOS_E_UNSUPPORTED;
    if (session->tok_vt == NULL || session->text == NULL) return EOS_E_UNSUPPORTED;

    {
        /* Stage 1: tokenize. */
        size_t        cap = (len > 0 ? len : 1) + 8;
        uint32_t     *ids = (uint32_t *)eosi_alloc(cap * sizeof(uint32_t), 4);
        size_t        n   = cap;
        eos_status_t  s;
        size_t        i;
        if (ids == NULL) return EOS_E_OUT_OF_MEMORY;
        s = session->tok_vt->encode(session->tok_state,
                                    (const char *)data, len, ids, &n);
        if (s != EOS_OK) { eosi_free(ids); return s; }
        if (n == 0) { eosi_free(ids); return EOS_OK; }

        /* Stage 2: prefill all tokens through the decoder so KV is
         * populated. The very last token becomes the seed for the
         * scheduler's next step. */
        {
            float *throwaway = (float *)eosi_alloc(
                session->text->n_vocab * sizeof(float), 16);
            if (throwaway == NULL) { eosi_free(ids); return EOS_E_OUT_OF_MEMORY; }
            for (i = 0; i < n; ++i) {
                eos_status_t r = eosi_text_step(session->text, ids[i], throwaway);
                if (r != EOS_OK) {
                    eosi_free(throwaway); eosi_free(ids); return r;
                }
            }
            eosi_free(throwaway);
        }
        /* Seed the scheduler with the last logits' implied token, i.e.
         * the last input token (for greedy this isn't strictly used —
         * the scheduler's next step computes from where pos points). */
        if (session->sched_state != NULL) {
            eosi_greedy_seed(session->sched_state, ids[n - 1]);
        }
        eosi_free(ids);
    }
    return EOS_OK;
}

eos_status_t eos_session_step(eos_session_t *session, uint32_t *out_token) {
    eosi_set_error(NULL);
    if (session == NULL || out_token == NULL) return EOS_E_INVALID_ARG;
    if (session->sched_vt == NULL) return EOS_E_UNSUPPORTED;
    return session->sched_vt->step(session->sched_state, out_token);
}

eos_status_t eos_session_set_deadline(eos_session_t *session,
                                      uint64_t       ns_from_now) {
    if (session == NULL) return EOS_E_INVALID_ARG;
    if (ns_from_now == 0) {
        session->deadline_abs_ns = 0;
    } else {
        session->deadline_abs_ns = eosi_now_ns() + ns_from_now;
    }
    return EOS_OK;
}

eos_status_t eos_session_decode(eos_session_t *session,
                                const uint32_t *ids, size_t n_ids,
                                char *out_text, size_t *io_len) {
    eosi_set_error(NULL);
    if (session == NULL || io_len == NULL) return EOS_E_INVALID_ARG;
    if (session->tok_vt == NULL || session->tok_state == NULL)
        return EOS_E_UNSUPPORTED;
    return session->tok_vt->decode(session->tok_state, ids, n_ids,
                                   out_text, io_len);
}

void eos_session_close(eos_session_t *session) {
    if (session == NULL) return;
    if (session->sched_vt != NULL && session->sched_vt->close != NULL) {
        session->sched_vt->close(session->sched_state);
    }
    if (session->tok_vt != NULL && session->tok_vt->close != NULL) {
        session->tok_vt->close(session->tok_state);
    }
    if (session->text != NULL) {
        eosi_text_close(session->text);
    }
    eosi_free(session);
}

void eos_free(void *ptr) {
    /* eosi_free already tolerates NULL. */
    eosi_free(ptr);
}
