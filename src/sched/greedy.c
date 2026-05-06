/*
 * src/sched/greedy.c — single-stream greedy / temperature sampler.
 *
 * The scheduler's `step` is the per-token entry point used by
 * eos_session_step. It calls into the active modality's decoder to
 * produce logits, then samples one token id (greedy when
 * temperature == 0, multinomial via top-k/top-p otherwise — Phase 1
 * implements greedy only; full top-k/top-p comes in Phase 2 alongside
 * the proper sampling utilities).
 *
 * Because Phase 1 has only one modality (text), the scheduler state
 * holds a pointer to the session's text decoder state; session.c sets
 * this up at open. A more general dispatcher comes in Phase 3.
 */
#include <stddef.h>
#include <stdint.h>

#include "eosllm/eosllm.h"
#include "eosllm/scheduler.h"

#include "../core/internal.h"
#include "../modality/text/text.h"
#include "sched_internal.h"

#if EOSLLM_HAVE_SCHED_GREEDY

typedef struct {
    eosi_text_t *text;        /* set by session.c after open */
    float       *logits;      /* size n_vocab */
    uint32_t     last_token;  /* token to feed next (the seed token) */
} greedy_state_t;

static eos_status_t greedy_open(eos_session_t *s, void **out_state) {
    greedy_state_t *gs;
    (void)s;
    gs = (greedy_state_t *)eosi_alloc(sizeof(*gs), 16);
    if (gs == NULL) {
        EOSI_LOG_ERROR("sched/greedy: alloc failed for greedy state");
        return EOS_E_OUT_OF_MEMORY;
    }
    gs->text       = NULL;
    gs->logits     = NULL;
    gs->last_token = 0;
    *out_state = gs;
    return EOS_OK;
}

static eos_status_t greedy_step(void *state, uint32_t *out_token) {
    greedy_state_t *gs = (greedy_state_t *)state;
    eos_status_t s;
    uint32_t i, best = 0;
    float    bestv;
    if (gs == NULL || out_token == NULL) {
        EOSI_LOG_ERROR("sched/greedy: NULL state or out_token argument");
        return EOS_E_INVALID_ARG;
    }
    if (gs->text == NULL || gs->logits == NULL) {
        EOSI_LOG_ERROR("sched/greedy: state not attached to a text decoder "
                       "(eosi_greedy_attach was not called)");
        return EOS_E_INTERNAL;
    }

    s = eosi_text_step(gs->text, gs->last_token, gs->logits);
    if (s != EOS_OK) return s;

    bestv = gs->logits[0];
    for (i = 1; i < gs->text->n_vocab; ++i) {
        if (gs->logits[i] > bestv) { bestv = gs->logits[i]; best = i; }
    }
    gs->last_token = best;
    *out_token     = best;
    return EOS_OK;
}

static void greedy_close(void *state) {
    greedy_state_t *gs = (greedy_state_t *)state;
    if (gs == NULL) return;
    eosi_free(gs->logits);
    eosi_free(gs);
}

/*
 * Internal helpers used by session.c to attach the text decoder state
 * and the logits scratch to a freshly-opened greedy scheduler.
 */
void eosi_greedy_attach(void *state, void *text, float *logits) {
    greedy_state_t *gs = (greedy_state_t *)state;
    if (gs == NULL) return;
    gs->text   = (eosi_text_t *)text;
    gs->logits = logits;
}

void eosi_greedy_seed(void *state, uint32_t token) {
    greedy_state_t *gs = (greedy_state_t *)state;
    if (gs == NULL) return;
    gs->last_token = token;
}

static const eos_sched_vt_t k_greedy = {
    "greedy", greedy_open, greedy_step, greedy_close
};

eos_status_t eosi_sched_greedy_register(void) {
    return eos_sched_register(&k_greedy);
}

#else /* !EOSLLM_HAVE_SCHED_GREEDY */
eos_status_t eosi_sched_greedy_register(void) { return EOS_E_UNSUPPORTED; }
void eosi_greedy_attach(void *s, void *t, float *l)
    { (void)s; (void)t; (void)l; }
void eosi_greedy_seed(void *s, uint32_t t) { (void)s; (void)t; }
#endif
