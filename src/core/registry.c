/*
 * src/core/registry.c — process-wide registry of pluggable modules.
 *
 * Single static instance. All registration entry points push into the
 * appropriate fixed-size table. Lookup is linear; the tables are tiny
 * by construction (≤ EOSI_MAX_*).
 */
#include <stddef.h>
#include <string.h>

#include "internal.h"
#include "eosllm/model.h"

static eosi_registry_t g_registry;

eosi_registry_t *eosi_registry(void) {
    return &g_registry;
}

/* ------------------------------------------------------------------ */
/* Backend                                                             */
/* ------------------------------------------------------------------ */

eos_status_t eos_backend_register(const eos_backend_vt_t *vt) {
    if (vt == NULL || vt->name == NULL) return EOS_E_INVALID_ARG;
    if (g_registry.frozen) return EOS_E_INVALID_ARG;
    if (g_registry.n_backends >= EOSI_MAX_BACKENDS) return EOS_E_OUT_OF_MEMORY;
    g_registry.backends[g_registry.n_backends++] = vt;
    return EOS_OK;
}

/* ------------------------------------------------------------------ */
/* Quant                                                               */
/* ------------------------------------------------------------------ */

eos_status_t eos_quant_register(const eos_quant_vt_t *vt) {
    if (vt == NULL || vt->name == NULL) return EOS_E_INVALID_ARG;
    if (g_registry.frozen) return EOS_E_INVALID_ARG;
    if (g_registry.n_quants >= EOSI_MAX_QUANTS) return EOS_E_OUT_OF_MEMORY;
    g_registry.quants[g_registry.n_quants++] = vt;
    return EOS_OK;
}

const eos_quant_vt_t *eos_quant_find(const char *name) {
    size_t i;
    if (name == NULL) return NULL;
    for (i = 0; i < g_registry.n_quants; ++i) {
        if (strcmp(g_registry.quants[i]->name, name) == 0) {
            return g_registry.quants[i];
        }
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Modality                                                            */
/* ------------------------------------------------------------------ */

eos_status_t eos_modality_register(const eos_modality_vt_t *vt) {
    if (vt == NULL || vt->name == NULL) return EOS_E_INVALID_ARG;
    if (g_registry.frozen) return EOS_E_INVALID_ARG;
    if (g_registry.n_modalities >= EOSI_MAX_MODALITIES) return EOS_E_OUT_OF_MEMORY;
    g_registry.modalities[g_registry.n_modalities++] = vt;
    return EOS_OK;
}

/* ------------------------------------------------------------------ */
/* Tokenizer                                                           */
/* ------------------------------------------------------------------ */

eos_status_t eos_tokenizer_register(const eos_tokenizer_vt_t *vt) {
    if (vt == NULL || vt->name == NULL) return EOS_E_INVALID_ARG;
    if (g_registry.frozen) return EOS_E_INVALID_ARG;
    if (g_registry.n_tokenizers >= EOSI_MAX_TOKENIZERS) return EOS_E_OUT_OF_MEMORY;
    g_registry.tokenizers[g_registry.n_tokenizers++] = vt;
    return EOS_OK;
}

/* ------------------------------------------------------------------ */
/* Scheduler                                                           */
/* ------------------------------------------------------------------ */

eos_status_t eos_sched_register(const eos_sched_vt_t *vt) {
    if (vt == NULL || vt->name == NULL) return EOS_E_INVALID_ARG;
    if (g_registry.frozen) return EOS_E_INVALID_ARG;
    if (g_registry.n_schedulers >= EOSI_MAX_SCHEDULERS) return EOS_E_OUT_OF_MEMORY;
    g_registry.schedulers[g_registry.n_schedulers++] = vt;
    return EOS_OK;
}

/* ------------------------------------------------------------------ */
/* File format                                                         */
/* ------------------------------------------------------------------ */

eos_status_t eos_format_register(const eos_format_vt_t *vt) {
    if (vt == NULL || vt->name == NULL) return EOS_E_INVALID_ARG;
    if (g_registry.frozen) return EOS_E_INVALID_ARG;
    if (g_registry.n_formats >= EOSI_MAX_FORMATS) return EOS_E_OUT_OF_MEMORY;
    g_registry.formats[g_registry.n_formats++] = vt;
    return EOS_OK;
}

int eos_backend_available(const char *name) {
    size_t i;
    if (name == NULL) return 0;
    for (i = 0; i < g_registry.n_backends; ++i) {
        const eos_backend_vt_t *vt = g_registry.backends[i];
        if (vt == NULL || vt->name == NULL) continue;
        if (strcmp(vt->name, name) != 0) continue;
        if (vt->probe == NULL) return 1;     /* always-available backend */
        return vt->probe() == EOS_OK ? 1 : 0;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* OS shim                                                             */
/* ------------------------------------------------------------------ */

eos_status_t eos_os_provide(const eos_os_shim_t *shim) {
    if (shim == NULL) return EOS_E_INVALID_ARG;
    if (g_registry.frozen) return EOS_E_INVALID_ARG;
    g_registry.os = *shim;
    g_registry.os_provided = 1;
    return EOS_OK;
}

void *eosi_alloc(size_t bytes, size_t alignment) {
    if (!g_registry.os_provided || g_registry.os.alloc == NULL) return NULL;
    return g_registry.os.alloc(bytes, alignment);
}

void eosi_free(void *ptr) {
    if (!g_registry.os_provided || g_registry.os.free == NULL) return;
    g_registry.os.free(ptr);
}

uint64_t eosi_now_ns(void) {
    if (!g_registry.os_provided || g_registry.os.now_ns == NULL) return 0;
    return g_registry.os.now_ns();
}

void eosi_log(int level, const char *msg) {
    if (!g_registry.os_provided || g_registry.os.log == NULL || msg == NULL) return;
    g_registry.os.log(level, msg);
}
