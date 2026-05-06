/*
 * tools/eosllm-convert/main.c — eosllm-convert C CLI.
 *
 * Implements `eosllm-convert from-gguf <input.gguf> -o <output.eosm>`
 * by:
 *   1. Opening the GGUF via the registered format reader.
 *   2. Probing a fixed Llama-style key table for which metadata
 *      entries are actually present.
 *   3. Calling eosi_eosm_from_model(...) with the present specs.
 *   4. Writing the resulting buffer to disk.
 *
 * Lossless pass-through for any built-in dtype (f32 / f16 / q8_0 / q4_k):
 * tensor bytes are copied byte-for-byte; metadata is filtered to the
 * keys this tool knows. New architectures are a one-line table addition.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eosllm/eosllm.h"
#include "eosllm/model.h"

/* Internal converter entry point (see src/format/eosm/eosm_internal.h). */
#include "../../src/format/eosm/eosm_internal.h"

typedef struct {
    const char *key;
    uint32_t    type;
    uint32_t    arr_elem_type;
} kv_descr_t;

/* Llama-3 / Llama-2 / TinyLlama key table. Optional keys are simply
 * skipped if absent in the source. Add new architectures as new tables. */
static const kv_descr_t k_llama_keys[] = {
    /* General. */
    { "general.architecture",                       EOSI_EOSM_KV_STRING, 0 },
    { "general.name",                               EOSI_EOSM_KV_STRING, 0 },
    { "general.file_type",                          EOSI_EOSM_KV_U32,    0 },
    /* Architecture hyperparameters. */
    { "llama.context_length",                       EOSI_EOSM_KV_U32,    0 },
    { "llama.embedding_length",                     EOSI_EOSM_KV_U32,    0 },
    { "llama.block_count",                          EOSI_EOSM_KV_U32,    0 },
    { "llama.feed_forward_length",                  EOSI_EOSM_KV_U32,    0 },
    { "llama.attention.head_count",                 EOSI_EOSM_KV_U32,    0 },
    { "llama.attention.head_count_kv",              EOSI_EOSM_KV_U32,    0 },
    { "llama.attention.layer_norm_rms_epsilon",     EOSI_EOSM_KV_F32,    0 },
    { "llama.rope.dimension_count",                 EOSI_EOSM_KV_U32,    0 },
    { "llama.rope.freq_base",                       EOSI_EOSM_KV_F32,    0 },
    /* Tokenizer. */
    { "tokenizer.ggml.model",                       EOSI_EOSM_KV_STRING, 0 },
    { "tokenizer.ggml.pre",                         EOSI_EOSM_KV_STRING, 0 },
    { "tokenizer.ggml.tokens",                      EOSI_EOSM_KV_ARRAY,
                                                    EOSI_EOSM_KV_STRING },
    { "tokenizer.ggml.merges",                      EOSI_EOSM_KV_ARRAY,
                                                    EOSI_EOSM_KV_STRING },
    { "tokenizer.ggml.token_type",                  EOSI_EOSM_KV_ARRAY,
                                                    EOSI_EOSM_KV_U32 },
    { "tokenizer.ggml.bos_token_id",                EOSI_EOSM_KV_U32,    0 },
    { "tokenizer.ggml.eos_token_id",                EOSI_EOSM_KV_U32,    0 },
    { "tokenizer.ggml.padding_token_id",            EOSI_EOSM_KV_U32,    0 },
};
#define N_LLAMA_KEYS (sizeof(k_llama_keys) / sizeof(k_llama_keys[0]))

/* Returns 1 if a key is present (using the right accessor for its type). */
static int key_present(const eos_model_t *m, const kv_descr_t *d) {
    switch (d->type) {
        case EOSI_EOSM_KV_STRING:
            return eos_model_meta_str(m, d->key) != NULL ? 1 : 0;
        case EOSI_EOSM_KV_U64:
        case EOSI_EOSM_KV_U32: {
            uint64_t v;
            return eos_model_meta_u64(m, d->key, &v) == EOS_OK ? 1 : 0;
        }
        case EOSI_EOSM_KV_F32: {
            float v;
            return eos_model_meta_f32(m, d->key, &v) == EOS_OK ? 1 : 0;
        }
        case EOSI_EOSM_KV_ARRAY: {
            size_t n = 0;
            if (d->arr_elem_type == EOSI_EOSM_KV_STRING)
                return eos_model_meta_arr_str_count(m, d->key, &n) == EOS_OK ? 1 : 0;
            if (d->arr_elem_type == EOSI_EOSM_KV_U32)
                return eos_model_meta_arr_u32_count(m, d->key, &n) == EOS_OK ? 1 : 0;
            return 0;
        }
        default: return 0;
    }
}

static void usage(FILE *to) {
    fprintf(to,
        "Usage: eosllm-convert from-gguf <input.gguf> -o <output.eosm>\n"
        "\n"
        "Lossless conversion of a GGUF v3 model file into the native\n"
        ".eosm v1 format. Tensor bytes are copied as-is; metadata is\n"
        "filtered to the Llama-family key table.\n");
}

static int do_from_gguf(const char *in_path, const char *out_path) {
    eos_model_t *src = NULL;
    eos_status_t rc;
    eosi_eosm_key_spec_t *specs = NULL;
    size_t n_specs = 0, i;
    void *buf = NULL;
    size_t bytes = 0;
    int exit_code = 1;
    FILE *out = NULL;
    uint64_t cap_bits;

    rc = eos_init_defaults();
    if (rc != EOS_OK) {
        fprintf(stderr, "eos_init_defaults failed: %s\n", eos_status_str(rc));
        return 1;
    }

    rc = eos_model_open(in_path, &src);
    if (rc != EOS_OK) {
        fprintf(stderr, "eos_model_open(\"%s\") failed: %s\n",
                in_path, eos_status_str(rc));
        return 1;
    }

    /* Filter keys to those actually present. */
    specs = (eosi_eosm_key_spec_t *)calloc(N_LLAMA_KEYS, sizeof(*specs));
    if (specs == NULL) { fprintf(stderr, "OOM\n"); goto cleanup; }
    for (i = 0; i < N_LLAMA_KEYS; ++i) {
        if (key_present(src, &k_llama_keys[i])) {
            specs[n_specs].key           = k_llama_keys[i].key;
            specs[n_specs].type          = k_llama_keys[i].type;
            specs[n_specs].arr_elem_type = k_llama_keys[i].arr_elem_type;
            n_specs++;
        }
    }
    fprintf(stdout, "[eosllm-convert] %s -> %s\n", in_path, out_path);
    fprintf(stdout, "[eosllm-convert] %zu metadata keys present, %zu tensors\n",
            n_specs, eos_model_num_tensors(src));

    /* Capability bits inferred from observed tensor dtypes. */
    cap_bits = EOSI_EOSM_CAP_MODALITY_TEXT;
    {
        size_t n = eos_model_num_tensors(src);
        size_t k;
        for (k = 0; k < n; ++k) {
            const eos_tensor_t *t = eos_model_tensor(src, k);
            if (t == NULL) continue;
            switch (eos_tensor_dtype(t)) {
                case EOS_DT_Q8_0: cap_bits |= EOSI_EOSM_CAP_QUANT_Q8_0; break;
                case EOS_DT_Q4_K: cap_bits |= EOSI_EOSM_CAP_QUANT_Q4_K; break;
                default: break;
            }
        }
    }

    rc = eosi_eosm_from_model(src, specs, n_specs, cap_bits, 64, &buf, &bytes);
    if (rc != EOS_OK) {
        fprintf(stderr, "eosi_eosm_from_model failed: %s\n", eos_status_str(rc));
        goto cleanup;
    }

    out = fopen(out_path, "wb");
    if (out == NULL) {
        fprintf(stderr, "cannot open output \"%s\" for writing\n", out_path);
        goto cleanup;
    }
    if (fwrite(buf, 1, bytes, out) != bytes) {
        fprintf(stderr, "short write to \"%s\"\n", out_path);
        goto cleanup;
    }
    fprintf(stdout, "[eosllm-convert] wrote %zu bytes\n", bytes);
    exit_code = 0;

cleanup:
    if (out  != NULL) fclose(out);
    free(specs);
    /* buf was allocated via the engine's OS shim; release it through
     * the matching public helper. */
    if (buf != NULL) eos_free(buf);
    eos_model_close(src);
    return exit_code;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(stderr); return 1; }
    if (strcmp(argv[1], "from-gguf") == 0) {
        const char *in = NULL, *out = NULL;
        int i;
        for (i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) { out = argv[++i]; }
            else if (in == NULL) { in = argv[i]; }
            else { fprintf(stderr, "unexpected positional arg: %s\n", argv[i]);
                   usage(stderr); return 1; }
        }
        if (in == NULL || out == NULL) { usage(stderr); return 1; }
        return do_from_gguf(in, out);
    }
    if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        usage(stdout); return 0;
    }
    fprintf(stderr, "unknown subcommand: %s\n", argv[1]);
    usage(stderr);
    return 1;
}
