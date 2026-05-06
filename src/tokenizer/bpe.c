/*
 * src/tokenizer/bpe.c — byte-level BPE tokenizer.
 *
 * Loads vocab + merges from GGUF metadata via the model handle.
 *
 * Encode pipeline:
 *   1. Walk the input. At each position, attempt longest-match against
 *      the registered special-token list. On hit, emit that token ID
 *      and skip; otherwise extend the current "regular" segment up to
 *      the next special-token boundary (or end).
 *   2. For each regular segment, map every byte to its byte-token ID
 *      (via st->byte_id[256]), then greedily apply highest-priority
 *      merges until none apply.
 *
 * Vocab lookup is an open-addressing hash table (linear probing, FNV-1a
 * key, capacity = next power-of-two ≥ 2 × vocab_size). Special-token
 * scan uses a 256-entry first-byte bitmap to skip non-candidates.
 *
 * The `tokenizer.ggml.pre` metadata key (when present) names the
 * pre-tokenizer the source model was trained with. We record it; the
 * Llama-3 / tiktoken regex pre-tokenizer
 * (`(?i:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`) is
 * NOT yet implemented — encodings produced for those models will not
 * be bit-exact vs. the reference Python `tokenizers` library. A WARN
 * is logged at open() time so the caller knows.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "eosllm/eosllm.h"
#include "eosllm/model.h"

#include "../core/internal.h"
#include "tokenizer_internal.h"

#if EOSLLM_HAVE_TOKENIZER_BPE

/* GGUF tokenizer.ggml.token_type values. */
#define BPE_TT_NORMAL        1
#define BPE_TT_UNKNOWN       2
#define BPE_TT_CONTROL       3
#define BPE_TT_USER_DEFINED  4
#define BPE_TT_UNUSED        5
#define BPE_TT_BYTE          6

typedef struct {
    eos_model_t *model;

    /* Vocab: token strings (pointers into the model's owned pool). */
    const char **vocab;
    uint32_t    *vocab_len;
    uint32_t     vocab_size;

    /* Open-addressing hash: hash_ids[h] = vocab id or UINT32_MAX (empty). */
    uint32_t    *hash_ids;
    uint32_t     hash_cap;        /* power of two, > vocab_size */
    uint32_t     hash_mask;       /* hash_cap - 1 */

    /* Per-byte token IDs (256). UINT32_MAX = "no direct byte token". */
    uint32_t     byte_id[256];

    /* Merges: ranked. Lower index = higher priority. */
    uint32_t    *merge_a;
    uint32_t    *merge_b;
    uint32_t    *merge_out;
    uint32_t     n_merges;

    /* Pair-hash: (a_id, b_id) -> merge rank. Open addressing, linear
     * probing. pair_rank == UINT32_MAX marks empty. Built once after
     * the merge table is resolved; turns best_merge from O(n·M) into
     * O(n) per pass. */
    uint32_t    *pair_a;
    uint32_t    *pair_b;
    uint32_t    *pair_rank;
    uint32_t     pair_cap;        /* power of two */
    uint32_t     pair_mask;       /* pair_cap - 1 */

    /* Special-token IDs sorted by length descending (longest-match). */
    uint32_t    *spec_idx;        /* indices into vocab[] */
    uint32_t     n_specials;
    uint8_t      spec_first_byte[256];

    /* Per-session scratch for byte→ID expansion + merge passes. */
    uint32_t    *scratch;
    size_t       scratch_cap;

    /* Special-token IDs (from metadata). UINT32_MAX = absent. */
    uint32_t     bos_id;
    uint32_t     eos_id;

    /* Recorded `tokenizer.ggml.pre` (interned in the model's pool). */
    const char  *pre_scheme;
} bpe_state_t;

/* ------------------------------------------------------------------ */
/* Hash table (FNV-1a, linear probing)                                 */
/* ------------------------------------------------------------------ */

static uint32_t hash_str(const char *s, uint32_t len) {
    uint32_t h = 0x811C9DC5u;
    uint32_t i;
    for (i = 0; i < len; ++i) {
        h ^= (uint8_t)s[i];
        h *= 0x01000193u;
    }
    /* Avoid landing on 0 too often; mix the length in. */
    h ^= len;
    return h;
}

static uint32_t round_up_pow2(uint32_t v) {
    uint32_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

/* SplitMix64-style mix on the 64-bit (a:b) pair, folded to 32 bits. */
static uint32_t hash_pair(uint32_t a, uint32_t b) {
    uint64_t k = ((uint64_t)a << 32) | (uint64_t)b;
    k ^= k >> 30;
    k *= 0xBF58476D1CE4E5B9ULL;
    k ^= k >> 27;
    k *= 0x94D049BB133111EBULL;
    k ^= k >> 31;
    return (uint32_t)k ^ (uint32_t)(k >> 32);
}

/* Find a token by string. Returns vocab id or UINT32_MAX. */
static uint32_t find_token(const bpe_state_t *st,
                           const char *s, uint32_t len) {
    uint32_t h;
    if (st->hash_ids == NULL) {
        /* Fallback for the degenerate empty-vocab case. */
        uint32_t i;
        for (i = 0; i < st->vocab_size; ++i) {
            if (st->vocab_len[i] == len
                && st->vocab[i] != NULL
                && memcmp(st->vocab[i], s, len) == 0) return i;
        }
        return UINT32_MAX;
    }
    h = hash_str(s, len) & st->hash_mask;
    /* Linear probe. Loop bound: hash_cap (table is < load factor 0.5). */
    {
        uint32_t probes;
        for (probes = 0; probes < st->hash_cap; ++probes) {
            uint32_t id = st->hash_ids[h];
            if (id == UINT32_MAX) return UINT32_MAX;
            if (st->vocab_len[id] == len
                && st->vocab[id] != NULL
                && memcmp(st->vocab[id], s, len) == 0) return id;
            h = (h + 1) & st->hash_mask;
        }
    }
    return UINT32_MAX;
}

static eos_status_t hash_build(bpe_state_t *st) {
    uint32_t i, h;
    if (st->vocab_size == 0) return EOS_OK;
    /* Load factor target ≤ 0.5. */
    st->hash_cap  = round_up_pow2(st->vocab_size * 2u + 1u);
    if (st->hash_cap < 8) st->hash_cap = 8;
    st->hash_mask = st->hash_cap - 1u;
    st->hash_ids  = (uint32_t *)eosi_alloc(
        st->hash_cap * sizeof(uint32_t), 8);
    if (st->hash_ids == NULL) return EOS_E_OUT_OF_MEMORY;
    for (i = 0; i < st->hash_cap; ++i) st->hash_ids[i] = UINT32_MAX;
    for (i = 0; i < st->vocab_size; ++i) {
        if (st->vocab[i] == NULL || st->vocab_len[i] == 0) continue;
        h = hash_str(st->vocab[i], st->vocab_len[i]) & st->hash_mask;
        while (st->hash_ids[h] != UINT32_MAX) {
            /* Duplicate vocab strings: keep the first (lowest-id) entry. */
            uint32_t exist = st->hash_ids[h];
            if (st->vocab_len[exist] == st->vocab_len[i]
                && memcmp(st->vocab[exist], st->vocab[i], st->vocab_len[i]) == 0) {
                /* dup; do not insert again */
                goto next;
            }
            h = (h + 1) & st->hash_mask;
        }
        st->hash_ids[h] = i;
    next: ;
    }
    return EOS_OK;
}

/* Build (a_id, b_id) -> rank lookup table over the resolved merges.
 * Skips merges with UINT32_MAX components (unparseable / unresolved).
 * Iterating merges in ascending rank order means the first insertion
 * wins, so duplicate pairs (rare but legal) keep the higher-priority
 * rank. */
static eos_status_t pair_hash_build(bpe_state_t *st) {
    uint32_t i, h;
    if (st->n_merges == 0) return EOS_OK;
    st->pair_cap  = round_up_pow2(st->n_merges * 2u + 1u);
    if (st->pair_cap < 8) st->pair_cap = 8;
    st->pair_mask = st->pair_cap - 1u;
    st->pair_a    = (uint32_t *)eosi_alloc(st->pair_cap * sizeof(uint32_t), 4);
    st->pair_b    = (uint32_t *)eosi_alloc(st->pair_cap * sizeof(uint32_t), 4);
    st->pair_rank = (uint32_t *)eosi_alloc(st->pair_cap * sizeof(uint32_t), 4);
    if (!st->pair_a || !st->pair_b || !st->pair_rank)
        return EOS_E_OUT_OF_MEMORY;
    for (i = 0; i < st->pair_cap; ++i) {
        st->pair_rank[i] = UINT32_MAX;
        st->pair_a[i]    = 0;
        st->pair_b[i]    = 0;
    }
    for (i = 0; i < st->n_merges; ++i) {
        uint32_t a, b;
        if (st->merge_out[i] == UINT32_MAX) continue;
        a = st->merge_a[i];
        b = st->merge_b[i];
        if (a == UINT32_MAX || b == UINT32_MAX) continue;
        h = hash_pair(a, b) & st->pair_mask;
        for (;;) {
            if (st->pair_rank[h] == UINT32_MAX) {
                st->pair_a[h] = a;
                st->pair_b[h] = b;
                st->pair_rank[h] = i;
                break;
            }
            if (st->pair_a[h] == a && st->pair_b[h] == b) {
                /* duplicate pair; keep first (lower rank). */
                break;
            }
            h = (h + 1) & st->pair_mask;
        }
    }
    return EOS_OK;
}

/* O(1) lookup: returns the merge index (rank) for pair (a,b), or
 * UINT32_MAX if no merge applies. */
static uint32_t pair_hash_lookup(const bpe_state_t *st,
                                 uint32_t a, uint32_t b) {
    uint32_t h, probes;
    if (st->pair_cap == 0) return UINT32_MAX;
    h = hash_pair(a, b) & st->pair_mask;
    for (probes = 0; probes < st->pair_cap; ++probes) {
        uint32_t r = st->pair_rank[h];
        if (r == UINT32_MAX) return UINT32_MAX;
        if (st->pair_a[h] == a && st->pair_b[h] == b) return r;
        h = (h + 1) & st->pair_mask;
    }
    return UINT32_MAX;
}

/* Decode "<0xNN>" → byte (0..255) or -1. */
static int parse_byte_token(const char *s, uint32_t len) {
    int hi, lo;
    if (s == NULL || len != 6) return -1;
    if (s[0] != '<' || s[1] != '0' || s[2] != 'x' || s[5] != '>') return -1;
    hi = s[3]; lo = s[4];
    if      (hi >= '0' && hi <= '9') hi -= '0';
    else if (hi >= 'A' && hi <= 'F') hi -= 'A' - 10;
    else if (hi >= 'a' && hi <= 'f') hi -= 'a' - 10;
    else return -1;
    if      (lo >= '0' && lo <= '9') lo -= '0';
    else if (lo >= 'A' && lo <= 'F') lo -= 'A' - 10;
    else if (lo >= 'a' && lo <= 'f') lo -= 'a' - 10;
    else return -1;
    return (hi << 4) | lo;
}

/* ------------------------------------------------------------------ */
/* Special-token table                                                 */
/* ------------------------------------------------------------------ */

/* Insertion sort spec_idx[] descending by vocab_len. */
static void sort_specials_desc(bpe_state_t *st) {
    uint32_t i, j, tmp;
    for (i = 1; i < st->n_specials; ++i) {
        tmp = st->spec_idx[i];
        j = i;
        while (j > 0 &&
               st->vocab_len[st->spec_idx[j - 1]] < st->vocab_len[tmp]) {
            st->spec_idx[j] = st->spec_idx[j - 1];
            --j;
        }
        st->spec_idx[j] = tmp;
    }
}

/* Mark the first byte of each special-token string in spec_first_byte. */
static void build_first_byte_set(bpe_state_t *st) {
    uint32_t i;
    memset(st->spec_first_byte, 0, sizeof(st->spec_first_byte));
    for (i = 0; i < st->n_specials; ++i) {
        uint32_t id = st->spec_idx[i];
        if (st->vocab_len[id] == 0 || st->vocab[id] == NULL) continue;
        st->spec_first_byte[(uint8_t)st->vocab[id][0]] = 1;
    }
}

/*
 * Discover special tokens.
 *
 *   - If `tokenizer.ggml.token_type` array is present (one u32 per
 *     vocab id), entries with type == CONTROL or == USER_DEFINED are
 *     special.
 *   - bos_id and eos_id (if known) are also added.
 *   - As a final fallback (no type array), mark only bos / eos.
 *
 * spec_idx[] is allocated and filled; n_specials is set.
 */
static eos_status_t resolve_specials(bpe_state_t *st) {
    eos_status_t s;
    size_t n_types = 0;
    uint32_t *types = NULL;
    uint32_t i;
    uint32_t cap = 0;

    s = eos_model_meta_arr_u32_count(st->model,
                                     "tokenizer.ggml.token_type", &n_types);
    if (s == EOS_OK && n_types == st->vocab_size && n_types > 0) {
        types = (uint32_t *)eosi_alloc(n_types * sizeof(uint32_t), 4);
        if (types == NULL) return EOS_E_OUT_OF_MEMORY;
        {
            size_t got = n_types;
            s = eos_model_meta_arr_u32_load(st->model,
                                            "tokenizer.ggml.token_type",
                                            types, &got);
            if (s != EOS_OK) { eosi_free(types); return s; }
        }
    }

    /* Count candidates first. */
    if (types != NULL) {
        for (i = 0; i < st->vocab_size; ++i) {
            if (types[i] == BPE_TT_CONTROL || types[i] == BPE_TT_USER_DEFINED)
                cap++;
        }
    }
    if (st->bos_id != UINT32_MAX) cap++;
    if (st->eos_id != UINT32_MAX) cap++;

    if (cap == 0) { eosi_free(types); return EOS_OK; }

    st->spec_idx = (uint32_t *)eosi_alloc(cap * sizeof(uint32_t), 4);
    if (st->spec_idx == NULL) { eosi_free(types); return EOS_E_OUT_OF_MEMORY; }
    st->n_specials = 0;

    if (types != NULL) {
        for (i = 0; i < st->vocab_size; ++i) {
            if (types[i] == BPE_TT_CONTROL || types[i] == BPE_TT_USER_DEFINED)
                st->spec_idx[st->n_specials++] = i;
        }
    }
    /* Make sure bos/eos are present even if type array didn't flag them. */
    {
        uint32_t k;
        int seen_bos = 0, seen_eos = 0;
        for (k = 0; k < st->n_specials; ++k) {
            if (st->spec_idx[k] == st->bos_id) seen_bos = 1;
            if (st->spec_idx[k] == st->eos_id) seen_eos = 1;
        }
        if (!seen_bos && st->bos_id != UINT32_MAX && st->bos_id < st->vocab_size)
            st->spec_idx[st->n_specials++] = st->bos_id;
        if (!seen_eos && st->eos_id != UINT32_MAX && st->eos_id < st->vocab_size)
            st->spec_idx[st->n_specials++] = st->eos_id;
    }

    eosi_free(types);
    sort_specials_desc(st);
    build_first_byte_set(st);
    return EOS_OK;
}

/* Try longest-match against special tokens at text[pos..end). */
static int try_special_at(const bpe_state_t *st,
                          const char *text, size_t pos, size_t end,
                          uint32_t *out_id, size_t *out_len) {
    uint32_t i;
    if (st->n_specials == 0) return 0;
    if (!st->spec_first_byte[(uint8_t)text[pos]]) return 0;
    for (i = 0; i < st->n_specials; ++i) {
        uint32_t id  = st->spec_idx[i];
        uint32_t len = st->vocab_len[id];
        if (len == 0 || st->vocab[id] == NULL) continue;
        if (pos + (size_t)len > end) continue;
        if (memcmp(text + pos, st->vocab[id], len) == 0) {
            *out_id  = id;
            *out_len = (size_t)len;
            return 1;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Module-private init from model handle                               */
/* ------------------------------------------------------------------ */

static eos_status_t bpe_open_from_model(eos_model_t *model, void **out_state) {
    bpe_state_t *st;
    size_t   n_vocab = 0, n_merges = 0;
    uint64_t bos = 0, eos = 0;
    uint32_t i;
    eos_status_t s;
    const char *pre_str;

    s = eos_model_meta_arr_str_count(model, "tokenizer.ggml.tokens", &n_vocab);
    if (s != EOS_OK) {
        EOSI_LOG_ERROR("bpe: model is missing tokenizer.ggml.tokens metadata");
        return s;
    }
    s = eos_model_meta_arr_str_count(model, "tokenizer.ggml.merges", &n_merges);
    if (s != EOS_OK) n_merges = 0;     /* merges optional */

    st = (bpe_state_t *)eosi_alloc(sizeof(*st), 16);
    if (st == NULL) return EOS_E_OUT_OF_MEMORY;
    memset(st, 0, sizeof(*st));
    st->model      = model;
    st->vocab_size = (uint32_t)n_vocab;
    st->n_merges   = (uint32_t)n_merges;
    st->bos_id     = UINT32_MAX;
    st->eos_id     = UINT32_MAX;
    for (i = 0; i < 256; ++i) st->byte_id[i] = UINT32_MAX;

    if (n_vocab > 0) {
        st->vocab     = (const char **)eosi_alloc(n_vocab * sizeof(char *), 8);
        st->vocab_len = (uint32_t   *)eosi_alloc(n_vocab * sizeof(uint32_t), 4);
        if (!st->vocab || !st->vocab_len) goto oom;
        {
            size_t got = n_vocab;
            s = eos_model_meta_arr_str_load(model, "tokenizer.ggml.tokens",
                                            st->vocab, &got);
            if (s != EOS_OK) goto fail;
            for (i = 0; i < n_vocab; ++i) {
                st->vocab_len[i] = st->vocab[i] != NULL
                                 ? (uint32_t)strlen(st->vocab[i]) : 0;
            }
        }
    }

    /* Build the vocab hash *before* we consult it for merge resolution. */
    s = hash_build(st);
    if (s != EOS_OK) goto fail;

    if (n_merges > 0) {
        const char **raw_merges = (const char **)eosi_alloc(
            n_merges * sizeof(char *), 8);
        if (raw_merges == NULL) goto oom;
        {
            size_t got = n_merges;
            s = eos_model_meta_arr_str_load(model, "tokenizer.ggml.merges",
                                            raw_merges, &got);
            if (s != EOS_OK) { eosi_free(raw_merges); goto fail; }
        }
        st->merge_a   = (uint32_t *)eosi_alloc(n_merges * sizeof(uint32_t), 4);
        st->merge_b   = (uint32_t *)eosi_alloc(n_merges * sizeof(uint32_t), 4);
        st->merge_out = (uint32_t *)eosi_alloc(n_merges * sizeof(uint32_t), 4);
        if (!st->merge_a || !st->merge_b || !st->merge_out) {
            eosi_free(raw_merges); goto oom;
        }
        for (i = 0; i < n_merges; ++i) {
            const char *m = raw_merges[i];
            const char *sp;
            uint32_t alen, blen, merged_len;
            char merged[1024];
            uint32_t a_id, b_id, out_id;
            sp = (m != NULL) ? strchr(m, ' ') : NULL;
            if (sp == NULL) {
                st->merge_a[i] = UINT32_MAX;
                st->merge_b[i] = UINT32_MAX;
                st->merge_out[i] = UINT32_MAX;
                continue;
            }
            alen = (uint32_t)(sp - m);
            blen = (uint32_t)strlen(sp + 1);
            a_id = find_token(st, m,    alen);
            b_id = find_token(st, sp+1, blen);
            merged_len = alen + blen;
            if (merged_len < sizeof(merged)) {
                memcpy(merged, m, alen);
                memcpy(merged + alen, sp + 1, blen);
                out_id = find_token(st, merged, merged_len);
            } else {
                out_id = UINT32_MAX;
            }
            st->merge_a[i]   = a_id;
            st->merge_b[i]   = b_id;
            st->merge_out[i] = out_id;
        }
        eosi_free(raw_merges);
    }

    /* Pair-hash over the resolved merges — turns best_merge() from
     * O(n·M) into O(n) per pass. Built after merge_a/b/out are filled. */
    s = pair_hash_build(st);
    if (s != EOS_OK) goto fail;

    /* Resolve byte tokens. Try "<0xNN>" form first (Llama-3 style). */
    for (i = 0; i < n_vocab; ++i) {
        int byte = parse_byte_token(st->vocab[i], st->vocab_len[i]);
        if (byte >= 0) st->byte_id[byte] = i;
    }
    /* Fallback: any single-byte ASCII vocab token. */
    for (i = 0; i < n_vocab; ++i) {
        if (st->vocab_len[i] == 1 && st->vocab[i] != NULL) {
            uint8_t b = (uint8_t)st->vocab[i][0];
            if (st->byte_id[b] == UINT32_MAX) st->byte_id[b] = i;
        }
    }

    /* Special token IDs (best effort). */
    if (eos_model_meta_u64(model, "tokenizer.ggml.bos_token_id", &bos) == EOS_OK)
        st->bos_id = (uint32_t)bos;
    if (eos_model_meta_u64(model, "tokenizer.ggml.eos_token_id", &eos) == EOS_OK)
        st->eos_id = (uint32_t)eos;

    /* Resolve full special-token list (control + user-defined + bos/eos). */
    s = resolve_specials(st);
    if (s != EOS_OK) goto fail;

    /* Record `tokenizer.ggml.pre` and warn for unimplemented schemes. */
    pre_str = eos_model_meta_str(model, "tokenizer.ggml.pre");
    st->pre_scheme = pre_str;
    if (pre_str != NULL && strcmp(pre_str, "default") != 0) {
        EOSI_LOG_WARN(
            "bpe: tokenizer.ggml.pre is set but the corresponding regex "
            "pre-tokenizer is not implemented in this build; encodings will "
            "not be bit-exact vs. the reference Python tokenizer for that model");
    }

    *out_state = st;
    return EOS_OK;

oom:
    s = EOS_E_OUT_OF_MEMORY;
fail:
    eosi_free(st->vocab);
    eosi_free(st->vocab_len);
    eosi_free(st->hash_ids);
    eosi_free(st->merge_a);
    eosi_free(st->merge_b);
    eosi_free(st->merge_out);
    eosi_free(st->pair_a);
    eosi_free(st->pair_b);
    eosi_free(st->pair_rank);
    eosi_free(st->spec_idx);
    eosi_free(st->scratch);
    eosi_free(st);
    return s;
}

static void bpe_close(void *state) {
    bpe_state_t *st = (bpe_state_t *)state;
    if (st == NULL) return;
    eosi_free(st->vocab);
    eosi_free(st->vocab_len);
    eosi_free(st->hash_ids);
    eosi_free(st->merge_a);
    eosi_free(st->merge_b);
    eosi_free(st->merge_out);
    eosi_free(st->pair_a);
    eosi_free(st->pair_b);
    eosi_free(st->pair_rank);
    eosi_free(st->spec_idx);
    eosi_free(st->scratch);
    eosi_free(st);
}

/* ------------------------------------------------------------------ */
/* Encode / decode                                                     */
/* ------------------------------------------------------------------ */

/* Find the highest-priority merge that applies to any adjacent pair in
 * `ids`. Returns the merge-table index or UINT32_MAX if none apply.
 * Uses the pair-hash table for O(1) per-pair lookup. */
static uint32_t best_merge(const bpe_state_t *st,
                           const uint32_t *ids, size_t n,
                           size_t *out_pos) {
    uint32_t best = UINT32_MAX;
    size_t   best_pos = 0;
    size_t   i;
    for (i = 0; i + 1 < n; ++i) {
        uint32_t r = pair_hash_lookup(st, ids[i], ids[i + 1]);
        if (r < best) { best = r; best_pos = i; }
    }
    *out_pos = best_pos;
    return best;
}

/* Ensure st->scratch has capacity ≥ need. */
static eos_status_t scratch_reserve(bpe_state_t *st, size_t need) {
    uint32_t *p;
    if (st->scratch_cap >= need) return EOS_OK;
    /* Grow with a 1.5× heuristic, minimum 64. */
    {
        size_t newcap = st->scratch_cap > 0 ? st->scratch_cap : 64;
        while (newcap < need) {
            size_t up = newcap + (newcap >> 1);
            newcap = up > newcap ? up : newcap + 64;
        }
        p = (uint32_t *)eosi_alloc(newcap * sizeof(uint32_t), 8);
        if (p == NULL) return EOS_E_OUT_OF_MEMORY;
        eosi_free(st->scratch);
        st->scratch     = p;
        st->scratch_cap = newcap;
    }
    return EOS_OK;
}

/* Encode a contiguous segment (no specials inside) into st->scratch.
 * Writes the count to *out_n. */
static eos_status_t encode_segment(bpe_state_t *st,
                                   const char *text, size_t text_len,
                                   size_t *out_n) {
    size_t n = 0;
    size_t i;
    eos_status_t s;
    if (text_len == 0) { *out_n = 0; return EOS_OK; }
    s = scratch_reserve(st, text_len);
    if (s != EOS_OK) return s;
    for (i = 0; i < text_len; ++i) {
        uint8_t b = (uint8_t)text[i];
        uint32_t id = st->byte_id[b];
        if (id == UINT32_MAX) {
            EOSI_LOG_ERROR("bpe_encode: input byte has no byte-token in vocab "
                           "(model is missing the <0xNN> fallback set)");
            return EOS_E_FORMAT;
        }
        st->scratch[n++] = id;
    }
    while (n > 1) {
        size_t pos;
        uint32_t r = best_merge(st, st->scratch, n, &pos);
        if (r == UINT32_MAX) break;
        st->scratch[pos] = st->merge_out[r];
        memmove(&st->scratch[pos + 1], &st->scratch[pos + 2],
                (n - pos - 2) * sizeof(uint32_t));
        n--;
    }
    *out_n = n;
    return EOS_OK;
}

/* Append `n` IDs from src into out_ids[*out..cap]; track required total. */
static void append_ids(uint32_t *out_ids, size_t cap, size_t *out,
                       const uint32_t *src, size_t n) {
    size_t i;
    if (out_ids != NULL) {
        for (i = 0; i < n && *out + i < cap; ++i) out_ids[*out + i] = src[i];
    }
    *out += n;
}

static eos_status_t bpe_encode(void *state, const char *text, size_t text_len,
                               uint32_t *out_ids, size_t *io_len) {
    bpe_state_t *st = (bpe_state_t *)state;
    size_t cap, out = 0, pos = 0;
    eos_status_t s;

    if (st == NULL || text == NULL || io_len == NULL) {
        EOSI_LOG_ERROR("bpe_encode: NULL state, text, or io_len argument");
        return EOS_E_INVALID_ARG;
    }
    if (out_ids == NULL && *io_len != 0) {
        EOSI_LOG_ERROR("bpe_encode: out_ids NULL but *io_len != 0 "
                       "(query-size mode requires *io_len == 0)");
        return EOS_E_INVALID_ARG;
    }
    cap = (out_ids != NULL) ? *io_len : 0;

    while (pos < text_len) {
        uint32_t sid; size_t slen;
        if (try_special_at(st, text, pos, text_len, &sid, &slen)) {
            uint32_t one = sid;
            append_ids(out_ids, cap, &out, &one, 1);
            pos += slen;
            continue;
        }
        /* Walk forward to next special-token boundary (or end). */
        {
            size_t seg_end = pos + 1;
            uint32_t sid2; size_t slen2;
            while (seg_end < text_len) {
                if (try_special_at(st, text, seg_end, text_len, &sid2, &slen2))
                    break;
                seg_end++;
            }
            s = encode_segment(st, text + pos, seg_end - pos, &slen);
            if (s != EOS_OK) return s;
            append_ids(out_ids, cap, &out, st->scratch, slen);
            pos = seg_end;
        }
    }

    if (out_ids != NULL && out > cap) {
        *io_len = out;
        return EOS_E_OUT_OF_MEMORY;
    }
    *io_len = out;
    return EOS_OK;
}

static eos_status_t bpe_decode(void *state, const uint32_t *ids, size_t n_ids,
                               char *out_text, size_t *io_len) {
    bpe_state_t *st = (bpe_state_t *)state;
    size_t pos = 0, i;
    if (st == NULL || ids == NULL || io_len == NULL) {
        EOSI_LOG_ERROR("bpe_decode: NULL state, ids, or io_len argument");
        return EOS_E_INVALID_ARG;
    }
    for (i = 0; i < n_ids; ++i) {
        uint32_t id = ids[i];
        const char *src; uint32_t len;
        int byte;
        if (id >= st->vocab_size) continue;
        src = st->vocab[id];
        len = st->vocab_len[id];
        byte = parse_byte_token(src, len);
        if (byte >= 0) {
            if (out_text != NULL && pos + 1 <= *io_len) out_text[pos] = (char)byte;
            pos++;
            continue;
        }
        if (out_text != NULL) {
            size_t cap = (pos + len <= *io_len) ? len : (*io_len - pos);
            if (src != NULL) memcpy(out_text + pos, src, cap);
        }
        pos += len;
    }
    {
        eos_status_t s = (out_text != NULL && pos > *io_len)
                         ? EOS_E_OUT_OF_MEMORY : EOS_OK;
        *io_len = pos;
        return s;
    }
}

/* The public eos_tokenizer_vt_t expects open(blob, blob_len, &state).
 * We accept the blob as a pointer-to-eos_model_t and ignore blob_len. */
static eos_status_t bpe_open_blob(const void *blob, size_t blob_len,
                                  void **out_state) {
    (void)blob_len;
    if (blob == NULL || out_state == NULL) {
        EOSI_LOG_ERROR("bpe vt open: NULL blob or out_state argument");
        return EOS_E_INVALID_ARG;
    }
    return bpe_open_from_model((eos_model_t *)blob, out_state);
}

static const eos_tokenizer_vt_t k_bpe = {
    "bpe", bpe_open_blob, bpe_encode, bpe_decode, bpe_close
};

eos_status_t eosi_tokenizer_bpe_register(void) {
    return eos_tokenizer_register(&k_bpe);
}

#else /* !EOSLLM_HAVE_TOKENIZER_BPE */

eos_status_t eosi_tokenizer_bpe_register(void) { return EOS_E_UNSUPPORTED; }

#endif
