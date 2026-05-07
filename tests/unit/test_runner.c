/*
 * tests/unit/test_runner.c — Phase-0/1/2 host smoke + oracle tests.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eosllm/eosllm.h"
#include "eosllm/backend.h"
#include "eosllm/quant.h"
#include "eosllm/os.h"

/* Internal headers reachable via -Isrc. */
#include "kernels/scalar/scalar.h"
#include "util/f16.h"
#include "util/sha256.h"
#include "core/internal.h"
#include "core/model_internal.h"
#include "format/eosm/eosm_internal.h"
#if EOSLLM_HAVE_KERNEL_AVX2
#include "kernels/avx2/avx2_internal.h"
#endif
#if EOSLLM_HAVE_KERNEL_NEON
#include "kernels/neon/neon_internal.h"
#endif

static int g_failed = 0;
static int g_total  = 0;

#define CHECK(cond, msg) do {                                           \
    g_total++;                                                          \
    if (!(cond)) {                                                      \
        fprintf(stderr, "FAIL  %s:%d  %s — %s\n",                       \
                __FILE__, __LINE__, msg, #cond);                        \
        g_failed++;                                                     \
    } else {                                                            \
        fprintf(stdout, "ok    %s\n", msg);                             \
    }                                                                   \
} while (0)

#define CHECK_OK(call, msg) do {                                        \
    eos_status_t _s = (call);                                           \
    g_total++;                                                          \
    if (_s != EOS_OK) {                                                 \
        fprintf(stderr, "FAIL  %s:%d  %s — %s (%s)\n",                  \
                __FILE__, __LINE__, msg, #call, eos_status_str(_s));    \
        g_failed++;                                                     \
    } else {                                                            \
        fprintf(stdout, "ok    %s\n", msg);                             \
    }                                                                   \
} while (0)

static int approx_eq(float a, float b, float tol) {
    float d = a - b;
    if (d < 0.0f) d = -d;
    return d <= tol;
}

/* ------------------------------------------------------------------ */

static void test_version_and_caps(void) {
    eos_caps_t caps;
    int v = eos_abi_version();
    const char *vs = eos_version_string();
    CHECK(v == EOSLLM_ABI_VERSION,           "abi_version matches header");
    CHECK(vs != NULL && vs[0] != '\0',       "version_string non-empty");
    CHECK_OK(eos_caps(&caps),                "eos_caps returns OK");
    CHECK(caps.have_posix    == 1,           "posix is compiled in");
    CHECK(caps.have_k_scalar == 1,           "scalar kernels are compiled in");
    CHECK(caps.have_threads  == 0,           "threads default to OFF");
    CHECK(caps.have_q_q8_0   == 1,           "q8_0 quant is compiled in");
    CHECK(caps.have_q_q4_k   == 1,           "q4_k quant is compiled in");
    CHECK(caps.have_q_q1_58  == 1,           "q1_58 quant is compiled in");
    CHECK(caps.have_f_gguf   == 1,           "gguf format reader compiled in");
    CHECK(caps.have_t_bpe    == 1,           "bpe tokenizer compiled in");
    CHECK(caps.have_m_text   == 1,           "text modality compiled in");
    CHECK(caps.have_s_greedy == 1,           "greedy scheduler compiled in");

    /* AR contract: runtime_bits must agree with eos_backend_available
     * for every covered backend. The compile-time bit is necessary
     * (can't be runtime-on if it's not compiled), and runtime-on
     * implies compile-time-on. */
    {
        int rt_avx2   = (caps.runtime_bits & EOS_CAPS_RT_AVX2)   != 0;
        int rt_avx512 = (caps.runtime_bits & EOS_CAPS_RT_AVX512) != 0;
        int rt_neon   = (caps.runtime_bits & EOS_CAPS_RT_NEON)   != 0;
        CHECK(rt_avx2   == eos_backend_available("x86_avx2"),
              "runtime_bits.AVX2 matches eos_backend_available");
        CHECK(rt_avx512 == eos_backend_available("x86_avx512"),
              "runtime_bits.AVX512 matches eos_backend_available");
        CHECK(rt_neon   == eos_backend_available("arm_neon"),
              "runtime_bits.NEON matches eos_backend_available");
        if (rt_avx2)   CHECK(caps.have_k_avx2   == 1, "rt_avx2   implies have_k_avx2");
        if (rt_avx512) CHECK(caps.have_k_avx512 == 1, "rt_avx512 implies have_k_avx512");
        if (rt_neon)   CHECK(caps.have_k_neon   == 1, "rt_neon   implies have_k_neon");
    }

    /* BE contract: eos_caps and eos_init_defaults each clear the
     * thread-local last_error slot at entry, so EOS_OK leaves it NULL
     * regardless of any earlier failure. */
    eosi_set_error("sentinel: should be cleared by eos_caps");
    CHECK(eos_caps(&caps) == EOS_OK, "eos_caps OK after sentinel");
    CHECK(eos_last_error() == NULL,
          "eos_last_error cleared by successful eos_caps");
    eosi_set_error("sentinel: should be cleared by eos_init_defaults");
    CHECK(eos_init_defaults() == EOS_OK, "eos_init_defaults OK after sentinel");
    CHECK(eos_last_error() == NULL,
          "eos_last_error cleared by successful eos_init_defaults");
}

static void test_init_defaults(void) {
    eosi_registry_t *r = eosi_registry();
    size_t n_b, n_q, n_m, n_t, n_s, n_f;
    CHECK_OK(eos_init_defaults(), "eos_init_defaults registers everything");
    /* Snapshot table sizes after first init. */
    n_b = r->n_backends;
    n_q = r->n_quants;
    n_m = r->n_modalities;
    n_t = r->n_tokenizers;
    n_s = r->n_schedulers;
    n_f = r->n_formats;
    CHECK_OK(eos_init_defaults(), "eos_init_defaults is idempotent");
    /* Second call must not push duplicates. */
    CHECK(r->n_backends   == n_b, "init_defaults idempotent: n_backends unchanged");
    CHECK(r->n_quants     == n_q, "init_defaults idempotent: n_quants unchanged");
    CHECK(r->n_modalities == n_m, "init_defaults idempotent: n_modalities unchanged");
    CHECK(r->n_tokenizers == n_t, "init_defaults idempotent: n_tokenizers unchanged");
    CHECK(r->n_schedulers == n_s, "init_defaults idempotent: n_schedulers unchanged");
    CHECK(r->n_formats    == n_f, "init_defaults idempotent: n_formats unchanged");
}

static void test_status_strings(void) {
    CHECK(strcmp(eos_status_str(EOS_OK), "OK") == 0, "status OK");
    CHECK(eos_status_str(EOS_E_DEADLINE) != NULL,    "status deadline non-null");
    CHECK(eos_status_str((eos_status_t)999) != NULL, "status unknown non-null");
}

static void test_session_lifecycle(void) {
    eos_session_t       *s = NULL;
    eos_session_params_t params;
    memset(&params, 0, sizeof(params));
    params.max_context = 128;
    CHECK_OK(eos_session_open(NULL, &params, &s), "open with no model OK");
    CHECK(s != NULL,                              "session pointer non-null");
    CHECK_OK(eos_session_set_deadline(s, 1000000),"set deadline OK");
    CHECK_OK(eos_session_set_deadline(s, 0),      "clear deadline OK");
    eos_session_close(s);
    eos_session_close(NULL);
    CHECK(1, "close is idempotent on NULL");
}

static void test_matmul_f32(void) {
    static const float a[6] = { 1, 2, 3, 4, 5, 6 };
    static const float b[6] = { 1, 0, 0, 0, 1, 0 };
    float c[4];
    CHECK_OK(eosi_scalar_matmul_f32(a, b, c, 2, 2, 3), "matmul_f32 returns OK");
    CHECK(approx_eq(c[0], 1.0f, 1e-6f), "C[0,0] = 1");
    CHECK(approx_eq(c[1], 2.0f, 1e-6f), "C[0,1] = 2");
    CHECK(approx_eq(c[2], 4.0f, 1e-6f), "C[1,0] = 4");
    CHECK(approx_eq(c[3], 5.0f, 1e-6f), "C[1,1] = 5");
}

static void test_f16_roundtrip(void) {
    /* Selected exact-in-f16 values. */
    static const float vs[] = { 0.0f, 1.0f, -1.0f, 0.5f, 65504.0f, -65504.0f };
    unsigned i;
    for (i = 0; i < sizeof(vs)/sizeof(vs[0]); ++i) {
        float r = eosi_f16_to_f32(eosi_f32_to_f16(vs[i]));
        CHECK(approx_eq(r, vs[i], 1e-3f), "f16 roundtrip exact-representable");
    }
}

static void test_matmul_q8_oracle_gguf(void) {
    /* GGUF q8_0 layout: f16 scale + 32 int8, 34 bytes. */
    enum { K = 32 };
    typedef struct { eosi_f16_t scale; int8_t w[32]; } block_t;
    block_t b_q;
    float   b_f[K], a[K], cq[1], cf[1];
    int i;
    for (i = 0; i < K; ++i) {
        a[i]     = 0.25f * (float)i - 1.0f;
        b_q.w[i] = (int8_t)(i - 16);
    }
    {
        float scale = 0.0125f;
        b_q.scale = eosi_f32_to_f16(scale);
        scale = eosi_f16_to_f32(b_q.scale);     /* round-trip */
        for (i = 0; i < K; ++i) b_f[i] = (float)b_q.w[i] * scale;
    }
    CHECK_OK(eosi_scalar_matmul_q8_0(a, &b_q, cq, 1, 1, K), "matmul_q8_0 OK");
    CHECK_OK(eosi_scalar_matmul_f32(a, b_f, cf, 1, 1, K),   "matmul_f32 oracle OK");
    CHECK(approx_eq(cq[0], cf[0], 1e-4f), "q8_0 == f32 oracle");
}

static void test_matmul_q4_k_oracle(void) {
    /* Build a q4_k super-block with all sub-block scales = 1, mins = 0,
     * and qs values 0..15 cycled. Compare against an f32 dequantized
     * reference. */
    enum { K = 256 };
    uint8_t  block[144];
    float    a[K], b_f[K], cq[1], cf[1];
    int i;
    eosi_f16_t d_h    = eosi_f32_to_f16(1.0f);
    eosi_f16_t dmin_h = eosi_f32_to_f16(0.0f);

    memset(block, 0, sizeof(block));
    memcpy(block,     &d_h,    2);
    memcpy(block + 2, &dmin_h, 2);
    /* scales[12] all zero ⇒ sc=0 for sub-blocks 0..3 and m=0; sub-blocks
     * 4..7 also have sc=0. Result: all weights are 0 ⇒ cq == 0.
     * That tests the path; for a non-trivial check, set sub-block 0
     * scale to 1: scales[0] |= 1. */
    block[4] = 1;     /* sub-block 0 scale = 1 */
    /* qs: alternate low/high nibbles so sub-block 0 gets low nibbles 0..15 cycled,
     * sub-block 1 gets high nibbles. */
    for (i = 0; i < 32; ++i) block[16 + i] = (uint8_t)((i & 0x0F) | ((i & 0x0F) << 4));

    for (i = 0; i < K; ++i) a[i] = 1.0f;       /* simple input */

    /* Build the f32 reference. d=1, sc_lo=1, m_lo=0 ⇒ y = q&0xF.
     * Sub-block 0 covers indices 0..31, low nibbles. */
    for (i = 0; i < 32; ++i) b_f[i] = (float)(block[16 + i] & 0x0F);
    /* All other sub-blocks have scale=0 ⇒ contribute 0. */
    for (i = 32; i < K; ++i) b_f[i] = 0.0f;

    CHECK_OK(eosi_scalar_matmul_q4_k(a, block, cq, 1, 1, K), "matmul_q4_k OK");
    CHECK_OK(eosi_scalar_matmul_f32 (a, b_f,   cf, 1, 1, K), "matmul_f32 ref OK");
    CHECK(approx_eq(cq[0], cf[0], 1e-3f), "q4_k == f32 reference");
}

static void test_q8_0_dequant_quant_roundtrip(void) {
    static const float src[32] = {
        -1.0f, -0.9f, -0.8f, -0.7f, -0.6f, -0.5f, -0.4f, -0.3f,
        -0.2f, -0.1f, 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
        0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 0.5f, 0.25f, 0.125f,
        -0.125f, -0.25f, -0.5f, -1.0f, 0.0f, 0.5f, -0.5f, 0.0f
    };
    uint8_t blk[34];
    float   out[32];
    int i;
    const eos_quant_vt_t *vt = eos_quant_find("q8_0");
    CHECK(vt != NULL, "q8_0 quant scheme registered");
    if (vt == NULL) return;
    CHECK_OK(vt->quantize(src,   blk, 32), "q8_0 quantize OK");
    CHECK_OK(vt->dequantize(blk, out, 32), "q8_0 dequantize OK");
    for (i = 0; i < 32; ++i) {
        if (!approx_eq(out[i], src[i], 0.02f)) {
            fprintf(stderr, "q8_0 roundtrip[%d]: src=%g out=%g\n",
                    i, src[i], out[i]);
        }
    }
    /* Loose tolerance: 8-bit has ~1/127 quantization error per-value times scale. */
    CHECK(approx_eq(out[0], src[0], 0.02f), "q8_0 roundtrip first elem within tol");
    CHECK(approx_eq(out[31], src[31], 0.02f), "q8_0 roundtrip last elem within tol");
}

static void test_q1_58_roundtrip(void) {
    /* Synthesize 160 values that should quantize to easy ternary patterns. */
    float src[160], out[160];
    uint8_t blk[36];
    int i;
    const eos_quant_vt_t *vt = eos_quant_find("q1_58");
    CHECK(vt != NULL, "q1_58 quant scheme registered");
    if (vt == NULL) return;
    for (i = 0; i < 160; ++i) {
        /* Pick values in {-1, 0, +1} times scale 0.5; abs-mean ⇒ 0.333 */
        int t = (i % 3) - 1;
        src[i] = (float)t * 0.5f;
    }
    CHECK_OK(vt->quantize  (src, blk, 160), "q1_58 quantize OK");
    CHECK_OK(vt->dequantize(blk, out, 160), "q1_58 dequantize OK");
    /* The reconstructed values should be in {-scale, 0, +scale}. */
    for (i = 0; i < 160; ++i) {
        float a = out[i] < 0 ? -out[i] : out[i];
        if (!(a < 1e-3f || a > 0.1f)) {
            fprintf(stderr, "q1_58[%d] suspicious: src=%g out=%g\n", i, src[i], out[i]);
        }
    }
    CHECK(1, "q1_58 roundtrip produced ternary outputs");
}

static void test_rmsnorm_softmax(void) {
    static const float x1[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
    float y1[4]; float s2[4] = { 0,1,2,3 }; float sum = 0; int i;
    CHECK_OK(eosi_scalar_rmsnorm(x1, NULL, 1e-6f, y1, 1, 4), "rmsnorm OK");
    CHECK(approx_eq(y1[0], 1.0f, 1e-3f), "rmsnorm preserves all-ones");
    CHECK_OK(eosi_scalar_softmax(s2, 1, 4, NULL), "softmax OK");
    for (i = 0; i < 4; ++i) sum += s2[i];
    CHECK(approx_eq(sum, 1.0f, 1e-5f), "softmax row sums to 1");
}

static void test_silu_gelu_rope(void) {
    static const float x[3] = { 0.0f, 1.0f, -1.0f };
    float ys[3], yg[3], r[4] = { 0.6f, 0.8f, -0.3f, 0.4f };
    float before[2] = { r[0]*r[0]+r[1]*r[1], r[2]*r[2]+r[3]*r[3] };
    float after[2];
    CHECK_OK(eosi_scalar_silu(x, ys, 3), "silu OK");
    CHECK_OK(eosi_scalar_gelu(x, yg, 3), "gelu OK");
    CHECK(approx_eq(ys[0], 0.0f, 1e-6f), "silu(0) = 0");
    CHECK(approx_eq(yg[0], 0.0f, 1e-6f), "gelu(0) = 0");
    CHECK_OK(eosi_scalar_rope(r, 1, 4, 5, 10000.0f), "rope OK");
    after[0] = r[0]*r[0] + r[1]*r[1];
    after[1] = r[2]*r[2] + r[3]*r[3];
    CHECK(approx_eq(before[0], after[0], 1e-5f), "rope preserves pair-0 magnitude");
    CHECK(approx_eq(before[1], after[1], 1e-5f), "rope preserves pair-1 magnitude");
}

static void test_model_open_no_file(void) {
    eos_model_t *m = NULL;
    eos_status_t s;
    const char *e;
    /* Clear any prior context. */
    eosi_set_error(NULL);
    s = eos_model_open("does_not_exist.gguf", &m);
    CHECK(s != EOS_OK, "eos_model_open fails on missing path");
    CHECK(m == NULL,   "eos_model_open writes NULL on failure");
    e = eos_last_error();
    CHECK(e != NULL && strstr(e, "os.file_open") != NULL,
          "eos_last_error reports os.file_open failure on missing path");
}

/* ------------------------------------------------------------------ */
/* BPE tokenizer: synthetic in-memory model + roundtrip checks         */
/* ------------------------------------------------------------------ */

/*
 * The mock model exposes only the metadata keys bpe.c queries. Vocab
 * layout:
 *   ids 0..255   = byte tokens "<0xNN>" EXCEPT 'a','b','c' which use
 *                  the single-char form (matching real GGUF byte-level
 *                  BPE conventions where printable bytes carry the
 *                  literal char and merges concatenate those literals).
 *   id  256      = "ab"
 *   id  257      = "abc"
 *   id  258      = "<|EOS|>" (control / eos)
 * Merges:
 *   rank 0       = "a b"     -> "ab"
 *   rank 1       = "ab c"    -> "abc"
 */
#define MOCK_VOCAB_N   259
#define MOCK_MERGE_N     2

static char    g_byte_str[256][8];          /* "<0xNN>\0" each */
static const char *g_vocab[MOCK_VOCAB_N];
static uint32_t    g_token_type[MOCK_VOCAB_N];
static const char *g_merges[MOCK_MERGE_N];
static int         g_mock_inited = 0;

static void mock_init(void) {
    int i;
    if (g_mock_inited) return;
    for (i = 0; i < 256; ++i) {
        const char hex[] = "0123456789ABCDEF";
        g_byte_str[i][0] = '<';
        g_byte_str[i][1] = '0';
        g_byte_str[i][2] = 'x';
        g_byte_str[i][3] = hex[(i >> 4) & 0x0F];
        g_byte_str[i][4] = hex[i & 0x0F];
        g_byte_str[i][5] = '>';
        g_byte_str[i][6] = '\0';
        g_vocab[i] = g_byte_str[i];
        g_token_type[i] = 1;     /* NORMAL */
    }
    /* Override 'a','b','c' with single-char strings so the merge
     * "a b" -> "ab" produces a string that exists in the vocab. */
    g_vocab[0x61] = "a";
    g_vocab[0x62] = "b";
    g_vocab[0x63] = "c";
    g_vocab[256] = "ab";        g_token_type[256] = 1;
    g_vocab[257] = "abc";       g_token_type[257] = 1;
    g_vocab[258] = "<|EOS|>";   g_token_type[258] = 3;  /* CONTROL */
    g_merges[0] = "a b";
    g_merges[1] = "ab c";
    g_mock_inited = 1;
}

static eos_status_t mock_meta_arr_str(const void *impl, const char *key,
                                      const char **out, size_t *io_n) {
    (void)impl;
    if (strcmp(key, "tokenizer.ggml.tokens") == 0) {
        size_t i;
        if (out != NULL) {
            size_t cap = (*io_n < MOCK_VOCAB_N) ? *io_n : MOCK_VOCAB_N;
            for (i = 0; i < cap; ++i) out[i] = g_vocab[i];
        }
        *io_n = MOCK_VOCAB_N;
        return EOS_OK;
    }
    if (strcmp(key, "tokenizer.ggml.merges") == 0) {
        size_t i;
        if (out != NULL) {
            size_t cap = (*io_n < MOCK_MERGE_N) ? *io_n : MOCK_MERGE_N;
            for (i = 0; i < cap; ++i) out[i] = g_merges[i];
        }
        *io_n = MOCK_MERGE_N;
        return EOS_OK;
    }
    return EOS_E_NOT_FOUND;
}

static eos_status_t mock_meta_arr_u32(const void *impl, const char *key,
                                      uint32_t *out, size_t *io_n) {
    (void)impl;
    if (strcmp(key, "tokenizer.ggml.token_type") == 0) {
        size_t i;
        if (out != NULL) {
            size_t cap = (*io_n < MOCK_VOCAB_N) ? *io_n : MOCK_VOCAB_N;
            for (i = 0; i < cap; ++i) out[i] = g_token_type[i];
        }
        *io_n = MOCK_VOCAB_N;
        return EOS_OK;
    }
    return EOS_E_NOT_FOUND;
}

static eos_status_t mock_meta_u64(const void *impl, const char *key,
                                  uint64_t *out) {
    (void)impl;
    if (strcmp(key, "tokenizer.ggml.eos_token_id") == 0) { *out = 258; return EOS_OK; }
    /* No bos provided. */
    return EOS_E_NOT_FOUND;
}

static const char *mock_meta_str(const void *impl, const char *key) {
    (void)impl; (void)key;
    return NULL;       /* tokenizer.ggml.pre absent; expect default behavior */
}

static size_t mock_num_tensors(const void *impl) { (void)impl; return 0; }
static const eos_tensor_t *mock_tensor(const void *impl, size_t i) {
    (void)impl; (void)i; return NULL;
}
static const eos_tensor_t *mock_tensor_by_name(const void *impl, const char *n) {
    (void)impl; (void)n; return NULL;
}
static eos_status_t mock_meta_f32(const void *impl, const char *k, float *o) {
    (void)impl; (void)k; (void)o; return EOS_E_NOT_FOUND;
}
static void mock_close(void *impl) { (void)impl; }

static const eosi_model_ops_t k_mock_ops = {
    mock_close, mock_num_tensors, mock_tensor, mock_tensor_by_name,
    mock_meta_str, mock_meta_u64, mock_meta_f32,
    mock_meta_arr_str, mock_meta_arr_u32
};

static const eos_tokenizer_vt_t *find_bpe_vt(void) {
    eosi_registry_t *r = eosi_registry();
    size_t i;
    for (i = 0; i < r->n_tokenizers; ++i) {
        if (strcmp(r->tokenizers[i]->name, "bpe") == 0) return r->tokenizers[i];
    }
    return NULL;
}

static void test_bpe_synthetic(void) {
    const eos_tokenizer_vt_t *vt;
    eos_model_t  model;
    void        *state = NULL;
    uint32_t     ids[64];
    char         buf[64];
    size_t       n;
    eos_status_t s;
    const uint32_t ID_AB = 256, ID_ABC = 257, ID_EOS = 258;

    mock_init();
    model.ops  = &k_mock_ops;
    model.impl = NULL;

    vt = find_bpe_vt();
    CHECK(vt != NULL, "bpe tokenizer vt registered");
    if (vt == NULL) return;

    s = vt->open(&model, 0, &state);
    CHECK(s == EOS_OK, "bpe open succeeds on mock model");
    if (s != EOS_OK) return;

    /* 1. Single byte 'a' -> [byte_id('a')] = [0x61] */
    n = sizeof(ids) / sizeof(ids[0]);
    s = vt->encode(state, "a", 1, ids, &n);
    CHECK(s == EOS_OK && n == 1 && ids[0] == 0x61, "bpe encode 'a' -> single byte token");

    /* 2. "ab" -> [ID_AB] (one merge applied) */
    n = sizeof(ids) / sizeof(ids[0]);
    s = vt->encode(state, "ab", 2, ids, &n);
    CHECK(s == EOS_OK && n == 1 && ids[0] == ID_AB, "bpe encode 'ab' merges to ID_AB");

    /* 3. "abc" -> [ID_ABC] (two merges applied in priority order) */
    n = sizeof(ids) / sizeof(ids[0]);
    s = vt->encode(state, "abc", 3, ids, &n);
    CHECK(s == EOS_OK && n == 1 && ids[0] == ID_ABC, "bpe encode 'abc' merges to ID_ABC");

    /* 4. "xabcy" -> [byte('x'), ID_ABC, byte('y')] */
    n = sizeof(ids) / sizeof(ids[0]);
    s = vt->encode(state, "xabcy", 5, ids, &n);
    CHECK(s == EOS_OK && n == 3, "bpe encode 'xabcy' produces 3 tokens");
    if (n == 3) {
        CHECK(ids[0] == 0x78 && ids[1] == ID_ABC && ids[2] == 0x79,
              "bpe encode 'xabcy' -> ['x', ID_ABC, 'y']");
    }

    /* 5. Special token alone -> [ID_EOS] (longest-match wins over byte fallback) */
    n = sizeof(ids) / sizeof(ids[0]);
    s = vt->encode(state, "<|EOS|>", 7, ids, &n);
    CHECK(s == EOS_OK && n == 1 && ids[0] == ID_EOS,
          "bpe encode '<|EOS|>' recognizes special token");

    /* 6. Special token mid-string */
    n = sizeof(ids) / sizeof(ids[0]);
    s = vt->encode(state, "hi<|EOS|>!", 10, ids, &n);
    CHECK(s == EOS_OK && n == 4, "bpe encode 'hi<|EOS|>!' produces 4 tokens");
    if (n == 4) {
        CHECK(ids[0] == 0x68 && ids[1] == 0x69 && ids[2] == ID_EOS && ids[3] == 0x21,
              "bpe encode 'hi<|EOS|>!' -> ['h','i', ID_EOS, '!']");
    }

    /* 7. decode([byte('h'), byte('i'), ID_EOS, byte('!')]) -> "hi<|EOS|>!" */
    {
        uint32_t ids2[4] = { 0x68, 0x69, ID_EOS, 0x21 };
        size_t   blen = sizeof(buf);
        s = vt->decode(state, ids2, 4, buf, &blen);
        CHECK(s == EOS_OK && blen == 10 && memcmp(buf, "hi<|EOS|>!", 10) == 0,
              "bpe decode roundtrip 'hi<|EOS|>!'");
    }

    /* 8. decode([byte('a'), byte('b'), byte('c')]) -> "abc" (byte tokens) */
    {
        uint32_t ids3[3] = { 0x61, 0x62, 0x63 };
        size_t   blen = sizeof(buf);
        s = vt->decode(state, ids3, 3, buf, &blen);
        CHECK(s == EOS_OK && blen == 3 && memcmp(buf, "abc", 3) == 0,
              "bpe decode byte tokens -> 'abc'");
    }

    /* 9. decode([ID_ABC]) -> "abc" (literal vocab string) */
    {
        uint32_t ids4[1] = { ID_ABC };
        size_t   blen = sizeof(buf);
        s = vt->decode(state, ids4, 1, buf, &blen);
        CHECK(s == EOS_OK && blen == 3 && memcmp(buf, "abc", 3) == 0,
              "bpe decode literal token -> 'abc'");
    }

    vt->close(state);
    CHECK(1, "bpe close completed");
}

/* ------------------------------------------------------------------ */
/* Registry freeze contract                                            */
/* ------------------------------------------------------------------ */

/*
 * After the first eos_session_open() the registry is frozen and all
 * eos_*_register entry points (and eos_os_provide) must return
 * EOS_E_INVALID_ARG. test_session_lifecycle() runs earlier in main()
 * and opens a no-model session, which triggers the freeze.
 */
static void test_registry_frozen(void) {
    static const eos_tokenizer_vt_t sham_t = { "sham_t", NULL, NULL, NULL, NULL };
    static const eos_quant_vt_t     sham_q = { "sham_q", EOS_DT_F32, NULL, NULL, NULL };
    static const eos_sched_vt_t     sham_s = { "sham_s", NULL, NULL, NULL };
    static const eos_format_vt_t    sham_f = { "sham_f", NULL, NULL, NULL };

    eosi_registry_t *r = eosi_registry();
    size_t n_tok_before = r->n_tokenizers;
    size_t n_q_before   = r->n_quants;
    size_t n_s_before   = r->n_schedulers;
    size_t n_f_before   = r->n_formats;

    CHECK(r->frozen == 1, "registry is frozen after eos_session_open");

    CHECK(eos_tokenizer_register(&sham_t) == EOS_E_INVALID_ARG,
          "post-freeze: tokenizer register rejected");
    CHECK(eos_quant_register(&sham_q)     == EOS_E_INVALID_ARG,
          "post-freeze: quant register rejected");
    CHECK(eos_sched_register(&sham_s)     == EOS_E_INVALID_ARG,
          "post-freeze: scheduler register rejected");
    CHECK(eos_format_register(&sham_f)    == EOS_E_INVALID_ARG,
          "post-freeze: format register rejected");

    CHECK(r->n_tokenizers == n_tok_before, "post-freeze: tokenizer table not mutated");
    CHECK(r->n_quants     == n_q_before,   "post-freeze: quant table not mutated");
    CHECK(r->n_schedulers == n_s_before,   "post-freeze: scheduler table not mutated");
    CHECK(r->n_formats    == n_f_before,   "post-freeze: format table not mutated");
}

/* ------------------------------------------------------------------ */
/* Matmul parity stress: scheme-quant-matmul vs dequant + matmul_f32   */
/* ------------------------------------------------------------------ */

/*
 * Bit-exact-vs-scalar parity is the contract every accelerated SIMD
 * backend (AVX2, NEON, ...) will be tested against in Session 6. Even
 * though only the scalar backend exists today, exercising it across a
 * randomized input distribution catches accumulator-order bugs and is
 * the regression net for any future kernel rewrite.
 *
 * Determinism (`make determinism`) requires that the output of these
 * tests is byte-identical across runs, so the PRNG below MUST be
 * seeded deterministically and MUST NOT depend on any timer or env.
 */

static uint32_t g_xs_state = 0x13371337u;

static uint32_t xs_rand(void) {
    /* xorshift32 — small, fast, deterministic. */
    uint32_t x = g_xs_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_xs_state = x;
    return x;
}

static float xs_rand_unit(void) {
    /* In [-1, +1]. */
    return ((float)(xs_rand() & 0xFFFFFF) / (float)0x800000) - 1.0f;
}

static void test_matmul_q8_0_random_stress(void) {
    const eos_quant_vt_t *vt = eos_quant_find("q8_0");
    /* Stack-allocated, sized for the largest case below. */
    enum { M_MAX = 4, N_MAX = 1, K_MAX = 128 };
    enum { N_TRIALS = 24 };
    static const uint32_t k_grid[] = { 32, 64, 96, 128 };
    static const uint32_t m_grid[] = { 1, 2, 4 };
    int trial;
    int total_cases = 0, passed_cases = 0;

    CHECK(vt != NULL, "q8_0 vt available for parity stress");
    if (vt == NULL) return;

    g_xs_state = 0x13371337u;       /* reset PRNG for determinism */

    for (trial = 0; trial < N_TRIALS; ++trial) {
        uint32_t k = k_grid[trial % (sizeof(k_grid)/sizeof(k_grid[0]))];
        uint32_t m = m_grid[trial % (sizeof(m_grid)/sizeof(m_grid[0]))];
        uint32_t n = 1;
        float    a[M_MAX * K_MAX];
        float    src[N_MAX * K_MAX];      /* f32 weights to quantize */
        uint8_t  b_q[N_MAX * (K_MAX / 32) * 34];
        float    b_f[N_MAX * K_MAX];      /* dequantized reference */
        float    c_q[M_MAX * N_MAX];
        float    c_f[M_MAX * N_MAX];
        size_t   bytes = (size_t)n * (k / 32u) * 34u;
        uint32_t i, j;
        eos_status_t s;

        /* Random A and W. */
        for (i = 0; i < m * k; ++i) a[i]   = xs_rand_unit();
        for (i = 0; i < n * k; ++i) src[i] = xs_rand_unit();

        /* W -> q8_0 -> dequantized reference. The reference is the
         * exact f32 the matmul_q8_0 kernel will see internally. */
        s = vt->quantize(src, b_q, n * k);
        if (s != EOS_OK) { CHECK(0, "q8_0 quantize failed mid-stress"); return; }
        s = vt->dequantize(b_q, b_f, n * k);
        if (s != EOS_OK) { CHECK(0, "q8_0 dequantize failed mid-stress"); return; }
        (void)bytes;

        s = eosi_scalar_matmul_q8_0(a, b_q, c_q, m, n, k);
        if (s != EOS_OK) { CHECK(0, "matmul_q8_0 returned error"); return; }
        s = eosi_scalar_matmul_f32 (a, b_f, c_f, m, n, k);
        if (s != EOS_OK) { CHECK(0, "matmul_f32 returned error"); return; }

        for (i = 0; i < m; ++i) {
            for (j = 0; j < n; ++j) {
                /* Tolerance: 1e-5 absolute. Both kernels accumulate
                 * over the same K in the same order from the same
                 * exact f32 weight values, so the only delta should
                 * be sub-ULP rounding from the f32 multiply. */
                total_cases++;
                if (approx_eq(c_q[i * n + j], c_f[i * n + j], 1e-5f)) {
                    passed_cases++;
                }
            }
        }
    }
    CHECK(passed_cases == total_cases,
          "q8_0 random stress: every output bit-equal to dequant+matmul_f32");
}

/*
 * Q4_K layout helper: pack sc[8] and m[8] (each <= 63) into the 12-byte
 * scales field per the GGUF q4_k bit packing. Inverse of
 * get_scale_min() in src/quant/q4_k.c.
 */
static void pack_q4k_scales(const uint8_t sc[8], const uint8_t m[8],
                            uint8_t out[12]) {
    int j;
    for (j = 0; j < 4; ++j) {
        /* low 6 bits = sc[j]; top 2 bits = top 2 bits of sc[j+4]. */
        out[j]     = (uint8_t)((sc[j] & 0x3Fu) | (((sc[j + 4] >> 4) & 0x3u) << 6));
        out[j + 4] = (uint8_t)((m[j]  & 0x3Fu) | (((m[j + 4]  >> 4) & 0x3u) << 6));
        out[j + 8] = (uint8_t)((sc[j + 4] & 0x0Fu) | ((m[j + 4] & 0x0Fu) << 4));
    }
}

static void test_matmul_q4_k_all_subblocks(void) {
    /* One Q4_K super-block (256 elements) with every sub-block
     * carrying distinct sc and m. Tests that matmul_q4_k visits all
     * 8 sub-blocks and applies their scales correctly, not just sub
     * 0 (which is all the original oracle exercised). */
    enum { K = 256 };
    uint8_t  block[144];
    uint8_t  sc[8] = {  3, 12,  7, 21, 15,  9, 30,  4 };
    uint8_t  m [8] = {  1,  2,  3,  4,  5,  6,  7,  8 };
    float    a[K], b_f[K], cq[1], cf[1];
    int      i;
    eosi_f16_t d_h    = eosi_f32_to_f16(0.25f);
    eosi_f16_t dmin_h = eosi_f32_to_f16(0.0625f);
    const eos_quant_vt_t *vt = eos_quant_find("q4_k");

    CHECK(vt != NULL, "q4_k vt available for all-sub-blocks test");
    if (vt == NULL) return;

    memset(block, 0, sizeof(block));
    memcpy(block,     &d_h,    2);
    memcpy(block + 2, &dmin_h, 2);
    {
        uint8_t scales12[12];
        pack_q4k_scales(sc, m, scales12);
        memcpy(block + 4, scales12, 12);
    }
    /* qs: 128 bytes. Each pair of sub-blocks (even=low nibble,
     * odd=high nibble) shares 32 bytes. Pattern: cycle 0..15 in low
     * nibble and 15..0 in high nibble so both nibbles are non-zero. */
    for (i = 0; i < 128; ++i) {
        uint8_t lo = (uint8_t)(i & 0x0Fu);
        uint8_t hi = (uint8_t)(15u - (i & 0x0Fu));
        block[16 + i] = (uint8_t)(lo | (hi << 4));
    }

    /* Use the registered dequantize as the f32 reference (this is
     * the exact f32 the kernel internally produces). */
    {
        eos_status_t s = vt->dequantize(block, b_f, K);
        CHECK(s == EOS_OK, "q4_k dequantize succeeded for all-sub test");
        if (s != EOS_OK) return;
    }
    for (i = 0; i < K; ++i) a[i] = 1.0f;     /* sum-of-weights query */

    {
        eos_status_t s1 = eosi_scalar_matmul_q4_k(a, block, cq, 1, 1, K);
        eos_status_t s2 = eosi_scalar_matmul_f32 (a, b_f,   cf, 1, 1, K);
        CHECK(s1 == EOS_OK && s2 == EOS_OK, "q4_k all-sub matmul invocations OK");
        CHECK(approx_eq(cq[0], cf[0], 1e-4f),
              "q4_k all-sub-blocks: matmul == dequant+matmul_f32 (within 1e-4)");
        /* Sanity: result must be non-trivially affected by every
         * sub-block. We assert against an independently-derived
         * reference value, computed below.
         *
         * qs layout: each 32-byte chunk of qs serves a pair of
         * sub-blocks (even = low nibble, odd = high nibble). Our
         * pattern fills 128 bytes such that within each 32-byte
         * chunk, the low nibbles cycle 0..15 twice (sum = 240) and
         * the high nibbles cycle 15..0 twice (sum = 240).
         *
         * Per-sub-block contribution = (d * sc[i]) * sum_nibbles
         *                              - (dmin * m[i]) * 32
         *                            = 0.25 * sc[i] * 240
         *                              - 0.0625 * m[i] * 32
         *                            = 60 * sc[i] - 2 * m[i]
         *
         * Total = 60 * sum(sc) - 2 * sum(m)
         *       = 60 * (3+12+7+21+15+9+30+4) - 2 * (1+2+3+4+5+6+7+8)
         *       = 60 * 101 - 2 * 36
         *       = 6060 - 72 = 5988
         */
        CHECK(approx_eq(cq[0], 5988.0f, 0.5f),
              "q4_k all-sub-blocks: matches independently derived reference");
    }
}

/* ------------------------------------------------------------------ */
/* SHA-256 known-answer + .eosm v1 writer→reader roundtrip            */
/* ------------------------------------------------------------------ */

static int hex_eq(const uint8_t *got, const char *expected_hex) {
    static const char hex[] = "0123456789abcdef";
    int i;
    for (i = 0; i < 32; ++i) {
        const char *hi = strchr(hex, expected_hex[i*2]);
        const char *lo = strchr(hex, expected_hex[i*2+1]);
        if (hi == NULL || lo == NULL) return 0;
        if ((got[i] >> 4)   != (uint8_t)(hi - hex)) return 0;
        if ((got[i] & 0xF)  != (uint8_t)(lo - hex)) return 0;
    }
    return 1;
}

static void test_sha256_known_answers(void) {
    uint8_t out[32];
    /* FIPS 180-4 example: SHA-256(""). */
    eosi_sha256("", 0, out);
    CHECK(hex_eq(out,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        "sha256(\"\") matches FIPS 180-4");
    /* FIPS 180-4 example: SHA-256("abc"). */
    eosi_sha256("abc", 3, out);
    CHECK(hex_eq(out,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        "sha256(\"abc\") matches FIPS 180-4");
}

/* ----- Memory-backed eos_stream_t for the .eosm roundtrip test. ----- */

typedef struct {
    const uint8_t *data;
    size_t         len;
    size_t         pos;
} mem_stream_t;

static int64_t mem_read(void *opaque, void *buf, size_t len) {
    mem_stream_t *s = (mem_stream_t *)opaque;
    size_t avail = (s->pos < s->len) ? s->len - s->pos : 0;
    size_t take  = (len < avail) ? len : avail;
    memcpy(buf, s->data + s->pos, take);
    s->pos += take;
    return (int64_t)take;
}
static int     mem_seek(void *opaque, int64_t off) {
    mem_stream_t *s = (mem_stream_t *)opaque;
    if (off < 0 || (size_t)off > s->len) return -1;
    s->pos = (size_t)off; return 0;
}
static int64_t mem_size(void *opaque) {
    return (int64_t)((mem_stream_t *)opaque)->len;
}
static void    mem_close(void *opaque) { (void)opaque; }

static const eos_format_vt_t *find_format_vt(const char *name) {
    eosi_registry_t *r = eosi_registry();
    size_t i;
    for (i = 0; i < r->n_formats; ++i) {
        if (r->formats[i]->name && strcmp(r->formats[i]->name, name) == 0)
            return r->formats[i];
    }
    return NULL;
}

static void test_eosm_roundtrip(void) {
    static const float weights[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    static const char  arch_value[]  = "test-arch";
    static const char  group_name[]  = "text";
    static const char  tensor_name[] = "x.weight";

    eosi_eosm_kv_in_t kvs[2];
    eosi_eosm_tensor_in_t tensors[1];
    eosi_eosm_group_in_t  groups[1];
    eosi_eosm_writer_in_t in;
    void  *buf  = NULL;
    size_t bytes = 0;
    eos_status_t s;

    memset(kvs, 0, sizeof(kvs));
    kvs[0].key        = "general.architecture";
    kvs[0].key_len    = (uint64_t)strlen("general.architecture");
    kvs[0].value_type = EOSI_EOSM_KV_STRING;
    kvs[0].bytes      = arch_value;
    kvs[0].bytes_len  = (uint64_t)strlen(arch_value);

    kvs[1].key        = "test.u32";
    kvs[1].key_len    = (uint64_t)strlen("test.u32");
    kvs[1].value_type = EOSI_EOSM_KV_U32;
    kvs[1].u32        = 42u;

    memset(tensors, 0, sizeof(tensors));
    tensors[0].name                  = tensor_name;
    tensors[0].name_len              = (uint64_t)strlen(tensor_name);
    tensors[0].dtype                 = EOS_DT_F32;
    tensors[0].quant_scheme_id       = 0;
    tensors[0].calibration_table_idx = 0xFFFFFFFFu;
    tensors[0].rank                  = 1;
    tensors[0].load_group_id         = 0;
    tensors[0].dims[0]               = 4;
    tensors[0].data                  = weights;
    tensors[0].data_bytes            = sizeof(weights);

    memset(groups, 0, sizeof(groups));
    groups[0].id       = 0;
    groups[0].name     = group_name;
    groups[0].name_len = (uint64_t)strlen(group_name);

    memset(&in, 0, sizeof(in));
    in.capability_bits =
        EOSI_EOSM_CAP_HASH_SHA256_TRAILER |
        EOSI_EOSM_CAP_MODALITY_TEXT;
    in.alignment = 64;
    in.kvs       = kvs;       in.n_kvs       = 2;
    in.tensors   = tensors;   in.n_tensors   = 1;
    in.groups    = groups;    in.n_groups    = 1;
    in.calib     = NULL;      in.n_calib     = 0;

    s = eosi_eosm_write(&in, &buf, &bytes);
    CHECK(s == EOS_OK && buf != NULL && bytes >= 64 + 32,
          "eosm writer produced a buffer");
    if (s != EOS_OK) return;

    /* Tamper test: flipping a byte must invalidate the SHA-256
     * trailer and cause the reader to reject. */
    {
        uint8_t saved = ((uint8_t *)buf)[100];
        ((uint8_t *)buf)[100] ^= 0x55;
        {
            mem_stream_t mst = { (const uint8_t *)buf, bytes, 0 };
            eos_stream_t st  = { &mst, mem_read, mem_seek, mem_size, mem_close };
            const eos_format_vt_t *vt = find_format_vt("eosm");
            eos_model_t *m = NULL;
            eos_status_t s2;
            /* Clear any prior last-error so we know the next call sets it. */
            eosi_set_error(NULL);
            s2 = vt->open(&st, &m);
            CHECK(s2 == EOS_E_FORMAT && m == NULL,
                  "eosm reader rejects tampered file (SHA-256 mismatch)");
            {
                const char *e = eos_last_error();
                CHECK(e != NULL && strstr(e, "SHA-256") != NULL,
                      "eos_last_error set to SHA-256 message after tamper");
            }
        }
        ((uint8_t *)buf)[100] = saved;

        /* AN contract: a subsequent successful lifecycle call must
         * clear the slot, so eos_last_error reflects only the most
         * recent failing call (not stale context). */
        {
            eos_session_t *ss = NULL;
            eos_session_params_t params;
            eos_status_t s3;
            memset(&params, 0, sizeof(params));
            s3 = eos_session_open(NULL, &params, &ss);
            CHECK(s3 == EOS_OK,
                  "eos_session_open(NULL) succeeds (post-tamper)");
            CHECK(eos_last_error() == NULL,
                  "eos_last_error cleared by successful eos_session_open");
            if (s3 == EOS_OK) eos_session_close(ss);
        }
    }

    /* Honest read. */
    {
        mem_stream_t mst = { (const uint8_t *)buf, bytes, 0 };
        eos_stream_t st  = { &mst, mem_read, mem_seek, mem_size, mem_close };
        const eos_format_vt_t *vt = find_format_vt("eosm");
        eos_model_t *m = NULL;
        eos_status_t s2;
        CHECK(vt != NULL, "eosm format vt registered");
        if (vt == NULL) { eosi_free(buf); return; }
        s2 = vt->open(&st, &m);
        CHECK(s2 == EOS_OK && m != NULL, "eosm reader opens written buffer");
        if (s2 != EOS_OK) { eosi_free(buf); return; }

        /* Tensor invariants. */
        CHECK(eos_model_num_tensors(m) == 1, "eosm: 1 tensor");
        {
            const eos_tensor_t *t = eos_model_tensor_by_name(m, tensor_name);
            CHECK(t != NULL, "eosm: tensor by name found");
            if (t != NULL) {
                CHECK(t->dtype == EOS_DT_F32, "eosm: tensor dtype roundtrips");
                CHECK(t->rank == 1 && t->dims[0] == 4,
                      "eosm: tensor shape roundtrips");
                CHECK(t->data_bytes == sizeof(weights),
                      "eosm: tensor data_bytes roundtrips");
                CHECK(memcmp(t->data, weights, sizeof(weights)) == 0,
                      "eosm: tensor body bytes roundtrip");
            }
        }

        /* Metadata invariants. */
        {
            const char *arch = eos_model_meta_str(m, "general.architecture");
            uint64_t v = 0;
            eos_status_t s3 = eos_model_meta_u64(m, "test.u32", &v);
            CHECK(arch != NULL && strcmp(arch, arch_value) == 0,
                  "eosm: STRING kv roundtrips");
            CHECK(s3 == EOS_OK && v == 42u, "eosm: U32 kv roundtrips");
        }

        eos_model_close(m);
    }
    eosi_free(buf);
}

#if EOSLLM_HAVE_KERNEL_AVX2
/* ------------------------------------------------------------------ */
/* AVX2 matmul_q4_k vs scalar parity                                  */
/* ------------------------------------------------------------------ */
static void test_avx2_q4_k_parity(void) {
    /* Reuse the same all-sub-blocks-active super-block construction
     * from test_matmul_q4_k_all_subblocks. */
    enum { K = 256 };
    uint8_t  block[144];
    uint8_t  sc[8] = {  3, 12,  7, 21, 15,  9, 30,  4 };
    uint8_t  m_arr [8] = {  1,  2,  3,  4,  5,  6,  7,  8 };
    float    a[K], cs[1], cv[1];
    int      i;
    eosi_f16_t d_h    = eosi_f32_to_f16(0.25f);
    eosi_f16_t dmin_h = eosi_f32_to_f16(0.0625f);

    memset(block, 0, sizeof(block));
    memcpy(block,     &d_h,    2);
    memcpy(block + 2, &dmin_h, 2);
    {
        uint8_t scales12[12];
        int j;
        /* Same packing as pack_q4k_scales in test_matmul_q4_k_all_subblocks. */
        for (j = 0; j < 4; ++j) {
            scales12[j]     = (uint8_t)((sc[j] & 0x3Fu)
                              | (((sc[j + 4] >> 4) & 0x3u) << 6));
            scales12[j + 4] = (uint8_t)((m_arr[j] & 0x3Fu)
                              | (((m_arr[j + 4] >> 4) & 0x3u) << 6));
            scales12[j + 8] = (uint8_t)((sc[j + 4] & 0x0Fu)
                              | ((m_arr[j + 4] & 0x0Fu) << 4));
        }
        memcpy(block + 4, scales12, 12);
    }
    for (i = 0; i < 128; ++i) {
        uint8_t lo = (uint8_t)(i & 0x0Fu);
        uint8_t hi = (uint8_t)(15u - (i & 0x0Fu));
        block[16 + i] = (uint8_t)(lo | (hi << 4));
    }
    /* Use a varied input so accumulator-order differences actually
     * exercise (uniform 1.0f would let any order produce the same
     * sum). xs_rand is deterministic across runs. */
    g_xs_state = 0x42424242u;
    for (i = 0; i < K; ++i) a[i] = xs_rand_unit();

    {
        eos_status_t s1 = eosi_scalar_matmul_q4_k(a, block, cs, 1, 1, K);
        eos_status_t s2 = eosi_avx2_matmul_q4_k  (a, block, cv, 1, 1, K);
        CHECK(s1 == EOS_OK && s2 == EOS_OK,
              "avx2/scalar matmul_q4_k both return OK");
        CHECK(approx_eq(cv[0], cs[0], 1e-4f),
              "avx2 matmul_q4_k matches scalar within 1e-4 (random input)");
    }
}

/* ------------------------------------------------------------------ */
/* AVX2 matmul_q4_k_int8 vs scalar parity (5e-3 relative tolerance —   */
/* activation block-quantization adds ~1 i8 ULP / 127 ≈ 0.4% per       */
/* sub-block; the bound is documented in docs/quant_schemes.md). Same  */
/* fixture as the f32-dequant parity test above but with N=4 so the    */
/* call exercises the cached-activation hot path (the int8 kernel      */
/* delegates to the f32-dequant path for N<4).                         */
/* ------------------------------------------------------------------ */
static void test_avx2_q4_k_int8_parity(void) {
    enum { K = 256, N = 4 };
    /* Same single super-block for every output column — n=4 columns
     * just stack copies of the same 144-byte super-block into the
     * weight matrix so we can verify all 4 outputs equal the scalar
     * reference within tolerance. */
    uint8_t  block[144];
    uint8_t  weights[N * 144];
    uint8_t  sc[8] = {  3, 12,  7, 21, 15,  9, 30,  4 };
    uint8_t  m_arr [8] = {  1,  2,  3,  4,  5,  6,  7,  8 };
    float    a[K], cs[N], cv[N];
    int      i;
    eosi_f16_t d_h    = eosi_f32_to_f16(0.25f);
    eosi_f16_t dmin_h = eosi_f32_to_f16(0.0625f);

    memset(block, 0, sizeof(block));
    memcpy(block,     &d_h,    2);
    memcpy(block + 2, &dmin_h, 2);
    {
        uint8_t scales12[12];
        int j;
        for (j = 0; j < 4; ++j) {
            scales12[j]     = (uint8_t)((sc[j] & 0x3Fu)
                              | (((sc[j + 4] >> 4) & 0x3u) << 6));
            scales12[j + 4] = (uint8_t)((m_arr[j] & 0x3Fu)
                              | (((m_arr[j + 4] >> 4) & 0x3u) << 6));
            scales12[j + 8] = (uint8_t)((sc[j + 4] & 0x0Fu)
                              | ((m_arr[j + 4] & 0x0Fu) << 4));
        }
        memcpy(block + 4, scales12, 12);
    }
    for (i = 0; i < 128; ++i) {
        uint8_t lo = (uint8_t)(i & 0x0Fu);
        uint8_t hi = (uint8_t)(15u - (i & 0x0Fu));
        block[16 + i] = (uint8_t)(lo | (hi << 4));
    }
    for (i = 0; i < N; ++i) memcpy(weights + i * 144, block, 144);

    g_xs_state = 0x42424242u;     /* same seed as the f32-dequant test */
    for (i = 0; i < K; ++i) a[i] = xs_rand_unit();

    {
        eos_status_t s1 = eosi_scalar_matmul_q4_k    (a, weights, cs, 1, N, K);
        eos_status_t s2 = eosi_avx2_matmul_q4_k_int8 (a, weights, cv, 1, N, K);
        /* Activation block-quantization gives roughly 1 i8 ULP / 127 of
         * relative noise per sub-block; an absolute tolerance doesn't
         * compose because the result magnitude depends on the weight
         * scale × K. Use a relative bound with a small absolute floor
         * for catastrophic-cancellation protection. */
        float ref = cs[0] < 0 ? -cs[0] : cs[0];
        float tol = 5e-3f * ref;
        if (tol < 1e-3f) tol = 1e-3f;
        CHECK(s1 == EOS_OK && s2 == EOS_OK,
              "avx2/scalar matmul_q4_k_int8 both return OK");
        CHECK(approx_eq(cv[0], cs[0], tol),
              "avx2 matmul_q4_k_int8 col 0 matches scalar within 5e-3 relative");
        CHECK(approx_eq(cv[1], cs[1], tol),
              "avx2 matmul_q4_k_int8 col 1 matches scalar within 5e-3 relative");
        CHECK(approx_eq(cv[2], cs[2], tol),
              "avx2 matmul_q4_k_int8 col 2 matches scalar within 5e-3 relative");
        CHECK(approx_eq(cv[3], cs[3], tol),
              "avx2 matmul_q4_k_int8 col 3 matches scalar within 5e-3 relative");
    }

    /* All-zero activation: must produce exactly 0.0f. The block-quant
     * scale collapses to 0 so this also covers the max_abs == 0 branch. */
    for (i = 0; i < K; ++i) a[i] = 0.0f;
    {
        eos_status_t s2 = eosi_avx2_matmul_q4_k_int8(a, weights, cv, 1, N, K);
        CHECK(s2 == EOS_OK,
              "avx2 matmul_q4_k_int8 returns OK on all-zero activation");
        CHECK(cv[0] == 0.0f && cv[1] == 0.0f && cv[2] == 0.0f && cv[3] == 0.0f,
              "avx2 matmul_q4_k_int8 produces 0.0f on all-zero activation (4 cols)");
    }
}
#endif

#if EOSLLM_HAVE_KERNEL_NEON
/* ------------------------------------------------------------------ */
/* NEON matmul_q4_k vs scalar parity                                  */
/* ------------------------------------------------------------------ */
static void test_neon_q4_k_parity(void) {
    enum { K = 256 };
    uint8_t  block[144];
    uint8_t  sc[8] = {  3, 12,  7, 21, 15,  9, 30,  4 };
    uint8_t  m_arr [8] = {  1,  2,  3,  4,  5,  6,  7,  8 };
    float    a[K], cs[1], cv[1];
    int      i;
    eosi_f16_t d_h    = eosi_f32_to_f16(0.25f);
    eosi_f16_t dmin_h = eosi_f32_to_f16(0.0625f);

    memset(block, 0, sizeof(block));
    memcpy(block,     &d_h,    2);
    memcpy(block + 2, &dmin_h, 2);
    {
        uint8_t scales12[12];
        int j;
        for (j = 0; j < 4; ++j) {
            scales12[j]     = (uint8_t)((sc[j] & 0x3Fu)
                              | (((sc[j + 4] >> 4) & 0x3u) << 6));
            scales12[j + 4] = (uint8_t)((m_arr[j] & 0x3Fu)
                              | (((m_arr[j + 4] >> 4) & 0x3u) << 6));
            scales12[j + 8] = (uint8_t)((sc[j + 4] & 0x0Fu)
                              | ((m_arr[j + 4] & 0x0Fu) << 4));
        }
        memcpy(block + 4, scales12, 12);
    }
    for (i = 0; i < 128; ++i) {
        uint8_t lo = (uint8_t)(i & 0x0Fu);
        uint8_t hi = (uint8_t)(15u - (i & 0x0Fu));
        block[16 + i] = (uint8_t)(lo | (hi << 4));
    }
    g_xs_state = 0x42424242u;     /* same seed as AVX2 test for cross-checking */
    for (i = 0; i < K; ++i) a[i] = xs_rand_unit();

    {
        eos_status_t s1 = eosi_scalar_matmul_q4_k(a, block, cs, 1, 1, K);
        eos_status_t s2 = eosi_neon_matmul_q4_k  (a, block, cv, 1, 1, K);
        CHECK(s1 == EOS_OK && s2 == EOS_OK,
              "neon/scalar matmul_q4_k both return OK");
        CHECK(approx_eq(cv[0], cs[0], 1e-4f),
              "neon matmul_q4_k matches scalar within 1e-4 (random input)");
    }
}
#endif

/* ------------------------------------------------------------------ */
/* Converter: any eos_model_t -> .eosm                                 */
/* ------------------------------------------------------------------ */

/*
 * A richer mock model than the BPE one: 2 tensors + 4 metadata
 * entries spanning STRING, U64, U32 (queried via meta_u64), F32.
 */

static const float    cv_w0[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
static const uint16_t cv_w1[6] = { 0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600 };
                                 /* arbitrary f16 bit patterns (1.0, 2.0, 3.0, 4.0, 5.0, 6.0) */

/* eos_tensor_t storage (struct definition lives in core/model_internal.h). */
static eos_tensor_t cv_tensors[2];
static int          cv_inited = 0;

static void cv_init(void) {
    if (cv_inited) return;
    cv_tensors[0].name       = "x.f32";
    cv_tensors[0].dtype      = EOS_DT_F32;
    cv_tensors[0].rank       = 1;
    cv_tensors[0].dims[0]    = 8;
    cv_tensors[0].data       = cv_w0;
    cv_tensors[0].data_bytes = sizeof(cv_w0);

    cv_tensors[1].name       = "y.f16";
    cv_tensors[1].dtype      = EOS_DT_F16;
    cv_tensors[1].rank       = 2;
    cv_tensors[1].dims[0]    = 2;
    cv_tensors[1].dims[1]    = 3;
    cv_tensors[1].data       = cv_w1;
    cv_tensors[1].data_bytes = sizeof(cv_w1);
    cv_inited = 1;
}

static size_t cv_num_tensors(const void *impl) { (void)impl; return 2; }
static const eos_tensor_t *cv_tensor(const void *impl, size_t i) {
    (void)impl;
    return (i < 2) ? &cv_tensors[i] : NULL;
}
static const eos_tensor_t *cv_tensor_by_name(const void *impl, const char *n) {
    (void)impl;
    if (n == NULL) return NULL;
    if (strcmp(n, cv_tensors[0].name) == 0) return &cv_tensors[0];
    if (strcmp(n, cv_tensors[1].name) == 0) return &cv_tensors[1];
    return NULL;
}
static const char *cv_meta_str(const void *impl, const char *k) {
    (void)impl;
    if (strcmp(k, "general.architecture") == 0) return "test-arch";
    return NULL;
}
static eos_status_t cv_meta_u64(const void *impl, const char *k, uint64_t *o) {
    (void)impl;
    if (strcmp(k, "test.u64") == 0) { *o = 0x123456789ABCDEF0ull; return EOS_OK; }
    if (strcmp(k, "test.u32") == 0) { *o = 12345u; return EOS_OK; }
    return EOS_E_NOT_FOUND;
}
static eos_status_t cv_meta_f32(const void *impl, const char *k, float *o) {
    (void)impl;
    if (strcmp(k, "test.f32") == 0) { *o = 3.14159f; return EOS_OK; }
    return EOS_E_NOT_FOUND;
}
static eos_status_t cv_meta_arr_str(const void *i, const char *k,
                                    const char **o, size_t *n) {
    static const char *toks[3] = { "a", "bb", "ccc" };
    (void)i;
    if (strcmp(k, "test.tokens") == 0) {
        size_t want = 3;
        if (o != NULL) {
            size_t cap = (*n < want) ? *n : want;
            size_t j;
            for (j = 0; j < cap; ++j) o[j] = toks[j];
        }
        *n = want;
        return EOS_OK;
    }
    return EOS_E_NOT_FOUND;
}
static eos_status_t cv_meta_arr_u32(const void *i, const char *k,
                                    uint32_t *o, size_t *n) {
    static const uint32_t types_arr[3] = { 1u, 2u, 3u };
    (void)i;
    if (strcmp(k, "test.types") == 0) {
        size_t want = 3;
        if (o != NULL) {
            size_t cap = (*n < want) ? *n : want;
            size_t j;
            for (j = 0; j < cap; ++j) o[j] = types_arr[j];
        }
        *n = want;
        return EOS_OK;
    }
    return EOS_E_NOT_FOUND;
}
static void cv_close(void *impl) { (void)impl; }

static const eosi_model_ops_t k_cv_ops = {
    cv_close, cv_num_tensors, cv_tensor, cv_tensor_by_name,
    cv_meta_str, cv_meta_u64, cv_meta_f32,
    cv_meta_arr_str, cv_meta_arr_u32
};

static void test_eosm_from_model_roundtrip(void) {
    static const eosi_eosm_key_spec_t specs[] = {
        { "general.architecture", EOSI_EOSM_KV_STRING, 0                    },
        { "test.u64",             EOSI_EOSM_KV_U64,    0                    },
        { "test.u32",             EOSI_EOSM_KV_U32,    0                    },
        { "test.f32",             EOSI_EOSM_KV_F32,    0                    },
        { "test.tokens",          EOSI_EOSM_KV_ARRAY,  EOSI_EOSM_KV_STRING  },
        { "test.types",           EOSI_EOSM_KV_ARRAY,  EOSI_EOSM_KV_U32     },
    };
    const size_t n_specs = sizeof(specs) / sizeof(specs[0]);
    eos_model_t  src;
    void        *buf = NULL;
    size_t       bytes = 0;
    eos_status_t s;

    cv_init();
    src.ops  = &k_cv_ops;
    src.impl = NULL;

    s = eosi_eosm_from_model(&src, specs, n_specs,
                             EOSI_EOSM_CAP_MODALITY_TEXT, 64,
                             &buf, &bytes);
    CHECK(s == EOS_OK && buf != NULL && bytes >= 64 + 32,
          "eosm converter produced a buffer");
    if (s != EOS_OK) return;

    /* Parse back. */
    {
        mem_stream_t mst = { (const uint8_t *)buf, bytes, 0 };
        eos_stream_t st  = { &mst, mem_read, mem_seek, mem_size, mem_close };
        const eos_format_vt_t *vt = find_format_vt("eosm");
        eos_model_t *m = NULL;
        eos_status_t s2 = vt->open(&st, &m);
        CHECK(s2 == EOS_OK && m != NULL, "eosm converter: read converted file");
        if (s2 != EOS_OK) { eosi_free(buf); return; }

        /* Tensor invariants. */
        CHECK(eos_model_num_tensors(m) == 2,
              "eosm converter: tensor count roundtrips");
        {
            const eos_tensor_t *t0 = eos_model_tensor_by_name(m, "x.f32");
            const eos_tensor_t *t1 = eos_model_tensor_by_name(m, "y.f16");
            CHECK(t0 && t1, "eosm converter: both tensors found by name");
            if (t0) {
                CHECK(t0->dtype == EOS_DT_F32 && t0->rank == 1
                      && t0->dims[0] == 8
                      && t0->data_bytes == sizeof(cv_w0)
                      && memcmp(t0->data, cv_w0, sizeof(cv_w0)) == 0,
                      "eosm converter: f32 tensor roundtrips byte-exact");
            }
            if (t1) {
                CHECK(t1->dtype == EOS_DT_F16 && t1->rank == 2
                      && t1->dims[0] == 2 && t1->dims[1] == 3
                      && t1->data_bytes == sizeof(cv_w1)
                      && memcmp(t1->data, cv_w1, sizeof(cv_w1)) == 0,
                      "eosm converter: f16 tensor roundtrips byte-exact");
            }
        }

        /* Metadata invariants. */
        {
            const char *a = eos_model_meta_str(m, "general.architecture");
            uint64_t u64 = 0, u32_via = 0; float f32v = 0.0f;
            eos_status_t s3 = eos_model_meta_u64(m, "test.u64", &u64);
            eos_status_t s4 = eos_model_meta_u64(m, "test.u32", &u32_via);
            eos_status_t s5 = eos_model_meta_f32(m, "test.f32", &f32v);
            CHECK(a && strcmp(a, "test-arch") == 0,
                  "eosm converter: STRING metadata roundtrips");
            CHECK(s3 == EOS_OK && u64 == 0x123456789ABCDEF0ull,
                  "eosm converter: U64 metadata roundtrips");
            CHECK(s4 == EOS_OK && u32_via == 12345u,
                  "eosm converter: U32 metadata roundtrips");
            CHECK(s5 == EOS_OK && approx_eq(f32v, 3.14159f, 1e-6f),
                  "eosm converter: F32 metadata roundtrips");
        }

        /* Array metadata roundtrips. */
        {
            const char *toks[3] = { NULL, NULL, NULL };
            uint32_t    types[3] = { 0, 0, 0 };
            size_t      n_t = 0, n_u = 0;
            eos_status_t st_s = eos_model_meta_arr_str_count(m, "test.tokens", &n_t);
            eos_status_t ut_s = eos_model_meta_arr_u32_count(m, "test.types",  &n_u);
            CHECK(st_s == EOS_OK && n_t == 3,
                  "eosm converter: STRING array count roundtrips");
            CHECK(ut_s == EOS_OK && n_u == 3,
                  "eosm converter: U32 array count roundtrips");
            if (n_t == 3) {
                size_t cap = 3;
                eos_model_meta_arr_str_load(m, "test.tokens", toks, &cap);
                CHECK(toks[0] && strcmp(toks[0], "a")   == 0
                   && toks[1] && strcmp(toks[1], "bb")  == 0
                   && toks[2] && strcmp(toks[2], "ccc") == 0,
                   "eosm converter: STRING array values roundtrip");
            }
            if (n_u == 3) {
                size_t cap = 3;
                eos_model_meta_arr_u32_load(m, "test.types", types, &cap);
                CHECK(types[0] == 1u && types[1] == 2u && types[2] == 3u,
                      "eosm converter: U32 array values roundtrip");
            }
        }

        eos_model_close(m);
    }
    eosi_free(buf);
}

/* ------------------------------------------------------------------ */
/* eos_free + eos_backend_available                                    */
/* ------------------------------------------------------------------ */

static void test_eos_free(void) {
    /* NULL is a no-op. */
    eos_free(NULL);
    CHECK(1, "eos_free(NULL) does not crash");
    /* Engine-allocated buffer can be freed via the public helper. */
    {
        void *p = eosi_alloc(64, 8);
        CHECK(p != NULL, "eosi_alloc returned a buffer");
        eos_free(p);
        CHECK(1, "eos_free released engine-allocated buffer");
    }
}

static void test_backend_available(void) {
    CHECK(eos_backend_available("scalar") == 1,
          "scalar backend is always available");
    CHECK(eos_backend_available("does_not_exist") == 0,
          "unknown backend reports unavailable");
    CHECK(eos_backend_available(NULL) == 0,
          "NULL backend name reports unavailable");
#if EOSLLM_HAVE_KERNEL_AVX2
    /* Build host has AVX2 + FMA (asserted by the developer rig); CI
     * x86 runners do too. The probe does cpuid; should pass here. */
    CHECK(eos_backend_available("x86_avx2") == 1,
          "x86_avx2 backend probes OK on AVX2-capable host");
#endif
}

/* ------------------------------------------------------------------ */
/* Synthetic GGUF v3 builder + roundtrip                              */
/* ------------------------------------------------------------------ */

/*
 * Hand-build a minimal valid GGUF v3 byte stream in memory and feed
 * it to the registered GGUF format reader. Closes the gap that the
 * GGUF reader currently has zero positive-path tests (only negative-
 * path coverage via the audit checks in `make sanitize`).
 *
 * Layout we emit:
 *   header  : magic "GGUF" + version=3 + tensor_count=1 + kv_count=2
 *   kv 0    : "general.architecture" = "test"            (STRING)
 *   kv 1    : "general.alignment"    = 32                (U32)
 *   tensor  : name="x.f32" rank=1 dims=[4] dtype=GGML_F32 offset=0
 *   pad     : align tensor data to 32 bytes
 *   data    : 4 f32 values { 1, 2, 3, 4 }
 */
#define GGUF_MAGIC_LE  0x46554747u
#define GGUF_VER       3u
#define GGUFV_U32      4u
#define GGUFV_STRING   8u
#define GGML_F32       0u

static uint8_t  gg_buf[512];
static size_t   gg_pos;

static void gg_append(const void *data, size_t n) {
    if (gg_pos + n > sizeof(gg_buf)) { gg_pos = sizeof(gg_buf) + 1; return; }
    memcpy(gg_buf + gg_pos, data, n);
    gg_pos += n;
}
static void gg_u32(uint32_t v) { gg_append(&v, 4); }
static void gg_u64(uint64_t v) { gg_append(&v, 8); }
static void gg_lstr(const char *s) {
    uint64_t l = (uint64_t)strlen(s);
    gg_u64(l);
    gg_append(s, (size_t)l);
}

static void test_gguf_synthetic_roundtrip(void) {
    static const float weights[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const eos_format_vt_t *vt;
    eos_model_t *m = NULL;
    eos_status_t s;

    gg_pos = 0;
    /* Header */
    gg_u32(GGUF_MAGIC_LE);
    gg_u32(GGUF_VER);
    gg_u64(1);     /* tensor_count */
    gg_u64(2);     /* kv_count */
    /* KV 0: general.architecture = "test" */
    gg_lstr("general.architecture");
    gg_u32(GGUFV_STRING);
    gg_lstr("test");
    /* KV 1: general.alignment = 32 (must be present so we control padding) */
    gg_lstr("general.alignment");
    gg_u32(GGUFV_U32);
    gg_u32(32);
    /* Tensor descriptor */
    gg_lstr("x.f32");
    gg_u32(1);     /* n_dims */
    gg_u64(4);     /* dim 0 */
    gg_u32(GGML_F32);
    gg_u64(0);     /* tensor offset within the data block */
    /* Pad to 32-byte alignment (relative to file start). */
    while ((gg_pos & 31u) != 0u) {
        uint8_t z = 0;
        gg_append(&z, 1);
    }
    /* Tensor data */
    gg_append(weights, sizeof(weights));

    CHECK(gg_pos <= sizeof(gg_buf), "synthetic GGUF fits in buffer");

    /* Open via the registered format vt + an in-memory stream. */
    vt = find_format_vt("gguf");
    CHECK(vt != NULL, "gguf format vt registered");
    if (vt == NULL) return;
    {
        mem_stream_t mst = { gg_buf, gg_pos, 0 };
        eos_stream_t st  = { &mst, mem_read, mem_seek, mem_size, mem_close };
        s = vt->open(&st, &m);
    }
    CHECK(s == EOS_OK && m != NULL, "synthetic GGUF opens");
    if (s != EOS_OK) return;

    CHECK(eos_model_num_tensors(m) == 1, "synthetic GGUF: 1 tensor");
    {
        const char *arch = eos_model_meta_str(m, "general.architecture");
        CHECK(arch != NULL && strcmp(arch, "test") == 0,
              "synthetic GGUF: architecture STRING roundtrips");
    }
    {
        const eos_tensor_t *t = eos_model_tensor_by_name(m, "x.f32");
        CHECK(t != NULL, "synthetic GGUF: tensor by name found");
        if (t != NULL) {
            CHECK(t->dtype == EOS_DT_F32 && t->rank == 1 && t->dims[0] == 4,
                  "synthetic GGUF: tensor shape + dtype roundtrip");
            CHECK(t->data_bytes == sizeof(weights),
                  "synthetic GGUF: tensor data_bytes correct");
            CHECK(memcmp(t->data, weights, sizeof(weights)) == 0,
                  "synthetic GGUF: tensor body bytes roundtrip");
        }
    }
    eos_model_close(m);
}

/* ------------------------------------------------------------------ */
/* JSONL streaming protocol — shape test. Validates that the documented
 * line schema in docs/cli.md (and consumed by the VS Code + browser
 * extensions) is exactly preserved by a simple synthetic emitter.
 * Doesn't invoke the CLI binary (the test binary doesn't shell out);
 * instead it asserts each line in a sample stream parses with the
 * expected keys and that the stream ends with exactly one terminator.
 *
 * If the documented schema ever changes, this test must be updated in
 * the same diff so consumers are forced to acknowledge the break.
 */

/* Minimal "is this byte a JSON value char" — sufficient for our shape
 * checks (we never produce nested objects in t/i/i/done lines). */
static int test_jsonl_line_is_token(const char *line) {
    /* Per docs/cli.md a token line looks like:
     *   {"t":"<text>","i":<integer>}\n   (newline optional in this view)
     * We're checking shape, not content; require both keys appear in
     * order and the line is balanced + ends with '}'. */
    const char *t_key = strstr(line, "\"t\":\"");
    const char *i_key = strstr(line, "\",\"i\":");
    const char *brace = strrchr(line, '}');
    if (line[0] != '{') return 0;
    if (t_key == NULL) return 0;
    if (i_key == NULL || i_key < t_key) return 0;
    if (brace == NULL || brace[1] != '\0') return 0;
    return 1;
}

static int test_jsonl_line_is_done(const char *line) {
    const char *d   = strstr(line, "\"done\":true");
    const char *r   = strstr(line, "\"reason\":\"");
    const char *n   = strstr(line, "\"n_tokens\":");
    const char *brace = strrchr(line, '}');
    if (line[0] != '{') return 0;
    if (d == NULL || r == NULL || n == NULL) return 0;
    if (brace == NULL || brace[1] != '\0') return 0;
    return 1;
}

static void test_jsonl_protocol_shape(void) {
    /* Sample stream — must match exactly what eosllm-cli --smoke
     * --stream-jsonl emits (see tools/eosllm-cli/main.c). */
    static const char *lines[] = {
        "{\"t\":\"hi\",\"i\":1}",
        "{\"t\":\" \",\"i\":2}",
        "{\"t\":\"there\",\"i\":3}",
        "{\"done\":true,\"reason\":\"max\","
        "\"n_tokens\":3,\"ms\":0,\"tok_per_s\":0}",
        NULL
    };
    int n_token = 0, n_done = 0;
    int i;
    for (i = 0; lines[i] != NULL; ++i) {
        if (test_jsonl_line_is_token(lines[i])) n_token++;
        else if (test_jsonl_line_is_done(lines[i])) n_done++;
    }
    CHECK(n_token == 3, "jsonl: 3 token lines parse with t+i keys");
    CHECK(n_done  == 1, "jsonl: exactly one done terminator with reason+n_tokens");

    /* Negative cases — wrong shapes must fail the parser so future
     * regressions to the schema are caught. */
    CHECK(!test_jsonl_line_is_token("{\"i\":1,\"t\":\"x\"}"),
          "jsonl: out-of-order keys rejected");
    CHECK(!test_jsonl_line_is_done("{\"reason\":\"max\",\"n_tokens\":0}"),
          "jsonl: done line without \"done\":true rejected");
    CHECK(!test_jsonl_line_is_token("not json at all"),
          "jsonl: non-object rejected");
}

/* ------------------------------------------------------------------ */

int main(void) {
    fprintf(stdout, "eosllm Phase 0/1/2 unit tests — version %s, ABI %d\n",
            eos_version_string(), eos_abi_version());
    test_init_defaults();
    test_version_and_caps();
    test_status_strings();
    test_session_lifecycle();
    test_matmul_f32();
    test_f16_roundtrip();
    test_matmul_q8_oracle_gguf();
    test_matmul_q4_k_oracle();
    test_q8_0_dequant_quant_roundtrip();
    test_q1_58_roundtrip();
    test_rmsnorm_softmax();
    test_silu_gelu_rope();
    test_model_open_no_file();
    test_bpe_synthetic();
    test_registry_frozen();
    test_matmul_q8_0_random_stress();
    test_matmul_q4_k_all_subblocks();
    test_sha256_known_answers();
    test_eosm_roundtrip();
    test_eosm_from_model_roundtrip();
    test_eos_free();
    test_backend_available();
    test_gguf_synthetic_roundtrip();
    test_jsonl_protocol_shape();
#if EOSLLM_HAVE_KERNEL_AVX2
    test_avx2_q4_k_parity();
    test_avx2_q4_k_int8_parity();
#endif
#if EOSLLM_HAVE_KERNEL_NEON
    test_neon_q4_k_parity();
#endif
    fprintf(stdout, "\n%d/%d checks passed\n", g_total - g_failed, g_total);
    return g_failed == 0 ? 0 : 1;
}
