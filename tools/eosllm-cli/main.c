/*
 * tools/eosllm-cli/main.c — minimal chat / generate CLI.
 *
 *   eosllm-cli --model <path.gguf> --prompt "Hello" --n 64
 *   eosllm-cli --smoke           # in-memory synthetic-GGUF lifecycle test
 *
 * Phase 1 features: load a Llama / Qwen2 GGUF, run greedy generation
 * for N tokens, print decoded text. The --smoke mode runs the full
 * model-open + session-open + close lifecycle against an in-memory
 * synthetic GGUF — useful as an integration smoke when no real model
 * file is at hand (CI nightly, devcontainers).
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eosllm/eosllm.h"
#include "eosllm/model.h"
#include "eosllm/os.h"

/* Internal — same convention as tools/eosllm-bench / tools/eosllm-convert. */
#include "../../src/core/internal.h"

static void usage(void) {
    fprintf(stderr,
        "Usage: eosllm-cli --model <path.gguf> --prompt \"<text>\" [--n N]\n"
        "                  [--scheduler greedy] [--ctx N]\n"
        "       eosllm-cli --smoke\n"
        "       eosllm-cli --smoke-bad-magic\n"
        "       eosllm-cli --last-error\n"
        "\n"
        "Phase 1 generates greedy text from a Llama-class GGUF.\n"
        "--smoke runs the full session lifecycle against an in-memory\n"
        "synthetic GGUF (no model file required).\n"
        "--last-error triggers a known-failing call and prints both\n"
        "eos_status_str() and eos_last_error() (sanity check for\n"
        "distro packagers that the TLS path linked correctly).\n");
}

/* ------------------------------------------------------------------ */
/* Synthetic GGUF builder (same byte-level layout as the unit-test     */
/* synthetic GGUF; duplicated here so the CLI is self-contained.)      */
/* ------------------------------------------------------------------ */

#define GGUF_MAGIC_LE  0x46554747u
#define GGUF_VER       3u
#define GGUFV_U32      4u
#define GGUFV_STRING   8u
#define GGML_F32       0u

static uint8_t  gg_buf[1024];
static size_t   gg_pos;

static int gg_put(const void *data, size_t n) {
    if (gg_pos + n > sizeof(gg_buf)) return -1;
    memcpy(gg_buf + gg_pos, data, n);
    gg_pos += n;
    return 0;
}
static int gg_u32(uint32_t v) { return gg_put(&v, 4); }
static int gg_u64(uint64_t v) { return gg_put(&v, 8); }
static int gg_lstr(const char *s) {
    uint64_t l = (uint64_t)strlen(s);
    if (gg_u64(l) != 0) return -1;
    return gg_put(s, (size_t)l);
}

/* ------------------------------------------------------------------ */
/* Memory-backed eos_stream_t                                          */
/* ------------------------------------------------------------------ */

typedef struct {
    const uint8_t *data;
    size_t         len;
    size_t         pos;
} mem_t;

static int64_t m_read(void *o, void *buf, size_t n) {
    mem_t *m = (mem_t *)o;
    size_t a = (m->pos < m->len) ? m->len - m->pos : 0;
    size_t t = (n < a) ? n : a;
    if (t > 0) memcpy(buf, m->data + m->pos, t);
    m->pos += t;
    return (int64_t)t;
}
static int     m_seek(void *o, int64_t off) {
    mem_t *m = (mem_t *)o;
    if (off < 0 || (size_t)off > m->len) return -1;
    m->pos = (size_t)off; return 0;
}
static int64_t m_size(void *o) { return (int64_t)((mem_t *)o)->len; }
static void    m_close(void *o) { (void)o; }

static const eos_format_vt_t *find_gguf_vt(void) {
    eosi_registry_t *r = eosi_registry();
    size_t i;
    for (i = 0; i < r->n_formats; ++i) {
        if (r->formats[i] && r->formats[i]->name
            && strcmp(r->formats[i]->name, "gguf") == 0) {
            return r->formats[i];
        }
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Smoke mode                                                          */
/* ------------------------------------------------------------------ */

static int run_smoke(void) {
    static const float weights[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    eos_status_t s;
    eos_model_t  *m  = NULL;
    eos_session_t *ss = NULL;
    eos_session_params_t params;
    const eos_format_vt_t *vt;
    int rc = 0;

    fprintf(stdout, "[smoke] eosllm %s, ABI %d\n",
            eos_version_string(), eos_abi_version());

    s = eos_init_defaults();
    fprintf(stdout, "[smoke] init_defaults: %s\n", eos_status_str(s));
    if (s != EOS_OK) return 1;

    /* Build a minimal valid GGUF v3 in memory. */
    gg_pos = 0;
    if (gg_u32(GGUF_MAGIC_LE) != 0
     || gg_u32(GGUF_VER)      != 0
     || gg_u64(1)             != 0     /* tensor_count */
     || gg_u64(2)             != 0     /* kv_count */
     || gg_lstr("general.architecture") != 0
     || gg_u32(GGUFV_STRING)             != 0
     || gg_lstr("smoke")                 != 0
     || gg_lstr("general.alignment")     != 0
     || gg_u32(GGUFV_U32)                != 0
     || gg_u32(32)                       != 0
     || gg_lstr("x.f32")                 != 0
     || gg_u32(1)                        != 0     /* n_dims */
     || gg_u64(4)                        != 0
     || gg_u32(GGML_F32)                 != 0
     || gg_u64(0)                        != 0) {  /* tensor offset */
        fprintf(stderr, "[smoke] gguf builder overflow\n");
        return 1;
    }
    while ((gg_pos & 31u) != 0u) {
        uint8_t z = 0;
        if (gg_put(&z, 1) != 0) { fprintf(stderr, "[smoke] pad overflow\n"); return 1; }
    }
    if (gg_put(weights, sizeof(weights)) != 0) {
        fprintf(stderr, "[smoke] data overflow\n"); return 1;
    }
    fprintf(stdout, "[smoke] synthetic GGUF: %zu bytes\n", gg_pos);

    vt = find_gguf_vt();
    if (vt == NULL) {
        fprintf(stderr, "[smoke] gguf format vt not registered\n");
        return 1;
    }

    /* Open the synthetic model via the in-memory stream. */
    {
        mem_t        mst = { gg_buf, gg_pos, 0 };
        eos_stream_t st  = { &mst, m_read, m_seek, m_size, m_close };
        s = vt->open(&st, &m);
    }
    fprintf(stdout, "[smoke] model_open(synthetic): %s%s\n",
            eos_status_str(s),
            (s == EOS_OK ? " (1 tensor, 2 metadata kvs)" : ""));
    if (s != EOS_OK || m == NULL) { rc = 1; goto cleanup_model; }
    fprintf(stdout, "[smoke]   num_tensors = %zu\n", eos_model_num_tensors(m));
    {
        const char *arch = eos_model_meta_str(m, "general.architecture");
        fprintf(stdout, "[smoke]   architecture = \"%s\"\n",
                arch ? arch : "<missing>");
    }

    /* Try to open a session against the synthetic model. Expected to
     * fail at BPE open (no tokenizer.ggml.tokens metadata) — a
     * controlled, documented error path. */
    memset(&params, 0, sizeof(params));
    params.max_context = 64;
    s = eos_session_open(m, &params, &ss);
    fprintf(stdout, "[smoke] session_open(synthetic): %s "
            "(error expected: no tokenizer metadata)\n",
            eos_status_str(s));
    if (s != EOS_OK) {
        const char *e = eos_last_error();
        if (e != NULL) fprintf(stdout, "[smoke]   last_error: %s\n", e);
    }
    if (s == EOS_OK && ss != NULL) {
        eos_session_close(ss);
        ss = NULL;
    }

cleanup_model:
    eos_model_close(m);
    fprintf(stdout, "[smoke] model_close: ok\n");

    /* Open + close a NULL-model session to demonstrate the lifecycle
     * is healthy independently of a loaded model. */
    s = eos_session_open(NULL, &params, &ss);
    fprintf(stdout, "[smoke] session_open(NULL): %s\n", eos_status_str(s));
    if (s == EOS_OK && ss != NULL) {
        /* feed + step are expected to return EOS_E_UNSUPPORTED with
         * a NULL model — exercise the negative paths. */
        s = eos_session_feed(ss, EOS_MODALITY_TEXT, "hi", 2);
        fprintf(stdout, "[smoke] session_feed(NULL model): %s "
                "(EOS_E_UNSUPPORTED expected)\n", eos_status_str(s));
        {
            uint32_t tok = 0;
            s = eos_session_step(ss, &tok);
            fprintf(stdout, "[smoke] session_step(NULL model): %s "
                    "(EOS_E_UNSUPPORTED expected)\n", eos_status_str(s));
        }
        eos_session_close(ss);
        fprintf(stdout, "[smoke] session_close: ok\n");
    } else {
        rc = 1;
    }

    fprintf(stdout, "[smoke] %s\n", rc == 0 ? "OK" : "FAILED");
    return rc;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt     = NULL;
    const char *sched      = "greedy";
    int         n_predict  = 64;
    int         max_ctx    = 2048;
    int         smoke      = 0;
    int         smoke_bad_magic = 0;
    int         show_last_error = 0;
    int         i;
    eos_status_t s;
    eos_model_t   *m  = NULL;
    eos_session_t *ss = NULL;
    eos_session_params_t params;

    for (i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--model")     && i + 1 < argc) model_path = argv[++i];
        else if (!strcmp(argv[i], "--prompt")    && i + 1 < argc) prompt     = argv[++i];
        else if (!strcmp(argv[i], "--n")         && i + 1 < argc) n_predict  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx")       && i + 1 < argc) max_ctx    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--scheduler") && i + 1 < argc) sched      = argv[++i];
        else if (!strcmp(argv[i], "--smoke"))                     smoke      = 1;
        else if (!strcmp(argv[i], "--smoke-bad-magic"))           smoke_bad_magic = 1;
        else if (!strcmp(argv[i], "--last-error"))                 show_last_error = 1;
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
            { usage(); return 0; }
        else { usage(); return 2; }
    }

    if (smoke) return run_smoke();

    if (smoke_bad_magic) {
        /* Demonstrate the GGUF reader's negative path through the CLI:
         * craft a 4-byte stream with magic "XXXX", feed it via the
         * registered format vt, surface status + last_error. */
        static const uint8_t bad_magic[4] = { 'X', 'X', 'X', 'X' };
        const eos_format_vt_t *vt;
        eos_status_t rc = eos_init_defaults();
        if (rc != EOS_OK) {
            fprintf(stderr, "init: %s\n", eos_status_str(rc)); return 1;
        }
        vt = find_gguf_vt();
        if (vt == NULL) {
            fprintf(stderr, "gguf vt not registered\n"); return 1;
        }
        {
            mem_t        mst = { bad_magic, sizeof(bad_magic), 0 };
            eos_stream_t st  = { &mst, m_read, m_seek, m_size, m_close };
            eos_model_t *bm  = NULL;
            const char  *e;
            rc = vt->probe(&st);
            fprintf(stdout, "probe(bad-magic): %s\n", eos_status_str(rc));
            (void)st.seek(&mst, 0);
            rc = vt->open(&st, &bm);
            e  = eos_last_error();
            fprintf(stdout, "open(bad-magic):  %s\n", eos_status_str(rc));
            fprintf(stdout, "last_error:       %s\n", e ? e : "(null)");
            if (bm != NULL) eos_model_close(bm);
            return (rc == EOS_OK) ? 1 : 0;
        }
    }

    if (show_last_error) {
        /* Sanity check for packagers: trigger a guaranteed-failing
         * call, then surface eos_last_error() to confirm the TLS
         * + EOSI_LOG_ERROR wiring is intact. */
        eos_model_t *dummy = NULL;
        const char *e;
        s = eos_init_defaults();
        if (s != EOS_OK) {
            fprintf(stderr, "init: %s\n", eos_status_str(s));
            return 1;
        }
        s = eos_model_open("/eosllm/no-such-file.gguf", &dummy);
        e = eos_last_error();
        fprintf(stdout, "status:     %s\n", eos_status_str(s));
        fprintf(stdout, "last_error: %s\n", e ? e : "(null)");
        return (s == EOS_OK || e == NULL) ? 1 : 0;
    }

    if (model_path == NULL || prompt == NULL) { usage(); return 2; }

    s = eos_init_defaults();
    if (s != EOS_OK) { fprintf(stderr, "init: %s\n", eos_status_str(s)); return 1; }

    s = eos_model_open(model_path, &m);
    if (s != EOS_OK) { fprintf(stderr, "open model: %s\n", eos_status_str(s)); return 1; }

    memset(&params, 0, sizeof(params));
    params.max_context = (uint32_t)max_ctx;
    params.scheduler   = sched;

    s = eos_session_open(m, &params, &ss);
    if (s != EOS_OK) {
        fprintf(stderr, "open session: %s\n", eos_status_str(s));
        eos_model_close(m); return 1;
    }

    s = eos_session_feed(ss, EOS_MODALITY_TEXT, prompt, strlen(prompt));
    if (s != EOS_OK) {
        fprintf(stderr, "feed: %s\n", eos_status_str(s));
        eos_session_close(ss); eos_model_close(m); return 1;
    }

    fputs(prompt, stdout);
    fflush(stdout);

    for (i = 0; i < n_predict; ++i) {
        uint32_t tok;
        char     buf[64];
        size_t   len = sizeof(buf);
        s = eos_session_step(ss, &tok);
        if (s != EOS_OK) {
            fprintf(stderr, "\nstep: %s\n", eos_status_str(s));
            break;
        }
        s = eos_session_decode(ss, &tok, 1, buf, &len);
        if (s == EOS_OK && len <= sizeof(buf)) {
            fwrite(buf, 1, len, stdout);
            fflush(stdout);
        }
    }
    fputc('\n', stdout);

    eos_session_close(ss);
    eos_model_close(m);
    return 0;
}
