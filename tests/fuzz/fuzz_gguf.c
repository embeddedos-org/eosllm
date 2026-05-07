/*
 * tests/fuzz/fuzz_gguf.c — libFuzzer harness for the GGUF v3 reader.
 *
 * Build with clang:
 *   make fuzz-gguf
 * Run with the seed corpus:
 *   ./tests/fuzz/fuzz_gguf tests/fuzz/corpus_gguf -max_total_time=30
 *
 * The harness mirrors fuzz_eosm.c: wrap input as an in-memory
 * eos_stream_t, dispatch via the registered "gguf" format vt's probe
 * + open, walk every tensor's name / first data byte, close. Any
 * crash, ASan/UBSan/LeakSan report is a libFuzzer find.
 *
 * Unlike .eosm, GGUF has no SHA-256 trailer to gate parsing, so
 * libFuzzer mutations reach the parser body easily. The seed corpus
 * provides positive-shape seeds so the fuzzer mutates valid layouts
 * rather than purely random noise.
 */
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "eosllm/eosllm.h"
#include "eosllm/model.h"

#include "core/internal.h"
#include "core/model_internal.h"  /* concrete eos_tensor layout for the data-touch */

typedef struct {
    const uint8_t *data;
    size_t         len;
    size_t         pos;
} mem_t;

static int64_t r_read(void *o, void *buf, size_t len) {
    mem_t *m = (mem_t *)o;
    size_t avail = (m->pos < m->len) ? m->len - m->pos : 0;
    size_t take  = (len < avail) ? len : avail;
    if (take > 0) memcpy(buf, m->data + m->pos, take);
    m->pos += take;
    return (int64_t)take;
}
static int     r_seek (void *o, int64_t off) {
    mem_t *m = (mem_t *)o;
    if (off < 0 || (size_t)off > m->len) return -1;
    m->pos = (size_t)off; return 0;
}
static int64_t r_size (void *o)               { return (int64_t)((mem_t *)o)->len; }
static void    r_close(void *o)               { (void)o; }

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

int LLVMFuzzerInitialize(int *argc, char ***argv);
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

int LLVMFuzzerInitialize(int *argc, char ***argv) {
    (void)argc; (void)argv;
    return (eos_init_defaults() == EOS_OK) ? 0 : 1;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    const eos_format_vt_t *vt = find_gguf_vt();
    if (vt == NULL) return 0;

    {
        mem_t        m   = { data, size, 0 };
        eos_stream_t st  = { &m, r_read, r_seek, r_size, r_close };
        eos_model_t *mdl = NULL;
        eos_status_t rc;

        if (size >= 4) {
            (void)vt->probe(&st);
            (void)st.seek(&m, 0);
        }

        rc = vt->open(&st, &mdl);
        if (rc == EOS_OK && mdl != NULL) {
            size_t n = eos_model_num_tensors(mdl);
            size_t i;
            for (i = 0; i < n; ++i) {
                const eos_tensor_t *t = eos_model_tensor(mdl, i);
                if (t == NULL) continue;
                if (t->name != NULL) (void)strlen(t->name);
                if (t->data != NULL && t->data_bytes > 0) {
                    volatile uint8_t b = ((const uint8_t *)t->data)[0];
                    (void)b;
                }
            }
            eos_model_close(mdl);
        }
    }
    return 0;
}
