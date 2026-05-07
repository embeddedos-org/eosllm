/*
 * tests/fuzz/fuzz_eosm.c — libFuzzer harness for the .eosm v1 reader.
 *
 * Build with clang:
 *   clang -fsanitize=fuzzer,address,undefined -g -O1 \
 *         -Iinclude -Isrc \
 *         tests/fuzz/fuzz_eosm.c libeosllm.a -lm \
 *         -o tests/fuzz/fuzz_eosm
 * Run with the seed corpus produced by `make fuzz-corpus`.
 *
 * The harness wraps the input bytes as an in-memory eos_stream_t and
 * feeds them to the registered .eosm format reader. Any crash, ASan,
 * or UBSan report is a libFuzzer find. Successful opens are also
 * exercised: we walk every tensor (touching its name + data pointer)
 * to flush any lazy bound checks, then close.
 *
 * The reader's mandatory SHA-256 trailer means that random inputs
 * almost never reach past the trailer-check; the seed corpus
 * (produced from a real writer output) is therefore essential to
 * drive coverage past the gate.
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

static const eos_format_vt_t *find_eosm_vt(void) {
    eosi_registry_t *r = eosi_registry();
    size_t i;
    for (i = 0; i < r->n_formats; ++i) {
        if (r->formats[i] && r->formats[i]->name
            && strcmp(r->formats[i]->name, "eosm") == 0) {
            return r->formats[i];
        }
    }
    return NULL;
}

int LLVMFuzzerInitialize(int *argc, char ***argv);
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

int LLVMFuzzerInitialize(int *argc, char ***argv) {
    (void)argc; (void)argv;
    /* Idempotent; safe to call once at process start. */
    return (eos_init_defaults() == EOS_OK) ? 0 : 1;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    const eos_format_vt_t *vt = find_eosm_vt();
    if (vt == NULL) return 0;

    {
        mem_t        m   = { data, size, 0 };
        eos_stream_t st  = { &m, r_read, r_seek, r_size, r_close };
        eos_model_t *mdl = NULL;
        eos_status_t rc;

        /* Probe (cheap; reads 4 bytes). */
        if (size >= 4) {
            (void)vt->probe(&st);
            (void)st.seek(&m, 0);
        }

        rc = vt->open(&st, &mdl);
        if (rc == EOS_OK && mdl != NULL) {
            /* Touch every tensor's name + first byte of data to flush
             * lazy out-of-bounds reads. */
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
