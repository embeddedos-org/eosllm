/*
 * tests/fuzz/seed_eosm_corpus.c — emit a tiny .eosm seed corpus into
 * tests/fuzz/corpus/ for the libFuzzer harness above.
 *
 * Run via `make fuzz-corpus`. Re-running is a no-op if the corpus
 * directory already contains seed files; otherwise it produces three
 * minimally-different valid .eosm buffers so libFuzzer has something
 * to mutate past the SHA-256 gate.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "eosllm/eosllm.h"

#include "format/eosm/eosm_internal.h"
#include "core/internal.h"

static int write_buffer(const char *path, const void *data, size_t len) {
    FILE *f = fopen(path, "wb");
    if (f == NULL) return -1;
    if (fwrite(data, 1, len, f) != len) { fclose(f); return -1; }
    fclose(f);
    return 0;
}

int main(int argc, char **argv) {
    const char *dir = (argc >= 2) ? argv[1] : "tests/fuzz/corpus";
    static const float weights[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    eosi_eosm_kv_in_t     kv;
    eosi_eosm_tensor_in_t t;
    eosi_eosm_group_in_t  g;
    eosi_eosm_writer_in_t in;
    void  *buf = NULL;
    size_t bytes = 0;
    eos_status_t rc;
    int seed;

    if (eos_init_defaults() != EOS_OK) {
        fprintf(stderr, "eos_init_defaults failed\n");
        return 1;
    }
    mkdir(dir, 0755);

    for (seed = 0; seed < 3; ++seed) {
        char path[256];
        memset(&kv, 0, sizeof(kv));
        kv.key        = "general.architecture";
        kv.key_len    = (uint64_t)strlen("general.architecture");
        kv.value_type = EOSI_EOSM_KV_STRING;
        kv.bytes      = (seed == 0) ? "test"
                      : (seed == 1) ? "tinyllama" : "qwen2";
        kv.bytes_len  = (uint64_t)strlen((const char *)kv.bytes);

        memset(&t, 0, sizeof(t));
        t.name                  = (seed == 0) ? "x.f32" : "y.f32";
        t.name_len              = (uint64_t)strlen(t.name);
        t.dtype                 = EOS_DT_F32;
        t.calibration_table_idx = 0xFFFFFFFFu;
        t.rank                  = 1;
        t.dims[0]               = (seed == 2) ? 2 : 4;
        t.data                  = weights;
        t.data_bytes            = (uint64_t)t.dims[0] * 4u;

        memset(&g, 0, sizeof(g));
        g.id = 0;
        g.name = "default";
        g.name_len = 7;

        memset(&in, 0, sizeof(in));
        in.capability_bits = EOSI_EOSM_CAP_HASH_SHA256_TRAILER
                           | EOSI_EOSM_CAP_MODALITY_TEXT;
        in.alignment = 64;
        in.kvs       = &kv; in.n_kvs     = 1;
        in.tensors   = &t;  in.n_tensors = 1;
        in.groups    = &g;  in.n_groups  = 1;

        rc = eosi_eosm_write(&in, &buf, &bytes);
        if (rc != EOS_OK) {
            fprintf(stderr, "writer failed for seed %d: %d\n", seed, rc);
            return 1;
        }
        snprintf(path, sizeof(path), "%s/seed_%02d.eosm", dir, seed);
        if (write_buffer(path, buf, bytes) != 0) {
            fprintf(stderr, "cannot write %s\n", path);
            return 1;
        }
        fprintf(stdout, "wrote %s (%zu bytes)\n", path, bytes);
        buf = NULL; bytes = 0;
    }
    return 0;
}
