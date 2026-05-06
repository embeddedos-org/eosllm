/*
 * tests/fuzz/seed_gguf_corpus.c — emit a tiny GGUF v3 seed corpus
 * into tests/fuzz/corpus_gguf/ for the libFuzzer harness.
 *
 * Each seed is a minimally-different valid v3 file built by the same
 * byte-emission helpers used by the unit-test synthetic builder. We
 * vary KV count, tensor dtype, dim count, and alignment so the fuzzer
 * has a few starting shapes to mutate.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define GGUF_MAGIC_LE  0x46554747u
#define GGUF_VER       3u
#define GGUFV_U32      4u
#define GGUFV_F32      6u
#define GGUFV_STRING   8u
#define GGML_F32       0u
#define GGML_Q8_0      8u

static uint8_t  buf[4096];
static size_t   pos;

static int put(const void *data, size_t n) {
    if (pos + n > sizeof(buf)) return -1;
    memcpy(buf + pos, data, n);
    pos += n;
    return 0;
}
static int put_u32(uint32_t v) { return put(&v, 4); }
static int put_u64(uint64_t v) { return put(&v, 8); }
static int put_f32(float v)    { return put(&v, 4); }
static int put_lstr(const char *s) {
    uint64_t l = (uint64_t)strlen(s);
    if (put_u64(l) != 0) return -1;
    return put(s, (size_t)l);
}
static int pad_to(uint32_t alignment) {
    while ((pos & (alignment - 1u)) != 0u) {
        uint8_t z = 0;
        if (put(&z, 1) != 0) return -1;
    }
    return 0;
}

static int write_buffer(const char *path, const void *data, size_t len) {
    FILE *f = fopen(path, "wb");
    if (f == NULL) return -1;
    if (fwrite(data, 1, len, f) != len) { fclose(f); return -1; }
    fclose(f);
    return 0;
}

/* ---------------------------------------------------------------- */
/* Seed builders — return 0 on success, -1 on overflow.             */
/* ---------------------------------------------------------------- */

static int build_seed_minimal_f32(uint32_t alignment) {
    static const float weights[4] = { 1, 2, 3, 4 };
    pos = 0;
    if (put_u32(GGUF_MAGIC_LE) != 0) return -1;
    if (put_u32(GGUF_VER)      != 0) return -1;
    if (put_u64(1)             != 0) return -1;   /* tensor_count */
    if (put_u64(2)             != 0) return -1;   /* kv_count */
    if (put_lstr("general.architecture")  != 0) return -1;
    if (put_u32(GGUFV_STRING)              != 0) return -1;
    if (put_lstr("test")                   != 0) return -1;
    if (put_lstr("general.alignment")      != 0) return -1;
    if (put_u32(GGUFV_U32)                 != 0) return -1;
    if (put_u32(alignment)                 != 0) return -1;
    /* tensor */
    if (put_lstr("x.f32")                  != 0) return -1;
    if (put_u32(1)                         != 0) return -1;     /* n_dims */
    if (put_u64(4)                         != 0) return -1;
    if (put_u32(GGML_F32)                  != 0) return -1;
    if (put_u64(0)                         != 0) return -1;     /* offset */
    if (pad_to(alignment)                  != 0) return -1;
    return put(weights, sizeof(weights));
}

static int build_seed_two_kv_f32_2d(void) {
    static const float weights[6] = { 1, 2, 3, 4, 5, 6 };
    pos = 0;
    if (put_u32(GGUF_MAGIC_LE) != 0) return -1;
    if (put_u32(GGUF_VER)      != 0) return -1;
    if (put_u64(1)             != 0) return -1;
    if (put_u64(3)             != 0) return -1;
    if (put_lstr("general.architecture") != 0) return -1;
    if (put_u32(GGUFV_STRING) != 0) return -1;
    if (put_lstr("synthetic") != 0) return -1;
    if (put_lstr("general.alignment") != 0) return -1;
    if (put_u32(GGUFV_U32) != 0) return -1;
    if (put_u32(32) != 0) return -1;
    if (put_lstr("test.f32") != 0) return -1;
    if (put_u32(GGUFV_F32) != 0) return -1;
    if (put_f32(0.125f) != 0) return -1;
    if (put_lstr("y.f32") != 0) return -1;
    if (put_u32(2) != 0) return -1;
    if (put_u64(2) != 0) return -1;
    if (put_u64(3) != 0) return -1;
    if (put_u32(GGML_F32) != 0) return -1;
    if (put_u64(0) != 0) return -1;
    if (pad_to(32) != 0) return -1;
    return put(weights, sizeof(weights));
}

/* A Q8_0 tensor: 32 elements per block, 34 bytes per block. One block. */
static int build_seed_q8_0(void) {
    uint8_t blk[34];
    int i;
    pos = 0;
    if (put_u32(GGUF_MAGIC_LE) != 0) return -1;
    if (put_u32(GGUF_VER)      != 0) return -1;
    if (put_u64(1)             != 0) return -1;
    if (put_u64(1)             != 0) return -1;
    if (put_lstr("general.alignment") != 0) return -1;
    if (put_u32(GGUFV_U32) != 0) return -1;
    if (put_u32(32) != 0) return -1;
    if (put_lstr("z.q8_0") != 0) return -1;
    if (put_u32(1) != 0) return -1;
    if (put_u64(32) != 0) return -1;
    if (put_u32(GGML_Q8_0) != 0) return -1;
    if (put_u64(0) != 0) return -1;
    if (pad_to(32) != 0) return -1;
    /* One Q8_0 block: f16 scale + 32 int8. Use scale=1.0f≈f16 0x3C00. */
    blk[0] = 0x00; blk[1] = 0x3C;
    for (i = 0; i < 32; ++i) blk[2 + i] = (uint8_t)(int8_t)(i - 16);
    return put(blk, sizeof(blk));
}

int main(int argc, char **argv) {
    const char *dir = (argc >= 2) ? argv[1] : "tests/fuzz/corpus_gguf";
    char path[256];

    mkdir(dir, 0755);

    if (build_seed_minimal_f32(32) != 0) {
        fprintf(stderr, "seed minimal_f32 build failed\n"); return 1;
    }
    snprintf(path, sizeof(path), "%s/seed_00.gguf", dir);
    if (write_buffer(path, buf, pos) != 0) { fprintf(stderr, "write fail\n"); return 1; }
    fprintf(stdout, "wrote %s (%zu bytes)\n", path, pos);

    if (build_seed_two_kv_f32_2d() != 0) {
        fprintf(stderr, "seed two_kv_f32_2d build failed\n"); return 1;
    }
    snprintf(path, sizeof(path), "%s/seed_01.gguf", dir);
    if (write_buffer(path, buf, pos) != 0) { fprintf(stderr, "write fail\n"); return 1; }
    fprintf(stdout, "wrote %s (%zu bytes)\n", path, pos);

    if (build_seed_q8_0() != 0) {
        fprintf(stderr, "seed q8_0 build failed\n"); return 1;
    }
    snprintf(path, sizeof(path), "%s/seed_02.gguf", dir);
    if (write_buffer(path, buf, pos) != 0) { fprintf(stderr, "write fail\n"); return 1; }
    fprintf(stdout, "wrote %s (%zu bytes)\n", path, pos);

    return 0;
}
