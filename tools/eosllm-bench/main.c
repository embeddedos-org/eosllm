/*
 * tools/eosllm-bench/main.c — matmul microbenchmark.
 *
 * Walks the registered backends, runs matmul_f32 / matmul_q8_0 /
 * matmul_q4_k at a handful of fixed shapes, times each via the
 * engine's OS shim now-ns, and emits JSON to stdout. Useful for:
 *   - confirming AVX2 / NEON / future kernels actually beat scalar;
 *   - regression-detecting kernel rewrites (e.g. before / after a
 *     SIMD refactor);
 *   - capacity planning even without a real LLM model file in place.
 *
 * Each run quantizes a fresh random weight matrix per-shape so the
 * cache state is realistic-ish. The reported "ops" count is the
 * canonical 2*M*N*K (count of fused-multiply-adds × 2 for the two
 * scalar ops per FMA). Higher is better.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "eosllm/eosllm.h"
#include "eosllm/backend.h"
#include "eosllm/quant.h"

/* Internal headers: bench is a project-internal tool, so it's allowed
 * to reach into eosi_* (same convention as eosllm-convert). */
#include "../../src/core/internal.h"

/* ---------- tiny deterministic PRNG ---------- */
static uint32_t g_xs = 0xCAFEBABEu;
static float    rand_unit(void) {
    uint32_t x = g_xs; x ^= x << 13; x ^= x >> 17; x ^= x << 5; g_xs = x;
    return ((float)(x & 0xFFFFFFu) / (float)0x800000) - 1.0f;
}

/* ---------- bench shapes ---------- */
typedef struct { uint32_t m, n, k; } shape_t;
static const shape_t k_shapes[] = {
    /* small / cache-resident */
    {  1,   1,    256 },
    {  1,   8,    512 },
    {  1,  64,   1024 },
    /* mid / approximate Llama-1B head sizes */
    {  1, 128,   2048 },
    {  1, 512,   2048 },
    /* one bigger one to flush L2 */
    {  1, 256,   4096 },
};
#define N_SHAPES (sizeof(k_shapes) / sizeof(k_shapes[0]))

static int n_iters_for(uint32_t m, uint32_t n, uint32_t k) {
    /* Aim for ~50ms of work per measurement: ops/iter ≈ 2*m*n*k. */
    double  ops = 2.0 * (double)m * (double)n * (double)k;
    double  target = 5e8;       /* ~5e8 ops ≈ 50ms on modern x86 */
    int n_it = (int)(target / (ops > 0 ? ops : 1));
    if (n_it < 1)   n_it = 1;
    if (n_it > 10000) n_it = 10000;
    return n_it;
}

/* ---------- emit one JSON record ---------- */
static int g_first_record = 1;
static void emit(const char *backend, const char *op,
                 uint32_t m, uint32_t n, uint32_t k,
                 int n_iters, uint64_t total_ns) {
    double ops_per_iter = 2.0 * (double)m * (double)n * (double)k;
    double total_ops    = ops_per_iter * (double)n_iters;
    double per_iter_ns  = (double)total_ns / (double)n_iters;
    double gops         = (total_ops / ((double)total_ns > 0 ? (double)total_ns : 1.0)); /* ops/ns = giga-ops/s */
    if (!g_first_record) printf(",\n");
    g_first_record = 0;
    printf("    {\"backend\":\"%s\",\"op\":\"%s\","
           "\"m\":%u,\"n\":%u,\"k\":%u,"
           "\"n_iters\":%d,\"total_ns\":%llu,"
           "\"ns_per_iter\":%.1f,\"giga_ops_per_s\":%.2f}",
           backend, op, m, n, k, n_iters,
           (unsigned long long)total_ns, per_iter_ns, gops);
}

/* ---------- the actual ops, looked up per-backend ---------- */
static const eos_backend_vt_t *find_backend(const char *name) {
    eosi_registry_t *r = eosi_registry();
    size_t i;
    for (i = 0; i < r->n_backends; ++i) {
        if (r->backends[i] && r->backends[i]->name
            && strcmp(r->backends[i]->name, name) == 0) {
            return r->backends[i];
        }
    }
    return NULL;
}

static void bench_matmul_f32(const char *backend_name) {
    const eos_backend_vt_t *vt = find_backend(backend_name);
    size_t s_idx;
    if (vt == NULL || vt->ops.matmul_f32 == NULL) return;
    if (vt->probe != NULL && vt->probe() != EOS_OK) return;

    for (s_idx = 0; s_idx < N_SHAPES; ++s_idx) {
        shape_t sh = k_shapes[s_idx];
        size_t a_n = (size_t)sh.m * sh.k;
        size_t b_n = (size_t)sh.n * sh.k;
        size_t c_n = (size_t)sh.m * sh.n;
        float *a = (float *)malloc(a_n * sizeof(float));
        float *b = (float *)malloc(b_n * sizeof(float));
        float *c = (float *)malloc(c_n * sizeof(float));
        int n_it = n_iters_for(sh.m, sh.n, sh.k);
        uint64_t t0, t1;
        int it;
        size_t i;
        if (!a || !b || !c) { free(a); free(b); free(c); continue; }
        for (i = 0; i < a_n; ++i) a[i] = rand_unit();
        for (i = 0; i < b_n; ++i) b[i] = rand_unit();
        /* warm-up */
        vt->ops.matmul_f32(a, b, c, sh.m, sh.n, sh.k);
        t0 = eosi_now_ns();
        for (it = 0; it < n_it; ++it) {
            vt->ops.matmul_f32(a, b, c, sh.m, sh.n, sh.k);
        }
        t1 = eosi_now_ns();
        emit(backend_name, "matmul_f32", sh.m, sh.n, sh.k,
             n_it, t1 - t0);
        free(a); free(b); free(c);
    }
}

static void bench_matmul_q(const char *backend_name,
                           const char *quant_name, eos_dtype_t dt,
                           uint32_t k_align) {
    const eos_backend_vt_t *vt  = find_backend(backend_name);
    const eos_quant_vt_t   *qvt = eos_quant_find(quant_name);
    size_t s_idx;
    if (vt == NULL || vt->ops.matmul_q == NULL) return;
    if (qvt == NULL) return;
    if (vt->probe != NULL && vt->probe() != EOS_OK) return;

    for (s_idx = 0; s_idx < N_SHAPES; ++s_idx) {
        shape_t sh = k_shapes[s_idx];
        eos_quant_layout_t layout;
        size_t a_n, b_blocks, b_bytes, c_n;
        float  *a = NULL, *src = NULL, *c = NULL;
        void   *b_q = NULL;
        int     n_it;
        uint64_t t0, t1;
        int it;
        size_t i;
        eos_status_t s;

        if ((sh.k % k_align) != 0) continue;     /* not divisible by block size */

        qvt->layout(&layout);
        a_n      = (size_t)sh.m * sh.k;
        b_blocks = (size_t)sh.n * sh.k / layout.elements_per_block;
        b_bytes  = b_blocks * layout.bytes_per_block;
        c_n      = (size_t)sh.m * sh.n;

        a   = (float *)malloc(a_n * sizeof(float));
        c   = (float *)malloc(c_n * sizeof(float));
        b_q = malloc(b_bytes);
        if (!a || !c || !b_q) goto cleanup;
        for (i = 0; i < a_n; ++i) a[i] = rand_unit();

        /* Populate b_q. If the scheme has a working quantize() we go
         * through it (most realistic). Otherwise (q4_k, today) fill
         * the bytes with deterministic noise — the kernel does the
         * same number of FMAs regardless of the actual values, so
         * timing is meaningful even if the math is meaningless. */
        if (qvt->quantize != NULL) {
            src = (float *)malloc((size_t)sh.n * sh.k * sizeof(float));
            if (!src) goto cleanup;
            for (i = 0; i < (size_t)sh.n * sh.k; ++i) src[i] = rand_unit() * 0.1f;
            s = qvt->quantize(src, b_q, (size_t)sh.n * sh.k);
            if (s != EOS_OK) {
                /* Quantize unsupported \u2014 fall through to noise fill. */
                uint8_t *p = (uint8_t *)b_q;
                for (i = 0; i < b_bytes; ++i) {
                    uint32_t r = (uint32_t)(rand_unit() * 127.0f + 128.0f);
                    p[i] = (uint8_t)(r & 0xFFu);
                }
            }
        } else {
            uint8_t *p = (uint8_t *)b_q;
            for (i = 0; i < b_bytes; ++i) {
                uint32_t r = (uint32_t)(rand_unit() * 127.0f + 128.0f);
                p[i] = (uint8_t)(r & 0xFFu);
            }
        }

        n_it = n_iters_for(sh.m, sh.n, sh.k);
        s = vt->ops.matmul_q(a, b_q, dt, c, sh.m, sh.n, sh.k);
        if (s != EOS_OK) goto cleanup;
        t0 = eosi_now_ns();
        for (it = 0; it < n_it; ++it) {
            vt->ops.matmul_q(a, b_q, dt, c, sh.m, sh.n, sh.k);
        }
        t1 = eosi_now_ns();
        {
            char op[32];
            snprintf(op, sizeof(op), "matmul_%s", quant_name);
            emit(backend_name, op, sh.m, sh.n, sh.k, n_it, t1 - t0);
        }
    cleanup:
        free(a); free(src); free(c); free(b_q);
    }
}

/* ---------- main ---------- */

static void usage(FILE *to) {
    fprintf(to,
        "Usage: eosllm-bench [--shapes-only]\n"
        "\n"
        "Runs a matmul microbenchmark across every registered backend\n"
        "whose runtime probe passes (scalar always; x86_avx2 / arm_neon\n"
        "if compiled in and the host CPU supports them). Emits a JSON\n"
        "object on stdout; one record per (backend, op, shape).\n");
}

int main(int argc, char **argv) {
    eos_status_t rc;
    int shapes_only = 0;
    int i;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--shapes-only") == 0) shapes_only = 1;
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(stdout); return 0;
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            usage(stderr); return 1;
        }
    }

    rc = eos_init_defaults();
    if (rc != EOS_OK) {
        fprintf(stderr, "eos_init_defaults failed: %s\n", eos_status_str(rc));
        return 1;
    }

    if (shapes_only) {
        size_t s;
        printf("{\n  \"shapes\": [\n");
        for (s = 0; s < N_SHAPES; ++s) {
            printf("    {\"m\":%u,\"n\":%u,\"k\":%u}%s\n",
                   k_shapes[s].m, k_shapes[s].n, k_shapes[s].k,
                   s + 1 < N_SHAPES ? "," : "");
        }
        printf("  ]\n}\n");
        return 0;
    }

    printf("{\n");
    printf("  \"library_version\": \"%s\",\n", eos_version_string());
    printf("  \"abi_version\": %d,\n", eos_abi_version());
    printf("  \"backends_probed\": {\n");
    printf("    \"scalar\":     %s,\n",
           eos_backend_available("scalar")    ? "true" : "false");
    printf("    \"x86_avx2\":   %s,\n",
           eos_backend_available("x86_avx2")  ? "true" : "false");
    printf("    \"x86_avx512\": %s,\n",
           eos_backend_available("x86_avx512")? "true" : "false");
    printf("    \"arm_neon\":   %s\n",
           eos_backend_available("arm_neon")  ? "true" : "false");
    printf("  },\n");

    printf("  \"records\": [\n");
    /* matmul_f32 across every available backend */
    bench_matmul_f32("scalar");
    bench_matmul_f32("x86_avx2");
    bench_matmul_f32("arm_neon");
    /* matmul_q across same */
    bench_matmul_q("scalar",   "q8_0", EOS_DT_Q8_0, 32);
    bench_matmul_q("scalar",   "q4_k", EOS_DT_Q4_K, 256);
    bench_matmul_q("x86_avx2", "q4_k", EOS_DT_Q4_K, 256);
    bench_matmul_q("arm_neon", "q4_k", EOS_DT_Q4_K, 256);
    printf("\n  ]\n}\n");
    return 0;
}
