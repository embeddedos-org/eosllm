/*
 * src/kernels/scalar/matmul_f32.c
 *
 * C = A · B^T, all f32, row-major.
 *   A is m×k, B is n×k, C is m×n.
 *
 * The triple-loop reference. No blocking, no SIMD, no threading.
 */
#include <stddef.h>
#include <stdint.h>

#include "scalar.h"

eos_status_t eosi_scalar_matmul_f32(const float *a, const float *b, float *c,
                                    uint32_t m, uint32_t n, uint32_t k) {
    uint32_t i, j, p;

    if (a == NULL || b == NULL || c == NULL) return EOS_E_INVALID_ARG;

    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            float acc = 0.0f;
            const float *ar = a + (size_t)i * k;
            const float *br = b + (size_t)j * k;
            for (p = 0; p < k; ++p) {
                acc += ar[p] * br[p];
            }
            c[(size_t)i * n + j] = acc;
        }
    }
    return EOS_OK;
}
