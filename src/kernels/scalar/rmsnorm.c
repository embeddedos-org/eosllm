/*
 * src/kernels/scalar/rmsnorm.c
 *
 * y[i,j] = x[i,j] * rsqrt(mean_j(x[i,j]^2) + eps) * w[j]
 *
 * w may be NULL ⇒ the per-feature scale is treated as 1.
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "scalar.h"

eos_status_t eosi_scalar_rmsnorm(const float *x, const float *w, float eps,
                                 float *y, uint32_t rows, uint32_t cols) {
    uint32_t i, j;

    if (x == NULL || y == NULL) return EOS_E_INVALID_ARG;
    if (cols == 0) return EOS_E_INVALID_ARG;

    for (i = 0; i < rows; ++i) {
        const float *xr = x + (size_t)i * cols;
        float       *yr = y + (size_t)i * cols;
        float mean_sq = 0.0f;
        float scale;
        for (j = 0; j < cols; ++j) {
            mean_sq += xr[j] * xr[j];
        }
        mean_sq /= (float)cols;
        scale = 1.0f / sqrtf(mean_sq + eps);
        for (j = 0; j < cols; ++j) {
            const float wj = (w != NULL) ? w[j] : 1.0f;
            yr[j] = xr[j] * scale * wj;
        }
    }
    return EOS_OK;
}
