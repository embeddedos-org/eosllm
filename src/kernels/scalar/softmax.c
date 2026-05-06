/*
 * src/kernels/scalar/softmax.c
 *
 * In-place row-wise softmax with a numerically stable max-subtract.
 * If mask_to_col is non-NULL, columns >= mask_to_col[row] are forced
 * to -inf before the exp (i.e. they contribute zero to the sum and
 * end up as zero in the output).
 */
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "scalar.h"

eos_status_t eosi_scalar_softmax(float *x, uint32_t rows, uint32_t cols,
                                 const int32_t *mask_to_col) {
    uint32_t i, j;

    if (x == NULL) return EOS_E_INVALID_ARG;
    if (cols == 0) return EOS_E_INVALID_ARG;

    for (i = 0; i < rows; ++i) {
        float *xr = x + (size_t)i * cols;
        const int32_t mask = (mask_to_col != NULL)
                             ? mask_to_col[i]
                             : (int32_t)cols;
        float maxv = -FLT_MAX;
        float sum  = 0.0f;
        int32_t valid = (mask < (int32_t)cols) ? mask : (int32_t)cols;
        if (valid <= 0) {
            /* Degenerate row: all masked. Zero it out. */
            for (j = 0; j < cols; ++j) xr[j] = 0.0f;
            continue;
        }
        for (j = 0; j < (uint32_t)valid; ++j) {
            if (xr[j] > maxv) maxv = xr[j];
        }
        for (j = 0; j < (uint32_t)valid; ++j) {
            xr[j] = expf(xr[j] - maxv);
            sum += xr[j];
        }
        if (sum > 0.0f) {
            const float inv = 1.0f / sum;
            for (j = 0; j < (uint32_t)valid; ++j) xr[j] *= inv;
        }
        for (j = (uint32_t)valid; j < cols; ++j) xr[j] = 0.0f;
    }
    return EOS_OK;
}
