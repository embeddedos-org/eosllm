/*
 * src/kernels/scalar/silu.c
 *
 * y = x * sigmoid(x)
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "scalar.h"

eos_status_t eosi_scalar_silu(const float *x, float *y, size_t n) {
    size_t i;

    if (x == NULL || y == NULL) return EOS_E_INVALID_ARG;

    for (i = 0; i < n; ++i) {
        const float xi = x[i];
        const float sig = 1.0f / (1.0f + expf(-xi));
        y[i] = xi * sig;
    }
    return EOS_OK;
}
