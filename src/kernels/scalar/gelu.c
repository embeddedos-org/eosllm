/*
 * src/kernels/scalar/gelu.c
 *
 * y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * (Tanh approximation, the same one used by Llama / Qwen.)
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "scalar.h"

eos_status_t eosi_scalar_gelu(const float *x, float *y, size_t n) {
    /* sqrt(2 / pi) */
    static const float k_a = 0.7978845608028654f;
    static const float k_b = 0.044715f;
    size_t i;

    if (x == NULL || y == NULL) return EOS_E_INVALID_ARG;

    for (i = 0; i < n; ++i) {
        const float xi = x[i];
        const float t  = k_a * (xi + k_b * xi * xi * xi);
        y[i] = 0.5f * xi * (1.0f + tanhf(t));
    }
    return EOS_OK;
}
