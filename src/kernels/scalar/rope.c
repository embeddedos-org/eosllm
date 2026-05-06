/*
 * src/kernels/scalar/rope.c
 *
 * Rotary positional embedding (Llama / Qwen flavor): each pair
 * (x[2i], x[2i+1]) is rotated by angle pos * base^(-2i/head_dim).
 *
 * Operates in place on a tensor of shape [n_heads, head_dim].
 * head_dim must be even.
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "scalar.h"

eos_status_t eosi_scalar_rope(float *x, uint32_t n_heads, uint32_t head_dim,
                              uint32_t pos, float base) {
    uint32_t h, i;

    if (x == NULL) return EOS_E_INVALID_ARG;
    if ((head_dim & 1u) != 0) return EOS_E_INVALID_ARG;
    if (head_dim == 0) return EOS_E_INVALID_ARG;

    for (h = 0; h < n_heads; ++h) {
        float *xh = x + (size_t)h * head_dim;
        for (i = 0; i + 1 < head_dim; i += 2) {
            const float exponent = -((float)i) / (float)head_dim;
            const float theta    = (float)pos * powf(base, exponent);
            const float c        = cosf(theta);
            const float s        = sinf(theta);
            const float a        = xh[i];
            const float b        = xh[i + 1];
            xh[i]     = a * c - b * s;
            xh[i + 1] = a * s + b * c;
        }
    }
    return EOS_OK;
}
