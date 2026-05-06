/*
 * src/util/f16.h — IEEE-754 binary16 ↔ binary32 conversion.
 *
 * Pure C99, no compiler intrinsics. Exact for finite values.
 * Subnormals and infinities/NaNs are handled.
 */
#ifndef EOSI_UTIL_F16_H
#define EOSI_UTIL_F16_H

#include <stdint.h>
#include <string.h>

/* Half-precision is represented as a uint16_t throughout the engine.
 * Storage is little-endian on disk (GGUF / .eosm). */
typedef uint16_t eosi_f16_t;

static inline float eosi_f16_to_f32(eosi_f16_t h) {
    /* Approach: branchless reconstruction of the 32-bit float bits.
     * Adapted from the public-domain "ieee_half" tables; rewritten in
     * arithmetic form for portability. */
    const uint32_t sign = ((uint32_t)(h & 0x8000u)) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                         /* ±0 */
        } else {
            /* subnormal — normalize */
            uint32_t e = 1u, m = mant;
            while ((m & 0x400u) == 0) { m <<= 1; e++; }
            m &= 0x3FFu;
            bits = sign | ((127u - 15u - e + 1u) << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13); /* inf / NaN */
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
    {
        float f;
        memcpy(&f, &bits, sizeof(f));
        return f;
    }
}

static inline eosi_f16_t eosi_f32_to_f16(float f) {
    uint32_t bits;
    uint32_t sign, exp, mant;
    memcpy(&bits, &f, sizeof(bits));
    sign = (bits >> 16) & 0x8000u;
    exp  = (bits >> 23) & 0xFFu;
    mant = bits & 0x7FFFFFu;

    if (exp == 0xFFu) {
        return (eosi_f16_t)(sign | 0x7C00u | (mant ? (mant >> 13) | 0x200u : 0u));
    }
    if (exp == 0) return (eosi_f16_t)sign;            /* ±0 / subnormal-flush */

    /* unbiased exponent */
    {
        int32_t e = (int32_t)exp - 127 + 15;
        if (e >= 0x1F) return (eosi_f16_t)(sign | 0x7C00u);    /* overflow → inf */
        if (e <= 0) {
            /* subnormal */
            if (e < -10) return (eosi_f16_t)sign;
            mant |= 0x800000u;
            {
                uint32_t shift = (uint32_t)(14 - e);
                uint32_t round = (mant >> (shift - 1)) & 1u;
                return (eosi_f16_t)(sign | (uint16_t)((mant >> shift) + round));
            }
        }
        {
            uint32_t round = (mant >> 12) & 1u;
            uint32_t out   = (uint32_t)(e << 10) | (mant >> 13);
            return (eosi_f16_t)(sign | (uint16_t)(out + round));
        }
    }
}

#endif /* EOSI_UTIL_F16_H */
