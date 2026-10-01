// cpu_avx2.cpp - the AVX2 expert kernels, for CPUs without AVX-512 VNNI (compiled with AVX2/FMA flags; chosen at run
// time by cpu_moe.cpp).
//
// The same arithmetic as cpu_avx512.cpp with VPMADDUBSW + VPMADDWD in place of VPDPBUSD: the unsigned weight quants
// times the signed int8 activations, summed in pairs to int16 (at most 2 * 63 * 127 = 16002 for Q6_K, so nothing
// saturates), then multiplied by the sub-block scale and summed to int32.  The activations are read in natural
// order (Q8K::nat).  Integer sums are exact, so the results match the AVX-512 kernels up to float summation order.
#include "cpu_kernels.hpp"

#include <immintrin.h>

#include <cstring>

namespace sq::avx2 {

void quantize_q8k(const float* x, Q8K* y, int n) {
    for (int i = 0; i < n / 256; ++i) {
        const float* xb = x + i * 256;
        Q8K& b = y[i];
        const __m256 sign = _mm256_set1_ps(-0.0f);
        __m256 vmax = _mm256_setzero_ps();
        for (int k = 0; k < 32; ++k) vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign, _mm256_loadu_ps(xb + 8 * k)));
        __m128 m = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
        m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        m = _mm_max_ss(m, _mm_movehdup_ps(m));
        const float amax = _mm_cvtss_f32(m);
        if (amax == 0.0f) {
            std::memset(&b, 0, sizeof(Q8K));
            continue;
        }
        // the AVX-512 kernel's rounding: x * (127 / amax), round to nearest even
        const __m256 id = _mm256_set1_ps(127.0f / amax);
        alignas(32) int32_t q[256];
        for (int k = 0; k < 32; ++k)
            _mm256_store_si256((__m256i*)(q + 8 * k), _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xb + 8 * k), id)));
        for (int k = 0; k < 16; ++k) {
            int s = 0;
            for (int j = 0; j < 16; ++j) {
                const int v = q[16 * k + j] > 127 ? 127 : q[16 * k + j] < -128 ? -128 : q[16 * k + j];
                b.nat[16 * k + j] = (int8_t)v;
                s += v;
            }
            b.bsums[k] = (int16_t)s;
        }
        for (int h = 0; h < 2; ++h) {
            const int8_t* s = b.nat + 128 * h;
            int8_t* d = b.pair + 128 * h;
            std::memcpy(d + 0, s + 0, 32);
            std::memcpy(d + 32, s + 64, 32);
            std::memcpy(d + 64, s + 32, 32);
            std::memcpy(d + 96, s + 96, 32);
        }
        b.d = amax / 127.0f;
    }
}

namespace {

// fp16 -> f32 with F16C (every AVX2 CPU has it).  quant.hpp's inline fp16_to_f32 is deliberately not used here: an
// inline function compiled in this file gets this file's instruction set, and the linker may keep that copy for the
// whole program.
inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))); }

inline float hsum_ps(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

inline int32_t hsum_epi32_128(__m128i v) {
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0x4E));
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0xB1));
    return _mm_cvtsi128_si32(v);
}

/// Unpacks the 12 packed scale bytes of Q4_K/Q5_K into 8 scales (bytes 0..7) and 8 mins (bytes 8..15).
inline void unpack_scales_k4(const uint8_t* s12, uint32_t utmp[4]) {
    constexpr uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    std::memcpy(utmp, s12, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
}

/// Sum over the 8 sub-blocks of mins[s] * (bsums[2s] + bsums[2s+1]).
inline int32_t min_dot(const uint32_t utmp[4], const int16_t* bsums) {
    const __m128i mins16 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*)(utmp + 2)));
    const __m128i bs = _mm_hadd_epi16(_mm_loadu_si128((const __m128i*)bsums), _mm_loadu_si128((const __m128i*)(bsums + 8)));
    return hsum_epi32_128(_mm_madd_epi16(mins16, bs));
}

/// sum(w * a) of 32 unsigned weight quants and 32 activations, times `scale`, as 8 int32 lanes.
inline __m256i dot32(__m256i w, const int8_t* a, __m256i scale16) {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(w, _mm256_loadu_si256((const __m256i*)a)), scale16);
}

}  // namespace

float dot_q4k(const void* wv, const Q8K* a, int nb) {
    const BlockQ4_K* w = (const BlockQ4_K*)wv;
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();
    float accm = 0.0f;
    for (int i = 0; i < nb; ++i) {
        uint32_t utmp[4];
        unpack_scales_k4(w[i].scales, utmp);
        const uint8_t* sc = (const uint8_t*)utmp;
        const float d = h2f(w[i].d) * a[i].d;
        const float dmin = h2f(w[i].dmin) * a[i].d;
        accm += dmin * (float)min_dot(utmp, a[i].bsums);
        __m256i s = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {   // 32 bytes of quants: values 64j.. (low nibbles) and 64j+32.. (high)
            const __m256i q = _mm256_loadu_si256((const __m256i*)(w[i].qs + 32 * j));
            const __m256i lo = _mm256_and_si256(q, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            s = _mm256_add_epi32(s, dot32(lo, a[i].nat + 64 * j, _mm256_set1_epi16(sc[2 * j])));
            s = _mm256_add_epi32(s, dot32(hi, a[i].nat + 64 * j + 32, _mm256_set1_epi16(sc[2 * j + 1])));
        }
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_set1_ps(d), acc);
    }
    return hsum_ps(acc) - accm;
}

float dot_q5k(const void* wv, const Q8K* a, int nb) {
    const BlockQ5_K* w = (const BlockQ5_K*)wv;
    const __m256i m4 = _mm256_set1_epi8(0x0F), v16 = _mm256_set1_epi8(16);
    __m256 acc = _mm256_setzero_ps();
    float accm = 0.0f;
    for (int i = 0; i < nb; ++i) {
        uint32_t utmp[4];
        unpack_scales_k4(w[i].scales, utmp);
        const uint8_t* sc = (const uint8_t*)utmp;
        const float d = h2f(w[i].d) * a[i].d;
        const float dmin = h2f(w[i].dmin) * a[i].d;
        accm += dmin * (float)min_dot(utmp, a[i].bsums);
        const __m256i qh = _mm256_loadu_si256((const __m256i*)w[i].qh);
        __m256i s = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {   // the fifth bit of value 64j + l is bit 2j of qh[l], of 64j + 32 + l bit 2j+1
            const __m256i q = _mm256_loadu_si256((const __m256i*)(w[i].qs + 32 * j));
            const __m256i blo = _mm256_set1_epi8((char)(1 << (2 * j))), bhi = _mm256_set1_epi8((char)(2 << (2 * j)));
            __m256i lo = _mm256_and_si256(q, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            lo = _mm256_add_epi8(lo, _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, blo), blo), v16));
            hi = _mm256_add_epi8(hi, _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, bhi), bhi), v16));
            s = _mm256_add_epi32(s, dot32(lo, a[i].nat + 64 * j, _mm256_set1_epi16(sc[2 * j])));
            s = _mm256_add_epi32(s, dot32(hi, a[i].nat + 64 * j + 32, _mm256_set1_epi16(sc[2 * j + 1])));
        }
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_set1_ps(d), acc);
    }
    return hsum_ps(acc) - accm;
}

float dot_q6k(const void* wv, const Q8K* a, int nb) {
    const BlockQ6_K* w = (const BlockQ6_K*)wv;
    const __m256i m4 = _mm256_set1_epi8(0x0F), m3 = _mm256_set1_epi8(0x03);
    __m256 acc = _mm256_setzero_ps();
    float accm = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const int8_t* sc = w[i].scales;
        const __m128i sc8 = _mm_loadu_si128((const __m128i*)sc);
        const __m256i off = _mm256_madd_epi16(_mm256_cvtepi8_epi16(sc8), _mm256_loadu_si256((const __m256i*)a[i].bsums));
        const int32_t isum_off = hsum_epi32_128(_mm_add_epi32(_mm256_castsi256_si128(off), _mm256_extracti128_si256(off, 1)));
        const float d = h2f(w[i].d) * a[i].d;
        accm += d * 32.0f * (float)isum_off;
        __m256i s = _mm256_setzero_si256();
        for (int h = 0; h < 2; ++h) {   // 128 values: ql[64h..], qh[32h..], scales 8h..8h+7 (one per 16 values)
            const __m256i L0 = _mm256_loadu_si256((const __m256i*)(w[i].ql + 64 * h));
            const __m256i L1 = _mm256_loadu_si256((const __m256i*)(w[i].ql + 64 * h + 32));
            const __m256i H = _mm256_loadu_si256((const __m256i*)(w[i].qh + 32 * h));
            const __m256i v[4] = {
                _mm256_or_si256(_mm256_and_si256(L0, m4), _mm256_slli_epi16(_mm256_and_si256(H, m3), 4)),
                _mm256_or_si256(_mm256_and_si256(L1, m4),
                                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(H, 2), m3), 4)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(L0, 4), m4),
                                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(H, 4), m3), 4)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(L1, 4), m4),
                                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(H, 6), m3), 4)),
            };
            for (int k = 0; k < 4; ++k) {   // values 128h + 32k .. +31: scales 8h + 2k (first 16), 8h + 2k + 1
                const __m256i scale = _mm256_set_m128i(_mm_set1_epi16(sc[8 * h + 2 * k + 1]), _mm_set1_epi16(sc[8 * h + 2 * k]));
                s = _mm256_add_epi32(s, dot32(v[k], a[i].nat + 128 * h + 32 * k, scale));
            }
        }
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_set1_ps(d), acc);
    }
    return hsum_ps(acc) - accm;
}

}  // namespace sq::avx2
