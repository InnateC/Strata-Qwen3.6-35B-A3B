// cpu_moe.cpp - AVX-512 VNNI expert kernels and the worker pool.  See cpu_moe.hpp.
//
// Every K-quant dot product uses the same trick: the weight quants are UNSIGNED (Q4_K 0..15, Q5_K 0..31,
// Q6_K 0..63 before its -32 offset) and the activations are signed int8, which is exactly the operand shape of
// VPDPBUSD.  Offsets (Q4_K/Q5_K mins, Q6_K's -32) are applied afterwards from the activation block sums.
#include "cpu_moe.hpp"

#include "common.hpp"

#include <immintrin.h>

#include <chrono>
#include <cmath>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#endif

namespace sq {

// ================================================================== quantization
void quantize_q8k(const float* x, Q8K* y, int n) {
    for (int i = 0; i < n / 256; ++i) {
        const float* xb = x + i * 256;
        Q8K& b = y[i];
        __m512 vmax = _mm512_setzero_ps();
        for (int k = 0; k < 16; ++k) vmax = _mm512_max_ps(vmax, _mm512_abs_ps(_mm512_loadu_ps(xb + 16 * k)));
        const float amax = _mm512_reduce_max_ps(vmax);
        if (amax == 0.0f) {
            std::memset(&b, 0, sizeof(Q8K));
            continue;
        }
        const __m512 id = _mm512_set1_ps(127.0f / amax);
        for (int k = 0; k < 16; ++k) {
            const __m512i iv = _mm512_cvt_roundps_epi32(_mm512_mul_ps(_mm512_loadu_ps(xb + 16 * k), id),
                                                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm_storeu_si128((__m128i*)(b.nat + 16 * k), _mm512_cvtsepi32_epi8(iv));
            b.bsums[k] = (int16_t)_mm512_reduce_add_epi32(iv);
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

// ================================================================== dot products
namespace {

inline __m512i perm_idx(int a, int b) {
    return _mm512_setr_epi32(a, a, a, a, a, a, a, a, b, b, b, b, b, b, b, b);
}

inline float hsum_epi32_128(__m128i v) {
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0x4E));
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0xB1));
    return (float)_mm_cvtsi128_si32(v);
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
    return (int32_t)hsum_epi32_128(_mm_madd_epi16(mins16, bs));
}

}  // namespace

float dot_q4k(const void* wv, const Q8K* a, int nb) {
    const BlockQ4_K* w = (const BlockQ4_K*)wv;
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    const __m512i i02 = perm_idx(0, 2), i13 = perm_idx(1, 3), i46 = perm_idx(4, 6), i57 = perm_idx(5, 7);
    __m512 acc = _mm512_setzero_ps();
    float accm = 0.0f;
    for (int i = 0; i < nb; ++i) {
        uint32_t utmp[4];
        unpack_scales_k4(w[i].scales, utmp);
        const __m512i sc32 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)utmp));
        const float d = fp16_to_f32(w[i].d) * a[i].d;
        const float dmin = fp16_to_f32(w[i].dmin) * a[i].d;
        accm += dmin * (float)min_dot(utmp, a[i].bsums);
        const __m512i q0 = _mm512_loadu_si512(w[i].qs), q1 = _mm512_loadu_si512(w[i].qs + 64);
        const __m512i* ap = (const __m512i*)a[i].pair;
        const __m512i z = _mm512_setzero_si512();
        const __m512i p0 = _mm512_dpbusd_epi32(z, _mm512_and_si512(q0, m4), _mm512_load_si512(ap + 0));
        const __m512i p1 = _mm512_dpbusd_epi32(z, _mm512_and_si512(_mm512_srli_epi16(q0, 4), m4), _mm512_load_si512(ap + 1));
        const __m512i p2 = _mm512_dpbusd_epi32(z, _mm512_and_si512(q1, m4), _mm512_load_si512(ap + 2));
        const __m512i p3 = _mm512_dpbusd_epi32(z, _mm512_and_si512(_mm512_srli_epi16(q1, 4), m4), _mm512_load_si512(ap + 3));
        __m512i s = _mm512_mullo_epi32(p0, _mm512_permutexvar_epi32(i02, sc32));
        s = _mm512_add_epi32(s, _mm512_mullo_epi32(p1, _mm512_permutexvar_epi32(i13, sc32)));
        s = _mm512_add_epi32(s, _mm512_mullo_epi32(p2, _mm512_permutexvar_epi32(i46, sc32)));
        s = _mm512_add_epi32(s, _mm512_mullo_epi32(p3, _mm512_permutexvar_epi32(i57, sc32)));
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), _mm512_set1_ps(d), acc);
    }
    return _mm512_reduce_add_ps(acc) - accm;
}

float dot_q5k(const void* wv, const Q8K* a, int nb) {
    const BlockQ5_K* w = (const BlockQ5_K*)wv;
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    const __m512i v16 = _mm512_set1_epi8(16);
    const __m512i i02 = perm_idx(0, 2), i13 = perm_idx(1, 3), i46 = perm_idx(4, 6), i57 = perm_idx(5, 7);
    auto sel = [](int lo, int hi) {
        return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi8((char)lo)), _mm256_set1_epi8((char)hi), 1);
    };
    const __m512i s02 = sel(0x01, 0x04), s13 = sel(0x02, 0x08), s46 = sel(0x10, 0x40), s57 = sel(0x20, 0x80);
    __m512 acc = _mm512_setzero_ps();
    float accm = 0.0f;
    for (int i = 0; i < nb; ++i) {
        uint32_t utmp[4];
        unpack_scales_k4(w[i].scales, utmp);
        const __m512i sc32 = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)utmp));
        const float d = fp16_to_f32(w[i].d) * a[i].d;
        const float dmin = fp16_to_f32(w[i].dmin) * a[i].d;
        accm += dmin * (float)min_dot(utmp, a[i].bsums);
        const __m512i qh = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*)w[i].qh));
        const __m512i q0 = _mm512_loadu_si512(w[i].qs), q1 = _mm512_loadu_si512(w[i].qs + 64);
        __m512i l0 = _mm512_and_si512(q0, m4), h0 = _mm512_and_si512(_mm512_srli_epi16(q0, 4), m4);
        __m512i l1 = _mm512_and_si512(q1, m4), h1 = _mm512_and_si512(_mm512_srli_epi16(q1, 4), m4);
        l0 = _mm512_mask_add_epi8(l0, _mm512_test_epi8_mask(qh, s02), l0, v16);
        h0 = _mm512_mask_add_epi8(h0, _mm512_test_epi8_mask(qh, s13), h0, v16);
        l1 = _mm512_mask_add_epi8(l1, _mm512_test_epi8_mask(qh, s46), l1, v16);
        h1 = _mm512_mask_add_epi8(h1, _mm512_test_epi8_mask(qh, s57), h1, v16);
        const __m512i* ap = (const __m512i*)a[i].pair;
        const __m512i z = _mm512_setzero_si512();
        const __m512i p0 = _mm512_dpbusd_epi32(z, l0, _mm512_load_si512(ap + 0));
        const __m512i p1 = _mm512_dpbusd_epi32(z, h0, _mm512_load_si512(ap + 1));
        const __m512i p2 = _mm512_dpbusd_epi32(z, l1, _mm512_load_si512(ap + 2));
        const __m512i p3 = _mm512_dpbusd_epi32(z, h1, _mm512_load_si512(ap + 3));
        __m512i s = _mm512_mullo_epi32(p0, _mm512_permutexvar_epi32(i02, sc32));
        s = _mm512_add_epi32(s, _mm512_mullo_epi32(p1, _mm512_permutexvar_epi32(i13, sc32)));
        s = _mm512_add_epi32(s, _mm512_mullo_epi32(p2, _mm512_permutexvar_epi32(i46, sc32)));
        s = _mm512_add_epi32(s, _mm512_mullo_epi32(p3, _mm512_permutexvar_epi32(i57, sc32)));
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), _mm512_set1_ps(d), acc);
    }
    return _mm512_reduce_add_ps(acc) - accm;
}

float dot_q6k(const void* wv, const Q8K* a, int nb) {
    const BlockQ6_K* w = (const BlockQ6_K*)wv;
    const __m512i m4 = _mm512_set1_epi8(0x0F), m3 = _mm512_set1_epi8(0x03);
    auto cnt = [](int lo, int hi) {
        return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi16((short)lo)), _mm256_set1_epi16((short)hi), 1);
    };
    const __m512i c_lo = cnt(0, 2), c_hi = cnt(4, 6);
    // scale index for lane l of the product covering values [64k + 4l, 64k + 4l + 4): (64k + 4l) / 16
    const __m512i ix[4] = {
        _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3),
        _mm512_setr_epi32(4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7),
        _mm512_setr_epi32(8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11),
        _mm512_setr_epi32(12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15),
    };
    __m512 acc = _mm512_setzero_ps();
    float accm = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const __m128i sc8 = _mm_loadu_si128((const __m128i*)w[i].scales);
        const __m512i sc32 = _mm512_cvtepi8_epi32(sc8);
        const __m256i off = _mm256_madd_epi16(_mm256_cvtepi8_epi16(sc8), _mm256_loadu_si256((const __m256i*)a[i].bsums));
        const int32_t isum_off = (int32_t)hsum_epi32_128(_mm_add_epi32(_mm256_castsi256_si128(off), _mm256_extracti128_si256(off, 1)));
        const float d = fp16_to_f32(w[i].d) * a[i].d;
        accm += d * 32.0f * (float)isum_off;
        __m512i s = _mm512_setzero_si512();
        for (int h = 0; h < 2; ++h) {
            const __m512i L = _mm512_loadu_si512(w[i].ql + 64 * h);
            const __m512i H = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i*)(w[i].qh + 32 * h)));
            const __m512i vlo = _mm512_or_si512(_mm512_and_si512(L, m4),
                                                _mm512_slli_epi16(_mm512_and_si512(_mm512_srlv_epi16(H, c_lo), m3), 4));
            const __m512i vhi = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(L, 4), m4),
                                                _mm512_slli_epi16(_mm512_and_si512(_mm512_srlv_epi16(H, c_hi), m3), 4));
            const __m512i z = _mm512_setzero_si512();
            const __m512i plo = _mm512_dpbusd_epi32(z, vlo, _mm512_load_si512((const __m512i*)(a[i].nat + 128 * h)));
            const __m512i phi = _mm512_dpbusd_epi32(z, vhi, _mm512_load_si512((const __m512i*)(a[i].nat + 128 * h + 64)));
            s = _mm512_add_epi32(s, _mm512_mullo_epi32(plo, _mm512_permutexvar_epi32(ix[2 * h + 0], sc32)));
            s = _mm512_add_epi32(s, _mm512_mullo_epi32(phi, _mm512_permutexvar_epi32(ix[2 * h + 1], sc32)));
        }
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(s), _mm512_set1_ps(d), acc);
    }
    return _mm512_reduce_add_ps(acc) - accm;
}

float dot_row(uint32_t type, const void* w, const Q8K* a, int nb) {
    switch (type) {
        case T_Q4_K: return dot_q4k(w, a, nb);
        case T_Q5_K: return dot_q5k(w, a, nb);
        case T_Q6_K: return dot_q6k(w, a, nb);
        default: return dot_row_ref(type, w, a, nb);
    }
}

float dot_row_ref(uint32_t type, const void* w, const Q8K* a, int nb) {
    std::vector<float> wf((size_t)nb * 256);
    dequant_row(type, w, wf.data(), (int64_t)nb * 256);
    double s = 0;
    for (int i = 0; i < nb; ++i)
        for (int k = 0; k < 256; ++k) s += (double)wf[(size_t)i * 256 + k] * a[i].nat[k] * a[i].d;
    return (float)s;
}

// ================================================================== worker pool
CpuMoe::CpuMoe() = default;
CpuMoe::~CpuMoe() { stop(); }

void CpuMoe::start(int n_threads, int n_layer, const std::vector<ExpertLayerDesc>& layers, Mailbox* mb, Result* res,
                   bool pin_threads) {
    n_threads_ = n_threads;
    n_layer_ = n_layer;
    layers_ = layers;
    mb_ = mb;
    res_ = res;
    scratch_.resize(n_threads);
    sync_ = std::make_unique<LayerSync[]>(n_layer);
    quit_ = false;
    const uint32_t gen0 = job_gen_.load();
    for (int t = 0; t < n_threads; ++t) {
        threads_.emplace_back([this, t, gen0] { worker(t, gen0); });
#ifdef _WIN32
        if (pin_threads) {
            const int ncpu = (int)std::thread::hardware_concurrency();
            // SMT siblings are adjacent logical processors on Windows: one worker per physical core.
            const int cpu = ncpu >= 2 ? (2 * t) % ncpu + (2 * t / ncpu) % 2 : 0;
            SetThreadAffinityMask(threads_.back().native_handle(), 1ull << cpu);
            SetThreadPriority(threads_.back().native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);
        }
#else
        (void)pin_threads;
#endif
    }
}

void CpuMoe::stop() {
    if (threads_.empty()) return;
    quit_ = true;
    job_gen_.fetch_add(1);
    job_gen_.notify_all();
    for (auto& t : threads_) t.join();
    threads_.clear();
}

void CpuMoe::dispatch(std::function<void(int)> job) {
    job_ = std::move(job);
    job_left_.store(n_threads_, std::memory_order_relaxed);
    job_gen_.fetch_add(1, std::memory_order_release);
    job_gen_.notify_all();
}

void CpuMoe::wait_job() {
    int spins = 0;
    while (job_left_.load(std::memory_order_acquire) > 0) {
        if (++spins < 1 << 16) _mm_pause();
        else std::this_thread::yield();
    }
}

void CpuMoe::worker(int tid, uint32_t seen) {
    for (;;) {
        // Spin briefly (decode steps arrive back to back), then sleep on the generation counter.
        uint32_t g;
        int spins = 0;
        while ((g = job_gen_.load(std::memory_order_acquire)) == seen) {
            if (++spins < 200000) _mm_pause();
            else job_gen_.wait(seen, std::memory_order_acquire);
        }
        seen = g;
        if (quit_) return;
        job_(tid);
        job_left_.fetch_sub(1, std::memory_order_acq_rel);
    }
}

void CpuMoe::reset_sync() {
    for (int l = 0; l < n_layer_; ++l) {
        LayerSync& Y = sync_[l];
        Y.take1.v.store(0, std::memory_order_relaxed);
        Y.done1.v.store(0, std::memory_order_relaxed);
        Y.take2.v.store(0, std::memory_order_relaxed);
        Y.done2.v.store(0, std::memory_order_relaxed);
    }
}

static inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

bool CpuMoe::run_layer(int tid, int layer, const float (*x)[2048], const int* ids, const float (*w)[kMaxT], int n,
                       int n_tok, float (*out)[2048]) {
    // h_ is shared by all layers.  It stays valid while this thread holds an unfinished chunk: the layer cannot
    // complete without it, and the next layer (which overwrites h_) only starts once the GPU has read this layer's
    // result.  So a late thread reads h_ only after it has taken a chunk.
    Scratch& S = scratch_[tid];
    LayerSync& Y = sync_[layer];
    const ExpertLayerDesc& L = layers_[layer];
    // (expert, token) pairs, expert-major; every thread derives the same list
    int pj[kMaxU], pt[kMaxU], first[kMaxU + 1];
    int np = 0;
    for (int j = 0; j < n; ++j) {
        first[j] = np;
        for (int t = 0; t < n_tok; ++t)
            if (w[j][t] != 0.0f && np < kMaxU) { pj[np] = j; pt[np] = t; ++np; }
    }
    first[n] = np;
    const int64_t rb_gu = row_bytes(L.t_gu, kE);
    const int R = n * kFF;
    bool have_x = false;
    for (;;) {   // gate/up rows: consecutive rows of one expert per chunk, each row dotted with its tokens
        const int c = Y.take1.v.fetch_add(kChunkGU, std::memory_order_relaxed);
        if (c >= R) break;
        if (!have_x) {
            for (int t = 0; t < n_tok; ++t) quantize_q8k(x[t], S.xq[t], kE);
            have_x = true;
        }
        const int ce = std::min(c + kChunkGU, R);
        for (int g = c; g < ce; ++g) {
            const int j = g / kFF, r = g % kFF;
            const uint8_t* blob = L.base + (int64_t)ids[j] * L.blob;
            for (int p = first[j]; p < first[j + 1]; ++p) {
                const float gt = dot_row(L.t_gu, blob + r * rb_gu, S.xq[pt[p]], kE / 256);
                const float up = dot_row(L.t_gu, blob + L.gu_bytes + r * rb_gu, S.xq[pt[p]], kE / 256);
                h_[p][r] = silu(gt) * up;
            }
        }
        Y.done1.v.fetch_add(ce - c, std::memory_order_release);
    }
    const int64_t rb_d = row_bytes(L.t_down, kFF);
    const uint8_t* down[kMaxU];
    for (int j = 0; j < n; ++j) down[j] = L.base + (int64_t)ids[j] * L.blob + 2 * L.gu_bytes;
    bool have_h = false;
    for (;;) {   // down rows: need every gate/up row
        const int c = Y.take2.v.fetch_add(kChunkD, std::memory_order_relaxed);
        if (c >= kE) return false;
        if (!have_h) {
            while (Y.done1.v.load(std::memory_order_acquire) < R) _mm_pause();
            for (int p = 0; p < np; ++p) quantize_q8k(h_[p], S.hq[p], kFF);
            have_h = true;
        }
        for (int r = c; r < c + kChunkD; ++r) {
            float s[kMaxT] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int p = 0; p < np; ++p) s[pt[p]] += w[pj[p]][pt[p]] * dot_row(L.t_down, down[pj[p]] + r * rb_d, S.hq[p], kFF / 256);
            for (int t = 0; t < n_tok; ++t) out[t][r] = s[t];
        }
        if (Y.done2.v.fetch_add(kChunkD, std::memory_order_acq_rel) + kChunkD == kE) return true;
    }
}

void CpuMoe::begin_step(uint32_t seq, int l0, int l1) {
    timed_out_ = false;
    reset_sync();
    dispatch([this, seq, l0, l1](int tid) {
        for (int l = l0; l < l1; ++l) {
            Mailbox& m = mb_[l];
            int spins = 0;
            auto t0 = std::chrono::steady_clock::now();
            while (m.ready_seq != seq) {
                _mm_pause();
                if (timed_out_.load(std::memory_order_relaxed)) return;
                if ((++spins & 0xFFFF) == 0 &&
                    std::chrono::steady_clock::now() - t0 > std::chrono::seconds(20)) {
                    timed_out_ = true;
                    return;
                }
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            const int n = m.n_miss;
            if (n <= 0) continue;
            if (run_layer(tid, l, m.x, m.ids, m.w, n, m.n_tok, res_[l].out)) {
                experts_computed += n;
                std::atomic_thread_fence(std::memory_order_release);
                res_[l].done_seq = seq;
            }
        }
    });
}

bool CpuMoe::end_step() {
    wait_job();
    return !timed_out_;
}

void CpuMoe::compute_layer_sync(int layer, const float* x, const int* ids, const float* w, int n, float* out) {
    auto xx = std::make_unique<float[][2048]>(1);
    auto oo = std::make_unique<float[][2048]>(1);
    std::memcpy(xx[0], x, sizeof(float) * 2048);
    float ww[kMaxU][kMaxT] = {};
    for (int j = 0; j < n; ++j) ww[j][0] = w[j];
    reset_sync();
    dispatch([&, this](int tid) { run_layer(tid, layer, xx.get(), ids, ww, n, 1, oo.get()); });
    wait_job();
    std::memcpy(out, oo[0], sizeof(float) * 2048);
}

}  // namespace sq
