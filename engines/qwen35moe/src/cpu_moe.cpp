// cpu_moe.cpp - the CPU half of the MoE: kernel dispatch and the worker pool.  See cpu_moe.hpp.
//
// The kernels themselves are in cpu_avx512.cpp and cpu_avx2.cpp, each compiled for its instruction set; this file is
// compiled for the baseline (x86-64) and picks one set at run time.
#include "cpu_moe.hpp"

#include "common.hpp"
#include "cpu_kernels.hpp"

#include <immintrin.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef _WIN32
#include <intrin.h>
#include <windows.h>
#else
#include <cpuid.h>
#include <pthread.h>
#include <sched.h>
#endif

namespace sq {

// ================================================================== instruction set
namespace {

void cpuid(int leaf, int sub, unsigned r[4]) {
#ifdef _WIN32
    int x[4];
    __cpuidex(x, leaf, sub);
    for (int i = 0; i < 4; ++i) r[i] = (unsigned)x[i];
#else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}

unsigned long long xgetbv0() {
#ifdef _WIN32
    return _xgetbv(0);
#else
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long)hi << 32) | lo;
#endif
}

struct Features { bool avx2 = false, avx512 = false; };

Features detect() {
    Features f;
    unsigned r[4];
    cpuid(0, 0, r);
    if (r[0] < 7) return f;
    cpuid(1, 0, r);
    const bool osxsave = r[2] & (1u << 27), fma = r[2] & (1u << 12), f16c = r[2] & (1u << 29);
    if (!osxsave) return f;
    const unsigned long long xcr0 = xgetbv0();
    const bool ymm = (xcr0 & 0x6) == 0x6, zmm = (xcr0 & 0xE6) == 0xE6;
    cpuid(7, 0, r);
    f.avx2 = ymm && fma && f16c && (r[1] & (1u << 5));
    // F, DQ, BW, VL and VNNI: what cpu_avx512.cpp is compiled for
    const unsigned ebx_need = (1u << 16) | (1u << 17) | (1u << 30) | (1u << 31);
    f.avx512 = f.avx2 && zmm && (r[1] & ebx_need) == ebx_need && (r[2] & (1u << 11));
    return f;
}

const CpuKernels kAvx512{"AVX-512 VNNI", avx512::quantize_q8k, avx512::dot_q4k, avx512::dot_q5k, avx512::dot_q6k};
const CpuKernels kAvx2{"AVX2", avx2::quantize_q8k, avx2::dot_q4k, avx2::dot_q5k, avx2::dot_q6k};
const CpuKernels* g_kernels = nullptr;

const CpuKernels& kernels() {
    if (!g_kernels) {
        const Features f = detect();
        SQ_CHECK(f.avx2, "this CPU has no AVX2 (with FMA and F16C): the expert kernels need it");
        const char* env = std::getenv("STRATA_CPU_ISA");
        g_kernels = f.avx512 && !(env && std::string(env) == "avx2") ? &kAvx512 : &kAvx2;
    }
    return *g_kernels;
}

}  // namespace

const char* cpu_isa() { return kernels().name; }

bool cpu_has_avx2() { return detect().avx2; }

bool cpu_force_isa(const char* name) {
    const Features f = detect();
    const std::string n(name);
    if (n == "avx512" && f.avx512) { g_kernels = &kAvx512; return true; }
    if (n == "avx2" && f.avx2) { g_kernels = &kAvx2; return true; }
    return false;
}

void quantize_q8k(const float* x, Q8K* y, int n) { kernels().quantize_q8k(x, y, n); }

DotFn dot_fn(uint32_t type) {
    const CpuKernels& k = kernels();
    switch (type) {
        case T_Q4_K: return k.dot_q4k;
        case T_Q5_K: return k.dot_q5k;
        case T_Q6_K: return k.dot_q6k;
        default: return nullptr;
    }
}

float dot_row(uint32_t type, const void* w, const Q8K* a, int nb) {
    const DotFn f = dot_fn(type);
    return f ? f(w, a, nb) : dot_row_ref(type, w, a, nb);
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
        if (pin_threads) {
            const int ncpu = (int)std::thread::hardware_concurrency();
            // SMT siblings are adjacent logical processors on Windows (and on most Linux systems with this
            // numbering, cores first is the other common one; pinning is only a hint there): one worker per core.
            const int cpu = ncpu >= 2 ? (2 * t) % ncpu + (2 * t / ncpu) % 2 : 0;
#ifdef _WIN32
            if (cpu < 64) SetThreadAffinityMask(threads_.back().native_handle(), 1ull << cpu);
            SetThreadPriority(threads_.back().native_handle(), THREAD_PRIORITY_ABOVE_NORMAL);
#else
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpu, &set);
            pthread_setaffinity_np(threads_.back().native_handle(), sizeof(set), &set);
#endif
        }
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
    const DotFn dot_gu = dot_fn(L.t_gu), dot_d = dot_fn(L.t_down);
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
                const float gt = dot_gu(blob + r * rb_gu, S.xq[pt[p]], kE / 256);
                const float up = dot_gu(blob + L.gu_bytes + r * rb_gu, S.xq[pt[p]], kE / 256);
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
            for (int p = 0; p < np; ++p) s[pt[p]] += w[pj[p]][pt[p]] * dot_d(down[pj[p]] + r * rb_d, S.hq[p], kFF / 256);
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
