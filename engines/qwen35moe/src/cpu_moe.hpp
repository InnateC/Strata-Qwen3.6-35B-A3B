// cpu_moe.hpp - the CPU half of the MoE: routed experts that are not resident in VRAM are computed here, in
// place in pinned RAM, while the GPU computes the resident ones.
//
// Hand-off with the GPU is through host-mapped memory, one mailbox per layer:
//   GPU router kernel  -> Mailbox{x, miss ids, weights}, then ready_seq = seq     (after __threadfence_system)
//   CPU workers        -> Result{out}, then done_seq = seq                        (after a release fence)
//   GPU combine kernel spins on done_seq when the layer had misses.
#pragma once

#include "quant.hpp"

#include <atomic>
#include <memory>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

namespace sq {

// ------------------------------------------------------------------ activation format for the CPU dot products
/// 256 activations quantized to int8 with one float scale.  `nat` is natural order; `pair` is the order the
/// Q4_K/Q5_K nibble layout wants: [0..31, 64..95 | 32..63, 96..127 | 128..159, 192..223 | 160..191, 224..255].
struct alignas(64) Q8K {
    int8_t nat[256];
    int8_t pair[256];
    int16_t bsums[16];   // sums of each 16 consecutive (natural order) quants
    float d;
    int32_t _pad[7];
};

void quantize_q8k(const float* x, Q8K* y, int n);
/// Dot products of `nb` super-blocks of one weight row against `a` (nb entries).
float dot_q4k(const void* w, const Q8K* a, int nb);
float dot_q5k(const void* w, const Q8K* a, int nb);
float dot_q6k(const void* w, const Q8K* a, int nb);
float dot_row(uint32_t type, const void* w, const Q8K* a, int nb);
/// Scalar reference: dequantize the row and dot with the dequantized activation.
float dot_row_ref(uint32_t type, const void* w, const Q8K* a, int nb);

// ------------------------------------------------------------------ mailboxes (host-mapped)
constexpr int kMaxK = 8;             // experts per token
constexpr int kMaxT = 4;             // tokens per step (1 = plain decode, more = speculative verification / MTP)
constexpr int kMaxU = kMaxK * kMaxT; // distinct experts per layer and step

/// One layer's missing experts for a step of n_tok tokens: the distinct experts, and for each of them the routing
/// weight of every token (0 = that token did not pick it).  x holds the tokens' normalized hidden states.
struct alignas(64) Mailbox {
    volatile uint32_t ready_seq;
    uint32_t _p0[15];
    int32_t n_miss;
    int32_t n_tok;
    int32_t _p1[14];
    int32_t ids[kMaxU];
    float w[kMaxU][kMaxT];
    float x[kMaxT][2048];
};
static_assert(sizeof(Mailbox) % 64 == 0);

struct alignas(64) Result {
    volatile uint32_t done_seq;
    uint32_t _p0[15];
    float out[kMaxT][2048];
};

struct ExpertLayerDesc {
    const uint8_t* base = nullptr;   // pinned blobs
    int64_t blob = 0, gu_bytes = 0;
    uint32_t t_gu = 0, t_down = 0;
};

class CpuMoe {
public:
    CpuMoe();
    ~CpuMoe();
    /// `mb`/`res` are host pointers to n_layer mailboxes/results (mapped memory).
    void start(int n_threads, int n_layer, const std::vector<ExpertLayerDesc>& layers, Mailbox* mb, Result* res,
               bool pin_threads);
    void stop();
    int threads() const { return n_threads_; }

    /// Arms the workers for one step: they walk layers l0..l1-1 waiting for the GPU's mailboxes.
    void begin_step(uint32_t seq, int l0, int l1);
    /// Blocks until the workers finished the step.  Returns false if they timed out waiting for the GPU.
    bool end_step();

    /// Synchronous single-layer compute (tests, fallbacks).  out = sum_j w_j * expert_j(x).
    void compute_layer_sync(int layer, const float* x, const int* ids, const float* w, int n, float* out);

    uint64_t experts_computed = 0;

private:
    void dispatch(std::function<void(int)> job);
    void wait_job();
    void worker(int tid, uint32_t gen0);
    /// Returns true on the thread that finished the layer's last output rows.  x / out: n_tok rows of 2048.
    bool run_layer(int tid, int layer, const float (*x)[2048], const int* ids, const float (*w)[kMaxT], int n, int n_tok,
                   float (*out)[2048]);
    void reset_sync();

    int n_threads_ = 0, n_layer_ = 0;
    std::vector<ExpertLayerDesc> layers_;
    Mailbox* mb_ = nullptr;
    Result* res_ = nullptr;
    std::vector<std::thread> threads_;

    std::function<void(int)> job_;
    std::atomic<uint32_t> job_gen_{0};
    std::atomic<int> job_left_{0};
    std::atomic<bool> quit_{false};
    std::atomic<bool> timed_out_{false};
    // Work is handed out in row chunks through per-layer counters, not split statically: a worker the OS preempts
    // only delays the chunk it holds, and the others finish the layer without it.
    struct alignas(64) Counter { std::atomic<int> v{0}; };
    struct LayerSync { Counter take1, done1, take2, done2; };
    std::unique_ptr<LayerSync[]> sync_;         // per layer, reset before each step

    static constexpr int kFF = 512, kE = 2048;
    static constexpr int kChunkGU = 32, kChunkD = 32;   // rows per work item (gate+up / down)
    struct alignas(64) Scratch {
        Q8K xq[kMaxT][kE / 256];
        Q8K hq[kMaxU][kFF / 256];
    };
    std::vector<Scratch> scratch_;               // per thread
    alignas(64) float h_[kMaxU][kFF];            // shared: gate/up output per (expert, token) pair
};

}  // namespace sq
