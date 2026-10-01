// engine.hpp - the inference engine: state (KV cache, delta-net states), the VRAM expert cache, the decode step
// (one CUDA graph per step, CPU experts in parallel), prefill, prefix reuse, and speculative decoding with the
// model's MTP (multi-token prediction) block.
//
// MTP: the MTP block predicts token i+2 from the trunk's normed hidden state h_i and the embedding of token i+1.
// A speculative step drafts D tokens with it (the first from the pending (h, token) pairs, the rest recursively
// from its own hidden state), verifies [token, drafts] in one decode step of D+1 tokens, keeps the longest accepted
// prefix plus one token sampled by the trunk, and rolls the delta-net state back to the last accepted token (its
// state after every token of the step is snapshotted).  The MTP layer has its own KV cache, filled for every
// position from (h_i, token i+1): during prefill for the prompt, and at the start of each step for the tokens the
// previous step accepted ("pending pairs": their hidden states wait in mtp_hin_ until the next token is known).
#pragma once

#include "cpu_moe.hpp"
#include "kernels.cuh"
#include "model.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace sq {

struct EngineOptions {
    std::string model_path;
    int ctx = 32768;
    int cpu_threads = 8;
    int64_t vram_reserve_mb = 1200;  // left free on the card once the engine is ready (other programs, driver)
    int64_t cache_mb = -1;           // expert cache size; -1 = everything that is free
    std::string profile_path;        // expert routing profile used to pick the resident experts
    bool use_graph = true;
    bool pin_threads = true;
    int prefill_chunk = 4096;
    int prefill_min = 8;             // shorter inputs go through the decode path
    int adapt_every = 32;            // decode steps between cache adaptation rounds (0 = off)
    int adapt_swaps = 24;            // max experts swapped in per round
    double profile_weight = 1.0;    // loaded profile vs live routing counts (see setup_cache)
    int ckpt_slots = 2;              // prefix-reuse checkpoints (delta-net state copies, ~63 MiB each)
    int mtp_draft = 3;               // speculative tokens drafted per step by the MTP block (0 = off; max kMaxT - 1)
    float draft_p = 0.8f;            // keep drafting while the MTP's probability of all drafts so far is >= this
                                     // (half of it for the first draft; below that the step is a plain one)
    bool verbose = false;
};

struct SamplingParams {
    float temperature = 0.0f;   // 0 = greedy
    int top_k = 20;
    float top_p = 0.95f;
    float min_p = 0.0f;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    float repetition_penalty = 1.0f;
    uint64_t seed = 0;
};

struct EngineStats {
    uint64_t decode_steps = 0;               // tokens fed through the decode path (a speculative step feeds several)
    uint64_t graph_steps = 0;                // decode-step graph launches (trunk)
    double decode_ms = 0, prefill_ms = 0;
    double adapt_ms = 0;                     // expert cache re-ranking + swaps after decode steps
    uint64_t prefill_tokens = 0;
    uint64_t swaps = 0;
    uint64_t spec_steps = 0, drafted = 0, accepted = 0;   // MTP speculative steps, drafts, accepted drafts
    uint64_t spec_skipped = 0;               // steps where the first draft was too unlikely to verify
    double mtp_ms = 0;                       // time in the MTP block (drafting + pending pairs)
    double verify_ms = 0;                    // time in verification steps (trunk over token + drafts)
    std::vector<float> step_ms;              // recent per-token decode times (ring), for a noise-robust median
};

class Prefill;

class Engine {
public:
    Engine();
    ~Engine();
    bool init(const EngineOptions& opt, std::string& err);

    const Config& cfg() const { return model_.cfg; }
    int ctx() const { return opt_.ctx; }
    int n_past() const { return n_past_; }
    const std::vector<int>& tokens() const { return history_; }

    /// Forget everything (positions restart at 0).
    void reset();
    /// Feed tokens (appending to the current sequence).  Leaves the logits of the LAST token ready.
    void feed(const int* tokens, int n);
    /// Sample the next token from the current logits.
    int sample(const SamplingParams& sp, std::mt19937_64& rng);
    /// Start of a request: reset penalty counters.
    void begin_request();

    /// True when the MTP block is loaded and drafting is on.
    bool mtp() const { return mtp_on_; }
    int mtp_draft() const { return mtp_on_ ? draft_ : 0; }
    /// Drafts per step from now on (0 = plain decoding; at most the --mtp value the engine started with).
    void set_mtp_draft(int d) { draft_ = std::clamp(d, 0, opt_.mtp_draft); }
    /// One speculative step.  `tok` is the token just sampled (not fed yet).  Feeds it plus the accepted drafts and
    /// returns in `out` the tokens that follow `tok`: the accepted drafts (fed) and last, one token sampled from the
    /// trunk (not fed, like the result of sample()).  Without MTP: feeds `tok` and samples one token.
    void spec_step(int tok, const SamplingParams& sp, std::mt19937_64& rng, std::vector<int>& out);

    /// Prefix reuse: how many leading tokens of `prompt` can be kept (the rest must be fed).  Picks the longest of
    /// the live state and the checkpoints and restores it when it returns > 0.
    int reuse_prefix(const std::vector<int>& prompt);
    /// Remember the current state in a checkpoint slot (least recently used one).  The delta-net state cannot be
    /// rolled back, so the server checkpoints the positions a follow-up request is likely to resume from.
    void save_checkpoint();

    /// Expert profile (routing frequencies seen so far, including the loaded profile) -> file.
    bool save_profile(const std::string& path, std::string& err);

    EngineStats stats;
    const Model& model() const { return model_; }
    int64_t cache_slots() const { return cache_slots_total_; }
    double cache_gib() const { return cache_bytes_ / 1073741824.0; }
    std::string status_line() const;

    // --- debugging
    std::vector<float> debug_hidden();   // copy of the current residual stream x (fp32)
    std::vector<float> logits_host();    // full logits of the last fed token
    /// A verification step over T tokens that keeps the first `keep` (MTP on); returns the T rows of logits.
    std::vector<float> debug_verify(const int* tokens, int T, int keep);
    std::vector<std::vector<float>>* debug_layers = nullptr;   // --no-graph only: token 0's x after every layer

private:
    friend class Prefill;
    void alloc_state();
    void setup_cache();
    void trim_cache(int64_t bytes);   // drop the coldest cached experts until at least `bytes` are freed
    void build_layer(cudaStream_t s, int il, int T, const StepParams* P, const float* next_w, float* snap);
    void build_decode(cudaStream_t s, int T);   // enqueue one decode step of T tokens (graph body)
    void build_mtp(cudaStream_t s, int T);      // enqueue one MTP step over T pending pairs (graph body)
    /// Runs a captured step (or builds it directly without graphs) with the CPU experts of layers l0..l1-1.
    void run_step(bool mtp, int T);
    void decode_one(int token);
    void decode_multi(const int* tokens, int T);   // trunk step over T tokens (verification)
    /// MTP over the pending pairs, `next` being the token after the last one.  Returns the drafted token.
    int mtp_catch_up(int next);
    int mtp_recur(int prev_draft, int pos);        // one more draft from the MTP's own hidden state
    /// After a verification step of T tokens: keep the first a (roll the delta net back), make them history.
    void commit_step(const int* toks, int a, int T);
    void adapt_cache();
    void apply_swap(int layer, int slot, int expert);

    EngineOptions opt_;
    Model model_;
    cudaStream_t stream_ = nullptr, copy_stream_ = nullptr;
    cudaGraphExec_t graph_[kMaxT + 1] = {};       // trunk step of T tokens
    cudaGraphExec_t mtp_graph_[kMaxT + 1] = {};   // MTP step over T pairs
    std::unique_ptr<CpuMoe> cpu_;
    std::unique_ptr<Prefill> prefill_;

    // host-mapped handshake memory
    Mailbox* mb_host_ = nullptr;
    Mailbox* mb_dev_ = nullptr;
    Result* res_host_ = nullptr;
    Result* res_dev_ = nullptr;
    StepParams* params_host_ = nullptr;       // pinned; copied into params_dev_ by each step
    StepParams* params_dev_ = nullptr;
    StepParams* mparams_host_ = nullptr;      // same for MTP steps
    StepParams* mparams_dev_ = nullptr;
    int32_t* out_host_ = nullptr;             // mapped: [0..kMaxT) verification argmax, [kMaxT] MTP draft
    int32_t* out_dev_ = nullptr;
    float* prob_host_ = nullptr;              // mapped: the MTP's probability of its draft
    float* prob_dev_ = nullptr;
    unsigned long long* ecount_ = nullptr;    // device: distinct experts resident / missing (all steps)
    Candidates* cand_host_ = nullptr;
    Candidates* cand_dev_ = nullptr;
    int32_t* tok_host_ = nullptr;   // mapped
    int32_t* tok_dev_ = nullptr;

    // device buffers (decode)
    float *x_ = nullptr, *xn_ = nullptr, *qkvz_ = nullptr, *ba_ = nullptr, *conv_out_ = nullptr, *o_ = nullptr;
    float *qkv_ = nullptr, *qbuf_ = nullptr, *part_o_ = nullptr, *part_ml_ = nullptr, *attn_out_ = nullptr;
    float *rlog_ = nullptr, *shgu_ = nullptr, *shout_ = nullptr, *h_gu_ = nullptr, *moe_gpu_ = nullptr;
    float *logits_ = nullptr, *d_sg_ = nullptr, *cand_tmp_ = nullptr;
    // MTP
    bool mtp_on_ = false;
    int draft_ = 0;
    float *mtp_hin_ = nullptr, *mtp_hlast_ = nullptr, *mtp_logits_ = nullptr;
    ActQ mtp_act_;                            // [enorm(e) | hnorm(h)] per pair, 4096 each
    int mtp_k_ = 0;                           // pending pairs: positions n_past_-mtp_k_ .. n_past_-1, hiddens in mtp_hin_
    float* snap_arena_ = nullptr;             // delta-net state after each token of a verification step (but the last)
    int32_t *d_nmiss_ = nullptr, *d_err_ = nullptr;
    unsigned long long* wait_ns_ = nullptr;     // GPU time spent waiting for the CPU experts (decode)
    HitList* hits_ = nullptr;
    ActQ xq_, yq_, shq_, hq_;
    uint32_t* route_counts_ = nullptr;   // [n_layer][256] (device, cumulative)
    uint32_t* tok_counts_ = nullptr;     // [vocab] generated-token counts for penalties

    // state
    std::vector<uint16_t*> kc_, vc_;          // per layer (null for delta-net layers)
    std::vector<float*> conv_st_, ssm_st_;    // per layer (null for attention layers)
    float* state_arena_ = nullptr;            // conv + ssm states of all delta-net layers, contiguous
    struct Checkpoint {
        float* arena = nullptr;               // copy of state_arena_
        float* hin = nullptr;                 // copy of mtp_hin_ (MTP pending pairs)
        int mtp_k = 0;
        std::vector<int> tokens;              // the tokens it covers
        uint64_t stamp = 0;                   // LRU
    };
    std::vector<Checkpoint> ckpts_;
    uint64_t ckpt_clock_ = 0;
    int64_t state_floats_ = 0;
    int n_past_ = 0;
    std::vector<int> history_;
    bool logits_valid_ = false;
    uint32_t seq_ = 0;

    // expert cache
    std::vector<uint8_t*> slot_base_;         // per layer
    std::vector<int> slots_per_layer_;
    std::vector<std::vector<int>> slot_expert_;   // [layer][slot] -> expert
    std::vector<int32_t> residency_host_;     // [layer][256] -> slot or -1
    int32_t* residency_dev_ = nullptr;
    CacheLayerInfo* cinfo_dev_ = nullptr;
    int64_t cache_slots_total_ = 0;
    int64_t cache_bytes_ = 0;
    std::vector<double> freq_;                // [layer][256] routing frequency (profile + decayed live counts)
    std::vector<uint32_t> counts_seen_;       // last copy of route_counts_
    std::vector<float> prior_;                // the profile as loaded (saved back plus this session's counts)
    int steps_since_adapt_ = 0;
    std::vector<uint32_t> route_host_;
};

}  // namespace sq
