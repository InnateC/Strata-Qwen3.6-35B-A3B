// engine.cpp - see engine.hpp.
#include "engine.hpp"

#include "common.hpp"
#include "prefill.hpp"
#include "sampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>

namespace sq {

namespace {

template <typename T>
T* dmalloc(size_t n) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    CUDA_CHECK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

ActQ alloc_act(int n) {
    ActQ a;
    a.q = dmalloc<int8_t>(n);
    a.d = dmalloc<float>(n / 32);
    a.s = dmalloc<float>(n / 32);
    return a;
}

template <typename T>
void host_mapped(size_t n, T*& host, T*& dev) {
    CUDA_CHECK(cudaHostAlloc((void**)&host, n * sizeof(T), cudaHostAllocMapped));
    std::memset((void*)host, 0, n * sizeof(T));
    CUDA_CHECK(cudaHostGetDevicePointer((void**)&dev, (void*)host, 0));
}

constexpr uint32_t kProfileMagic = 0x50455153;   // "SQEP"
constexpr double kDecay = 0.97;                  // routing-frequency decay per adapt_cache() call

}  // namespace

Engine::Engine() = default;

Engine::~Engine() {
    if (cpu_) cpu_->stop();
    prefill_.reset();
    for (auto* g : graph_)
        if (g) cudaGraphExecDestroy(g);
    for (auto* g : mtp_graph_)
        if (g) cudaGraphExecDestroy(g);
}

bool Engine::init(const EngineOptions& opt, std::string& err) {
    opt_ = opt;
    opt_.mtp_draft = std::clamp(opt_.mtp_draft, 0, kMaxT - 1);
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin | cudaDeviceMapHost));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    log("GPU: %s, %.1f GiB, sm_%d%d", prop.name, prop.totalGlobalMem / 1073741824.0, prop.major, prop.minor);
    if (!model_.load(opt.model_path, opt_.mtp_draft > 0, err)) return false;
    const Config& c = model_.cfg;
    mtp_on_ = c.n_mtp > 0 && opt_.mtp_draft > 0;
    if (opt_.mtp_draft > 0 && !mtp_on_) log("no MTP block in this model: speculative decoding off");
    opt_.ctx = std::min(opt_.ctx, c.ctx_train);
    opt_.ctx = (opt_.ctx + 255) / 256 * 256;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
    alloc_state();
    prefill_ = std::make_unique<Prefill>(*this);
    if (!prefill_->init(err)) return false;
    setup_cache();

    const int NL = c.n_moe_layers();
    std::vector<ExpertLayerDesc> descs(NL);
    for (int il = 0; il < NL; ++il) {
        const LayerW& L = model_.layers[il];
        descs[il] = ExpertLayerDesc{L.experts, L.blob_bytes, L.gu_bytes, L.t_gu, L.t_down};
    }
    cpu_ = std::make_unique<CpuMoe>();
    cpu_->start(opt_.cpu_threads, NL, descs, mb_host_, res_host_, opt_.pin_threads);

    {   // warm-up: loads the cuBLAS / prefill kernels now instead of on the first request
        std::vector<int> w(std::max(opt_.prefill_min, 16), 198);
        const auto saved = opt_.adapt_every;
        opt_.adapt_every = 0;
        feed(w.data(), (int)w.size());
        opt_.adapt_every = saved;
        CUDA_CHECK(cudaMemset(route_counts_, 0, (size_t)NL * c.n_expert * 4));
        stats = EngineStats{};
        CUDA_CHECK(cudaMemset(wait_ns_, 0, 8));
        CUDA_CHECK(cudaMemset(ecount_, 0, 16));
    }
    if (opt_.use_graph) {
        auto capture = [&](cudaGraphExec_t& ge, auto&& body) {
            cudaGraph_t g;
            CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
            body();
            CUDA_CHECK(cudaStreamEndCapture(stream_, &g));
            CUDA_CHECK(cudaGraphInstantiate(&ge, g, 0));
            CUDA_CHECK(cudaGraphDestroy(g));
        };
        const int tmax = mtp_on_ ? opt_.mtp_draft + 1 : 1;
        for (int T = 1; T <= tmax; ++T) {
            capture(graph_[T], [&] { build_decode(stream_, T); });
            if (mtp_on_) capture(mtp_graph_[T], [&] { build_mtp(stream_, T); });
        }
    }
    reset();
    size_t fr = 0, tot = 0;
    CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
    // The warm-up and the graphs allocated more (cuBLAS workspace, lazily loaded kernels, graph memory) than
    // setup_cache could see: give experts back until the reserve is really free.
    const int64_t short_by = opt_.vram_reserve_mb * 1048576ll - (int64_t)fr;
    if (short_by > 0) {
        trim_cache(short_by);
        CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
    }
    log("ready: ctx %d, expert cache %lld slots (%.2f GiB, %.1f%% of experts), %d CPU threads, VRAM free %.2f GiB",
        opt_.ctx, (long long)cache_slots_total_, cache_gib(), 100.0 * cache_slots_total_ / (NL * c.n_expert),
        opt_.cpu_threads, fr / 1073741824.0);
    draft_ = mtp_on_ ? opt_.mtp_draft : 0;
    if (mtp_on_) log("MTP speculative decoding: %d draft token(s) per step", opt_.mtp_draft);
    return true;
}

void Engine::alloc_state() {
    const Config& c = model_.cfg;
    const int NL = c.n_moe_layers();
    constexpr int T = kMaxT;
    host_mapped(NL, mb_host_, mb_dev_);
    host_mapped(NL, res_host_, res_dev_);
    host_mapped(1, cand_host_, cand_dev_);
    host_mapped(4, tok_host_, tok_dev_);
    host_mapped(kMaxT + 4, out_host_, out_dev_);
    host_mapped(4, prob_host_, prob_dev_);
    d_err_ = tok_dev_ + 1;   // tok_host_[1] is the GPU error flag
    wait_ns_ = dmalloc<unsigned long long>(1);
    ecount_ = dmalloc<unsigned long long>(2);
    // the params structs must live in device memory for the kernels: each step copies them in as its first node
    CUDA_CHECK(cudaHostAlloc((void**)&params_host_, sizeof(StepParams), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&mparams_host_, sizeof(StepParams), cudaHostAllocDefault));
    std::memset(params_host_, 0, sizeof(StepParams));
    std::memset(mparams_host_, 0, sizeof(StepParams));
    params_dev_ = dmalloc<StepParams>(1);
    mparams_dev_ = dmalloc<StepParams>(1);

    x_ = dmalloc<float>((size_t)T * c.n_embd);
    xn_ = dmalloc<float>((size_t)T * c.n_embd);
    qkvz_ = dmalloc<float>((size_t)T * (c.conv_dim() + c.ssm_d_inner));
    ba_ = dmalloc<float>((size_t)T * 64);
    conv_out_ = dmalloc<float>((size_t)T * c.conv_dim());
    o_ = dmalloc<float>((size_t)T * c.ssm_d_inner);
    qkv_ = dmalloc<float>((size_t)T * (c.n_head * c.head_dim * 2 + 2 * c.n_head_kv * c.head_dim));
    qbuf_ = dmalloc<float>((size_t)T * c.n_head * c.head_dim);
    const int ms = attn_max_splits(opt_.ctx);
    part_o_ = dmalloc<float>((size_t)T * c.n_head * ms * c.head_dim);
    part_ml_ = dmalloc<float>((size_t)T * c.n_head * ms * 2);
    attn_out_ = dmalloc<float>((size_t)T * c.n_embd);
    rlog_ = dmalloc<float>((size_t)T * (c.n_expert + 1));
    shgu_ = dmalloc<float>((size_t)T * 2 * c.n_ff_shexp);
    shout_ = dmalloc<float>((size_t)T * c.n_embd);
    h_gu_ = dmalloc<float>((size_t)kMaxU * 2 * c.n_ff_exp);
    moe_gpu_ = dmalloc<float>((size_t)T * c.n_embd);
    logits_ = dmalloc<float>((size_t)T * c.n_vocab);
    d_sg_ = dmalloc<float>(T);
    cand_tmp_ = dmalloc<float>(128 * kCand * 2);
    d_nmiss_ = dmalloc<int32_t>(1);
    hits_ = dmalloc<HitList>(1);
    xq_ = alloc_act(T * c.n_embd);
    yq_ = alloc_act(T * c.n_head * c.head_dim);   // 4096: delta-net y and attention output
    shq_ = alloc_act(T * c.n_ff_shexp);
    hq_ = alloc_act(kMaxU * c.n_ff_exp);
    route_counts_ = dmalloc<uint32_t>((size_t)NL * c.n_expert);
    tok_counts_ = dmalloc<uint32_t>(c.n_vocab);

    kc_.assign(NL, nullptr);
    vc_.assign(NL, nullptr);
    conv_st_.assign(NL, nullptr);
    ssm_st_.assign(NL, nullptr);
    const int64_t conv_f = (int64_t)c.conv_dim() * (c.ssm_d_conv - 1);
    const int64_t ssm_f = (int64_t)c.ssm_n_vh * c.ssm_d_state * c.ssm_d_state;
    state_floats_ = (conv_f + ssm_f) * c.n_gdn_layers();
    state_arena_ = dmalloc<float>(state_floats_);
    ckpts_.resize(std::max(opt_.ckpt_slots, 1));
    for (auto& ck : ckpts_) {
        ck.arena = dmalloc<float>(state_floats_);
        ck.hin = dmalloc<float>((size_t)kMaxT * c.n_embd);
    }
    int64_t off = 0;
    for (int il = 0; il < NL; ++il) {
        if (c.is_attn(il)) {
            kc_[il] = dmalloc<uint16_t>((size_t)c.n_head_kv * opt_.ctx * c.head_dim);
            vc_[il] = dmalloc<uint16_t>((size_t)c.n_head_kv * opt_.ctx * c.head_dim);
        } else {
            conv_st_[il] = state_arena_ + off;
            off += conv_f;
            ssm_st_[il] = state_arena_ + off;
            off += ssm_f;
        }
    }
    mtp_hin_ = dmalloc<float>((size_t)kMaxT * c.n_embd);
    if (mtp_on_) {
        mtp_hlast_ = dmalloc<float>(c.n_embd);
        mtp_logits_ = dmalloc<float>(c.n_vocab);
        mtp_act_ = alloc_act(T * 2 * c.n_embd);
        snap_arena_ = dmalloc<float>((size_t)opt_.mtp_draft * state_floats_);
    }
    const double kv_gib = (double)(c.n_attn_layers() + c.n_mtp) * 2 * c.n_head_kv * opt_.ctx * c.head_dim * 2 / 1073741824.0;
    log("KV cache %.2f GiB (ctx %d), delta-net state %.0f MiB%s", kv_gib, opt_.ctx, state_floats_ * 4 / 1048576.0,
        mtp_on_ ? " (+ rollback snapshots)" : "");
}

// ================================================================================================= expert cache
void Engine::setup_cache() {
    const Config& c = model_.cfg;
    const int NL = c.n_moe_layers(), NE = c.n_expert;
    freq_.assign((size_t)NL * NE, 0.0);
    counts_seen_.assign((size_t)NL * NE, 0);
    route_host_.assign((size_t)NL * NE, 0);
    prior_.assign((size_t)NL * NE, 0.0f);
    bool have_profile = false;
    if (!opt_.profile_path.empty()) {
        std::ifstream f(opt_.profile_path, std::ios::binary);
        uint32_t hdr[3] = {0, 0, 0};
        // profiles written without / with the MTP layer both fit: the layers they share are used
        if (f && f.read((char*)hdr, sizeof(hdr)) && hdr[0] == kProfileMagic && (int)hdr[1] >= c.n_layer &&
            (int)hdr[1] <= c.n_layer + 1 && (int)hdr[2] == NE) {
            const int FL = (int)hdr[1];
            std::vector<float> fr((size_t)FL * NE);
            if (f.read((char*)fr.data(), fr.size() * 4)) {
                // The profile is a long-run count; scale each layer to profile_weight times the mass the decayed live
                // counts settle at (n_expert_used * adapt_every / (1 - decay)), so that it is a prior the current
                // conversation can override within a few hundred tokens instead of freezing the cache.
                const double live = (double)c.n_expert_used * std::max(opt_.adapt_every, 1) / (1.0 - kDecay);
                for (int l = 0; l < std::min(NL, FL); ++l) {
                    double sum = 0;
                    for (int e = 0; e < NE; ++e) sum += fr[(size_t)l * NE + e];
                    const double k = sum > 0 ? opt_.profile_weight * live / sum : 0.0;
                    for (int e = 0; e < NE; ++e) {
                        const size_t i = (size_t)l * NE + e;
                        prior_[i] = fr[i];
                        freq_[i] = fr[i] * k;
                    }
                }
                have_profile = true;
                log("expert profile: %s", opt_.profile_path.c_str());
            }
        }
        if (!have_profile) log("expert profile %s not usable; starting from a uniform cache", opt_.profile_path.c_str());
    }
    size_t fr = 0, tot = 0;
    CUDA_CHECK(cudaMemGetInfo(&fr, &tot));
    // kInitOverhead: what the warm-up and graph capture still allocate after this point (init() trims the cache if
    // it turns out to be more).  Without it that memory would come out of the reserve, and under WDDM an
    // over-full card spills to system memory instead of failing.
    constexpr int64_t kInitOverhead = 768ll << 20;
    int64_t budget = (int64_t)fr - opt_.vram_reserve_mb * 1048576ll - kInitOverhead;
    if (opt_.cache_mb >= 0) budget = std::min<int64_t>(budget, opt_.cache_mb * 1048576ll);
    budget = std::max<int64_t>(budget, 0);

    // Rank (layer, expert) by frequency; take greedily while the bytes fit.  Without a profile every layer gets
    // the same share (frequencies are all zero; the tie-break spreads the slots across layers).
    std::vector<int> order((size_t)NL * NE);
    std::iota(order.begin(), order.end(), 0);
    std::vector<int> rank_in_layer((size_t)NL * NE);
    for (int i = 0; i < NL * NE; ++i) rank_in_layer[i] = i % NE;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (freq_[a] != freq_[b]) return freq_[a] > freq_[b];
        return rank_in_layer[a] < rank_in_layer[b];
    });
    slots_per_layer_.assign(NL, 0);
    int64_t used = 0;
    std::vector<std::vector<int>> chosen(NL);
    for (int idx : order) {
        const int l = idx / NE;
        const int64_t b = model_.layers[l].blob_bytes;
        if (used + b > budget) continue;
        used += b;
        chosen[l].push_back(idx % NE);
    }
    residency_host_.assign((size_t)NL * NE, -1);
    slot_base_.assign(NL, nullptr);
    slot_expert_.assign(NL, {});
    std::vector<CacheLayerInfo> ci(NL);
    cache_slots_total_ = 0;
    cache_bytes_ = 0;
    for (int l = 0; l < NL; ++l) {
        const int n = (int)chosen[l].size();
        slots_per_layer_[l] = n;
        const int64_t b = model_.layers[l].blob_bytes;
        if (n > 0) {
            cudaError_t e = cudaMalloc(&slot_base_[l], (size_t)(b * n));
            if (e != cudaSuccess) {   // fragmentation or another program took memory: shrink this layer
                cudaGetLastError();
                slots_per_layer_[l] = 0;
                chosen[l].clear();
                continue;
            }
        }
        ci[l] = CacheLayerInfo{slot_base_[l], (long long)b};
        for (int s = 0; s < n; ++s) {
            const int e = chosen[l][s];
            residency_host_[(size_t)l * NE + e] = s;
            slot_expert_[l].push_back(e);
            CUDA_CHECK(cudaMemcpyAsync(slot_base_[l] + s * b, model_.layers[l].expert_blob(e), (size_t)b,
                                       cudaMemcpyHostToDevice, stream_));
        }
        cache_slots_total_ += n;
        cache_bytes_ += b * n;
    }
    residency_dev_ = dmalloc<int32_t>((size_t)NL * NE);
    cinfo_dev_ = dmalloc<CacheLayerInfo>(NL);
    CUDA_CHECK(cudaMemcpyAsync(residency_dev_, residency_host_.data(), residency_host_.size() * 4, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(cinfo_dev_, ci.data(), ci.size() * sizeof(CacheLayerInfo), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Engine::apply_swap(int layer, int slot, int expert) {
    const int NE = model_.cfg.n_expert;
    const int64_t b = model_.layers[layer].blob_bytes;
    const int old = slot_expert_[layer][slot];
    residency_host_[(size_t)layer * NE + old] = -1;
    residency_host_[(size_t)layer * NE + expert] = slot;
    slot_expert_[layer][slot] = expert;
    CUDA_CHECK(cudaMemcpyAsync(slot_base_[layer] + slot * b, model_.layers[layer].expert_blob(expert), (size_t)b,
                               cudaMemcpyHostToDevice, stream_));
    ++stats.swaps;
}

// Shrinks the cache in place: the coldest resident experts go, and each layer that loses slots gets a smaller slab
// (freed first, so this works on a full card) with its remaining experts uploaded again from host memory.  Only
// between requests: nothing may be in flight on the GPU or the CPU workers.
void Engine::trim_cache(int64_t bytes) {
    const Config& c = model_.cfg;
    const int NL = c.n_moe_layers(), NE = c.n_expert;
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
    std::vector<int> res;   // resident (layer, expert), coldest first
    for (int l = 0; l < NL; ++l)
        for (int e : slot_expert_[l]) res.push_back(l * NE + e);
    std::stable_sort(res.begin(), res.end(), [&](int a, int b) { return freq_[a] < freq_[b]; });
    std::vector<char> drop((size_t)NL * NE, 0), touched(NL, 0);
    int64_t freed = 0;
    int dropped = 0;
    for (int idx : res) {
        if (freed >= bytes) break;
        drop[idx] = 1;
        touched[idx / NE] = 1;
        freed += model_.layers[idx / NE].blob_bytes;
        ++dropped;
    }
    for (int l = 0; l < NL; ++l) {
        if (!touched[l]) continue;
        const int64_t b = model_.layers[l].blob_bytes;
        std::vector<int> keep;
        for (int e : slot_expert_[l]) {
            residency_host_[(size_t)l * NE + e] = -1;
            if (!drop[(size_t)l * NE + e]) keep.push_back(e);
        }
        CUDA_CHECK(cudaFree(slot_base_[l]));
        slot_base_[l] = nullptr;
        cache_bytes_ -= b * slots_per_layer_[l];
        cache_slots_total_ -= slots_per_layer_[l];
        if (!keep.empty() && cudaMalloc(&slot_base_[l], (size_t)(b * keep.size())) != cudaSuccess) {
            cudaGetLastError();
            slot_base_[l] = nullptr;
            keep.clear();
        }
        slot_expert_[l] = keep;
        slots_per_layer_[l] = (int)keep.size();
        for (int s = 0; s < (int)keep.size(); ++s) {
            residency_host_[(size_t)l * NE + keep[s]] = s;
            CUDA_CHECK(cudaMemcpyAsync(slot_base_[l] + s * b, model_.layers[l].expert_blob(keep[s]), (size_t)b,
                                       cudaMemcpyHostToDevice, stream_));
        }
        cache_bytes_ += b * slots_per_layer_[l];
        cache_slots_total_ += slots_per_layer_[l];
    }
    std::vector<CacheLayerInfo> ci(NL);
    for (int l = 0; l < NL; ++l) ci[l] = CacheLayerInfo{slot_base_[l], (long long)model_.layers[l].blob_bytes};
    CUDA_CHECK(cudaMemcpyAsync(residency_dev_, residency_host_.data(), residency_host_.size() * 4, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(cinfo_dev_, ci.data(), ci.size() * sizeof(CacheLayerInfo), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    log("expert cache trimmed by %d slots (%.0f MiB) to keep %lld MiB of VRAM free", dropped, freed / 1048576.0,
        (long long)opt_.vram_reserve_mb);
}

void Engine::adapt_cache() {
    const Config& c = model_.cfg;
    const int NL = c.n_moe_layers(), NE = c.n_expert;
    CUDA_CHECK(cudaMemcpyAsync(route_host_.data(), route_counts_, route_host_.size() * 4, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    // Exponentially decayed routing frequency: the profile fades, the conversation takes over.
    for (size_t i = 0; i < freq_.size(); ++i) {
        const uint32_t d = route_host_[i] - counts_seen_[i];
        counts_seen_[i] = route_host_[i];
        freq_[i] = freq_[i] * kDecay + d;
    }
    // Candidate swaps: across layers, the hottest non-resident expert vs its layer's coldest resident one.
    struct Cand { double gain; int layer, slot, expert; };
    std::vector<Cand> cands;
    for (int l = 0; l < NL; ++l) {
        const int n = slots_per_layer_[l];
        if (n == 0) continue;
        std::vector<int> res(slot_expert_[l]);
        std::vector<int> non;
        for (int e = 0; e < NE; ++e)
            if (residency_host_[(size_t)l * NE + e] < 0) non.push_back(e);
        const double* F = freq_.data() + (size_t)l * NE;
        std::sort(non.begin(), non.end(), [&](int a, int b) { return F[a] > F[b]; });
        std::vector<int> slots(n);
        std::iota(slots.begin(), slots.end(), 0);
        std::sort(slots.begin(), slots.end(), [&](int a, int b) { return F[slot_expert_[l][a]] < F[slot_expert_[l][b]]; });
        for (size_t k = 0; k < non.size() && k < slots.size() && k < 8; ++k) {
            const double gain = F[non[k]] - F[slot_expert_[l][slots[k]]];
            if (gain <= 1.0) break;   // hysteresis: only clear wins
            cands.push_back({gain, l, slots[k], non[k]});
        }
    }
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.gain > b.gain; });
    int done = 0;
    for (const Cand& cd : cands) {
        if (done >= opt_.adapt_swaps) break;
        if (residency_host_[(size_t)cd.layer * NE + cd.expert] >= 0) continue;
        apply_swap(cd.layer, cd.slot, cd.expert);
        ++done;
    }
    if (done > 0)
        CUDA_CHECK(cudaMemcpyAsync(residency_dev_, residency_host_.data(), residency_host_.size() * 4, cudaMemcpyHostToDevice, stream_));
    // the residency upload reads pageable host memory: make it complete before the table changes again
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

bool Engine::save_profile(const std::string& path, std::string& err) {
    const Config& c = model_.cfg;
    const int NL = c.n_moe_layers();
    CUDA_CHECK(cudaMemcpy(route_host_.data(), route_counts_, route_host_.size() * 4, cudaMemcpyDeviceToHost));
    // Undecayed: the loaded profile plus every routing decision since start-up (the warm-up counts are cleared).
    // Each layer is capped at kProfileMaxPerLayer so that long use keeps the profile a moving average.
    std::vector<float> fr(freq_.size());
    constexpr double kProfileMaxPerLayer = 1e8;
    for (int l = 0; l < NL; ++l) {
        double sum = 0;
        for (int e = 0; e < c.n_expert; ++e) {
            const size_t i = (size_t)l * c.n_expert + e;
            sum += (double)prior_[i] + route_host_[i];
        }
        const double k = sum > kProfileMaxPerLayer ? kProfileMaxPerLayer / sum : 1.0;
        for (int e = 0; e < c.n_expert; ++e) {
            const size_t i = (size_t)l * c.n_expert + e;
            fr[i] = (float)(((double)prior_[i] + route_host_[i]) * k);
        }
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) { err = "cannot write " + path; return false; }
    const uint32_t hdr[3] = {kProfileMagic, (uint32_t)NL, (uint32_t)c.n_expert};
    f.write((const char*)hdr, sizeof(hdr));
    f.write((const char*)fr.data(), fr.size() * 4);
    return true;
}

// ================================================================================================= decode
void Engine::build_layer(cudaStream_t s, int il, int T, const StepParams* P, const float* next_w, float* snap) {
    const Config& c = model_.cfg;
    const float eps = c.eps;
    const LayerW& L = model_.layers[il];
    if (!L.attn) {
        float* sc = snap ? snap + (conv_st_[il] - state_arena_) : nullptr;
        float* ss = snap ? snap + (ssm_st_[il] - state_arena_) : nullptr;
        k_gemv_q8(L.qkvz, xq_, qkvz_, T, s);
        k_gemv_f32(L.ba, xn_, ba_, T, s);
        k_gdn_conv(qkvz_, conv_st_[il], L.conv_w, conv_out_, T, sc, state_floats_, s);
        k_gdn_recur(conv_out_, ba_, L.dt_bias, L.ssm_a, ssm_st_[il], o_, T, ss, state_floats_, s);
        k_gdn_norm_gate(o_, qkvz_, L.ssm_norm, eps, yq_, T, s);
        k_gemv_q8(L.ssm_out, yq_, attn_out_, T, s);
    } else {
        k_gemv_q8(L.qkv, xq_, qkv_, T, s);
        k_attn_prep(qkv_, L.q_norm, L.k_norm, eps, P, c.rope_base, qbuf_, kc_[il], vc_[il], opt_.ctx, T, s);
        k_attn_decode(qbuf_, kc_[il], vc_[il], P, opt_.ctx, part_o_, part_ml_, T, s);
        k_attn_combine(part_o_, part_ml_, qkv_, P, opt_.ctx, yq_, T, s);
        k_gemv_q8(L.wo, yq_, attn_out_, T, s);
    }
    k_add_rmsnorm_q(x_, attn_out_, L.post_norm, eps, xn_, xq_, T, s);
    k_gemv_f32(L.router, xn_, rlog_, T, s);
    k_router(rlog_, il, residency_dev_, cinfo_dev_, xn_, hits_, mb_dev_ + il, P, d_nmiss_, d_sg_, route_counts_, ecount_, T, s);
    k_gemv_q8(L.sh_gu, xq_, shgu_, T, s);
    k_swiglu_q(shgu_, shq_, c.n_ff_shexp, T, s);
    k_gemv_q8(L.sh_down, shq_, shout_, T, s);
    k_moe_gu(L.t_gu, xq_, hits_, h_gu_, T, s);
    k_moe_act(h_gu_, hits_, hq_, T, s);
    k_moe_down(L.t_down, 2 * L.gu_bytes, hq_, hits_, moe_gpu_, T, s);
    k_combine(x_, moe_gpu_, shout_, d_sg_, d_nmiss_, res_dev_ + il, P, next_w, eps, xn_, xq_, d_err_, wait_ns_, T, s);
    if (debug_layers && il < c.n_layer) {
        std::vector<float> h((size_t)c.n_embd * 3);
        CUDA_CHECK(cudaMemcpyAsync(h.data(), x_, (size_t)c.n_embd * 4, cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaMemcpyAsync(h.data() + c.n_embd, attn_out_, (size_t)c.n_embd * 4, cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaMemcpyAsync(h.data() + 2 * c.n_embd, moe_gpu_, (size_t)c.n_embd * 4, cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        debug_layers->push_back(std::move(h));
    }
}

void Engine::build_decode(cudaStream_t s, int T) {
    const Config& c = model_.cfg;
    CUDA_CHECK(cudaMemcpyAsync(params_dev_, params_host_, step_params_bytes(T), cudaMemcpyHostToDevice, s));
    k_embed_in(params_dev_, x_, T, s);
    k_rmsnorm_q(x_, model_.layers[0].attn_norm, c.eps, xn_, xq_, T, s);
    // a verification step keeps the delta-net state after each of its tokens, for the rollback
    float* snap = T > 1 ? snap_arena_ : nullptr;
    for (int il = 0; il < c.n_layer; ++il) {
        const float* next_w = il + 1 < c.n_layer ? model_.layers[il + 1].attn_norm : model_.output_norm;
        build_layer(s, il, T, params_dev_, next_w, snap);
    }
    k_gemv_q8(model_.lm_head, xq_, logits_, T, s);
    if (T > 1) k_argmax(logits_, c.n_vocab, out_dev_, T, s);
    // the normed final hidden states seed the MTP pairs of these positions
    if (mtp_on_) CUDA_CHECK(cudaMemcpyAsync(mtp_hin_, xn_, (size_t)T * c.n_embd * 4, cudaMemcpyDeviceToDevice, s));
}

void Engine::build_mtp(cudaStream_t s, int T) {
    const Config& c = model_.cfg;
    const MtpW& M = model_.mtp;
    const int il = c.n_layer;
    CUDA_CHECK(cudaMemcpyAsync(mparams_dev_, mparams_host_, step_params_bytes(T), cudaMemcpyHostToDevice, s));
    k_mtp_in(mparams_dev_, mtp_hin_, M.enorm, M.hnorm, c.eps, mtp_act_, T, s);
    k_gemv_q8(M.eh_proj, mtp_act_, x_, T, s);
    k_rmsnorm_q(x_, model_.layers[il].attn_norm, c.eps, xn_, xq_, T, s);
    build_layer(s, il, T, mparams_dev_, M.head_norm, nullptr);
    // only the last pair drafts: its normed hidden state seeds the next (recursive) draft
    CUDA_CHECK(cudaMemcpyAsync(mtp_hlast_, xn_ + (size_t)(T - 1) * c.n_embd, (size_t)c.n_embd * 4, cudaMemcpyDeviceToDevice, s));
    k_gemv_q8(model_.lm_head, xq_.row(T - 1, c.n_embd), mtp_logits_, 1, s);
    k_argmax(mtp_logits_, c.n_vocab, out_dev_ + kMaxT, 1, s, prob_dev_);
}

void Engine::run_step(bool mtp, int T) {
    const Config& c = model_.cfg;
    StepParams& P = mtp ? *mparams_host_ : *params_host_;
    P.seq = ++seq_;
    P.n_tok = T;
    if (mtp) cpu_->begin_step(seq_, c.n_layer, c.n_layer + 1);
    else cpu_->begin_step(seq_, 0, c.n_layer);
    cudaGraphExec_t g = mtp ? mtp_graph_[T] : graph_[T];
    if (g) CUDA_CHECK(cudaGraphLaunch(g, stream_));
    else if (mtp) build_mtp(stream_, T);
    else build_decode(stream_, T);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    const bool cpu_ok = cpu_->end_step();
    if (!cpu_ok || tok_host_[1] != 0) {
        std::string lay;
        for (int l = 0; l < c.n_moe_layers(); ++l) {
            char b[64];
            std::snprintf(b, sizeof(b), " L%d:r%u/d%u/n%d", l, mb_host_[l].ready_seq, res_host_[l].done_seq,
                          mb_host_[l].n_miss);
            lay += b;
        }
        die("CPU/GPU expert hand-off failed (seq %u, %s step of %d, cpu %s, gpu flag %d):%s", seq_, mtp ? "MTP" : "trunk", T,
            cpu_ok ? "ok" : "timed out", tok_host_[1], lay.c_str());
    }
    if (!mtp) ++stats.graph_steps;
}

void Engine::decode_one(int token) {
    SQ_CHECK(n_past_ < opt_.ctx, "context full (%d tokens)", opt_.ctx);
    // the pending MTP pairs end with this token: fill their MTP KV entries before the trunk moves on
    if (mtp_on_ && mtp_k_ > 0) {
        const double tm = now_ms();
        mtp_catch_up(token);
        stats.mtp_ms += now_ms() - tm;
    }
    StepParams& P = *params_host_;
    P.pos = n_past_;
    P.token[0] = token;
    model_.embed(token, P.emb[0]);
    const double t0 = now_ms();
    run_step(false, 1);
    const double dt = now_ms() - t0;
    stats.decode_ms += dt;
    if (stats.step_ms.size() < 4096) stats.step_ms.push_back((float)dt);
    else stats.step_ms[stats.decode_steps % 4096] = (float)dt;
    ++stats.decode_steps;
    ++n_past_;
    history_.push_back(token);
    logits_valid_ = true;
    if (mtp_on_) mtp_k_ = 1;
    if (opt_.adapt_every > 0 && cache_slots_total_ > 0 && ++steps_since_adapt_ >= opt_.adapt_every) {
        steps_since_adapt_ = 0;
        const double ta = now_ms();
        adapt_cache();
        stats.adapt_ms += now_ms() - ta;
    }
}

void Engine::decode_multi(const int* tokens, int T) {
    SQ_CHECK(n_past_ + T <= opt_.ctx, "context full (%d tokens)", opt_.ctx);
    StepParams& P = *params_host_;
    P.pos = n_past_;
    for (int t = 0; t < T; ++t) {
        P.token[t] = tokens[t];
        model_.embed(tokens[t], P.emb[t]);
    }
    run_step(false, T);
}

int Engine::mtp_catch_up(int next) {
    const int k = mtp_k_;
    SQ_CHECK(k >= 1 && k <= kMaxT, "MTP: %d pending pairs", k);
    const int q = n_past_ - k;
    StepParams& P = *mparams_host_;
    P.pos = q;
    for (int j = 0; j < k; ++j) {
        // pair j: hidden state of position q + j, token of position q + j + 1
        const int t = q + j + 1 < n_past_ ? history_[q + j + 1] : next;
        P.token[j] = t;
        model_.embed(t, P.emb[j]);
    }
    run_step(true, k);
    mtp_k_ = 0;
    return out_host_[kMaxT];
}

int Engine::mtp_recur(int prev_draft, int pos) {
    CUDA_CHECK(cudaMemcpyAsync(mtp_hin_, mtp_hlast_, (size_t)model_.cfg.n_embd * 4, cudaMemcpyDeviceToDevice, stream_));
    StepParams& P = *mparams_host_;
    P.pos = pos;
    P.token[0] = prev_draft;
    model_.embed(prev_draft, P.emb[0]);
    run_step(true, 1);
    return out_host_[kMaxT];
}

void Engine::spec_step(int tok, const SamplingParams& sp, std::mt19937_64& rng, std::vector<int>& out) {
    out.clear();
    const Config& c = model_.cfg;
    int D = std::min(mtp_draft(), opt_.ctx - n_past_ - 1);
    if (D <= 0 || mtp_k_ <= 0) {
        feed(&tok, 1);
        out.push_back(sample(sp, rng));
        return;
    }
    const double t0 = now_ms();
    // ---- draft: the pending pairs (ending with tok) give the first draft, the MTP's own hidden state the others.
    // Drafting stops once the MTP's own probability that all drafts so far are right drops below draft_p: a draft
    // that is likely rejected costs a verification slot (and its CPU experts) for nothing.
    int toks[kMaxT];
    toks[0] = tok;
    toks[1] = mtp_catch_up(tok);
    double conf = *prob_host_;
    int nd = 1;
    if (conf < 0.5 * opt_.draft_p) nd = 0;   // not even the first one: a plain step
    for (int i = 2; i <= D && nd == i - 1 && conf >= opt_.draft_p; ++i) {
        toks[i] = mtp_recur(toks[i - 1], n_past_ + i - 2);
        conf *= *prob_host_;
        nd = i;
    }
    const double t1 = now_ms();
    stats.mtp_ms += t1 - t0;
    if (nd == 0) {   // the catch-up is done (mtp_k_ = 0): this feeds tok alone
        ++stats.spec_skipped;
        feed(&tok, 1);
        out.push_back(sample(sp, rng));
        return;
    }
    D = nd;
    // ---- verify [tok, drafts] in one trunk step
    const int T = D + 1;
    decode_multi(toks, T);
    stats.verify_ms += now_ms() - t1;
    // ---- accept: row i of the logits predicts the token after toks[i]
    int a = 0;         // tokens of the step kept (tok + accepted drafts)
    int bonus = -1;    // the trunk's own token after them
    const bool greedy = sp.greedy();
    for (int i = 0; i < T; ++i) {
        a = i + 1;
        int t;
        bool acc = false;
        if (greedy) {
            t = out_host_[i];
            acc = i < D && t == toks[i + 1];
        } else {
            SamplePen pen{sp.presence_penalty, sp.frequency_penalty, sp.repetition_penalty};
            k_candidates(logits_ + (size_t)i * c.n_vocab, c.n_vocab, tok_counts_, pen, cand_tmp_, cand_dev_, stream_);
            CUDA_CHECK(cudaStreamSynchronize(stream_));
            t = i < D ? sample_speculative(*cand_host_, sp, rng, toks[i + 1], acc) : sample_candidates(*cand_host_, sp, rng);
            // count it for the penalties of the following rows (in stream order before their candidates; the id
            // is read from mapped memory, which is rewritten only after the next synchronisation)
            count_sampled(t);
        }
        if (!acc) { bonus = t; break; }
    }
    commit_step(toks, a, T);
    for (int i = 1; i < a; ++i) out.push_back(toks[i]);
    out.push_back(bonus);

    const double dt = now_ms() - t0;
    stats.decode_ms += dt;
    for (int i = 0; i < a; ++i) {
        const float per = (float)(dt / a);
        if (stats.step_ms.size() < 4096) stats.step_ms.push_back(per);
        else stats.step_ms[(stats.decode_steps + i) % 4096] = per;
    }
    stats.decode_steps += a;
    ++stats.spec_steps;
    stats.drafted += D;
    stats.accepted += a - 1;
    steps_since_adapt_ += a;
    if (opt_.adapt_every > 0 && cache_slots_total_ > 0 && steps_since_adapt_ >= opt_.adapt_every) {
        steps_since_adapt_ = 0;
        const double ta = now_ms();
        adapt_cache();
        stats.adapt_ms += now_ms() - ta;
    }
}

void Engine::commit_step(const int* toks, int a, int T) {
    const Config& c = model_.cfg;
    // roll the delta net back to the last kept token; logits row 0 = the last kept token's
    if (a < T)
        CUDA_CHECK(cudaMemcpyAsync(state_arena_, snap_arena_ + (size_t)(a - 1) * state_floats_, state_floats_ * 4,
                                   cudaMemcpyDeviceToDevice, stream_));
    if (a > 1)
        CUDA_CHECK(cudaMemcpyAsync(logits_, logits_ + (size_t)(a - 1) * c.n_vocab, (size_t)c.n_vocab * 4,
                                   cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    n_past_ += a;
    history_.insert(history_.end(), toks, toks + a);
    logits_valid_ = true;
    mtp_k_ = a;   // the step copied the kept tokens' hidden states to mtp_hin_[0 .. a)
}

std::vector<float> Engine::debug_verify(const int* tokens, int T, int keep) {
    SQ_CHECK(mtp_on_ && T >= 2 && T <= opt_.mtp_draft + 1 && keep >= 1 && keep <= T, "debug_verify: bad arguments");
    const size_t V = model_.cfg.n_vocab;
    if (mtp_k_ > 0) mtp_catch_up(tokens[0]);
    decode_multi(tokens, T);
    std::vector<float> h(V * T);
    CUDA_CHECK(cudaMemcpy(h.data(), logits_, h.size() * 4, cudaMemcpyDeviceToHost));
    commit_step(tokens, keep, T);
    return h;
}

void Engine::feed(const int* tokens, int n) {
    if (n <= 0) return;
    SQ_CHECK(n_past_ + n <= opt_.ctx, "prompt exceeds the context (%d + %d > %d)", n_past_, n, opt_.ctx);
    if (n >= opt_.prefill_min && prefill_ && prefill_->ready()) {
        const double t0 = now_ms();
        prefill_->run(tokens, n);   // also runs the MTP layer over the new positions (and leaves one pair pending)
        stats.prefill_ms += now_ms() - t0;
        stats.prefill_tokens += n;
        n_past_ += n;
        history_.insert(history_.end(), tokens, tokens + n);
        logits_valid_ = true;
        if (opt_.adapt_every > 0 && cache_slots_total_ > 0) adapt_cache();
        return;
    }
    for (int i = 0; i < n; ++i) decode_one(tokens[i]);
}

void Engine::reset() {
    CUDA_CHECK(cudaMemsetAsync(state_arena_, 0, state_floats_ * 4, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    n_past_ = 0;
    history_.clear();
    logits_valid_ = false;
    mtp_k_ = 0;
}

void Engine::begin_request(const SamplingParams& sp) {
    pen_n_ = sp.penalized() ? (sp.penalty_last_n > 0 ? sp.penalty_last_n : opt_.ctx) : 0;
    pen_hist_.clear();
    if (pen_n_ == 0) return;
    const int from = std::max(0, (int)history_.size() - pen_n_);
    pen_hist_.assign(history_.begin() + from, history_.end());
    std::vector<uint32_t> c((size_t)model_.cfg.n_vocab, 0);
    for (int t : pen_hist_) ++c[(size_t)t];
    CUDA_CHECK(cudaMemcpyAsync(tok_counts_, c.data(), c.size() * 4, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Engine::count_sampled(int tok) {
    if (pen_n_ == 0) return;
    pen_hist_.push_back(tok);
    const int n = (int)pen_hist_.size();
    tok_host_[2] = tok;
    tok_host_[3] = n > pen_n_ ? pen_hist_[(size_t)(n - 1 - pen_n_)] : -1;
    k_count_token(tok_counts_, tok_dev_ + 2, stream_);
}

void Engine::expert_counts(uint64_t& resident, uint64_t& missing) const {
    unsigned long long ec[2] = {0, 0};
    CUDA_CHECK(cudaMemcpy(ec, ecount_, 16, cudaMemcpyDeviceToHost));
    resident = ec[0];
    missing = ec[1];
}

int Engine::sample(const SamplingParams& sp, std::mt19937_64& rng) {
    SQ_CHECK(logits_valid_, "sample() without logits");
    int tok;
    if (sp.greedy()) {
        k_argmax(logits_, model_.cfg.n_vocab, tok_dev_, 1, stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        tok = tok_host_[0];
    } else {
        SamplePen pen{sp.presence_penalty, sp.frequency_penalty, sp.repetition_penalty};
        k_candidates(logits_, model_.cfg.n_vocab, tok_counts_, pen, cand_tmp_, cand_dev_, stream_);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        tok = sample_candidates(*cand_host_, sp, rng);
    }
    // count it for the penalties: the kernel reads the id from mapped host memory
    count_sampled(tok);
    return tok;
}

// ================================================================================================= prefix reuse
static bool is_prefix(const std::vector<int>& p, const std::vector<int>& s) {
    return p.size() <= s.size() && std::equal(p.begin(), p.end(), s.begin());
}

int Engine::reuse_prefix(const std::vector<int>& prompt) {
    // Positions past the common prefix are about to be rewritten (KV caches), so a checkpoint that is not a prefix
    // of this prompt can no longer be restored.
    for (auto& ck : ckpts_)
        if (!ck.tokens.empty() && !is_prefix(ck.tokens, prompt)) ck.tokens.clear();
    // the live state works if it is a prefix (and, when it covers the whole prompt, still has its logits)
    int best = 0;
    if (!history_.empty() && is_prefix(history_, prompt) && (history_.size() < prompt.size() || logits_valid_))
        best = (int)history_.size();
    Checkpoint* pick = nullptr;
    for (auto& ck : ckpts_)
        if (!ck.tokens.empty() && (int)ck.tokens.size() > best && ck.tokens.size() < prompt.size()) {
            best = (int)ck.tokens.size();
            pick = &ck;
        }
    if (!pick) {
        if (best > 0) return best;
        reset();
        return 0;
    }
    CUDA_CHECK(cudaMemcpyAsync(state_arena_, pick->arena, state_floats_ * 4, cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(mtp_hin_, pick->hin, (size_t)kMaxT * model_.cfg.n_embd * 4, cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    pick->stamp = ++ckpt_clock_;
    history_ = pick->tokens;
    n_past_ = (int)history_.size();
    logits_valid_ = false;
    mtp_k_ = pick->mtp_k;
    return n_past_;
}

void Engine::save_checkpoint(bool base) {
    // slot 0 is the base slot when there are two or more; the others rotate (least recently used)
    const size_t first = ckpts_.size() > 1 && !base ? 1 : 0;
    for (auto& ck : ckpts_)
        if (ck.tokens == history_) { ck.stamp = ++ckpt_clock_; return; }
    Checkpoint* slot = &ckpts_[first];
    if (!base || ckpts_.size() == 1)
        for (size_t i = first; i < ckpts_.size(); ++i)
            if (ckpts_[i].stamp < slot->stamp) slot = &ckpts_[i];
    CUDA_CHECK(cudaMemcpyAsync(slot->arena, state_arena_, state_floats_ * 4, cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(slot->hin, mtp_hin_, (size_t)kMaxT * model_.cfg.n_embd * 4, cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    slot->tokens = history_;
    slot->mtp_k = mtp_k_;
    slot->stamp = ++ckpt_clock_;
}

std::string Engine::status_line() const {
    char buf[640];
    unsigned long long wait = 0, ec[2] = {0, 0};
    CUDA_CHECK(cudaMemcpy(&wait, wait_ns_, 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ec, ecount_, 16, cudaMemcpyDeviceToHost));
    const double hr = ec[0] + ec[1] ? 100.0 * ec[0] / (ec[0] + ec[1]) : 0.0;
    const double toks = std::max<double>((double)stats.decode_steps, 1.0);
    std::vector<float> st = stats.step_ms;
    double median = 0;
    if (!st.empty()) {
        std::nth_element(st.begin(), st.begin() + st.size() / 2, st.end());
        median = st[st.size() / 2];
    }
    int n = std::snprintf(buf, sizeof(buf),
                          "decode %.1f tok/s (median %.1f), prefill %.0f tok/s, cache hit %.1f%%, swaps %llu "
                          "(per token: %.2f ms, GPU waits for CPU %.2f ms, cache adapt %.3f ms)",
                          stats.decode_ms > 0 ? stats.decode_steps * 1000.0 / stats.decode_ms : 0.0,
                          median > 0 ? 1000.0 / median : 0.0,
                          stats.prefill_ms > 0 ? stats.prefill_tokens * 1000.0 / stats.prefill_ms : 0.0, hr,
                          (unsigned long long)stats.swaps, stats.decode_ms / toks, wait * 1e-6 / toks, stats.adapt_ms / toks);
    if (stats.spec_steps > 0 && n > 0 && n < (int)sizeof(buf))
        std::snprintf(buf + n, sizeof(buf) - n, "; MTP: %.1f%% of drafts accepted, %.2f tokens per step (%.1f%% of steps without a draft), "
                      "drafting %.2f ms + verifying %.2f ms per step",
                      stats.drafted ? 100.0 * stats.accepted / stats.drafted : 0.0,
                      1.0 + (double)stats.accepted / stats.spec_steps,
                      100.0 * stats.spec_skipped / (stats.spec_steps + stats.spec_skipped),
                      stats.mtp_ms / (stats.spec_steps + stats.spec_skipped), stats.verify_ms / stats.spec_steps);
    return buf;
}

std::vector<float> Engine::debug_hidden() {
    std::vector<float> h(model_.cfg.n_embd);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    CUDA_CHECK(cudaMemcpy(h.data(), x_, h.size() * 4, cudaMemcpyDeviceToHost));
    return h;
}

std::vector<float> Engine::logits_host() {
    std::vector<float> h(model_.cfg.n_vocab);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    CUDA_CHECK(cudaMemcpy(h.data(), logits_, h.size() * 4, cudaMemcpyDeviceToHost));
    return h;
}

}  // namespace sq
