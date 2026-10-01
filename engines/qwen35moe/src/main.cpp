// main.cpp - command line: benchmark / greedy generation from token ids / the --serve line protocol.
//
//   strata-qwen35moe -m model.gguf --ids 1,2,3 -n 64            greedy continuation, prints ids and speed
//   strata-qwen35moe -m model.gguf --bench                       prefill + decode benchmark on a synthetic prompt
//   strata-qwen35moe -m model.gguf --serve                       Strata's engine protocol (serve/server.py)
//   strata-qwen35moe -m model.gguf --ids 1,2,3 -n 64 --spec-check   greedy with and without MTP drafts must agree
//
// --serve speaks the protocol of Strata's own engine (src/program/generate.cpp), so serve/server.py, the web app
// and the tools drive this engine unchanged.  One line each way; logs go to stderr:
//   <- INFO key=value ...             facts for the Monitor tab (context, kv, expert_slots, ...), before READY
//   <- READY <context> stop            ready; "stop": a STOP line ends the running request
//   -> GEN <max_new> [key=value ...] <id,id,...>
//        keys: temperature top_p top_k min_p penalty_last_n penalty_repeat penalty_freq penalty_present seed;
//        unknown keys are skipped (the ids start at the first token without '=')
//   <- PP <position reached> <prompt tokens> <ms> <fresh tokens/s>   once per prompt slice
//   <- T <id>                          each generated token (an end-of-turn token is sent, then the request ends)
//   <- DONE <generated> <prompt> <prompt ms> <decode ms> <stop|length|cancel> <drafts accepted> <drafts offered>
//           <reused> <expert hits> <expert lookups> <RAM experts> <file experts> <file MB>
//   <- ERR <message>                   the request was refused
//   -> STOP                            end the running request (finish "cancel")
//   -> QUIT                            save the expert profile (--profile-save) and exit
//
// Prefix reuse: the delta-net state cannot be rolled back, so the engine checkpoints positions a follow-up request
// is likely to resume from - the end of the prompt, the start of its last turn (Qwen's template drops the reasoning
// of earlier turns, so the next request diverges right after it) and the end of the first (system) message.
#include "common.hpp"
#include "engine.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#ifndef SQ_VERSION
#define SQ_VERSION "0.0.0"
#endif

using namespace sq;

namespace {

// the qwen35 vocabulary's turn markers (Qwen3.5 / Qwen3.6 GGUFs; --eos-ids overrides the end-of-turn ids)
constexpr int kImStart = 248045, kImEnd = 248046, kEndOfText = 248044, kNewline = 198;

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
        if (!tok.empty()) v.push_back(std::stoi(tok));
    return v;
}

void usage() {
    std::fprintf(stderr,
        "usage: strata-qwen35moe -m MODEL.gguf [options]\n"
        "  --max-context N    context length (default 32768; also --ctx)\n"
        "  --threads N        CPU expert threads (default 8)\n"
        "  --vram-reserve-mib N  VRAM left free once ready (default 1200; also --reserve-mb)\n"
        "  --cache-mb N       cap the VRAM expert cache\n"
        "  --expert-profile FILE  expert routing profile (default data/expert-profile-qwen36.bin; also --profile)\n"
        "  --profile-save FILE    keep the profile plus this run's routing here (read instead of --expert-profile\n"
        "                     when it exists; written on QUIT and after requests)\n"
        "  --no-graph         launch kernels one by one (debug)\n"
        "  --no-adapt         keep the expert cache fixed\n"
        "  --adapt-every N --adapt-swaps M   re-rank the cache every N tokens, swapping up to M experts\n"
        "  --profile-weight W weight of the profile against live routing (default 1)\n"
        "  --prefill-chunk N  tokens per prefill pass (default 4096; smaller leaves more VRAM for experts)\n"
        "  --ckpt-slots N     prefix-reuse checkpoints (default 3, ~66 MiB of VRAM each)\n"
        "  --mtp-draft N      speculative decoding: up to N tokens drafted per step by the MTP block (default 3,\n"
        "                     max 3; 0 = off and the MTP block is not loaded; also --mtp)\n"
        "  --draft-p P        keep drafting while the MTP's probability of the drafts so far is >= P (default 0.8)\n"
        "  --eos-ids a,b      end-of-turn token ids for --serve (default 248044,248046)\n"
        "  --ids a,b,c|@file -n N   generation from token ids (greedy unless --temp)\n"
        "  --temp T --top-k K --top-p P --pres X --seed S   sampling for --ids / --bench\n"
        "  --dump-logits F    write the logits after --ids to F (float32)\n"
        "  --bench [--bench-prompt N] [-n N]\n"
        "  --selfcheck N      logits of a random N-token prompt: batched prefill vs prefill(N-1) + decode(1)\n"
        "  --spec-check       with --ids: greedy generation with MTP drafts vs without, token by token\n"
        "  --serve            Strata's engine protocol on stdin/stdout\n");
}

void emit(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stdout, fmt, ap);
    va_end(ap);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

struct ServeOptions {
    std::vector<int> eos = {kEndOfText, kImEnd};
    std::string profile_save;
};

/// The sampling keys of a GEN line, Strata's spelling.  `p` points after max_new; on return it points at the ids.
void parse_keys(const char*& p, SamplingParams& sp) {
    for (;;) {
        while (*p == ' ') ++p;
        const char* start = p;
        while (*p != '\0' && *p != ' ') ++p;
        if (p == start) return;
        const std::string tok(start, (size_t)(p - start));
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) { p = start; return; }
        const std::string k = tok.substr(0, eq);
        const char* v = tok.c_str() + eq + 1;
        if (k == "temperature") sp.temperature = std::strtof(v, nullptr);
        else if (k == "top_p") sp.top_p = std::strtof(v, nullptr);
        else if (k == "top_k") sp.top_k = std::atoi(v);
        else if (k == "min_p") sp.min_p = std::strtof(v, nullptr);
        else if (k == "penalty_last_n") sp.penalty_last_n = std::atoi(v);
        else if (k == "penalty_repeat") sp.repetition_penalty = std::strtof(v, nullptr);
        else if (k == "penalty_freq") sp.frequency_penalty = std::strtof(v, nullptr);
        else if (k == "penalty_present") sp.presence_penalty = std::strtof(v, nullptr);
        else if (k == "seed") sp.seed = std::strtoull(v, nullptr, 10);
        // pcie_frac, spec_min_p, cvec: settings of Strata's own engine that this one does not have
    }
}

int serve(Engine& eng, const ServeOptions& so) {
#ifdef _WIN32
    // Ctrl+C in the console reaches every process in it: the server (our parent) handles it and sends QUIT
    SetConsoleCtrlHandler(nullptr, TRUE);
#endif
    // stdin on its own thread: a STOP must be seen while a request runs
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop_req{false};
    std::thread([&] {
        std::string l;
        while (std::getline(std::cin, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") { stop_req.store(true); continue; }
            std::lock_guard<std::mutex> lk(mu);
            lines.push_back(l);
            cv.notify_one();
        }
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        cv.notify_one();
    }).detach();
    auto next_line = [&](std::string& out) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return !lines.empty() || eof; });
        if (lines.empty()) return false;
        out = std::move(lines.front());
        lines.pop_front();
        return true;
    };
    auto save_profile = [&] {
        if (so.profile_save.empty()) return;
        std::string err;
        if (!eng.save_profile(so.profile_save, err)) log("strata-qwen35moe: %s", err.c_str());
    };

    const int ctx = eng.ctx(), vocab = eng.cfg().n_vocab;
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    log("strata-qwen35moe: expert cache %lld slots, %.2f GiB; %lld MiB of VRAM free with everything loaded",
        (long long)eng.cache_slots(), eng.cache_gib(), (long long)(free_b >> 20));
    log("strata-qwen35moe: session is up");
    emit("INFO context=%d kv=fp16 kv_resident=0 expert_slots=%lld expert_cache_mib=%lld expert_slots_primary=%lld "
         "expert_cache_primary_mib=%lld spec=%d mtp_max=%d lookup=0 vram_free_mib=%lld arena_mib=%lld pool_workers=%d "
         "conversation_cache_slots=%d engine=" SQ_VERSION "+qwen35moe",
         ctx, (long long)eng.cache_slots(), (long long)(eng.cache_gib() * 1024), (long long)eng.cache_slots(),
         (long long)(eng.cache_gib() * 1024), eng.mtp_draft() + 1, eng.mtp_draft() + 1, (long long)(free_b >> 20),
         (long long)(eng.expert_arena_bytes() >> 20), eng.cpu_threads(), eng.ckpt_slots());
    emit("READY %d stop", ctx);
    using Clock = std::chrono::steady_clock;
    auto last_save = Clock::now();
    std::string line;
    while (next_line(line)) {
        if (line == "QUIT") break;
        stop_req.store(false);   // a STOP between requests is stale
        if (line == "STATS") { log("strata-qwen35moe: %s", eng.status_line().c_str()); continue; }
        if (line.rfind("GENI ", 0) == 0) { emit("ERR this engine was started without --vision"); continue; }
        if (line.rfind("GEN ", 0) != 0) { emit("ERR expected: GEN <max_new> <id,id,...>"); continue; }
        const char* p = line.c_str() + 4;
        char* endp = nullptr;
        const long long max_new = std::strtoll(p, &endp, 10);
        SamplingParams sp;
        sp.top_k = 20;       // the sampled path's default, as in Strata's engine (requests normally set it)
        sp.top_p = 1.0f;
        p = endp;
        parse_keys(p, sp);
        if (sp.top_k < 1 || sp.top_k > kCand) sp.top_k = kCand;
        std::vector<int> ids;
        bool bad = max_new < 1;
        try { ids = parse_ids(p); } catch (...) { bad = true; }
        if (bad || ids.empty()) { emit("ERR bad request: max_new or ids"); continue; }
        const int n = (int)ids.size();
        if (n + max_new > ctx) {
            emit("ERR prompt (%d tokens) + max_new (%lld) exceeds the context (%d)", n, max_new, ctx);
            continue;
        }
        if (std::any_of(ids.begin(), ids.end(), [&](int t) { return t < 0 || t >= vocab; })) {
            emit("ERR a token id is outside the vocabulary");
            continue;
        }
        std::mt19937_64 rng(sp.seed ? sp.seed : (uint64_t)Clock::now().time_since_epoch().count());
        uint64_t hits0 = 0, miss0 = 0;
        eng.expert_counts(hits0, miss0);
        const uint64_t acc0 = eng.stats.accepted, off0 = eng.stats.drafted;

        // ---- the prompt: what a checkpoint or the live state already holds, then the rest in slices
        const auto t0 = Clock::now();
        const int reused = eng.reuse_prefix(ids);
        std::vector<int> cks;
        int base_ck = -1;   // the end of the first (system) message: its own checkpoint slot
        auto add_ck = [&](int pos) { if (pos > reused && pos < n) cks.push_back(pos); };
        for (int i = n - 1; i >= 0; --i)
            if (ids[i] == kImStart) { add_ck(i); break; }
        if (ids[0] == kImStart)
            for (int i = 1; i < n; ++i)
                if (ids[i] == kImEnd) {
                    base_ck = i + 1 < n && ids[i + 1] == kNewline ? i + 2 : i + 1;
                    add_ck(base_ck);
                    break;
                }
        std::sort(cks.begin(), cks.end());
        int pos = reused;
        while (pos < n) {
            int end = std::min(pos + 4096, n);
            for (int c : cks)
                if (c > pos && c < end) { end = c; break; }
            eng.feed(ids.data() + pos, end - pos);
            pos = end;
            if (std::find(cks.begin(), cks.end(), pos) != cks.end()) eng.save_checkpoint(pos == base_ck);
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            emit("PP %d %d %.0f %.1f", pos, n, ms, ms > 0 ? 1000.0 * (pos - reused) / ms : 0.0);
        }
        if (reused < n) eng.save_checkpoint();
        const auto t1 = Clock::now();

        // ---- the answer: each round emits the tokens after the last one emitted (one sampled token, or with MTP the
        // accepted drafts and the trunk's next token); the last of them is fed by the next round
        eng.begin_request(sp);
        long long produced = 0;
        const char* finish = "length";
        std::vector<int> next{eng.sample(sp, rng)};
        for (bool done = false; !done;) {
            for (int tok : next) {
                emit("T %d", tok);
                ++produced;
                if (std::find(so.eos.begin(), so.eos.end(), tok) != so.eos.end()) { finish = "stop"; done = true; break; }
                if (produced >= max_new) { done = true; break; }
            }
            if (done) break;
            if (stop_req.load()) { finish = "cancel"; break; }
            eng.spec_step(next.back(), sp, rng, next);
        }
        const auto t2 = Clock::now();
        const double prompt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double decode_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        uint64_t hits = 0, miss = 0;
        eng.expert_counts(hits, miss);
        hits -= hits0;
        miss -= miss0;
        emit("DONE %lld %d %.1f %.1f %s %llu %llu %d %llu %llu %llu 0 0.0", produced, n, prompt_ms, decode_ms, finish,
             (unsigned long long)(eng.stats.accepted - acc0), (unsigned long long)(eng.stats.drafted - off0), reused,
             (unsigned long long)hits, (unsigned long long)(hits + miss), (unsigned long long)miss);
        log("strata-qwen35moe: prompt %d tokens = %d reused + %d read in %.0f ms (%.1f tok/s), %lld generated in %.0f ms "
            "(%.1f tok/s), drafts accepted %llu of %llu%s", n, reused, n - reused, prompt_ms,
            prompt_ms > 0 ? 1000.0 * (n - reused) / prompt_ms : 0.0, produced, decode_ms,
            decode_ms > 0 ? 1000.0 * produced / decode_ms : 0.0, (unsigned long long)(eng.stats.accepted - acc0),
            (unsigned long long)(eng.stats.drafted - off0), std::strcmp(finish, "cancel") == 0 ? " (cancelled)" : "");
        if (Clock::now() - last_save > std::chrono::seconds(60)) {
            save_profile();
            last_save = Clock::now();
        }
    }
    save_profile();
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    EngineOptions opt;
    opt.profile_path = "data/expert-profile-qwen36.bin";
    ServeOptions so;
    std::string ids_s, dump_logits;
    int n_gen = 32, bench_prompt = 512, selfcheck = 0;
    bool do_serve = false, do_bench = false, spec_check = false;
    SamplingParams sp;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(); die("missing value for %s", a.c_str()); }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") opt.model_path = next();
        else if (a == "--ctx" || a == "--max-context") opt.ctx = std::stoi(next());
        else if (a == "--threads") opt.cpu_threads = std::stoi(next());
        else if (a == "--reserve-mb" || a == "--vram-reserve-mib") opt.vram_reserve_mb = std::stoll(next());
        else if (a == "--cache-mb") opt.cache_mb = std::stoll(next());
        else if (a == "--profile" || a == "--expert-profile") opt.profile_path = next();
        else if (a == "--profile-save") so.profile_save = next();
        else if (a == "--ckpt-slots") opt.ckpt_slots = std::stoi(next());
        else if (a == "--eos-ids") so.eos = parse_ids(next());
        else if (a == "--no-graph") opt.use_graph = false;
        else if (a == "--no-adapt") opt.adapt_every = 0;
        else if (a == "--profile-weight") opt.profile_weight = std::stod(next());
        else if (a == "--adapt-every") opt.adapt_every = std::stoi(next());
        else if (a == "--adapt-swaps") opt.adapt_swaps = std::stoi(next());
        else if (a == "--prefill-chunk") opt.prefill_chunk = std::stoi(next());
        else if (a == "--no-pin") opt.pin_threads = false;
        else if (a == "--mtp" || a == "--mtp-draft") opt.mtp_draft = std::stoi(next());
        else if (a == "--no-mtp") opt.mtp_draft = 0;
        else if (a == "--draft-p") opt.draft_p = std::stof(next());
        else if (a == "--spec-check") spec_check = true;
        else if (a == "--ids") ids_s = next();
        else if (a == "-n") n_gen = std::stoi(next());
        else if (a == "--dump-logits") dump_logits = next();
        else if (a == "--serve") do_serve = true;
        else if (a == "--bench") do_bench = true;
        else if (a == "--bench-prompt") bench_prompt = std::stoi(next());
        else if (a == "--selfcheck") selfcheck = std::stoi(next());
        else if (a == "--temp") sp.temperature = std::stof(next());
        else if (a == "--top-k") sp.top_k = std::stoi(next());
        else if (a == "--top-p") sp.top_p = std::stof(next());
        else if (a == "--pres") sp.presence_penalty = std::stof(next());
        else if (a == "--seed") sp.seed = std::stoull(next());
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { usage(); die("unknown option %s", a.c_str()); }
    }
    if (opt.model_path.empty()) { usage(); return 1; }
    if (!so.profile_save.empty() && std::ifstream(so.profile_save).good()) opt.profile_path = so.profile_save;
    Engine eng;
    std::string err;
    if (!eng.init(opt, err)) die("%s", err.c_str());
    if (do_serve) return serve(eng, so);
    if (selfcheck > 1) {
        std::mt19937 rng(7);
        std::vector<int> p;
        for (int i = 0; i < selfcheck; ++i) p.push_back(1000 + (int)(rng() % 50000));
        eng.feed(p.data(), selfcheck);
        const auto a = eng.logits_host();
        eng.reset();
        eng.feed(p.data(), selfcheck - 1);
        eng.feed(p.data() + selfcheck - 1, 1);
        const auto b = eng.logits_host();
        double ma = -1e30, mb = -1e30, da = 0, db = 0, kl = 0, maxd = 0;
        for (size_t i = 0; i < a.size(); ++i) { ma = std::max(ma, (double)a[i]); mb = std::max(mb, (double)b[i]); }
        for (size_t i = 0; i < a.size(); ++i) { da += std::exp(a[i] - ma); db += std::exp(b[i] - mb); }
        size_t am = 0, bm = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double la = a[i] - ma - std::log(da), lb = b[i] - mb - std::log(db);
            kl += std::exp(la) * (la - lb);
            if (la > -10) maxd = std::max(maxd, std::abs(la - lb));
            if (a[i] > a[am]) am = i;
            if (b[i] > b[bm]) bm = i;
        }
        std::printf("selfcheck %d: KL(prefill||decode) = %.2e, max |dlogp| (p > e-10) = %.4f, argmax %zu / %zu\n", selfcheck,
                    kl, maxd, am, bm);
        const int T = eng.mtp_draft() + 1;
        if (T >= 2 && selfcheck > 2 * T) {
            // verification step vs one token at a time: rows of the T-token step, then a rollback to 1 kept token
            const int n0 = selfcheck - 2 * T;
            auto cmp = [&](const float* x, const std::vector<float>& y, const char* what) {
                double mx = -1e30, my = -1e30, sx = 0, sy = 0, k = 0, md = 0;
                size_t ax = 0, ay = 0;
                for (size_t i = 0; i < y.size(); ++i) { mx = std::max(mx, (double)x[i]); my = std::max(my, (double)y[i]); }
                for (size_t i = 0; i < y.size(); ++i) { sx += std::exp(x[i] - mx); sy += std::exp(y[i] - my); }
                for (size_t i = 0; i < y.size(); ++i) {
                    const double lx = x[i] - mx - std::log(sx), ly = y[i] - my - std::log(sy);
                    k += std::exp(lx) * (lx - ly);
                    md = std::max(md, std::abs((double)x[i] - y[i]));
                    if (x[i] > x[ax]) ax = i;
                    if (y[i] > y[ay]) ay = i;
                }
                std::printf("  %-26s KL %.2e, max |dlogit| %.2e, argmax %zu / %zu\n", what, k, md, ax, ay);
            };
            eng.reset();
            eng.feed(p.data(), n0);
            std::vector<std::vector<float>> seq;
            for (int t = 0; t < 2 * T; ++t) { eng.feed(p.data() + n0 + t, 1); seq.push_back(eng.logits_host()); }
            eng.reset();
            eng.feed(p.data(), n0);
            char w0[64];
            std::printf("single-token decoding, repeated:\n");
            for (int t = 0; t < 2; ++t) {
                eng.feed(p.data() + n0 + t, 1);
                std::snprintf(w0, sizeof(w0), "row %d", t);
                cmp(eng.logits_host().data(), seq[t], w0);
            }
            eng.reset();
            eng.feed(p.data(), n0);
            const auto v = eng.debug_verify(p.data() + n0, T, 1);   // keep 1: rolled back
            std::printf("verification step of %d tokens vs single-token decoding:\n", T);
            char w[64];
            for (int t = 0; t < T; ++t) {
                std::snprintf(w, sizeof(w), "row %d", t);
                cmp(v.data() + (size_t)t * seq[0].size(), seq[t], w);
            }
            const auto v2 = eng.debug_verify(p.data() + n0 + 1, T, T);   // after the rollback, keep all
            for (int t = 0; t < T; ++t) {
                std::snprintf(w, sizeof(w), "after rollback, row %d", t);
                cmp(v2.data() + (size_t)t * seq[0].size(), seq[1 + t], w);
            }
            eng.feed(p.data() + n0 + 1 + T, 1);
            cmp(eng.logits_host().data(), seq[1 + T], "then single decode");
            if (!opt.use_graph) {   // where the rows start to differ: token 0 after each layer
                std::vector<std::vector<float>> A, B;
                eng.reset();
                eng.feed(p.data(), n0);
                eng.debug_layers = &A;
                eng.feed(p.data() + n0, 1);
                eng.reset();
                eng.feed(p.data(), n0);
                eng.debug_layers = &B;
                eng.debug_verify(p.data() + n0, T, 1);
                eng.debug_layers = nullptr;
                for (size_t l = 0; l < std::min(A.size(), B.size()); ++l) {
                    double d[3] = {0, 0, 0};
                    for (int part = 0; part < 3; ++part)
                        for (int i = 0; i < 2048; ++i)
                            d[part] = std::max(d[part], (double)std::abs(A[l][part * 2048 + i] - B[l][part * 2048 + i]));
                    std::printf("  layer %2zu: max |dx| %.3e  |d mixer| %.3e  |d moe_gpu| %.3e\n", l, d[0], d[1], d[2]);
                }
            }
        }
        return 0;
    }

    if (!ids_s.empty() && ids_s[0] == '@') {   // --ids @file: comma-separated ids from a file (long prompts)
        std::ifstream f(ids_s.substr(1));
        if (!f) die("cannot read %s", ids_s.c_str() + 1);
        ids_s.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    std::vector<int> ids = parse_ids(ids_s);
    if (do_bench) {
        ids.clear();
        std::mt19937 rng(1);
        for (int i = 0; i < bench_prompt; ++i) ids.push_back(1000 + (int)(rng() % 50000));
    }
    if (ids.empty()) { usage(); return 1; }
    const double t0 = now_ms();
    eng.feed(ids.data(), (int)ids.size());
    const double t1 = now_ms();
    if (!dump_logits.empty()) {
        auto lg = eng.logits_host();
        FILE* f = std::fopen(dump_logits.c_str(), "wb");
        std::fwrite(lg.data(), 4, lg.size(), f);
        std::fclose(f);
    }
    std::mt19937_64 rng(sp.seed ? sp.seed : 1);
    std::vector<int> out;
    auto generate = [&] {
        out.clear();
        eng.begin_request(sp);
        std::vector<int> next{eng.sample(sp, rng)};
        for (;;) {
            for (int t : next)
                if ((int)out.size() < n_gen) out.push_back(t);
            if ((int)out.size() >= n_gen) break;
            eng.spec_step(next.back(), sp, rng, next);
        }
    };
    if (spec_check) {   // the same prompt without drafts, then with them: greedy outputs should agree
        const int d = eng.mtp_draft();
        SQ_CHECK(d > 0, "--spec-check needs a model with an MTP block and --mtp > 0");
        sp = SamplingParams{};
        eng.set_mtp_draft(0);
        const double tp = now_ms();
        generate();
        log("plain: %d tokens in %.1f ms (%.2f tok/s)", n_gen, now_ms() - tp, n_gen * 1000.0 / (now_ms() - tp));
        const std::vector<int> ref = out;
        eng.stats = EngineStats{};
        eng.reset();
        eng.feed(ids.data(), (int)ids.size());
        eng.set_mtp_draft(d);
        const double ts = now_ms();
        generate();
        const double te = now_ms();
        int same = 0;
        while (same < (int)out.size() && out[same] == ref[same]) ++same;
        std::printf("spec-check: %d / %zu tokens identical to plain greedy decoding%s\n", same, out.size(),
                    same == (int)out.size() ? "" : " (first difference shown below)");
        if (same < (int)out.size()) std::printf("  at %d: plain %d, with drafts %d\n", same, ref[same], out[same]);
        log("with drafts: %d tokens in %.1f ms (%.2f tok/s)", n_gen, te - ts, n_gen * 1000.0 / (te - ts));
        log("%s", eng.status_line().c_str());
        return same == (int)out.size() ? 0 : 2;
    }
    const double t2 = now_ms();
    generate();
    const double t3 = now_ms();
    std::printf("generated:");
    for (int t : out) std::printf(" %d", t);
    std::printf("\n");
    log("prompt %zu tokens in %.1f ms (%.1f tok/s); generated %d tokens in %.1f ms (%.2f tok/s)", ids.size(), t1 - t0,
        ids.size() * 1000.0 / (t1 - t0), n_gen, t3 - t2, (n_gen - 1) * 1000.0 / (t3 - t2));
    log("%s", eng.status_line().c_str());
    return 0;
}
