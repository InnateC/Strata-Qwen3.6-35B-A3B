// main.cpp - command line: benchmark / greedy generation from token ids / the --serve line protocol.
//
//   strataq -m model.gguf --ids 1,2,3 -n 64            greedy continuation, prints ids and speed
//   strataq -m model.gguf --bench                       prefill + decode benchmark on a synthetic prompt
//   strataq -m model.gguf --serve                       stdin/stdout protocol used by serve/server.py
//   strataq -m model.gguf --ids 1,2,3 -n 64 --spec-check   greedy with and without MTP drafts must agree
//
// Protocol (one line each way; logs go to stderr):
//   -> GEN max_new=N temp=T top_k=K top_p=P min_p=M pres=X freq=Y rep=R seed=S stop=a,b [ckpt=p,q] [draft=D] ids=1,2,3
//      (ckpt: extra prompt positions to checkpoint for prefix reuse; the end of the prompt always is;
//       draft: MTP drafts per step for this request, default the --mtp value)
//   <- PROG done total            (prompt processing progress)
//   <- TOK id                     (each generated token)
//   <- DONE n_prompt=.. n_reused=.. n_gen=.. prefill_ms=.. decode_ms=.. reason=stop|length|cancel|error
//   -> STOP                       (cancel the running generation)
//   -> STATS / SAVEPROFILE path / RESET / QUIT
#include "common.hpp"
#include "engine.hpp"

#include <atomic>
#include <condition_variable>
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

using namespace sq;

namespace {

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
        "usage: strataq -m MODEL.gguf [options]\n"
        "  --ctx N            context length (default 32768)\n"
        "  --threads N        CPU expert threads (default 8)\n"
        "  --reserve-mb N     VRAM left free once ready (default 1200)\n"
        "  --cache-mb N       cap the VRAM expert cache\n"
        "  --profile FILE     expert routing profile (default data/expert_profile.bin)\n"
        "  --no-graph         launch kernels one by one (debug)\n"
        "  --no-adapt         keep the expert cache fixed\n"
        "  --adapt-every N --adapt-swaps M   re-rank the cache every N tokens, swapping up to M experts\n"
        "  --profile-weight W weight of the profile against live routing (default 1)\n"
        "  --prefill-chunk N  tokens per prefill pass (default 4096; smaller leaves more VRAM for experts)\n"
        "  --mtp N            speculative decoding: up to N tokens drafted per step by the MTP block (default 3,\n"
        "                     max 3; 0 = off and the MTP block is not loaded)\n"
        "  --draft-p P        keep drafting while the MTP's probability of the drafts so far is >= P (default 0.8)\n"
        "  --ids a,b,c|@file -n N   generation from token ids (greedy unless --temp)\n"
        "  --temp T --top-k K --top-p P --pres X --seed S   sampling for --ids / --bench\n"
        "  --dump-logits F    write the logits after --ids to F (float32)\n"
        "  --bench [--bench-prompt N] [-n N]\n"
        "  --selfcheck N      logits of a random N-token prompt: batched prefill vs prefill(N-1) + decode(1)\n"
        "  --spec-check       with --ids: greedy generation with MTP drafts vs without, token by token\n"
        "  --serve            line protocol on stdin/stdout\n");
}

struct LineReader {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::string> q;
    bool eof = false;
    std::thread th;
    void start() {
        th = std::thread([this] {
            std::string line;
            while (std::getline(std::cin, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                std::lock_guard<std::mutex> lk(m);
                q.push_back(line);
                cv.notify_all();
            }
            std::lock_guard<std::mutex> lk(m);
            eof = true;
            cv.notify_all();
        });
        th.detach();
    }
    bool next(std::string& out) {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, [&] { return !q.empty() || eof; });
        if (q.empty()) return false;
        out = q.front();
        q.pop_front();
        return true;
    }
    bool poll_stop() {   // consumes a pending STOP
        std::lock_guard<std::mutex> lk(m);
        for (auto it = q.begin(); it != q.end(); ++it)
            if (*it == "STOP") { q.erase(it); return true; }
        return false;
    }
};

void emit(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stdout, fmt, ap);
    va_end(ap);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

int serve(Engine& eng) {
    const int default_draft = eng.mtp_draft();
#ifdef _WIN32
    // Ctrl+C in the console reaches every process in it: let the server (our parent) handle it, save the profile
    // and close our stdin, which ends this loop.
    SetConsoleCtrlHandler(nullptr, TRUE);
#endif
    LineReader rd;
    rd.start();
    emit("READY ctx=%d vocab=%d mtp=%d", eng.ctx(), eng.cfg().n_vocab, default_draft);
    std::string line;
    while (rd.next(line)) {
        if (line == "QUIT") break;
        if (line == "STOP") continue;
        if (line == "STATS") { emit("STATS %s", eng.status_line().c_str()); continue; }
        if (line == "RESET") { eng.reset(); emit("OK"); continue; }
        if (line.rfind("SAVEPROFILE ", 0) == 0) {
            std::string err;
            if (eng.save_profile(line.substr(12), err)) emit("OK");
            else emit("ERR %s", err.c_str());
            continue;
        }
        if (line.rfind("GEN ", 0) != 0) { emit("ERR unknown command"); continue; }
        SamplingParams sp;
        int max_new = 256, draft = -1;
        std::vector<int> ids, stop, ckpt;
        std::stringstream ss(line.substr(4));
        std::string kv;
        while (ss >> kv) {
            const size_t eq = kv.find('=');
            if (eq == std::string::npos) continue;
            const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
            if (k == "max_new") max_new = std::stoi(v);
            else if (k == "temp") sp.temperature = std::stof(v);
            else if (k == "top_k") sp.top_k = std::stoi(v);
            else if (k == "top_p") sp.top_p = std::stof(v);
            else if (k == "min_p") sp.min_p = std::stof(v);
            else if (k == "pres") sp.presence_penalty = std::stof(v);
            else if (k == "freq") sp.frequency_penalty = std::stof(v);
            else if (k == "rep") sp.repetition_penalty = std::stof(v);
            else if (k == "seed") sp.seed = std::stoull(v);
            else if (k == "stop") stop = parse_ids(v);
            else if (k == "ids") ids = parse_ids(v);
            else if (k == "ckpt") ckpt = parse_ids(v);
            else if (k == "draft") draft = std::stoi(v);
        }
        if (ids.empty()) { emit("DONE n_prompt=0 n_reused=0 n_gen=0 prefill_ms=0 decode_ms=0 reason=error"); continue; }
        if ((int)ids.size() >= eng.ctx()) {
            emit("DONE n_prompt=%zu n_reused=0 n_gen=0 prefill_ms=0 decode_ms=0 reason=context", ids.size());
            continue;
        }
        std::mt19937_64 rng(sp.seed ? sp.seed : (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count());
        const double t0 = now_ms();
        const int reused = eng.reuse_prefix(ids);
        if (reused < (int)ids.size()) {
            // feed in slices so the client sees progress on long prompts; stop at the requested checkpoints
            const int total = (int)ids.size() - reused;
            const int slice = 4096;
            std::sort(ckpt.begin(), ckpt.end());
            int pos = reused;
            while (pos < (int)ids.size()) {
                int end = std::min(pos + slice, (int)ids.size());
                bool at_ckpt = false;
                for (int c : ckpt)
                    if (c > pos && c < end) { end = c; at_ckpt = true; break; }
                    else if (c == end) at_ckpt = true;
                eng.feed(ids.data() + pos, end - pos);
                pos = end;
                if (at_ckpt) eng.save_checkpoint();
                emit("PROG %d %d", pos - reused, total);
            }
            eng.save_checkpoint();
        }
        const double t1 = now_ms();
        eng.begin_request();
        eng.set_mtp_draft(draft < 0 ? default_draft : draft);
        int n_gen = 0;
        const char* reason = "length";
        const int room = eng.ctx() - (int)ids.size();
        max_new = std::min(max_new, room);
        // Each round emits the tokens that follow the last emitted one: one sampled token, or with MTP the accepted
        // drafts and the trunk's next token.  The last of them is not fed yet: spec_step feeds it.
        std::vector<int> next{eng.sample(sp, rng)};
        while (max_new > 0) {
            bool finished = false;
            for (const int tok : next) {
                emit("TOK %d", tok);
                if (std::find(stop.begin(), stop.end(), tok) != stop.end()) { reason = "stop"; finished = true; break; }
                if (++n_gen >= max_new) { finished = true; break; }
            }
            if (finished) break;
            if (rd.poll_stop()) { reason = "cancel"; break; }
            eng.spec_step(next.back(), sp, rng, next);
        }
        const double t2 = now_ms();
        emit("DONE n_prompt=%zu n_reused=%d n_gen=%d prefill_ms=%.1f decode_ms=%.1f reason=%s", ids.size(), reused, n_gen,
             t1 - t0, t2 - t1, reason);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    EngineOptions opt;
    opt.profile_path = "data/expert_profile.bin";
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
        else if (a == "--ctx") opt.ctx = std::stoi(next());
        else if (a == "--threads") opt.cpu_threads = std::stoi(next());
        else if (a == "--reserve-mb") opt.vram_reserve_mb = std::stoll(next());
        else if (a == "--cache-mb") opt.cache_mb = std::stoll(next());
        else if (a == "--profile") opt.profile_path = next();
        else if (a == "--no-graph") opt.use_graph = false;
        else if (a == "--no-adapt") opt.adapt_every = 0;
        else if (a == "--profile-weight") opt.profile_weight = std::stod(next());
        else if (a == "--adapt-every") opt.adapt_every = std::stoi(next());
        else if (a == "--adapt-swaps") opt.adapt_swaps = std::stoi(next());
        else if (a == "--prefill-chunk") opt.prefill_chunk = std::stoi(next());
        else if (a == "--no-pin") opt.pin_threads = false;
        else if (a == "--mtp") opt.mtp_draft = std::stoi(next());
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
    Engine eng;
    std::string err;
    if (!eng.init(opt, err)) die("%s", err.c_str());
    if (do_serve) return serve(eng);
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
        eng.begin_request();
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
