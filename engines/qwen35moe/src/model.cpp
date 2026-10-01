// model.cpp - see model.hpp.
#include "model.hpp"

#include "common.hpp"
#include "quant.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <thread>

namespace sq {

namespace {

template <typename F>
void parallel_for(int64_t n, F&& f, int n_threads = 0) {
    if (n_threads <= 0) n_threads = (int)std::max(1u, std::thread::hardware_concurrency());
    n_threads = (int)std::min<int64_t>(n_threads, std::max<int64_t>(1, n));
    std::vector<std::thread> th;
    for (int t = 0; t < n_threads; ++t) {
        th.emplace_back([&, t] {
            const int64_t a = n * t / n_threads, b = n * (t + 1) / n_threads;
            for (int64_t i = a; i < b; ++i) f(i);
        });
    }
    for (auto& x : th) x.join();
}

constexpr int64_t kAlign = 256;
int64_t align_up(int64_t x) { return (x + kAlign - 1) / kAlign * kAlign; }

int64_t q8_dev_bytes(int64_t rows, int64_t cols) { return align_up(rows * cols) + align_up(rows * cols / 32 * 2); }

std::string blk(int il, const char* name) { return "blk." + std::to_string(il) + "." + name; }

}  // namespace

Model::~Model() {
    if (dev_arena_) cudaFree(dev_arena_);
    for (auto* p : pinned_) cudaFreeHost(p);
}

void* Model::dev_alloc(int64_t bytes) {
    bytes = align_up(bytes);
    SQ_CHECK(dev_used_ + bytes <= dev_cap_, "device weight arena overflow");
    void* p = dev_arena_ + dev_used_;
    dev_used_ += bytes;
    return p;
}

DQ8 Model::upload_q8(const std::vector<const GgufTensor*>& parts) {
    const int64_t cols = parts[0]->ne[0];
    int64_t rows = 0;
    for (auto* t : parts) {
        // Q8_0 is copied as stored; any other type the dequantizer knows (Q4_K..Q6_K, F16, BF16, F32 - the dense
        // tensors of the other quantizations, e.g. Q4_K_M's Q6_K output head) is converted to Q8_0 at load
        SQ_CHECK(t->type == T_Q8_0 || can_dequant(t->type), "%s: unsupported type %s", t->name.c_str(),
                 type_name(t->type));
        SQ_CHECK(t->ne[0] == cols && cols % 32 == 0, "%s: column mismatch", t->name.c_str());
        rows += t->n_elements() / cols;
    }
    const int64_t nb = cols / 32;
    std::vector<int8_t> qs((size_t)(rows * cols));
    std::vector<uint16_t> d((size_t)(rows * nb));
    int64_t r0 = 0;
    for (auto* t : parts) {
        const int64_t tr = t->n_elements() / cols;
        if (t->type == T_Q8_0) {
            const BlockQ8_0* src = (const BlockQ8_0*)t->data;
            parallel_for(tr, [&](int64_t r) {
                const BlockQ8_0* b = src + r * nb;
                int8_t* q = qs.data() + (r0 + r) * cols;
                uint16_t* dd = d.data() + (r0 + r) * nb;
                for (int64_t i = 0; i < nb; ++i) {
                    dd[i] = b[i].d;
                    std::memcpy(q + i * 32, b[i].qs, 32);
                }
            });
        } else {
            const int64_t rb = row_bytes(t->type, cols);
            parallel_for(tr, [&](int64_t r) {
                std::vector<float> f((size_t)cols);
                dequant_row(t->type, t->data + r * rb, f.data(), cols);
                quantize_row_q8_0(f.data(), qs.data() + (r0 + r) * cols, d.data() + (r0 + r) * nb, cols);
            });
            log("%s: %s converted to Q8_0 for the GPU", t->name.c_str(), type_name(t->type));
        }
        r0 += tr;
    }
    DQ8 m;
    m.rows = (int)rows;
    m.cols = (int)cols;
    int8_t* dq = (int8_t*)dev_alloc(rows * cols);
    uint16_t* dd = (uint16_t*)dev_alloc(rows * nb * 2);
    CUDA_CHECK(cudaMemcpy(dq, qs.data(), qs.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dd, d.data(), d.size() * 2, cudaMemcpyHostToDevice));
    m.qs = dq;
    m.d = dd;
    return m;
}

DF32 Model::upload_f32(const std::vector<const GgufTensor*>& parts) {
    const int64_t cols = parts[0]->ne[0];
    int64_t rows = 0;
    for (auto* t : parts) {
        SQ_CHECK(t->type == T_F32 || t->type == T_BF16 || t->type == T_F16, "%s: expected F32", t->name.c_str());
        SQ_CHECK(t->ne[0] == cols || t->n_elements() == cols, "%s: column mismatch", t->name.c_str());
        rows += t->n_elements() / cols;
    }
    float* w = (float*)dev_alloc(rows * cols * 4);
    int64_t off = 0;
    for (auto* t : parts) {
        if (t->type == T_F32) {
            CUDA_CHECK(cudaMemcpy(w + off, t->data, t->n_elements() * 4, cudaMemcpyHostToDevice));
        } else {   // the MTP block's router is stored as BF16
            std::vector<float> f((size_t)t->n_elements());
            dequant_row(t->type, t->data, f.data(), t->n_elements());
            CUDA_CHECK(cudaMemcpy(w + off, f.data(), f.size() * 4, cudaMemcpyHostToDevice));
        }
        off += t->n_elements();
    }
    DF32 m;
    m.w = w;
    m.rows = (int)rows;
    m.cols = (int)cols;
    return m;
}

const float* Model::upload_vec(const GgufTensor& t) {
    SQ_CHECK(t.type == T_F32 || t.type == T_F16 || t.type == T_BF16, "%s: expected F32, got %s", t.name.c_str(),
             type_name(t.type));
    float* w = (float*)dev_alloc(t.n_elements() * 4);
    if (t.type == T_F32) {
        CUDA_CHECK(cudaMemcpy(w, t.data, t.n_elements() * 4, cudaMemcpyHostToDevice));
    } else {
        std::vector<float> f((size_t)t.n_elements());
        dequant_row(t.type, t.data, f.data(), t.n_elements());
        CUDA_CHECK(cudaMemcpy(w, f.data(), f.size() * 4, cudaMemcpyHostToDevice));
    }
    return w;
}

bool Model::load(const std::string& path, bool with_mtp, std::string& err) {
    const double t0 = now_ms();
    if (!gguf.open(path, err)) return false;
    const std::string arch = gguf.str("general.architecture");
    if (arch != "qwen35moe") { err = "unsupported architecture '" + arch + "' (this engine runs qwen35moe only)"; return false; }
    Config& c = cfg;
    c.n_layer = (int)gguf.i64("qwen35moe.block_count");
    const int nextn = (int)gguf.num("qwen35moe.nextn_predict_layers", 0);
    c.n_layer -= nextn;   // the MTP blocks come after the main layers
    c.n_mtp = with_mtp && nextn > 0 && gguf.tensor(blk(c.n_layer, "nextn.eh_proj.weight")) ? 1 : 0;
    c.n_embd = (int)gguf.i64("qwen35moe.embedding_length");
    c.n_head = (int)gguf.i64("qwen35moe.attention.head_count");
    c.n_head_kv = (int)gguf.i64("qwen35moe.attention.head_count_kv");
    c.head_dim = (int)gguf.i64("qwen35moe.attention.key_length");
    c.n_rot = (int)gguf.i64("qwen35moe.rope.dimension_count");
    c.rope_base = (float)gguf.num("qwen35moe.rope.freq_base", 1e7);
    c.eps = (float)gguf.num("qwen35moe.attention.layer_norm_rms_epsilon", 1e-6);
    c.n_expert = (int)gguf.i64("qwen35moe.expert_count");
    c.n_expert_used = (int)gguf.i64("qwen35moe.expert_used_count");
    c.n_ff_exp = (int)gguf.i64("qwen35moe.expert_feed_forward_length");
    c.n_ff_shexp = (int)gguf.i64("qwen35moe.expert_shared_feed_forward_length");
    c.ssm_d_conv = (int)gguf.i64("qwen35moe.ssm.conv_kernel");
    c.ssm_d_state = (int)gguf.i64("qwen35moe.ssm.state_size");
    c.ssm_n_kh = (int)gguf.i64("qwen35moe.ssm.group_count");
    c.ssm_n_vh = (int)gguf.i64("qwen35moe.ssm.time_step_rank");
    c.ssm_d_inner = (int)gguf.i64("qwen35moe.ssm.inner_size");
    c.full_attn_interval = (int)gguf.num("qwen35moe.full_attention_interval", 4);
    c.ctx_train = (int)gguf.num("qwen35moe.context_length", 262144);
    const GgufTensor& te = gguf.need("token_embd.weight");
    c.n_vocab = (int)te.ne[1];

    // The kernels are written for this model's shapes; refuse anything else rather than mis-index.
    SQ_CHECK(c.n_embd == 2048 && c.head_dim == 256 && c.n_head == 16 && c.n_head_kv == 2 && c.n_rot == 64 &&
             c.n_expert == 256 && c.n_expert_used == 8 && c.n_ff_exp == 512 && c.n_ff_shexp == 512 &&
             c.ssm_d_state == 128 && c.ssm_n_kh == 16 && c.ssm_n_vh == 32 && c.ssm_d_conv == 4,
             "model shape differs from Qwen3.6-35B-A3B; this engine is specialised for it");

    // ---------------------------------------------------------------- size the device arena
    int64_t need = 0;
    auto q8 = [&](int64_t r, int64_t cc) { need += q8_dev_bytes(r, cc); };
    auto f32 = [&](int64_t n) { need += align_up(n * 4); };
    for (int il = 0; il < c.n_moe_layers(); ++il) {
        f32(c.n_embd); f32(c.n_embd);
        if (c.is_attn(il)) {
            q8(c.n_head * c.head_dim * 2 + 2 * c.n_head_kv * c.head_dim, c.n_embd);
            f32(c.head_dim); f32(c.head_dim);
            q8(c.n_embd, c.n_head * c.head_dim);
        } else {
            q8(c.conv_dim() + c.ssm_d_inner, c.n_embd);
            f32(2 * c.ssm_n_vh * c.n_embd);
            f32(c.conv_dim() * c.ssm_d_conv); f32(c.ssm_n_vh); f32(c.ssm_n_vh); f32(c.ssm_d_state);
            q8(c.n_embd, c.ssm_d_inner);
        }
        f32((c.n_expert + 1) * c.n_embd);
        q8(2 * c.n_ff_shexp, c.n_embd);
        q8(c.n_embd, c.n_ff_shexp);
    }
    f32(c.n_embd);
    q8(c.n_vocab, c.n_embd);
    if (c.n_mtp) {
        f32(c.n_embd); f32(c.n_embd); f32(c.n_embd);
        q8(c.n_embd, 2 * c.n_embd);
    }
    dev_cap_ = need + (1 << 20);
    CUDA_CHECK(cudaMalloc(&dev_arena_, dev_cap_));
    dense_bytes = dev_cap_;

    tok_embd = te.data;
    tok_embd_type = te.type;
    tok_embd_row_bytes = row_bytes(te.type, c.n_embd);
    SQ_CHECK(te.type == T_Q8_0 || te.type == T_F32 || te.type == T_F16 || te.type == T_Q6_K || te.type == T_Q4_K ||
             te.type == T_Q5_K, "unsupported token_embd type");

    layers.resize(c.n_moe_layers());
    for (int il = 0; il < c.n_moe_layers(); ++il) {
        LayerW& L = layers[il];
        L.attn = c.is_attn(il);
        L.attn_norm = upload_vec(gguf.need(blk(il, "attn_norm.weight")));
        L.post_norm = upload_vec(gguf.need(blk(il, "post_attention_norm.weight")));
        if (L.attn) {
            L.qkv = upload_q8({&gguf.need(blk(il, "attn_q.weight")), &gguf.need(blk(il, "attn_k.weight")),
                               &gguf.need(blk(il, "attn_v.weight"))});
            L.q_norm = upload_vec(gguf.need(blk(il, "attn_q_norm.weight")));
            L.k_norm = upload_vec(gguf.need(blk(il, "attn_k_norm.weight")));
            L.wo = upload_q8({&gguf.need(blk(il, "attn_output.weight"))});
        } else {
            L.qkvz = upload_q8({&gguf.need(blk(il, "attn_qkv.weight")), &gguf.need(blk(il, "attn_gate.weight"))});
            L.ba = upload_f32({&gguf.need(blk(il, "ssm_beta.weight")), &gguf.need(blk(il, "ssm_alpha.weight"))});
            L.conv_w = upload_vec(gguf.need(blk(il, "ssm_conv1d.weight")));
            L.dt_bias = upload_vec(gguf.need(blk(il, "ssm_dt.bias")));
            L.ssm_a = upload_vec(gguf.need(blk(il, "ssm_a")));
            L.ssm_norm = upload_vec(gguf.need(blk(il, "ssm_norm.weight")));
            L.ssm_out = upload_q8({&gguf.need(blk(il, "ssm_out.weight"))});
        }
        L.router = upload_f32({&gguf.need(blk(il, "ffn_gate_inp.weight")), &gguf.need(blk(il, "ffn_gate_inp_shexp.weight"))});
        L.sh_gu = upload_q8({&gguf.need(blk(il, "ffn_gate_shexp.weight")), &gguf.need(blk(il, "ffn_up_shexp.weight"))});
        L.sh_down = upload_q8({&gguf.need(blk(il, "ffn_down_shexp.weight"))});
    }
    output_norm = upload_vec(gguf.need("output_norm.weight"));
    const GgufTensor* out = gguf.tensor("output.weight");
    lm_head = upload_q8({out ? out : &te});
    if (c.n_mtp) {
        const int il = c.n_layer;
        mtp.enorm = upload_vec(gguf.need(blk(il, "nextn.enorm.weight")));
        mtp.hnorm = upload_vec(gguf.need(blk(il, "nextn.hnorm.weight")));
        const GgufTensor* hn = gguf.tensor(blk(il, "nextn.shared_head_norm.weight"));
        mtp.head_norm = hn ? upload_vec(*hn) : output_norm;
        mtp.eh_proj = upload_q8({&gguf.need(blk(il, "nextn.eh_proj.weight"))});
        SQ_CHECK(mtp.eh_proj.rows == c.n_embd && mtp.eh_proj.cols == 2 * c.n_embd, "MTP eh_proj shape");
        SQ_CHECK(!gguf.tensor(blk(il, "nextn.embed_tokens.weight")) && !gguf.tensor(blk(il, "nextn.shared_head_head.weight")),
                 "MTP block with its own embedding / head is not supported");
        log("MTP block loaded (layer %d)", il);
    }
    const double t1 = now_ms();
    log("dense weights on GPU: %.2f GiB (%.1f s)", dense_bytes / 1073741824.0, (t1 - t0) / 1000);

    // ---------------------------------------------------------------- routed experts -> pinned RAM
    expert_bytes = 0;
    for (int il = 0; il < c.n_moe_layers(); ++il) {
        LayerW& L = layers[il];
        const GgufTensor& g = gguf.need(blk(il, "ffn_gate_exps.weight"));
        const GgufTensor& u = gguf.need(blk(il, "ffn_up_exps.weight"));
        const GgufTensor& d = gguf.need(blk(il, "ffn_down_exps.weight"));
        SQ_CHECK(g.type == u.type, "layer %d: gate/up types differ", il);
        for (uint32_t t : {g.type, d.type})
            SQ_CHECK(t == T_Q4_K || t == T_Q5_K || t == T_Q6_K || t == T_Q8_0, "layer %d: unsupported expert type %s", il, type_name(t));
        SQ_CHECK(g.ne[0] == c.n_embd && g.ne[1] == c.n_ff_exp && d.ne[0] == c.n_ff_exp && d.ne[1] == c.n_embd, "expert shape");
        L.t_gu = g.type;
        L.t_down = d.type;
        L.gu_bytes = row_bytes(g.type, c.n_embd) * c.n_ff_exp;
        L.down_bytes = row_bytes(d.type, c.n_ff_exp) * c.n_embd;
        L.blob_bytes = 2 * L.gu_bytes + L.down_bytes;
        expert_bytes += L.blob_bytes * c.n_expert;
    }
    for (int il = 0; il < c.n_moe_layers(); ++il) {
        LayerW& L = layers[il];
        uint8_t* p = nullptr;
        cudaError_t e = cudaHostAlloc((void**)&p, (size_t)(L.blob_bytes * c.n_expert), cudaHostAllocPortable);
        if (e != cudaSuccess) {
            err = "cannot pin " + std::to_string(L.blob_bytes * c.n_expert >> 20) + " MiB of RAM for layer " +
                  std::to_string(il) + " experts: " + cudaGetErrorString(e);
            return false;
        }
        pinned_.push_back(p);
        L.experts = p;
    }
    // Copy expert matrices into [gate | up | down] blobs, one layer at a time, many threads (page faults dominate).
    for (int il = 0; il < c.n_moe_layers(); ++il) {
        LayerW& L = layers[il];
        const uint8_t* g = gguf.need(blk(il, "ffn_gate_exps.weight")).data;
        const uint8_t* u = gguf.need(blk(il, "ffn_up_exps.weight")).data;
        const uint8_t* d = gguf.need(blk(il, "ffn_down_exps.weight")).data;
        parallel_for(c.n_expert, [&](int64_t e) {
            uint8_t* dst = L.experts + e * L.blob_bytes;
            std::memcpy(dst, g + e * L.gu_bytes, (size_t)L.gu_bytes);
            std::memcpy(dst + L.gu_bytes, u + e * L.gu_bytes, (size_t)L.gu_bytes);
            std::memcpy(dst + 2 * L.gu_bytes, d + e * L.down_bytes, (size_t)L.down_bytes);
        }, 16);
    }
    log("experts in pinned RAM: %.2f GiB (%.1f s)", expert_bytes / 1073741824.0, (now_ms() - t1) / 1000);
    return true;
}

void Model::embed(int token, float* out) const {
    SQ_CHECK(token >= 0 && token < cfg.n_vocab, "token id %d out of range", token);
    dequant_row(tok_embd_type, tok_embd + (int64_t)token * tok_embd_row_bytes, out, cfg.n_embd);
}

}  // namespace sq
