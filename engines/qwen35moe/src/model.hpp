// model.hpp - Qwen3.6-35B-A3B (qwen35moe): hyper-parameters and where every weight lives.
//
//   GPU  : every dense weight (Q8_0 repacked to SoA int8 + fp16 scales, F32 as-is), fused where two matrices
//          read the same input (qkv|z, q|k|v, gate|up, router|shared-gate, beta|alpha), plus the LM head.
//   RAM  : all 40 x 256 routed experts, pinned, one contiguous blob per expert: [gate | up | down].
//   mmap : the token embedding (one row is dequantized per token on the CPU).
//
// A GGUF with a multi-token-prediction block (qwen35moe.nextn_predict_layers = 1, block `n_layer`) gets it loaded as
// one more attention + MoE layer (layers[n_layer]) plus its input projection and norms (Model::mtp).
#pragma once

#include "gguf.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sq {

struct Config {
    int n_layer = 40;
    int n_embd = 2048;
    int n_vocab = 248320;
    int n_head = 16;
    int n_head_kv = 2;
    int head_dim = 256;
    int n_rot = 64;
    float rope_base = 1e7f;
    float eps = 1e-6f;
    int n_expert = 256;
    int n_expert_used = 8;
    int n_ff_exp = 512;
    int n_ff_shexp = 512;
    int ssm_d_conv = 4;
    int ssm_d_state = 128;   // head dim of k / v in the delta net
    int ssm_n_kh = 16;       // key heads
    int ssm_n_vh = 32;       // value heads
    int ssm_d_inner = 4096;  // n_vh * d_state
    int full_attn_interval = 4;
    int ctx_train = 262144;
    int n_mtp = 0;           // MTP (nextn) blocks loaded after the main layers (0 or 1)

    bool is_attn(int il) const { return il >= n_layer || (il + 1) % full_attn_interval == 0; }
    int n_moe_layers() const { return n_layer + n_mtp; }   // layers with routed experts (main + MTP)
    int conv_dim() const { return 2 * ssm_n_kh * ssm_d_state + ssm_d_inner; }  // 8192
    int n_attn_layers() const { return n_layer / full_attn_interval; }
    int n_gdn_layers() const { return n_layer - n_attn_layers(); }
};

/// A Q8_0 matrix repacked for the GPU: `qs` is rows x cols int8 (row-major, 16-byte aligned rows), `d` is
/// rows x cols/32 fp16 scales.  Two arrays instead of 34-byte blocks so every load is 16-byte aligned.
struct DQ8 {
    const int8_t* qs = nullptr;
    const uint16_t* d = nullptr;
    int rows = 0, cols = 0;
};

struct DF32 {
    const float* w = nullptr;
    int rows = 0, cols = 0;
};

struct LayerW {
    bool attn = false;
    const float* attn_norm = nullptr;
    const float* post_norm = nullptr;
    // gated delta net
    DQ8 qkvz;                 // 12288 x 2048 : q(2048) k(2048) v(4096) | z(4096)
    DF32 ba;                  // 64 x 2048    : beta(32) | alpha(32)
    const float* conv_w = nullptr;   // 8192 x 4
    const float* dt_bias = nullptr;  // 32
    const float* ssm_a = nullptr;    // 32 (= -exp(A_log))
    const float* ssm_norm = nullptr; // 128
    DQ8 ssm_out;              // 2048 x 4096
    // gated attention
    DQ8 qkv;                  // 9216 x 2048 : q+gate (16 x 512, interleaved per head) | k (512) | v (512)
    const float* q_norm = nullptr;   // 256
    const float* k_norm = nullptr;   // 256
    DQ8 wo;                   // 2048 x 4096
    // MoE
    DF32 router;              // 257 x 2048 : 256 expert logits | shared-expert gate
    DQ8 sh_gu;                // 1024 x 2048 : gate(512) | up(512)
    DQ8 sh_down;              // 2048 x 512
    // routed experts (host)
    uint32_t t_gu = 0, t_down = 0;          // ggml types of gate/up and down
    int64_t gu_bytes = 0;                   // bytes of ONE of gate/up (512 rows)
    int64_t down_bytes = 0;                 // bytes of down (2048 rows)
    int64_t blob_bytes = 0;                 // gate + up + down
    uint8_t* experts = nullptr;             // pinned host, n_expert * blob_bytes
    const uint8_t* expert_blob(int e) const { return experts + (int64_t)e * blob_bytes; }
};

/// The MTP block's own weights; its decoder layer is layers[n_layer].
struct MtpW {
    const float* enorm = nullptr;      // rmsnorm of the next token's embedding
    const float* hnorm = nullptr;      // rmsnorm of the trunk's final (normed) hidden state
    const float* head_norm = nullptr;  // norm before the shared LM head
    DQ8 eh_proj;                       // 2048 x 4096 : [enorm(e) | hnorm(h)] -> 2048
};

class Model {
public:
    Config cfg;
    std::vector<LayerW> layers;        // n_layer main layers, then the MTP layer if loaded
    MtpW mtp;
    const float* output_norm = nullptr;
    DQ8 lm_head;
    // the token embedding, copied out of the file (rows of n_embd in the GGUF's type; one is dequantized per token)
    const uint8_t* tok_embd = nullptr;
    uint32_t tok_embd_type = 0;
    int64_t tok_embd_row_bytes = 0;

    int64_t dense_bytes = 0;    // device bytes of dense weights
    int64_t expert_bytes = 0;   // pinned host bytes of experts

    /// with_mtp: load the MTP block when the file has one.
    bool load(const std::string& path, bool with_mtp, std::string& err);
    void embed(int token, float* out) const;   // dequantize one embedding row (host)
    ~Model();

    GgufFile gguf;

private:
    uint8_t* dev_arena_ = nullptr;
    int64_t dev_used_ = 0, dev_cap_ = 0;
    std::vector<uint8_t*> pinned_;
    std::vector<uint8_t> tok_embd_copy_;
    void* dev_alloc(int64_t bytes);
    DQ8 upload_q8(const std::vector<const GgufTensor*>& parts);
    DF32 upload_f32(const std::vector<const GgufTensor*>& parts);
    const float* upload_vec(const GgufTensor& t);
};

}  // namespace sq
