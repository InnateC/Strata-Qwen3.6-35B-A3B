// prefill.cu - batched prompt processing.  See prefill.hpp.
//
// Everything runs on the GPU in chunks of up to C tokens:
//   dense matmuls   : Q8_0 weights dequantized to fp16 per layer, cuBLAS tensor-core GEMM (fp32 accumulate/output)
//   routed experts  : per layer, tokens are grouped by expert on the host; an expert's weights come from its VRAM
//                     slot when resident, otherwise its pinned blob is streamed over PCIe into a staging ring
//                     (copy engine, overlapped with the GEMMs of the previous experts)
//   delta net       : causal conv and the gated delta rule scanned token by token (one warp per state column)
//   attention       : tiled causal attention over the fp16 KV cache
//   MTP             : the MTP layer runs over the chunk's (h_i, token i+1) pairs to fill its KV cache
#include "prefill.hpp"

#include "common.hpp"
#include "engine.hpp"
#include "quant.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace sq {

#define CUBLAS_CHECK(x) do { cublasStatus_t st_ = (x); if (st_ != CUBLAS_STATUS_SUCCESS) die("cuBLAS %s failed: %d", #x, (int)st_); } while (0)

namespace {

constexpr int kE = 2048, kHD = 256, kNH = 16, kNKV = 2, kVH = 32, kKH = 16, kDS = 128, kConv = 8192, kFF = 512;
constexpr int kStage = 6;   // staging ring slots for streamed experts

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float block_sum(float v, float* sh) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    v = lane < nw ? sh[lane] : 0.0f;
    return warp_sum(v);
}
__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ __half to_h(float x) { return __float2half_rn(fminf(fmaxf(x, -65504.0f), 65504.0f)); }

// ---------------------------------------------------------------------------------------------- dequantization
__global__ void dequant_q8_kernel(const int8_t* __restrict__ qs, const __half* __restrict__ d, __half* __restrict__ out,
                                  int64_t n /* elements */) {
    const int64_t i = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i >= n) return;
    const float s = __half2float(d[i >> 5]);
    const int2 q = *(const int2*)(qs + i);
    const int8_t* qb = (const int8_t*)&q;
    __align__(16) __half o[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) o[k] = __float2half_rn(s * (float)qb[k]);
    *(int4*)(out + i) = *(int4*)o;
}

__device__ __forceinline__ void sm_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4); }
}

/// One warp per 256-value super-block; lane l writes values [8l, 8l+8).
template <uint32_t T>
__global__ void dequant_k_kernel(const uint8_t* __restrict__ src, __half* __restrict__ out, int64_t n_sb) {
    const int64_t sb = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (sb >= n_sb) return;
    const int lane = threadIdx.x & 31;
    const int v0 = lane * 8;                 // first value
    __align__(16) __half o[8];
    if constexpr (T == T_Q4_K || T == T_Q5_K) {
        constexpr int BB = T == T_Q4_K ? 144 : 176;
        const uint8_t* b = src + sb * BB;
        const float2 dm = __half22float2(*(const __half2*)b);
        const int j = v0 / 64, hi = (v0 / 32) & 1, l0 = v0 % 32;   // pair, nibble, position in sub-block
        int sc, m;
        sm_k4(2 * j + hi, b + 4, sc, m);
        const float d = dm.x * sc, mn = dm.y * m;
        const uint8_t* qs = b + (T == T_Q4_K ? 16 : 48) + 32 * j + l0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            int q = hi ? (qs[k] >> 4) : (qs[k] & 0xF);
            if constexpr (T == T_Q5_K) q += ((b[16 + l0 + k] >> (2 * j + hi)) & 1) << 4;
            o[k] = __float2half_rn(d * q - mn);
        }
    } else {   // Q6_K
        const uint8_t* b = src + sb * 210;
        const float d = __half2float(*(const __half*)(b + 208));
        const int h = v0 / 128, r = v0 % 128, qd = r / 32, l0 = r % 32;
        const uint8_t* ql = b + 64 * h + 32 * (qd & 1) + l0;
        const uint8_t* qh = b + 128 + 32 * h + l0;
        const int8_t sc = ((const int8_t*)(b + 192))[8 * h + 2 * qd + l0 / 16];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int lo = (qd >> 1) ? (ql[k] >> 4) : (ql[k] & 0xF);
            const int q = (lo | (((qh[k] >> (2 * qd)) & 3) << 4)) - 32;
            o[k] = __float2half_rn(d * sc * q);
        }
    }
    *(int4*)(out + sb * 256 + v0) = *(int4*)o;
}

void dequant_expert(uint32_t t, const uint8_t* src, __half* out, int64_t n_values, cudaStream_t s) {
    const int64_t nsb = n_values / 256;
    const int blocks = (int)((nsb + 7) / 8);
    switch (t) {
        case T_Q4_K: dequant_k_kernel<T_Q4_K><<<blocks, 256, 0, s>>>(src, out, nsb); break;
        case T_Q5_K: dequant_k_kernel<T_Q5_K><<<blocks, 256, 0, s>>>(src, out, nsb); break;
        case T_Q6_K: dequant_k_kernel<T_Q6_K><<<blocks, 256, 0, s>>>(src, out, nsb); break;
        default: die("dequant_expert: type %u", t);
    }
}

// ---------------------------------------------------------------------------------------------- row kernels
/// Hn = rmsnorm(X) * w (fp32, optional), Hh = same in fp16.  One block of 256 threads per token.
__global__ void rmsnorm_rows_kernel(const float* __restrict__ X, const float* __restrict__ w, float eps,
                                    float* __restrict__ Hn, __half* __restrict__ Hh) {
    __shared__ float sh[32];
    const int t = blockIdx.x;
    const float* x = X + (size_t)t * kE;
    float v[8];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = x[threadIdx.x + 256 * i];
        ss += v[i] * v[i];
    }
    ss = block_sum(ss, sh);
    const float r = rsqrtf(ss / kE + eps);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int c = threadIdx.x + 256 * i;
        const float y = v[i] * r * w[c];
        if (Hn) Hn[(size_t)t * kE + c] = y;
        Hh[(size_t)t * kE + c] = to_h(y);
    }
}

/// Hh[t * ldo + c] = fp16(rmsnorm(X[t]) * w)  (one half of the MTP's [enorm(e) | hnorm(h)] input)
__global__ void rmsnorm_rows_ld_kernel(const float* __restrict__ X, const float* __restrict__ w, float eps,
                                       __half* __restrict__ Hh, int ldo) {
    __shared__ float sh[32];
    const int t = blockIdx.x;
    const float* x = X + (size_t)t * kE;
    float v[8];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] = x[threadIdx.x + 256 * i];
        ss += v[i] * v[i];
    }
    ss = block_sum(ss, sh);
    const float r = rsqrtf(ss / kE + eps);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int c = threadIdx.x + 256 * i;
        Hh[(size_t)t * ldo + c] = to_h(v[i] * r * w[c]);
    }
}

__global__ void gather_rows_kernel(const __half* __restrict__ src, const int* __restrict__ idx, __half* __restrict__ dst) {
    const int r = blockIdx.x;
    const int4* s = (const int4*)(src + (size_t)idx[r] * kE);
    int4* d = (int4*)(dst + (size_t)r * kE);
    d[threadIdx.x] = s[threadIdx.x];   // 256 threads x 8 halves
}

/// out[r][i] = silu(gu[r][i]) * gu[r][n + i]  (fp16 out)
__global__ void swiglu_rows_kernel(const float* __restrict__ gu, __half* __restrict__ out, int n) {
    const int r = blockIdx.x;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        out[(size_t)r * n + i] = to_h(silu(gu[(size_t)r * 2 * n + i]) * gu[(size_t)r * 2 * n + n + i]);
}

/// X[t] += sum_k w[t,k] * moe[inv[t,k]] + sigmoid(sg[t]) * sh[t]
__global__ void moe_combine_rows_kernel(float* __restrict__ X, const float* __restrict__ moe, const int* __restrict__ inv,
                                        const float* __restrict__ w, const float* __restrict__ sh,
                                        const float* __restrict__ rlog, int ldr) {
    const int t = blockIdx.x;
    const float sg = sigm(rlog[(size_t)t * ldr + 256]);
    int a[8];
    float ww[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) { a[k] = inv[t * 8 + k]; ww[k] = w[t * 8 + k]; }
    for (int c = threadIdx.x; c < kE; c += blockDim.x) {
        float acc = sg * sh[(size_t)t * kE + c];
#pragma unroll
        for (int k = 0; k < 8; ++k) acc += ww[k] * moe[(size_t)a[k] * kE + c];
        X[(size_t)t * kE + c] += acc;
    }
}

/// Router for a batch: softmax over 256 experts, top-8, weights renormalized.  One block (256) per token.
__global__ void router_rows_kernel(const float* __restrict__ rlog, int ldr, int* __restrict__ ids, float* __restrict__ wts,
                                   uint32_t* __restrict__ counts, int layer) {
    __shared__ float sh[32];
    __shared__ float bv[8];
    __shared__ int bi[8];
    __shared__ int sel[8];
    __shared__ float selp[8];
    const int t = blockIdx.x, e = threadIdx.x, lane = e & 31, warp = e >> 5;
    const float l = rlog[(size_t)t * ldr + e];
    float m = warp_max(l);
    if (lane == 0) bv[warp] = m;
    __syncthreads();
    m = bv[0];
    for (int i = 1; i < 8; ++i) m = fmaxf(m, bv[i]);
    const float ex = __expf(l - m);
    const float tot = block_sum(ex, sh);
    float p = ex / tot;
    for (int k = 0; k < 8; ++k) {
        float v = p;
        int idx = e;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        __syncthreads();
        if (lane == 0) { bv[warp] = v; bi[warp] = idx; }
        __syncthreads();
        if (e == 0) {
            float bvv = bv[0];
            int bii = bi[0];
            for (int i = 1; i < 8; ++i)
                if (bv[i] > bvv || (bv[i] == bvv && bi[i] < bii)) { bvv = bv[i]; bii = bi[i]; }
            sel[k] = bii;
            selp[k] = bvv;
        }
        __syncthreads();
        if (e == sel[k]) p = -1.0f;
    }
    if (e < 8) {
        float s = 0.0f;
        for (int k = 0; k < 8; ++k) s += selp[k];
        ids[t * 8 + e] = sel[e];
        wts[t * 8 + e] = selp[e] / s;
        atomicAdd(&counts[layer * 256 + sel[e]], 1u);
    }
}

// ---------------------------------------------------------------------------------------------- delta net
/// Causal depthwise conv over the chunk (kernel 4) + SiLU; the conv state holds the previous 3 inputs.
__global__ void conv_seq_kernel(const float* __restrict__ qkvz, int ld, float* __restrict__ cs, const float* __restrict__ cw,
                                float* __restrict__ out, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kConv) return;
    const float4 w = *(const float4*)(cw + c * 4);
    float s0 = cs[c * 3 + 0], s1 = cs[c * 3 + 1], s2 = cs[c * 3 + 2];
    for (int t = 0; t < T; ++t) {
        const float in = qkvz[(size_t)t * ld + c];
        const float y = w.x * s0 + w.y * s1 + w.z * s2 + w.w * in;
        s0 = s1;
        s1 = s2;
        s2 = in;
        out[(size_t)t * kConv + c] = silu(y);
    }
    cs[c * 3 + 0] = s0;
    cs[c * 3 + 1] = s1;
    cs[c * 3 + 2] = s2;
}

/// Per token: l2-normalize q and k heads in place (q also takes the 1/sqrt(128) scale); decay and beta per v head.
__global__ void gdn_prep_kernel(float* __restrict__ conv, const float* __restrict__ ba, const float* __restrict__ dt_bias,
                                const float* __restrict__ ssm_a, float* __restrict__ gb) {
    const int t = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;   // 32 warps: 16 q heads + 16 k heads
    float* p = conv + (size_t)t * kConv + warp * kDS;
    float v[4];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) { v[i] = p[lane + 32 * i]; ss += v[i] * v[i]; }
    ss = warp_sum(ss);
    const float r = rsqrtf(ss + 1e-6f) * (warp < kKH ? rsqrtf((float)kDS) : 1.0f);
#pragma unroll
    for (int i = 0; i < 4; ++i) p[lane + 32 * i] = v[i] * r;
    if (threadIdx.x < kVH) {
        const int h = threadIdx.x;
        const float a = ba[t * 64 + kVH + h] + dt_bias[h];
        const float sp = a > 20.0f ? a : log1pf(__expf(a));
        gb[t * 64 + h] = __expf(sp * ssm_a[h]);         // decay
        gb[t * 64 + kVH + h] = sigm(ba[t * 64 + h]);   // beta
    }
}

/// grid (32 v heads, 32 column groups), 4 warps; each warp owns one state column and scans the T tokens.
__global__ void __launch_bounds__(128) gdn_scan_kernel(const float* __restrict__ conv, const float* __restrict__ gb,
                                                       float* __restrict__ state, float* __restrict__ o, int T) {
    const int h = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = blockIdx.y * 4 + warp;
    const int kh = h % kKH;
    float* S = state + ((size_t)h * kDS + col) * kDS;
    float s[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) s[i] = S[lane + 32 * i];
    for (int t = 0; t < T; ++t) {
        const float* ct = conv + (size_t)t * kConv;
        float q[4], k[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            q[i] = ct[kh * kDS + lane + 32 * i];
            k[i] = ct[kKH * kDS + kh * kDS + lane + 32 * i];
        }
        const float v = ct[2 * kKH * kDS + h * kDS + col];
        const float decay = gb[t * 64 + h], beta = gb[t * 64 + kVH + h];
        float kv = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) kv += s[i] * k[i];
        kv = warp_sum(kv);
        const float delta = (v - decay * kv) * beta;
        float out = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[i] = decay * s[i] + k[i] * delta;
            out += s[i] * q[i];
        }
        out = warp_sum(out);
        if (lane == 0) o[(size_t)t * (kVH * kDS) + h * kDS + col] = out;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) S[lane + 32 * i] = s[i];
}

/// Y[t][h] = rmsnorm(o[t][h]) * w * silu(z[t][h]) in fp16.  grid (T, 32), 128 threads.
__global__ void gdn_normgate_rows_kernel(const float* __restrict__ o, const float* __restrict__ qkvz, int ld,
                                         const float* __restrict__ nw, float eps, __half* __restrict__ Y) {
    __shared__ float sh[32];
    const int t = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    const float v = o[(size_t)t * 4096 + h * kDS + i];
    const float ss = block_sum(v * v, sh);
    const float r = rsqrtf(ss / kDS + eps);
    const float z = qkvz[(size_t)t * ld + kConv + h * kDS + i];
    Y[(size_t)t * 4096 + h * kDS + i] = to_h(v * r * nw[i] * silu(z));
}

// ---------------------------------------------------------------------------------------------- attention
/// Per token: q/k rmsnorm + rope; q (pre-scaled) -> Qh fp16 [T][16][256]; k, v -> caches at p0 + t.
/// grid (T, 20): 16 q heads, 2 k heads, 2 v heads.
__global__ void attn_prep_rows_kernel(const float* __restrict__ qkv, int ld, const float* __restrict__ qnw,
                                      const float* __restrict__ knw, float eps, int p0, float base,
                                      __half* __restrict__ Qh, __half* __restrict__ kc, __half* __restrict__ vc, int max_ctx) {
    __shared__ float sh[32];
    __shared__ float sv[kHD];
    const int t = blockIdx.x, b = blockIdx.y, i = threadIdx.x;
    const float* row = qkv + (size_t)t * ld;
    const int pos = p0 + t;
    if (b >= kNH + kNKV) {
        const int kh = b - kNH - kNKV;
        vc[((size_t)kh * max_ctx + pos) * kHD + i] = __float2half_rn(row[kNH * 2 * kHD + kNKV * kHD + kh * kHD + i]);
        return;
    }
    const bool isq = b < kNH;
    const float x = isq ? row[b * 2 * kHD + i] : row[kNH * 2 * kHD + (b - kNH) * kHD + i];
    const float ss = block_sum(x * x, sh);
    const float r = rsqrtf(ss / kHD + eps);
    sv[i] = x * r * (isq ? qnw[i] : knw[i]);
    __syncthreads();
    if (i < 32) {
        const double inv = pow((double)base, -2.0 * i / 64.0);
        double sn, cs;
        sincos((double)pos * (double)(float)inv, &sn, &cs);
        const float x0 = sv[i], x1 = sv[i + 32];
        sv[i] = x0 * (float)cs - x1 * (float)sn;
        sv[i + 32] = x0 * (float)sn + x1 * (float)cs;
    }
    __syncthreads();
    if (isq) Qh[((size_t)t * kNH + b) * kHD + i] = __float2half_rn(sv[i] * 0.0625f);
    else kc[((size_t)(b - kNH) * max_ctx + pos) * kHD + i] = __float2half_rn(sv[i]);
}

constexpr int kAQ = 8;                // query tokens per block
constexpr int kAR = kAQ * 8;          // query rows per block (8 q heads per kv head)
constexpr int kAK = 32;               // keys per tile
constexpr int kKPad = kHD + 8;        // padded K row (halves) against bank conflicts

/// Causal attention for a chunk.  grid (ceil(T/8), 2 kv heads), 256 threads.  Rows r = token*8 + head.
__global__ void __launch_bounds__(256) attn_prefill_kernel(const __half* __restrict__ Qh, const __half* __restrict__ kc,
                                                           const __half* __restrict__ vc, int T, int p0, int max_ctx,
                                                           const float* __restrict__ qkv, int ld, __half* __restrict__ out) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half* sQ = (__half*)smem_raw;                          // [64][256]
    __half* sK = sQ + kAR * kHD;                              // [32][264]
    __half* sV = sK + kAK * kKPad;                            // [32][256]
    float* sS = (float*)(sV + kAK * kHD);                     // [64][32]
    float* sM = sS + kAR * kAK;                               // [64] running max
    float* sL = sM + kAR;                                     // [64] running sum
    float* sA = sL + kAR;                                     // [64] rescale factor
    const int kh = blockIdx.y, tq0 = blockIdx.x * kAQ, tid = threadIdx.x;
    const int nq = min(kAQ, T - tq0);
    // load Q rows: r = tok*8 + h  ->  Qh[tq0+tok][kh*8+h]
    for (int i = tid; i < kAR * kHD / 8; i += 256) {
        const int r = i / (kHD / 8), c8 = i % (kHD / 8);
        const int tok = r >> 3, h = r & 7;
        int4 v = make_int4(0, 0, 0, 0);
        if (tok < nq) v = *(const int4*)(Qh + ((size_t)(tq0 + tok) * kNH + kh * 8 + h) * kHD + c8 * 8);
        *(int4*)(sQ + r * kHD + c8 * 8) = v;
    }
    if (tid < kAR) { sM[tid] = -INFINITY; sL[tid] = 0.0f; }
    // output accumulators: thread -> rows rg*8 .. rg*8+7 (rg = warp), dims lane*8 .. lane*8+7
    const int warp = tid >> 5, lane = tid & 31;
    float acc[8][8];
#pragma unroll
    for (int a = 0; a < 8; ++a)
#pragma unroll
        for (int b = 0; b < 8; ++b) acc[a][b] = 0.0f;
    const int last_pos = p0 + tq0 + nq - 1;
    const int n_keys = last_pos + 1;
    // score micro-tile: rows (rp*2, rp*2+1), keys kq*4 .. kq*4+3
    const int rp = tid >> 3, kq = tid & 7;
    for (int k0 = 0; k0 < n_keys; k0 += kAK) {
        __syncthreads();
        for (int i = tid; i < kAK * kHD / 8; i += 256) {
            const int j = i / (kHD / 8), c8 = i % (kHD / 8);
            int4 kv4 = make_int4(0, 0, 0, 0), vv4 = make_int4(0, 0, 0, 0);
            if (k0 + j < n_keys) {
                kv4 = *(const int4*)(kc + ((size_t)kh * max_ctx + k0 + j) * kHD + c8 * 8);
                vv4 = *(const int4*)(vc + ((size_t)kh * max_ctx + k0 + j) * kHD + c8 * 8);
            }
            *(int4*)(sK + j * kKPad + c8 * 8) = kv4;
            *(int4*)(sV + j * kHD + c8 * 8) = vv4;
        }
        __syncthreads();
        {
            float s[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
            const __half2* q0 = (const __half2*)(sQ + (rp * 2) * kHD);
            const __half2* q1 = (const __half2*)(sQ + (rp * 2 + 1) * kHD);
#pragma unroll 8
            for (int d2 = 0; d2 < kHD / 2; ++d2) {
                const float2 a0 = __half22float2(q0[d2]), a1 = __half22float2(q1[d2]);
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const float2 kf = __half22float2(((const __half2*)(sK + (kq * 4 + u) * kKPad))[d2]);
                    s[0][u] += a0.x * kf.x + a0.y * kf.y;
                    s[1][u] += a1.x * kf.x + a1.y * kf.y;
                }
            }
#pragma unroll
            for (int a = 0; a < 2; ++a) {
                const int r = rp * 2 + a;
                const int qpos = p0 + tq0 + (r >> 3);
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int kp = k0 + kq * 4 + u;
                    sS[r * kAK + kq * 4 + u] = (kp <= qpos && (r >> 3) < nq) ? s[a][u] : -INFINITY;
                }
            }
        }
        __syncthreads();
        {   // online softmax: 4 threads per row, 8 scores each
            const int r = tid >> 2, part = tid & 3;
            float* row = sS + r * kAK + part * 8;
            float mx = -INFINITY;
#pragma unroll
            for (int u = 0; u < 8; ++u) mx = fmaxf(mx, row[u]);
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float m_old = sM[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const float e = m_new == -INFINITY ? 0.0f : __expf(row[u] - m_new);
                row[u] = e;
                sum += e;
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            __syncwarp();
            if (part == 0) {
                const float alpha = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);
                sA[r] = alpha;
                sL[r] = sL[r] * alpha + sum;
                sM[r] = m_new;
            }
        }
        __syncthreads();
#pragma unroll
        for (int a = 0; a < 8; ++a) {
            const float al = sA[warp * 8 + a];
#pragma unroll
            for (int b = 0; b < 8; ++b) acc[a][b] *= al;
        }
        const int kn = min(kAK, n_keys - k0);
        for (int j = 0; j < kn; ++j) {
            const int4 raw = *(const int4*)(sV + j * kHD + lane * 8);
            const __half2* hv = (const __half2*)&raw;
            float vf[8];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const float2 f = __half22float2(hv[u]);
                vf[2 * u] = f.x;
                vf[2 * u + 1] = f.y;
            }
#pragma unroll
            for (int a = 0; a < 8; ++a) {
                const float p = sS[(warp * 8 + a) * kAK + j];
#pragma unroll
                for (int b = 0; b < 8; ++b) acc[a][b] += p * vf[b];
            }
        }
    }
    __syncthreads();
    // write: out[t][h*256 + d] = acc / l * sigmoid(gate)
#pragma unroll
    for (int a = 0; a < 8; ++a) {
        const int r = warp * 8 + a;
        const int tok = r >> 3, h = kh * 8 + (r & 7);
        if (tok >= nq) continue;
        const float inv_l = 1.0f / sL[r];
        const float* gate = qkv + (size_t)(tq0 + tok) * ld + h * 2 * kHD + kHD;
        __align__(16) __half o[8];
#pragma unroll
        for (int b = 0; b < 8; ++b) o[b] = to_h(acc[a][b] * inv_l * sigm(gate[lane * 8 + b]));
        *(int4*)(out + (size_t)(tq0 + tok) * (kNH * kHD) + h * kHD + lane * 8) = *(int4*)o;
    }
}

constexpr size_t kAttnSmem = (size_t)kAR * kHD * 2 + (size_t)kAK * kKPad * 2 + (size_t)kAK * kHD * 2 +
                             (size_t)kAR * kAK * 4 + 3 * kAR * 4;

}  // namespace

// =================================================================================================================
struct Prefill::Impl {
    int C = 0;
    cublasHandle_t cb = nullptr;
    cudaStream_t s = nullptr, cs = nullptr;
    // activations
    float *X = nullptr, *Hn = nullptr, *big = nullptr, *ba = nullptr, *conv = nullptr, *gb = nullptr, *O = nullptr;
    float *rlog = nullptr, *moe = nullptr, *gu = nullptr, *shgu = nullptr, *shout = nullptr, *wts = nullptr;
    __half *Hh = nullptr, *Y = nullptr, *W16 = nullptr, *Xe = nullptr, *act = nullptr, *Wgu = nullptr, *Wdn = nullptr, *Qh = nullptr;
    int *ids = nullptr, *perm = nullptr, *inv = nullptr;
    // host (pinned)
    int* h_ids = nullptr;
    int* h_perm = nullptr;   // [C*8] perm then [C*8] inv
    // staging ring for streamed experts
    uint8_t* stage = nullptr;
    int64_t stage_bytes = 0;
    cudaEvent_t ev_copied[kStage], ev_free[kStage];
};

Prefill::~Prefill() {
    if (p_) {
        if (p_->cb) cublasDestroy(p_->cb);
        delete p_;
    }
}

bool Prefill::init(std::string& err) {
    (void)err;
    p_ = new Impl;
    Impl& P = *p_;
    const Config& c = e_.model_.cfg;
    // with MTP, a chunk's MTP pass also carries up to kMaxT pending pairs of the previous step: room for them
    P.C = std::max(16, e_.opt_.prefill_chunk) + (e_.mtp_on_ ? kMaxT : 0);
    const int C = P.C;
    P.s = e_.stream_;
    P.cs = e_.copy_stream_;
    CUBLAS_CHECK(cublasCreate(&P.cb));
    CUBLAS_CHECK(cublasSetStream(P.cb, P.s));
    CUBLAS_CHECK(cublasSetMathMode(P.cb, CUBLAS_TENSOR_OP_MATH));
    auto fal = [](size_t n) { float* p; CUDA_CHECK(cudaMalloc(&p, n * 4)); return p; };
    auto hal = [](size_t n) { __half* p; CUDA_CHECK(cudaMalloc(&p, n * 2)); return p; };
    auto ial = [](size_t n) { int* p; CUDA_CHECK(cudaMalloc(&p, n * 4)); return p; };
    const size_t big_cols = std::max<size_t>(c.conv_dim() + c.ssm_d_inner, (size_t)c.n_head * c.head_dim * 2 + 2 * c.n_head_kv * c.head_dim);
    P.X = fal((size_t)C * kE);
    P.Hn = fal((size_t)C * kE);
    // big (projections) and O (mixer output) are dead during the MoE half of a layer and moe (per-slot expert
    // outputs) is dead outside it: one allocation serves both, which keeps large chunks affordable.
    P.big = fal((size_t)C * std::max<size_t>(big_cols + 4096, (size_t)8 * kE));
    P.O = P.big + (size_t)C * big_cols;
    P.moe = P.big;
    P.ba = fal((size_t)C * 64);
    P.conv = fal((size_t)C * kConv);
    P.gb = fal((size_t)C * 64);
    P.rlog = fal((size_t)C * 257);
    P.gu = fal((size_t)C * 2 * kFF);
    P.shgu = fal((size_t)C * 2 * kFF);
    P.shout = fal((size_t)C * kE);
    P.wts = fal((size_t)C * 8);
    P.Hh = hal((size_t)C * kE);
    P.Y = hal((size_t)C * 4096);
    P.W16 = hal(big_cols * kE);
    P.Xe = hal((size_t)C * 8 * kE);
    P.act = hal((size_t)C * kFF);
    P.Wgu = hal((size_t)2 * kFF * kE);
    P.Wdn = hal((size_t)kE * kFF);
    P.Qh = hal((size_t)C * kNH * kHD);
    P.ids = ial((size_t)C * 8);
    P.perm = ial((size_t)C * 16);
    P.inv = P.perm + (size_t)C * 8;
    CUDA_CHECK(cudaHostAlloc((void**)&P.h_ids, (size_t)C * 8 * 4, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&P.h_perm, (size_t)C * 16 * 4, cudaHostAllocDefault));
    for (const LayerW& L : e_.model_.layers) P.stage_bytes = std::max(P.stage_bytes, L.blob_bytes);
    CUDA_CHECK(cudaMalloc(&P.stage, (size_t)P.stage_bytes * kStage));
    for (int i = 0; i < kStage; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&P.ev_copied[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&P.ev_free[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(P.ev_free[i], P.s));
    }
    CUDA_CHECK(cudaFuncSetAttribute(attn_prefill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kAttnSmem));
    ready_ = true;
    return true;
}

namespace {

/// Y[T][rows] (fp32) = X[T][cols] (fp16) * W[rows][cols]^T (fp16); beta 1 accumulates into Y.
void gemm(cublasHandle_t cb, const __half* W, const __half* X, float* Y, int rows, int cols, int T, int ldy, float beta) {
    const float alpha = 1.0f;
    CUBLAS_CHECK(cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, rows, T, cols, &alpha, W, CUDA_R_16F, cols, X, CUDA_R_16F, cols,
                              &beta, Y, CUDA_R_32F, ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

void gemm_f32(cublasHandle_t cb, const float* W, const float* X, float* Y, int rows, int cols, int T, int ldy) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(cb, CUBLAS_OP_T, CUBLAS_OP_N, rows, T, cols, &alpha, W, cols, X, cols, &beta, Y, ldy));
}

void dequant_q8(const DQ8& W, __half* out, cudaStream_t s) {
    const int64_t n = (int64_t)W.rows * W.cols;
    dequant_q8_kernel<<<(unsigned)((n / 8 + 255) / 256), 256, 0, s>>>(W.qs, (const __half*)W.d, out, n);
}

}  // namespace

void Prefill::layer(int il, int T, int p0) {
    Impl& P = *p_;
    Engine& E = e_;
    const Config& c = E.model_.cfg;
    const float eps = c.eps;
    cudaStream_t s = P.s;
    const LayerW& L = E.model_.layers[il];
    rmsnorm_rows_kernel<<<T, 256, 0, s>>>(P.X, L.attn_norm, eps, L.attn ? nullptr : P.Hn, P.Hh);
    if (!L.attn) {
        const int ld = L.qkvz.rows;   // 12288
        dequant_q8(L.qkvz, P.W16, s);
        gemm(P.cb, P.W16, P.Hh, P.big, L.qkvz.rows, kE, T, ld, 0.0f);
        gemm_f32(P.cb, L.ba.w, P.Hn, P.ba, 64, kE, T, 64);
        conv_seq_kernel<<<kConv / 256, 256, 0, s>>>(P.big, ld, E.conv_st_[il], L.conv_w, P.conv, T);
        gdn_prep_kernel<<<T, 1024, 0, s>>>(P.conv, P.ba, L.dt_bias, L.ssm_a, P.gb);
        gdn_scan_kernel<<<dim3(kVH, kDS / 4), 128, 0, s>>>(P.conv, P.gb, E.ssm_st_[il], P.O, T);
        gdn_normgate_rows_kernel<<<dim3(T, kVH), kDS, 0, s>>>(P.O, P.big, ld, L.ssm_norm, eps, P.Y);
        dequant_q8(L.ssm_out, P.W16, s);
        gemm(P.cb, P.W16, P.Y, P.X, kE, 4096, T, kE, 1.0f);
    } else {
        const int ld = L.qkv.rows;    // 9216
        dequant_q8(L.qkv, P.W16, s);
        gemm(P.cb, P.W16, P.Hh, P.big, L.qkv.rows, kE, T, ld, 0.0f);
        attn_prep_rows_kernel<<<dim3(T, kNH + 2 * kNKV), 256, 0, s>>>(P.big, ld, L.q_norm, L.k_norm, eps, p0, c.rope_base,
                                                                      P.Qh, (__half*)E.kc_[il], (__half*)E.vc_[il], E.opt_.ctx);
        attn_prefill_kernel<<<dim3((T + kAQ - 1) / kAQ, kNKV), 256, kAttnSmem, s>>>(
            P.Qh, (const __half*)E.kc_[il], (const __half*)E.vc_[il], T, p0, E.opt_.ctx, P.big, ld, P.Y);
        dequant_q8(L.wo, P.W16, s);
        gemm(P.cb, P.W16, P.Y, P.X, kE, 4096, T, kE, 1.0f);
    }
    // ---- MoE
    rmsnorm_rows_kernel<<<T, 256, 0, s>>>(P.X, L.post_norm, eps, P.Hn, P.Hh);
    gemm_f32(P.cb, L.router.w, P.Hn, P.rlog, 257, kE, T, 257);
    router_rows_kernel<<<T, 256, 0, s>>>(P.rlog, 257, P.ids, P.wts, E.route_counts_, il);
    CUDA_CHECK(cudaMemcpyAsync(P.h_ids, P.ids, (size_t)T * 8 * 4, cudaMemcpyDeviceToHost, s));
    // shared expert while the host waits for the routing
    dequant_q8(L.sh_gu, P.W16, s);
    gemm(P.cb, P.W16, P.Hh, P.shgu, 2 * kFF, kE, T, 2 * kFF, 0.0f);
    swiglu_rows_kernel<<<T, 256, 0, s>>>(P.shgu, P.act, kFF);
    dequant_q8(L.sh_down, P.W16, s);
    gemm(P.cb, P.W16, P.act, P.shout, kE, kFF, T, kE, 0.0f);
    CUDA_CHECK(cudaStreamSynchronize(s));
    // group assignments by expert (counting sort)
    int cnt[257] = {0};
    for (int a = 0; a < T * 8; ++a) ++cnt[P.h_ids[a] + 1];
    for (int e = 0; e < 256; ++e) cnt[e + 1] += cnt[e];
    int off[256];
    std::memcpy(off, cnt, sizeof(off));
    int* perm = P.h_perm;
    int* inv = P.h_perm + (size_t)T * 8;
    for (int a = 0; a < T * 8; ++a) {
        const int pos = off[P.h_ids[a]]++;
        perm[pos] = a >> 3;   // token of this assignment
        inv[a] = pos;
    }
    CUDA_CHECK(cudaMemcpyAsync(P.perm, perm, (size_t)T * 8 * 4, cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(P.inv, inv, (size_t)T * 8 * 4, cudaMemcpyHostToDevice, s));
    gather_rows_kernel<<<T * 8, 256, 0, s>>>(P.Hh, P.perm, P.Xe);
    // stream non-resident experts through the staging ring, a few ahead of the compute
    std::vector<int> order;
    for (int e = 0; e < 256; ++e)
        if (cnt[e + 1] > cnt[e]) order.push_back(e);
    std::vector<int> stage_of(order.size(), -1);
    size_t next_copy = 0;
    int ring = 0;
    auto issue_copies = [&](size_t upto) {
        for (; next_copy < order.size() && next_copy < upto; ++next_copy) {
            const int e = order[next_copy];
            if (E.residency_host_[(size_t)il * 256 + e] >= 0) continue;
            const int sl = ring++ % kStage;
            stage_of[next_copy] = sl;
            CUDA_CHECK(cudaStreamWaitEvent(P.cs, P.ev_free[sl], 0));
            CUDA_CHECK(cudaMemcpyAsync(P.stage + (size_t)sl * P.stage_bytes, L.expert_blob(e), (size_t)L.blob_bytes,
                                       cudaMemcpyHostToDevice, P.cs));
            CUDA_CHECK(cudaEventRecord(P.ev_copied[sl], P.cs));
        }
    };
    // how many experts ahead may be in flight: limited by the ring
    auto ahead_limit = [&](size_t i) {
        size_t k = i, used = 0;
        while (k < order.size() && used < (size_t)kStage) {
            if (E.residency_host_[(size_t)il * 256 + order[k]] < 0) ++used;
            ++k;
        }
        return k;
    };
    for (size_t i = 0; i < order.size(); ++i) {
        issue_copies(ahead_limit(i));
        const int e = order[i];
        const int ne = cnt[e + 1] - cnt[e], o0 = cnt[e];
        const uint8_t* blob;
        const int slot = E.residency_host_[(size_t)il * 256 + e];
        if (slot >= 0) {
            blob = E.slot_base_[il] + (size_t)slot * L.blob_bytes;
        } else {
            const int sl = stage_of[i];
            CUDA_CHECK(cudaStreamWaitEvent(s, P.ev_copied[sl], 0));
            blob = P.stage + (size_t)sl * P.stage_bytes;
        }
        dequant_expert(L.t_gu, blob, P.Wgu, (int64_t)2 * kFF * kE, s);
        dequant_expert(L.t_down, blob + 2 * L.gu_bytes, P.Wdn, (int64_t)kE * kFF, s);
        if (slot < 0) CUDA_CHECK(cudaEventRecord(P.ev_free[stage_of[i]], s));
        gemm(P.cb, P.Wgu, P.Xe + (size_t)o0 * kE, P.gu, 2 * kFF, kE, ne, 2 * kFF, 0.0f);
        swiglu_rows_kernel<<<ne, 256, 0, s>>>(P.gu, P.act, kFF);
        gemm(P.cb, P.Wdn, P.act, P.moe + (size_t)o0 * kE, kE, kFF, ne, kE, 0.0f);
    }
    moe_combine_rows_kernel<<<T, 256, 0, s>>>(P.X, P.moe, P.inv, P.wts, P.shout, P.rlog, 257);
}

void Prefill::mtp(const int* tokens, int n, int done, int T, int p0) {
    Impl& P = *p_;
    Engine& E = e_;
    const Config& c = E.model_.cfg;
    const MtpW& M = E.model_.mtp;
    cudaStream_t s = P.s;
    const int k = E.mtp_k_;                  // pending pairs (positions p0-k .. p0-1), hiddens in mtp_hin_
    const bool last = done + T == n;
    const int m = last ? T - 1 : T;          // chunk rows whose next token is known
    const int Tm = k + m, q = p0 - k;
    // the trunk's normed final hidden states of the chunk
    rmsnorm_rows_kernel<<<T, 256, 0, s>>>(P.X, E.model_.output_norm, c.eps, P.Hn, P.Hh);
    float* MH = P.big;                       // [Tm][2048] pair hiddens (P.big is free until the layer's projections)
    if (k > 0) CUDA_CHECK(cudaMemcpyAsync(MH, E.mtp_hin_, (size_t)k * kE * 4, cudaMemcpyDeviceToDevice, s));
    if (m > 0) CUDA_CHECK(cudaMemcpyAsync(MH + (size_t)k * kE, P.Hn, (size_t)m * kE * 4, cudaMemcpyDeviceToDevice, s));
    if (last) CUDA_CHECK(cudaMemcpyAsync(E.mtp_hin_, P.Hn + (size_t)(T - 1) * kE, (size_t)kE * 4, cudaMemcpyDeviceToDevice, s));
    E.mtp_k_ = last ? 1 : 0;
    if (Tm == 0) return;
    // embeddings of the pairs' next tokens (position q + r + 1)
    const int base = E.n_past_;              // tokens[0] is at position base
    std::vector<float> emb((size_t)Tm * kE);
    for (int r = 0; r < Tm; ++r) {
        const int g = q + r + 1;
        const int t = g < base ? E.history_[g] : tokens[g - base];
        E.model_.embed(t, emb.data() + (size_t)r * kE);
    }
    float* EM = P.shout;                     // free until the layer's shared expert
    CUDA_CHECK(cudaMemcpyAsync(EM, emb.data(), emb.size() * 4, cudaMemcpyHostToDevice, s));
    rmsnorm_rows_ld_kernel<<<Tm, 256, 0, s>>>(EM, M.enorm, c.eps, P.Y, 2 * kE);
    rmsnorm_rows_ld_kernel<<<Tm, 256, 0, s>>>(MH, M.hnorm, c.eps, P.Y + kE, 2 * kE);
    dequant_q8(M.eh_proj, P.W16, s);
    gemm(P.cb, P.W16, P.Y, P.X, kE, 2 * kE, Tm, kE, 0.0f);
    CUDA_CHECK(cudaStreamSynchronize(s));    // emb (pageable) is released below
    layer(c.n_layer, Tm, q);
}

void Prefill::run(const int* tokens, int n) {
    // with MTP, a chunk's MTP pass also carries the pending pairs of the previous one
    const int cmax = e_.mtp_on_ ? p_->C - kMaxT : p_->C;
    for (int done = 0; done < n;) {
        const int T = std::min(cmax, n - done);
        Impl& P = *p_;
        Engine& E = e_;
        const Config& c = E.model_.cfg;
        const float eps = c.eps;
        const int p0 = E.n_past_ + done;
        cudaStream_t s = P.s;
        // ---- embeddings (host dequant -> device)
        {
            std::vector<float> emb((size_t)T * kE);
            for (int t = 0; t < T; ++t) E.model_.embed(tokens[done + t], emb.data() + (size_t)t * kE);
            CUDA_CHECK(cudaMemcpyAsync(P.X, emb.data(), emb.size() * 4, cudaMemcpyHostToDevice, s));
            CUDA_CHECK(cudaStreamSynchronize(s));
        }
        for (int il = 0; il < c.n_layer; ++il) layer(il, T, p0);
        // ---- logits of the last token of the chunk (only the final chunk's matter)
        if (done + T == n) {
            k_rmsnorm_q(P.X + (size_t)(T - 1) * kE, E.model_.output_norm, eps, E.xn_, E.xq_, 1, s);
            k_gemv_q8(E.model_.lm_head, E.xq_, E.logits_, 1, s);
        }
        if (E.mtp_on_) mtp(tokens, n, done, T, p0);
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaGetLastError());
        done += T;
    }
}

}  // namespace sq
