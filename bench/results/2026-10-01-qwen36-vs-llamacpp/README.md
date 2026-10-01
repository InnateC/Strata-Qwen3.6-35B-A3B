# Qwen3.6-35B-A3B UD-Q4_K_M: Strata-Qwen36 vs llama.cpp (2026-10-01)

One PC, one model file, the same client requests to both servers.

- **PC:** RTX 5080 16 GB (driver 616.92), Ryzen 7 9800X3D (8 cores, AVX-512), 64 GB DDR5, Windows 11.
- **Model:** `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` from `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` at `5bc3e238`
  (SHA-256 `0b21525e...ac58b`), the MTP layer included.
- **Strata-Qwen36:** engine 0.1.0+qwen35moe as `setup_qwen36.py` configures it (64K context, fp16 KV, MTP up to 3
  drafts, the shipped expert profile), through `serve/server.py` on port 8081.
- **llama.cpp:** LM Studio's `llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.26.0` (`llama-server`, build 86d86ed), all
  layers on the GPU except the experts of the first N layers (`--n-cpu-moe N`), `-c 65536 -fa on -t 8`. N was swept
  for the fastest output that fits the 16 GB: without MTP 19 (18 or less spills into shared memory: 2.9-4.8 tok/s),
  with MTP (`--spec-type draft-mtp`, its default 3 drafts) 21-23 are alike, 22 taken. `-ub 2048 -b 2048` (bigger
  prompt batches) reads prompts 2.3x faster and needs N = 24.

The client is `tools/bench_vs_llamacpp.py` (the same file for both): greedy, every prompt begins with a fresh random
id so neither server reuses a cached prefix (llama.cpp also gets `cache_prompt: false`), and the speeds are the
`timings` each server reports.

| | Strata-Qwen36 | llama.cpp | llama.cpp + MTP | llama.cpp + MTP, `-ub 2048` |
| --- | ---: | ---: | ---: | ---: |
| | | `--n-cpu-moe 19` | `--n-cpu-moe 22` | `--n-cpu-moe 24` |
| Output, short chat (tok/s) | **187.6** | 84.1 | 94.7 | 90.8 |
| Output after a 28.7K-token prompt (tok/s) | **157.3** | 77.6 | 98.6 | 87.8 |
| Reading a 7.9K-token prompt (tok/s) | **3,269** | 1,011 | 911 | 2,283 |
| Reading a 29.8K-token prompt (tok/s) | **2,196** | 1,000 | 898 | 2,156 |

- Short chat: 3 prompts (a Python function, Japanese prose, a proof), thinking on, 384 tokens, twice each; the mean of
  the 6. Strata's first pass over the Japanese and the proof prompts ran at 152-156 tok/s and the second at 186-202:
  its VRAM expert cache adapts to the conversation.
- After a long prompt: this repository's docs and code (~28.7K tokens), then a 400-token summary, thinking off.
- Against the best llama.cpp configuration for each row: output 2.0x (short chat) and 1.6x (after 28.7K tokens),
  prompt reading 1.4x at 8K and on par at 30K.

Files: `strata.json`, `llamacpp-ncmoe19.json`, `llamacpp-mtp-ncmoe22.json`, `llamacpp-mtp-ncmoe24-ub2048.json` (every
request with its token counts, speeds and MTP draft counts).
