# Single-server private inference for open-weight LLMs (prototype)

The server learns nothing about the prompt. No second server and no trusted hardware are involved.

## Idea
- **Linear layers (server):** the client sends Ring-LWE-masked activations `c = a*alpha + e + x` (mod 2^32, n=1024). The server runs an ordinary int8 GEMV `y = M c`, which is the same work as plaintext.
- **Unmasking (client):** the client computes `M x ~= y - H alpha`, where `H = M A` is public and precomputed by the server.
- **`H alpha` ahead of time:** this is computed in advance under BFV encryption (batched over 4096 future tokens, then compressed to one ~30-bit limb). The approach follows Tiptoe's "token" (Henzinger et al., SOSP 2023), applied here to every layer of an LLM.
- **Nonlinear parts (client):** the client does the embedding lookup (public table), RMSNorm, RoPE, SwiGLU gating, softmax and GQA attention, and holds the KV cache.

## v3: c0 folded into the online reply (`fold.py`, `run.py v3`)
In v2 the server sends back a BFV ciphertext `(C0, C1)` of `H alpha` for every weight row and every batch (15 GB per batch for Qwen3-0.6B). v3 sends neither part per batch:
- **C1 is public and one-time.** The BFV `a_j` are fixed and public (from a seed); only the secret `s` is fresh per batch. So `C1 = M (A a)` depends on the model alone. The server publishes `C1' = round(C1 / 2^48) mod 2^32` once (4 bytes x rows x 4096).
- **C0 is never sent.** `C0[t] = M (A b[t])`, so the server subtracts it inside the online GEMV: `y = M (c - u)` with `u = round(A b[t] / 2^48) mod 2^32`. This is the mod-switched `M (Delta c - A b[t])`. Switching before the GEMV instead of after keeps the GEMV at 32 bits, and only adds `M eps` with `|eps| <= 1/2` next to the `M e` (sigma 6.4) that is there anyway.
- **The reply is 16 bits per row, not 32.** The client scales so that `|M x| < 2^30`, which makes bit 31 redundant, and the low 15 bits are at the level of the noise `M e`. The server sends bits 15-30 (`bits=20` sends 20).
- **The client subtracts `(C1' s)[t]`.** Per batch that is one FFT product per weight row, from the one-time download.
- **Per batch, the offline phase is only the client uploading the `b_j`** (42 MB, about 10 KB per token). The server does no offline work per batch.
- **No primes and no NTTs.** The BFV modulus is now `Q = 2^80` (N = 4096, plaintext modulus 2^32, Delta = 2^48), so every ring product is a float64 FFT on 16-bit limbs. The inner masks (n = 1024, mod 2^32, sigma 6.4) are unchanged, and so is what the server sees: `b` offline and `c` online.
- The public hint `H = M A` of v2 is no longer needed at all.

## Measured, v3: Qwen3-0.6B (`run.py v3`; same machine, prompts and 134 tokens as the v2 run below)
| | v2 | v3 |
|---|---|---|
| Greedy tokens, private == W8 == fp32 == HF, 4 prompts | yes | yes |
| Logit rel. error vs W8, mean / max | 0.22% / 0.58% | 0.30% / 0.93% (20-bit reply: 0.21% / 0.41%) |
| Online download per token | 1984 KB | 992 KB (20-bit reply: 1240 KB) |
| Online upload per token, rounds | 807 KB, 113 | same |
| Offline download per batch | 15.2 GB | 0 |
| Offline upload per batch | 94 MB | 42 MB |
| Offline server time per batch (GPU) | 50 s | 0 |
| Offline client time per batch | 14.4 s (GPU) | 3.1 s (GPU), 12.5 s (CPU, `cpuclient`) |
| One-time per model, server | hint `H`: 22.6 s (CPU) | `C1'`: 10.7 s (GPU) |
| One-time per model, client download | none | 8.1 GB (`C1'`, public) |
| Server per token, compute only (`bench_gpu.py`) | 6.7 ms | 9.0 ms (native int8: 4.5 ms) |
| Server per token inside `run.py` | 23.1 ms | 25.0 ms |

- The extra server compute is 0.6 ms for the pads `u` of all 113 layers (one batched FFT per token) plus about 1.7 ms of per-layer
  GPU call overhead (subtract `u`, cut the reply to 16 bits). The pads grow with the total input width of the layers, the call
  overhead with the number of layers; neither grows with the weight count.
- The 16-bit reply rounds with std 2^15/sqrt(12) = 9.5k, about the same as the existing noise `M e` (7k-11k on these weights), which is
  why the logit error goes from 0.22% to 0.30%. A 20-bit reply brings it back to the v2 level.

## Two round trips per block instead of four (`run.py v3 merge`)
RMSNorm only divides by one number, so "residual branch `R`, add, RMSNorm with weight `w`, next matrix `G`" fits in one trip:
`G rmsnorm(x + R v) = (G diag(w) R v + G (w x)) / rms(x + R v)`. The client sends `[lam v; w x]`, the server applies the merged
int8 matrix `[[R, 0], [G diag(w) R, G]]` (the `v` columns divided by `lam`), and the client, who then knows `x + R v`, divides by its rms.
- Merged pairs: `o` + `gate_up`, and `down` + the next block's `qkv`. The first `qkv`, the last `down` and the head stay alone, so a
  token takes `2 L + 2` trips instead of `4 L + 1` (58 instead of 113 for Qwen3-0.6B, 74 instead of 145 for 36 layers).
- `lam` is one public number per merged layer. It starts at the value that is best for the int8 rows and is raised until `lam v`
  is as large as `w x` on a public calibration text (`prompts.CALIB`, cached in `merge_calib.json`). Without that step the `R v`
  rows sit low in the range and the logit error is 5x higher (1.7% instead of 0.41%).
- The merged weights are 2.26x larger (1.35 G instead of 0.60 G int8 entries for 0.6B). Rows, upload and download are unchanged.

Measured on Qwen3-0.6B, v3, same machine, prompts and 134 tokens:

| | v3 | v3 + merge |
|---|---|---|
| Round trips per token | 113 | 58 |
| Greedy tokens, private == W8 == fp32 == HF, 4 prompts | yes | yes |
| Logit rel. error vs W8, mean / max | 0.30% / 0.93% | 0.41% / 0.72% |
| Logit rel. error vs fp32, mean | 3.7% | 4.2% |
| Server per token, compute only (`bench_gpu.py merge`; native int8 4.3-4.5 ms) | 9.0 ms | 9.3 ms |
| Server per token inside `run.py` | 25.0 ms | 17.5 ms |
| Whole private step per token, no network | 53.6 ms | 38.6 ms |
| Server weights | 0.60 GB | 1.35 GB |

- Not run on Qwen3-4B: its merged weights are about 9.5 GB and do not fit on the 8 GB GPU.

## Measured, v3 + merge: Qwen3-1.7B (`hf_reference.py 1.7b && run.py v3 merge 1.7b`; int8, everything on the GPU, same 4 prompts, 134 tokens)
| | v2 | v3 | v3 + merge | v3 + merge, `bits=20` |
|---|---|---|---|---|
| Round trips per token | 113 | 113 | 58 | 58 |
| Greedy tokens, private == W8 (prompts) | 4 of 4 | 3 of 4 | 3 of 4 | 4 of 4 |
| Logit rel. error vs W8, mean / max | 0.78% / 2.5% | 1.05% / 5.0% | 0.95% / 2.8% | 0.77% / 2.3% |
| Online download per token | 2901 KB | 1451 KB | 1451 KB | 1813 KB |
| Server per token, compute only (native int8: 10.1 ms) | 11.9 ms | 12.9 ms | 20.6 ms | |
| Server per token inside `run.py` | | 28.6 ms | 28.7 ms | |
| Whole private step per token, no network | | 62.7 ms | 57.9 ms | |
| Server weights | 1.7 GB | 1.7 GB | 3.3 GB | 3.3 GB |

- This model is noisier under the protocol than 0.6B or 4B: v2 already has 0.78% logit error (0.22% on 0.6B). Its int8 version is also
  less faithful: W8 plaintext matches HF fp32 on only 2 of 4 prompts, merged or not.
- Each private != W8 case is one prompt where the W8 run has a near-tie between its top two tokens (logit gaps of 0.006 and 0.014).
  The 20-bit reply removes the extra noise of the 16-bit reply and matched on all 4 prompts here; near-ties that small can still flip.
- Per batch: 42 MB up, 0 down, 4.8 s client. One-time: `C1'` 11.9 GB public download, 23 s server.

## Whole-prompt passes and speculative decoding (`spec.py 1.7b k=4 bits=20`)
One pass of the merged model costs `2 L + 2` round trips however many tokens go through it (`forward_batch`, `FoldClient.lin_batch`,
`FoldServer.online_batch`; each token of a pass uses its own secret). Two uses:
- **Whole prompt:** all prompt tokens go through one pass instead of one pass each.
- **Speculative decoding:** a small draft model on the client (Qwen3-0.6B, plaintext, public weights) guesses `k` tokens and the private
  model checks `[last token, guess 1..k]` in one pass. Guesses are kept while they equal the private model's own greedy choice, so the
  text is the private model's greedy text. Rejected guesses still cost data and a secret each.

Measured on Qwen3-1.7B (v3 + merge, 20-bit reply, 4 prompts of 42 tokens in total, 24 new tokens each). Compute is measured; fiber
(10 ms round trip, 1 Gbit/s) and home (30 ms, 100/20 Mbit/s) are computed from the measured compute time, round trips and bytes.

| Tokens per second | no network | fiber | home |
|---|---|---|---|
| Prompt, one token per pass | 11.8 | 1.45 | 0.40 |
| Prompt, whole prompt in one pass | 32-37 | 9.2-9.5 | 1.16 |
| Generation, one token per pass | 11-13 | 1.4-1.5 | 0.40 |
| Generation, speculative k=4 | 8.2 | 2.8 | 0.57 |

| End to end per message (10.5 prompt tokens, 24 new tokens), fiber | first token after | reply done after |
|---|---|---|
| One token per pass | 7.3 s | 22.9 s |
| Whole prompt | 1.1 s | 17.1 s |
| Whole prompt + speculative k=4 | 1.1 s | 9.3 s |

- Speculative: 2.97 tokens per pass, 56% of guesses kept, about 50% more data per generated token. The draft model ran on the CPU and
  took about half of the compute time, which is why the no-network speed drops (`draftgpu` moves it to the GPU; not measured).
- Private == W8 on 2, 3 and 4 of 4 prompts in the three configurations; this model has near-ties (see above), so that varies per run.
- For large models the gain shrinks: round trips grow with the layer count (254 for 126 layers) and the data per token (about 90 MB
  for a 405B model) costs more than the trips that speculation saves. Estimated 0.3-0.4 tokens per second on fiber at 405B.

## Measured, v3: Qwen3-4B (`run.py v3 4b`; int8, everything on the GPU, same 4 prompts, 134 tokens)
| | v2 | v3 |
|---|---|---|
| Greedy tokens, private == W8 == HF fp32, 4 prompts | yes | yes |
| Logit rel. error vs W8, mean / max | 0.54% / 1.58% | 0.70% / 2.57% |
| Online download per token | 5031 KB | 2516 KB |
| Online upload per token, rounds | 2738 KB, 145 | same |
| Offline download per batch | 38.6 GB | 0 |
| Offline upload per batch | 94 MB | 42 MB |
| Offline server time per batch (GPU) | 126 s | 0 |
| Offline client time per batch (GPU) | 31.7 s | 8.1 s |
| One-time per model, server | hint `H`: 123 s (CPU) | `C1'`: 39.6 s (GPU) |
| One-time per model, client download | none | 20.6 GB (`C1'`, public) |
| Server per token, compute only (`bench_gpu.py 4b`) | 23.8 ms (1.10x native) | 27.8 ms (1.29x native; native int8: 21.6 ms) |
| Server per token inside `run.py` | 43.4 ms | 48.2 ms |

- Of the 4 ms of extra server compute, 2.9 ms is the pads for all 145 layers. They depend only on `b[t]`, so the server could
  also compute them ahead of time per batch (2.7 MB of server storage per token) instead of online.

## Measured, v2: Qwen3-0.6B (`run.py v2`; i7-13700H + RTX 2000 Ada laptop GPU, 4 prompts, 134 tokens)
- Greedy tokens: private == W8 plaintext == fp32 numpy == HF transformers fp32, on all 4 prompts (24 new tokens each).
- Logit relative error, private vs W8 plaintext: 0.22% mean, 0.58% max. Private vs fp32 is 3.7% mean, mostly W8 weight quantization.
- Online, per token: 113 rounds, 807 KB up, 1984 KB down. Client: 15 ms mask + 6 ms unmask (CPU).
- Server, all on the GPU (`bench_gpu.py`, compute only): private 5.6 ms vs native int8 4.4 ms (1.27x) and native bf16 GEMVs 5.6 ms.
  For reference, a full HF bf16 decode step takes 10.3 ms. Inside `run.py`, where every layer is a host<->GPU round trip interleaved
  with client work, the private server takes 17.4 ms against 4.5 ms native int8 (3.9x).
- Offline batch (4096 token slots), on the GPU via exact int8 GEMMs: 49 s server, 14 s client, 94 MB up, 15.2 GB down.
  That is 12 ms server and 3.7 MB down per token, but only if all 4096 slots are used (this run used 134, so 363 ms/token).
  The numpy CPU path (`run.py v2 cpu`) takes about 23 min per batch.

## Measured, v2: Qwen3-8B (`hf_reference.py 8b && run.py v2 8b`; same machine, same 4 prompts, 134 tokens)
- The int8 weights (7.6 GB) don't fit next to anything else on the 8 GB GPU, so the online server runs on the CPU.
  The offline BFV step still runs on the GPU. The reference is HF bf16 on the CPU, because fp32 needs about 33 GB.
- Greedy tokens: private == W8 plaintext on all 4 prompts. Private == HF bf16 on 3 of 4. On the first prompt, W8 plaintext
  diverges from bf16 in exactly the same way (a near-tie at step 6, "Italy" vs "France"), so the cause is weight quantization, not the protocol.
- Logit relative error, private vs W8: 0.60% mean, 3.2% max.
- Online, per token: 145 rounds, 3.6 MB up, 6.2 MB down. Server on the CPU: 0.67 s against 0.38 s for plaintext int8 (1.76x, CPU vs CPU).
  Client: 74 ms mask + 21 ms unmask.
- Offline batch on the GPU: 152 s server, 37 s client, 47.7 GB down. That is 37 ms server and 11.6 MB down per token at a full 4096-token batch.
  The one-time public hint setup takes 201 s on the CPU.

## Measured, v2: Qwen3-8B, GPTQ 4-bit (`quant4.py && run.py v2 8b w4`; everything on the GPU)
- Weights: GPTQ 4-bit with one scale per output row (the protocol can't use per-group scales), act-order, lm_head int8.
  The server holds 4.1 GB. WikiText-2 perplexity (512-token windows, 20k tokens): bf16 11.50, W8 RTN 11.49, W4 RTN 15.46, **W4 GPTQ 12.05**.
  GPTQ was calibrated on WikiText-2 train, so this flatters it a little.
- Greedy tokens: private == plain W4 on all 4 prompts. Logit rel. error 0.13% mean, 0.28% max. Private == bf16 on 2 of 4 prompts;
  plain W4 diverges from bf16 in exactly the same way, so that is weight quantization.
- Server on the GPU: private 268 ms vs native 248 ms per token, compute only (`bench_gpu.py 8b w4`, 1.08x). Both are slow because the
  4-bit weights are unpacked to int8 in separate passes on every call; a fused int4 kernel would cut both.
- Offline batch on the GPU: 159 s server, 38 s client (39 ms + 9 ms per token at a full 4096-token batch).

## Measured (random-weight GPT, d=1024, 4 layers, vocab 4096, 64 tokens; 2-core CPU)
- Top-1 token agreement with the plaintext W8 model is 100%. Logit relative error is 0.11%.
- Online server time per token is 0.99x the plaintext int8 GEMV time.
- Per-layer noise is about 1e-3 with guaranteed no-wraparound scaling, and about 1.2e-4 with 8-sigma scaling.
- Ahead-of-time cost per token: 315 KB download, 23 KB upload, 61 ms server (numpy), 11 ms client (numpy).

## Files
- `crypto.py`: NTT, ring-LWE masks, hint, BFV offline scheme (v1 BGV-batched, v2 compressed, v2 on GPU via exact int8 GEMMs)
- `fold.py`: v3 crypto (public one-time `C1'`, per-token pads, per-batch upload `b`, 16-bit reply); `test_v3.py` checks it against exact big-integer arithmetic
- `protocol.py`: Qwen3 model (safetensors loader), client, server (`Server`/`Client` = v2, `FoldServer`/`FoldClient` = v3)
- `spec.py [1.7b] [k=N] [bits=N] [draftgpu]`: whole-prompt passes and speculative decoding on v3 + merge, written to `results_spec*.json`
- `prompts.py`, `hf_reference.py`: prompts, the calibration text for `merge`, and the HF transformers fp32 greedy reference (`hf_reference.json`, `hf_logits.npy`)
- `kernels.c`: int8 GEMV kernels
- `run.py [v2|v3] [merge] [4b|8b] [w4] [cpu] [bits=N] [cpuclient]`: end-to-end run (`merge` = 2 round trips per block; v2/v3 put the server GEMVs and the offline or one-time step on the GPU when CUDA is available, unless `cpu` is given; `bits=N` sets the v3 reply width, `cpuclient` keeps the v3 client off the GPU). Results go to `results_qwen*_v2.json` / `results_qwen*_v3.json`
- `bench_gpu.py [merge]`: same-GPU timing of private (v2 and v3) vs native int8/bf16 server work, written to `bench_gpu*_v3.json`
- `costmodel.py`, `qwen_cost.py`: per-token projections for 7B, 70B, DeepSeek-V3 and larger Qwen models. These still model v2 (7.5 offline bytes and 4 online bytes per row and token)
- `test_*.py`: correctness tests

Build: `gcc -O3 -march=native -shared -fPIC kernels.c -o libk.so`, then in a venv with numpy, tokenizers, huggingface_hub (+ torch, transformers for the GPU path and the reference): `python hf_reference.py && python run.py v3`

## Security estimates (`lattice_estimate.py`, malb/lattice-estimator commit 7d59f86)
Bits of work for the best attack found. "Full" is the estimator's default cost model, "lower bound" is its `rough` mode (core-SVP).

| Parameter set | full | lower bound |
|---|---|---|
| Inner masks: n = 1024, q = 2^32, secret and error sigma 6.4 (same for 2^18 to 2^24 samples per token) | 121.5 | 95.2 |
| v3 outer: N = 4096, Q = 2^80, ternary secret, sigma 3.2 | 174.9 | 146.9 |
| v2 outer: N = 4096, Q ~ 2^90, ternary secret, sigma 3.2 | 153.8 | 125.0 |
| v1 outer: N = 2048, Q ~ 2^54, ternary secret, sigma 3.2 | 126.3 | 97.8 |

- The inner masks are the weak point: about 6 bits short of 128 on the full model. They are the SimplePIR parameters the prototype started with.
- What would fix them (not implemented): sigma 12.8 gives 130.5 full / 104.5 lower bound and doubles the mask noise `M e`;
  n = 2048 at sigma 6.4 gives 257 / 237 and doubles the per-batch upload, the per-token pad work and the client's mask work.

## Caveats
- The inner mask parameters (n = 1024, q = 2^32, sigma 6.4) estimate to about 121 bits, not 128 (see Security estimates).
- The server is semi-honest. Privacy should hold against a malicious server, but integrity is not provided.
- The Qwen3 runs also do the client's per-batch work on the GPU. A real client may not have one (v3 on the CPU: `cpuclient`, 12.5 s per batch for 0.6B).
- v2: the offline download is 15.2 GB per batch, so it is only amortized if most of the 4096 token slots get used. v3 removes it, but the client has to download `C1'` once per model (8.1 GB for 0.6B, 20.6 GB for 4B), and `run.py` streams it from the server instead of storing it.
- v3 uses a power-of-two BFV modulus (N = 4096, log Q = 80, about 175 bits by the estimator) and reuses the public `a_j` across batches with a fresh `s` each time. The estimator covers the plain LWE problem; it does not prove anything about that reuse or about the protocol as a whole.
- v3 alone does not change the round trips per token (113 for 0.6B, 145 for 4B); `merge` halves them but needs 2.26x the server memory, and its `lam` values come from a short calibration text.
