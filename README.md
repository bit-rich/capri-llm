# CAPRI: Client-Assisted PRivate Inference

Run an open-weight LLM on someone else's GPU without them seeing your prompt or the answer. There is one untrusted
server, with no second server and no trusted hardware. Privacy comes from lattice cryptography alone.

**Status: research prototype.** It works end to end on real models over a real internet connection, but it has not
been audited, and the security assumptions below are weaker than a production system should accept.

## How it works

- **The server does the heavy part, on masked data.** For every weight matrix, the client sends its activation
  hidden under a fresh Ring-LWE mask (`c = A·alpha + e + x mod 2^32`). The server runs an ordinary int8 GEMV on it,
  the same work as normal inference, and sends the result back. It never sees `x`.
- **The client does the light part in the clear.** That covers the embedding lookup (a public table), RMSNorm, RoPE,
  softmax attention, SwiGLU and the KV cache. This is small next to the weight matrices.
- **The client removes the mask.** That needs `M·A·alpha` for each layer and token, which the client gets from a
  one-time public download plus a cheap computation of its own (the BFV `c0` part is folded into the online reply).
  Per batch of 4,096 tokens, the only offline step is a 42 MB upload.
- **Two round trips per transformer block.** RMSNorm only divides by one number, so "branch, add, norm, next matrix"
  merges into one trip. A whole prompt goes through in one pass, and speculative decoding with a small local draft
  model checks several guessed tokens per pass.

Details, derivations and every measurement are in [`pi/README.md`](pi/README.md).

## Results

| Model | Setup | Speed while writing | Matches plain int8 |
|---|---|---|---|
| Qwen3-0.6B | laptop, client and server on one machine | ~10 tokens/s | 4 of 4 prompts |
| Qwen3-1.7B | laptop, whole prompt + speculative decoding | 8.2 tokens/s (2.8 projected on fiber) | 4 of 4 prompts |
| Qwen3-4B | laptop GPU (RTX 2000 Ada, 8 GB), one token per pass | ~9 tokens/s | 4 of 4 prompts |
| **Qwen3-32B** | **client: laptop on home internet (17 ms to AWS); server: AWS g4dn.metal (8x T4)** | **0.18 tokens/s** | 1 of 2 (the other split at a near-tie: top-2 logit gap 0.044) |

- The server's extra work is about **2x** a normal int8 forward pass (Qwen3-1.7B, same GPU, compute only).
- Most of the remaining slowdown is the network: about 130 round trips and about 11 MB up and 12 MB down per token
  for Qwen3-32B. That is why fewer round trips and less data per token are the main open problems.
- "Matches plain int8" means the private run picked exactly the same tokens as the same int8 model without privacy.

## Security

- **Privacy does not depend on the server being honest.** The server only ever sees masked numbers, so even a server
  that breaks the rules cannot read your prompt or the answer. What it does see is how many tokens go in and come out.
- **Correctness does.** There is no integrity protection: a cheating server could return wrong numbers and make the
  answer wrong without the client noticing.
- **Lattice estimates** (malb/lattice-estimator): the per-token masks (n = 1024, q = 2^32, sigma 6.4) come out at
  about 121 bits (95 on the conservative core-SVP measure). That is below the usual 128; n = 2048 would give over 230.
  The outer BFV scheme (N = 4096, Q = 2^80) comes out at about 175 bits. The estimator covers the underlying LWE
  problems, not the protocol as a whole.
- **The client never takes public randomness from the server.** A server that chose the `a`'s could switch the masks off.
- **Mixture-of-experts models need all routing hidden.** Expert selections alone reveal about 91% of tokens
  ([arXiv:2602.04105](https://arxiv.org/abs/2602.04105)). This prototype only supports dense models.

## Quick start (one machine)

```bash
cd pi
python -m venv .venv && . .venv/bin/activate
pip install numpy torch tokenizers huggingface_hub transformers safetensors
gcc -O3 -march=native -shared -fPIC kernels.c -o libk.so
python test_v3.py                       # crypto checks against exact big-integer arithmetic
python run.py v3 merge                  # Qwen3-0.6B, private vs plain, all on this machine
python server.py &                      # the same over TCP: start the server, wait for "listening",
python client.py check                  #   then run the client (check compares with the plain model)
python chat.py                          # chat with the model through the private protocol
```

- **Over a network:** set the same password on both sides (environment variable `PI_TOKEN`, or the file
  `~/.config/pi/token` with mode 600), then run `python server.py host=0.0.0.0` and
  `python client.py host=<server>`. `pi/aws/` has the scripts used for the AWS runs.
- **Mistral3 models:** supported too (FP8 checkpoints included). `python plan.py <HF id>` prints the GPU memory,
  data per token and round trips a model needs.

## Credits

Built with the assistance of Claude.

MIT License.
