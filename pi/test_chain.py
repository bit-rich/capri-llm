"""Stage-1 check of chain mode (chain.py): exactness of the 56-bit GEMM and the masks, then the private chained
evaluation of a real model against the plain (unmerged) int8 model, token by token."""
import sys, time, numpy as np, torch
from protocol import *
import chain

rng = np.random.default_rng(0)
W = rng.integers(-127, 128, (96, 300)).astype(np.int8); C = rng.integers(0, chain.Q, (5, 300), dtype=np.int64)
ref = np.array([[sum(int(W[i, j]) * int(C[t, j]) for j in range(300)) % chain.Q for i in range(96)] for t in range(5)], dtype=object)
got = chain.gemm56(torch.as_tensor(W, device="cuda"), torch.as_tensor(C, device="cuda")).cpu().numpy()
assert (got.astype(object) == ref).all(), "gemm56"
a = rng.integers(0, chain.Q, (2, 64), dtype=np.int64); al = rng.integers(-20, 21, 64)
sch = np.zeros((2, 64), dtype=object)
for b in range(2):
    for i in range(64):
        for j in range(64):
            k = i + j; v = int(a[b, i]) * int(al[j])
            if k >= 64: sch[b, k - 64] -= v
            else: sch[b, k] += v
assert (chain.negmul56(a, al).astype(object) == sch % chain.Q).all(), "negmul56"
print("56-bit GEMM and negacyclic masks exact")

repo = next((x[5:] for x in sys.argv if x.startswith("repo=")), "Qwen/Qwen3-0.6B")
path = qwen_path(repo)
from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
from prompts import PROMPTS
m = Qwen3(path, keep_float=False)
ch = chain.Chain(m)
for p in PROMPTS[:3]:
    ids = tok.encode(p).ids
    c1 = [([], []) for _ in m.blocks]; w8 = np.array([forward_token(m, t, c1, lambda n, L, x: L.plain(x)) for t in ids])
    c2 = [([], []) for _ in m.blocks]; r0 = ch.rounds
    pv = np.concatenate([ch.forward(ids[:1], c2)] + [ch.forward([t], c2) for t in ids[1:]])
    rel = np.linalg.norm(pv - w8, axis=1) / np.linalg.norm(w8, axis=1)
    print(f"{p[:30]!r}: rel logit err vs plain W8 mean {rel.mean():.2e} max {rel.max():.2e}, argmax equal "
          f"{(pv.argmax(1) == w8.argmax(1)).mean():.0%}, trips/token {(ch.rounds - r0) / len(ids):.0f}")
print(f"server per token: {1e3 * ch.server_s / ch.rounds * (2 * m.L + 2) :.1f} ms (chain, incl. Python overhead)")
