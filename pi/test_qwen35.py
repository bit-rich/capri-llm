"""Hybrid Qwen3.5-family support (gated DeltaNet + gated full attention), checked on the real Qwen/Qwen3.5-0.8B
against HF transformers in fp32: token-by-token logits of our float forward pass (unmerged, merged, merged in batched
passes, and after a speculative-decoding rollback), then one private pass (v3 + merge) against the merged int8 model.
`python test_qwen35.py [repo=Qwen/Qwen3.5-2B]`"""
import sys, numpy as np, torch
from protocol import *

repo = next((a[5:] for a in sys.argv if a.startswith("repo=")), "Qwen/Qwen3.5-0.8B")
path = qwen_path(repo)
from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
ids = tok.encode("The capital of France is Paris. The capital of Germany is").ids
print(repo, len(ids), "tokens")

from transformers import AutoModelForImageTextToText
hf = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.float32).eval()
with torch.no_grad(): ref = hf(input_ids=torch.tensor([ids])).logits[0].double().numpy()
del hf
rel = lambda a, b: float((np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)).max())
fl = lambda n, L, x: L.float_(x)
flb = lambda n, L, X: np.stack([L.float_(x) for x in X])

m = Qwen3(path, keep_float=True)
c = [([], []) for _ in m.blocks]
ours = np.array([forward_token(m, t, c, fl) for t in ids])
print(f"fp32, unmerged, token by token vs HF: max rel logit diff {rel(ours, ref):.1e}, argmax equal {(ours.argmax(1) == ref.argmax(1)).mean():.0%}")
assert rel(ours, ref) < 1e-3
del m

mm = Qwen3(path, keep_float=True, merge=True)
c = [([], []) for _ in mm.blocks]
mer = np.array([forward_token(mm, t, c, fl) for t in ids])
c = [([], []) for _ in mm.blocks]
bat = np.concatenate([forward_batch(mm, ids[:5], c, flb), forward_batch(mm, ids[5:], c, flb)])
c = [([], []) for _ in mm.blocks]                               # speculative rollback: run 3 extra tokens, drop them
forward_batch(mm, ids[:6], c, flb); forward_batch(mm, ids[6:9] + [17, 18, 19], c, flb); cache_keep(c, 9)
rb = np.concatenate([np.zeros((9, ref.shape[1])), forward_batch(mm, ids[9:], c, flb)])
print(f"fp32 merged vs HF {rel(mer, ref):.1e}; merged, two batched passes {rel(bat, ref):.1e}; after rollback {rel(rb[9:], ref[9:]):.1e}")
assert max(rel(mer, ref), rel(bat, ref), rel(rb[9:], ref[9:])) < 1e-3
del mm

mq = Qwen3(path, keep_float=False, merge=True)
c = [([], []) for _ in mq.blocks]; w8 = np.array([forward_token(mq, t, c, lambda n, L, x: L.plain(x)) for t in ids])
gpu = torch.cuda.is_available()
lwe = InnerLWE(); srv = FoldServer(mq, lwe, gpu=gpu, out_bits=20); cl = FoldClient(srv, lwe, None, gpu=gpu)
cl.offline(len(ids) + 2); srv.hint = None
if gpu: torch.cuda.empty_cache(); srv.upload_gpu()
c = [([], []) for _ in mq.blocks]; cl.begin_pass(len(ids)); pv = forward_batch(mq, ids, c, cl.lin_batch)
print(f"private (v3 + merge, {cl.rounds} round trips for the whole sequence) vs merged W8: max rel logit diff {rel(pv, w8):.1e}, "
      f"argmax equal {(pv.argmax(1) == w8.argmax(1)).mean():.0%}; W8 vs HF fp32 argmax equal {(w8.argmax(1) == ref.argmax(1)).mean():.0%}")
assert cl.rounds == 2 * mq.L + 2 and rel(pv, w8) < 0.05
print("qwen3.5 tests passed")
