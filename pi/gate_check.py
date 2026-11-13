"""Which output gate does a Qwen3.6/3.8 checkpoint want? Their config says output_gate_type="swish", which HF
transformers ignores (it always uses attn * sigmoid(gate)). The gate is client-side, so the same prepared weights
serve both; the right one gives the lower perplexity on ordinary text. Runs the plain merged int8 model.
  python gate_check.py <prepared weights dir> [tokens=64]"""
import os, sys, numpy as np
from protocol import load_prepared, forward_token
from tokenizers import Tokenizer
from prompts import CALIB, PROMPTS
d = sys.argv[1]; n = int(next((a[7:] for a in sys.argv if a.startswith("tokens=")), 64))
m = load_prepared(d); tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
ids = tok.encode(CALIB + " " + PROMPTS[2] + " sunlight, water and carbon dioxide into glucose and oxygen.").ids[:n]
for gate in ("sigmoid", "swish"):
    os.environ["PI_GATE"] = gate; cache = [([], []) for _ in m.blocks]; nll = []
    for t in range(len(ids) - 1):
        lg = forward_token(m, ids[t], cache, lambda nm, L, x: L.plain(x))
        lg = lg - lg.max(); nll.append(np.log(np.exp(lg).sum()) - lg[ids[t + 1]])
    print(f"gate {gate:7s}: perplexity {np.exp(np.mean(nll)):8.2f} over {len(nll)} tokens", flush=True)
