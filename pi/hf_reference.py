"""Plaintext ground truth: HF transformers Qwen3-0.6B in float32, manual greedy loop (fixed length,
no EOS stop, so it lines up step by step with run.py). Writes hf_reference.json (+ hf_logits.npy).
`python hf_reference.py 4b`: Qwen3-4B in float32 (~16 GB RAM). `python hf_reference.py 8b`: Qwen3-8B in bfloat16 (fp32 needs ~33 GB RAM), writes hf_reference_8b.json."""
import json, sys, numpy as np, torch
from transformers import AutoModelForCausalLM
from prompts import PROMPTS, NEW_TOKENS, encode
from protocol import qwen_path

repo, tag, dtype = (("Qwen/Qwen3-8B", "_8b", torch.bfloat16) if "8b" in sys.argv else
                    ("Qwen/Qwen3-4B", "_4b", torch.float32) if "4b" in sys.argv else
                    ("Qwen/Qwen3-1.7B", "_1.7b", torch.float32) if "1.7b" in sys.argv else ("Qwen/Qwen3-0.6B", "", torch.float32))
path = qwen_path(repo)
model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).eval()
out, all_logits = [], []
with torch.no_grad():
    for p in PROMPTS:
        ids = encode(path, p); past = None; inp = torch.tensor([ids]); gen, logits = [], []
        for _ in range(NEW_TOKENS):
            r = model(input_ids=inp, past_key_values=past, use_cache=True)
            past, lg = r.past_key_values, r.logits[0, -1].double()
            gen.append(int(lg.argmax())); logits.append(lg.float().numpy())
            inp = torch.tensor([[gen[-1]]])
        out.append(dict(prompt=p, prompt_ids=ids, tokens=gen)); all_logits.append(logits)
        print(repr(p), "->", gen)
json.dump(out, open(f"hf_reference{tag}.json", "w"), indent=1)
np.save(f"hf_logits{tag}.npy", np.array(all_logits, dtype=np.float32))   # (prompts, NEW_TOKENS, vocab)
