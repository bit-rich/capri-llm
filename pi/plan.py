"""Sizes of a v3 run for any supported model, from its config.json alone (nothing else is downloaded):
`python plan.py mistralai/Mistral-Medium-3.5-128B [bits=20]`. Use it to pick the machine before a big run."""
import json, sys
from huggingface_hub import hf_hub_download

repo = sys.argv[1]; bits = next((int(a[5:]) for a in sys.argv if a.startswith("bits=")), 20)
c = json.load(open(hf_hub_download(repo, "config.json"))); c = c.get("text_config", c)
d, L, ff, V, hq, hkv = (c[k] for k in ("hidden_size", "num_hidden_layers", "intermediate_size", "vocab_size",
                                       "num_attention_heads", "num_key_value_heads"))
hd = c.get("head_dim") or d // hq
types = c.get("layer_types") or ["full_attention"] * L
def block_io(i):                                            # (rows of the input projection, input width of the output projection)
    if types[i] == "linear_attention":                      # Qwen3.5 family: gated DeltaNet [q k v | z | b | a], out_proj
        nk, nv, dk, dv = (c[k] for k in ("linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim"))
        return 2 * nk * dk + 2 * nv * dv + 2 * nv, nv * dv
    return (hq * (2 if c.get("attn_output_gate") else 1) + 2 * hkv) * hd, hq * hd
def tot(layers):                                            # (weights, output rows, input width) per token
    return [sum(r * k for r, k in layers), sum(r for r, k in layers), sum(k for r, k in layers)]
head = (V, d)
plain, merged = [], [(block_io(0)[0], d)]
for i in range(L):
    rin, oin = block_io(i)
    plain += [(rin, d), (d, oin), (2 * ff, d), (d, ff)]
    merged.append((d + 2 * ff, oin + d))
    merged.append((d + block_io(i + 1)[0], ff + d) if i < L - 1 else (d, ff))
runs = {"4 trips per block": (tot(plain + [head]), 4 * L + 1), "2 trips per block (merge)": (tot(merged + [head]), 2 * L + 2)}
print(f"{repo}: d {d}, {L} layers, mlp {ff}, vocab {V}, heads {hq}/{hkv} x {hd}")
for name, ((w, rows, win), trips) in runs.items():
    print(f"\n{name}: {trips} round trips per pass")
    print(f"  server weights (int8), GPU memory needed   {w / 1e9:8.1f} GB")
    print(f"  upload per token                           {4 * win / 1e6:8.1f} MB")
    print(f"  download per token ({bits}-bit reply)          {rows * bits / 8 / 1e6:8.1f} MB")
    print(f"  one-time public download C1'               {4 * rows * 4096 / 1e9:8.1f} GB   (run.py streams it, never stored)")
    print(f"  client RAM for unmasking values            {4 * rows * 100 / 1e9:8.1f} GB per 100 tokens of the batch in use")
    print(f"  one-time C1' work                          {5 * w * 4096 / 1e15:8.2f} e15 int8 multiply-adds")
print(f"\nper batch of 4096 tokens: 41.9 MB upload, nothing downloaded")
print(f"host RAM: the int8 weights are also held in RAM (plus the unmerged copy if the native baseline is timed)")
