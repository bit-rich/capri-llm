"""Client work per token at a big model's real sizes (default Qwen/Qwen3.8-27B; `repo=` for others), on this machine.
Times each part: masks, noise, packing, unpacking, unmasking, the token mixers (linear and full attention)."""
import sys, time, json, numpy as np, torch
import chainproto as CP
from protocol import delta_mix, attend, dgauss
from net import pack_bits, unpack_bits
from huggingface_hub import hf_hub_download
repo = next((a[5:] for a in sys.argv if a.startswith("repo=")), "Qwen/Qwen3.8-27B")
c = json.load(open(hf_hub_download(repo, "config.json"))); c = c.get("text_config", c)
d, ff, L, V = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"], c["vocab_size"]
hq, hkv = c["num_attention_heads"], c["num_key_value_heads"]; hd = c.get("head_dim") or d // hq
types = c.get("layer_types") or ["full_attention"] * L
gate = 2 if c.get("attn_output_gate") else 1
if "linear_attention" in types:
    nk, nv, dk, dv = c["linear_num_key_heads"], c["linear_num_value_heads"], c["linear_key_head_dim"], c["linear_value_head_dim"]
    lin_in, lin_out = 2 * nk * dk + 2 * nv * dv + 2 * nv, nv * dv
att_in, att_out = (gate * hq + 2 * hkv) * hd, hq * hd
ins = [d] + sum([[(lin_out if t == "linear_attention" else att_out), d, ff, d] for t in types], []) + [d]
outs_res = 2 * L * d; outs_chain = sum(2 * ff + (lin_in if t == "linear_attention" else att_in) for t in types) + V
rng = np.random.default_rng(0); dev = "cuda"
def timeit(f, n=3):
    f(); torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n
a = {i: CP.inner_a(i, w) for i, w in enumerate(ins)}; M = CP.Masks(a, dev); al = dgauss(rng, (1, CP.N_LWE), 6.4)
X = [rng.integers(0, 1 << 50, (1, w)) for w in ins]
yc = rng.integers(0, 1 << 24, outs_chain); yr = rng.integers(0, 1 << 55, outs_res); ycb, yrb = pack_bits(yc, 24), pack_bits(yr, 56)
z = rng.integers(0, 1 << 56, outs_chain + outs_res); y = rng.integers(0, 1 << 56, outs_chain + outs_res)
gen = torch.Generator(device=dev); gen.manual_seed(1)
def masked_upload(pre, i, w, x):                              # what ChainClient._mask does for a network link
    e = torch.round(torch.randn((1, w), generator=gen, device=dev, dtype=torch.float32) * 6.4).long()
    C = (pre[i][:, :w] + e + torch.as_tensor(x, device=dev)) & ((1 << 56) - 1)
    return C.view(torch.uint8).view(-1, 8)[:, :7].contiguous().cpu()
def all_uploads():
    pre = M.all(al)                                          # every layer's A alpha, once per pass
    return [masked_upload(pre, i, w, x) for (i, w), x in zip(enumerate(ins), X)]
res = {"masks + noise + packing (GPU)": timeit(all_uploads),
       "unpacking replies": timeit(lambda: (unpack_bits(ycb, 24, outs_chain), unpack_bits(yrb, 56, outs_res))),
       "unmasking": timeit(lambda: (((y - z + (1 << 54)) & ((1 << 55) - 1)) - (1 << 54)) / 3.0)}
m = type("M", (), {})(); m.eps = 1e-6
if "linear_attention" in types:
    m.lin_dims = [nk, nv, dk, dv, 4]
    b = dict(conv=rng.normal(size=(2 * nk * dk + nv * dv, 4)), A_log=rng.normal(size=nv), dt_bias=rng.normal(size=nv), gnorm=np.ones(dv))
    rows = rng.normal(size=(1, lin_in)); cache = ([], [])
    res[f"linear-attention layers ({types.count('linear_attention')})"] = timeit(lambda: delta_mix(m, b, rows, cache, len(cache[0]))) * types.count("linear_attention")
K = rng.normal(size=(500, hkv, hd)); q = rng.normal(size=(hq, hd))
res[f"full-attention layers ({types.count('full_attention')}, 500-token context)"] = timeit(lambda: attend(q, K, K)) * types.count("full_attention")
print(f"{repo}: {sum(ins) / 1e6:.2f} M masked inputs, {(outs_res + outs_chain) / 1e6:.2f} M outputs, {len(ins) // 2} trips per token")
for k, v in res.items(): print(f"  {k:46s} {1e3 * v:7.1f} ms")
print(f"  {'total client work per token':46s} {1e3 * sum(res.values()):7.1f} ms")
