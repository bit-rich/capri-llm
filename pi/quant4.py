"""4-bit weights for Qwen3-8B that the private protocol can use: GPTQ (Frantar et al. 2022) with ONE scale per
output row (protocol constraint: the server computes plain integer M c, so per-group scales would need one
masked output per group). Act-order, per-row clip search, lm_head stays int8 (RTN, as in protocol.Linear).
Runs layer by layer on the GPU (bf16 model stays on the CPU), calibrated on 128 x 512 WikiText-2 train tokens.
Also reports WikiText-2 test perplexity for bf16 / W8 RTN / W4 RTN / W4 GPTQ.
Writes qwen3_8b_w4.npz: per fused linear ("{i}.qkv", "{i}.o", "{i}.gate_up", "{i}.down") packed nibbles + scales."""
import glob, time, numpy as np, torch, pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM
from prompts import encode
from protocol import qwen_path

dev, SEQ, N_CAL, N_EVAL = "cuda", 512, 128, 40
path = qwen_path("Qwen/Qwen3-8B")
wt = snapshot_download("Salesforce/wikitext", repo_type="dataset", allow_patterns=["wikitext-2-raw-v1/*"])
text = lambda split: "".join(pq.read_table(glob.glob(f"{wt}/wikitext-2-raw-v1/{split}-*")[0]).column("text").to_pylist())
cal = torch.tensor(encode(path, text("train")))
g = torch.Generator().manual_seed(0)
cal = torch.stack([cal[s:s + SEQ] for s in torch.randint(0, len(cal) - SEQ, (N_CAL,), generator=g).tolist()])
ev = torch.tensor(encode(path, text("test"))[:N_EVAL * SEQ]).reshape(N_EVAL, SEQ)

model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
layers = model.model.layers
pos = torch.arange(SEQ, device=dev)[None]
rot = model.model.rotary_emb.to(dev)

def fused(layer):
    a, m = layer.self_attn, layer.mlp
    return {"qkv": [a.q_proj, a.k_proj, a.v_proj], "o": [a.o_proj], "gate_up": [m.gate_proj, m.up_proj], "down": [m.down_proj]}

def quant_rtn(W, maxq):                          # per-row symmetric round-to-nearest (what protocol.Linear does)
    s = W.abs().amax(1).clamp_min(1e-30) / maxq
    return torch.clamp(torch.round(W / s[:, None]), -maxq, maxq), s

def gptq(W, H, maxq=7, block=128, damp=0.01):
    """W (rows, cols) fp32, H = sum x x^T over calibration inputs. Returns integer Q in [-maxq, maxq] and row scales."""
    W = W.clone(); cols = W.shape[1]
    dead = torch.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
    best, s = None, None                         # per-row clip ratio minimising diag(H)-weighted RTN error
    for r in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7):
        sr = W.abs().amax(1).clamp_min(1e-30) * r / maxq
        err = ((W - torch.clamp(torch.round(W / sr[:, None]), -maxq, maxq) * sr[:, None]) ** 2 * torch.diag(H)).sum(1)
        if best is None: best, s = err, sr
        else: s = torch.where(err < best, sr, s); best = torch.minimum(err, best)
    perm = torch.argsort(torch.diag(H), descending=True); W, H = W[:, perm], H[perm][:, perm]
    H += damp * torch.diag(H).mean() * torch.eye(cols, device=W.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    for i1 in range(0, cols, block):
        i2 = min(i1 + block, cols); W1 = W[:, i1:i2].clone(); E1 = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            q = torch.clamp(torch.round(W1[:, i] / s), -maxq, maxq); Q[:, i1 + i] = q
            e = (W1[:, i] - q * s) / Hi[i, i]
            W1[:, i:] -= e[:, None] * Hi[i, i:][None]; E1[:, i] = e
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q[:, torch.argsort(perm)], s

def run_layer(layer, h, bs=8):
    cos, sin = rot(h[:1].to(dev), pos)
    out = []
    for b in range(0, len(h), bs):
        o = layer(h[b:b + bs].to(dev), attention_mask=None, position_ids=pos, position_embeddings=(cos, sin))
        out.append((o[0] if isinstance(o, tuple) else o).cpu())
    return torch.cat(out)

_head = []
def head_int8():                                 # lm_head int8 RTN (dequantized, bf16), on the CPU in row chunks
    if not _head:
        W = model.lm_head.weight.data
        _head.append(torch.cat([(lambda q, s: (q * s[:, None]).bfloat16())(*quant_rtn(W[r:r + 16384].float(), 127))
                                for r in range(0, W.shape[0], 16384)]))
    return _head[0]

@torch.no_grad()
def ppl(qfn):
    """WikiText-2 test perplexity; qfn(W fp32) -> dequantized W (None = bf16 weights as they are)."""
    h = model.model.embed_tokens(ev)
    for layer in layers:
        layer.to(dev); saved = {}
        if qfn:
            for lins in fused(layer).values():
                for l in lins: saved[l] = l.weight.data.clone(); l.weight.data = qfn(l.weight.data.float()).bfloat16()
        h = run_layer(layer, h)
        for l, w in saved.items(): l.weight.data = w
        layer.to("cpu")
    model.model.norm.to(dev); nll = 0.0
    head = (head_int8() if qfn else model.lm_head.weight.data).to(dev)
    for b in range(N_EVAL):
        lg = (model.model.norm(h[b:b + 1].to(dev)) @ head.T).float()[0]
        nll += torch.nn.functional.cross_entropy(lg[:-1], ev[b, 1:].to(dev), reduction="sum").item()
    model.model.norm.to("cpu"); del head; torch.cuda.empty_cache()
    return float(np.exp(nll / (N_EVAL * (SEQ - 1))))

deq = lambda maxq: (lambda W: (lambda q, s: q * s[:, None])(*quant_rtn(W, maxq)))
res = {}
for name, f in (("bf16", None), ("w8_rtn", deq(127)), ("w4_rtn", deq(7))):
    t = time.perf_counter(); res[name] = ppl(f); print(name, round(res[name], 3), f"({time.perf_counter() - t:.0f}s)", flush=True)

# ---- GPTQ, sequential: each layer is calibrated on the outputs of the already-quantized layers before it
t0 = time.perf_counter(); out = {}
with torch.no_grad():
    h = model.model.embed_tokens(cal)
    for i, layer in enumerate(layers):
        layer.to(dev); Hs, hooks = {}, []
        for key, lins in fused(layer).items():
            def hook(mod, inp, _o, key=key):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                Hs[key] = Hs.get(key, 0) + x.T @ x
            hooks.append(lins[0].register_forward_hook(hook))
        run_layer(layer, h)
        for hk in hooks: hk.remove()
        for key, lins in fused(layer).items():
            W = torch.cat([l.weight.data.float() for l in lins])
            Q, s = gptq(W, Hs.pop(key))
            Wq = (Q * s[:, None]).bfloat16(); r = 0
            for l in lins: n = l.weight.shape[0]; l.weight.data = Wq[r:r + n].clone(); r += n
            q = Q.to(torch.int8).cpu().numpy()
            out[f"{i}.{key}.packed"] = ((q[:, 0::2] & 0xF) | (q[:, 1::2] << 4)).astype(np.uint8)
            out[f"{i}.{key}.scale"] = s.cpu().numpy().astype(np.float32)
        del Hs; h = run_layer(layer, h); layer.to("cpu"); torch.cuda.empty_cache()
        print(f"layer {i} done ({time.perf_counter() - t0:.0f}s)", flush=True)
np.savez("qwen3_8b_w4.npz", **out)
res["gptq_s"] = time.perf_counter() - t0
res["w4_gptq"] = ppl(lambda W: W)               # layers now hold the GPTQ-dequantized weights; head int8 RTN
print({k: round(v, 3) for k, v in res.items()})
import json; json.dump(res, open("quant4_ppl.json", "w"), indent=1)
