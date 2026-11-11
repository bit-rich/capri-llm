"""Apples-to-apples GPU timing per token over all linear layers, server side only.
Compute-only numbers: inputs already on the GPU, one synchronize per token.
`bench_gpu.py`: Qwen3-0.6B int8 (+ bf16 GEMVs and a full HF bf16 decode step).
`bench_gpu.py 4b`: Qwen3-4B int8 (int8 native vs private only).
`bench_gpu.py 8b w4`: Qwen3-8B GPTQ 4-bit; native and private both unpack the 4-bit weights per call.
private_ms is the v2 server (GEMV on c, 32-bit reply); private_v3_ms is the v3 server (pads from b[t] for all layers,
GEMV on c - u, 16-bit reply). `merge`: the private numbers use the merged 2-trips-per-block weights (after `run.py v3 merge`
has written merge_calib.json); native stays on the ordinary weights. Results go to bench_gpu*_v3.json."""
import json, sys, time, numpy as np, torch
import fold
from crypto import InnerLWE
from protocol import Qwen3, gemv_u32_gpu, gemv_i8_gpu, qwen_path, unpack4

w4, b4, merge = "w4" in sys.argv, "4b" in sys.argv or "1.7b" in sys.argv, "merge" in sys.argv   # b4: int8 native vs private only
repo, tag = ("Qwen/Qwen3-1.7B", "_1.7b") if "1.7b" in sys.argv else ("Qwen/Qwen3-4B", "_4b") if b4 else ("Qwen/Qwen3-0.6B", "")

def per_tok(step, n=200):
    for _ in range(5): step()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): step(); torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3

m = (Qwen3(qwen_path("Qwen/Qwen3-8B"), keep_float=False, w4="qwen3_8b_w4.npz") if w4 else
     Qwen3(qwen_path(repo), keep_float=False) if b4 else Qwen3())
def load(model):
    global lins, Wg, din, x8, c, c_host, idx
    lins = [L for _, L in model.linears()]
    Wg = [torch.as_tensor(L.packed if L.packed is not None else L.q, device="cuda") for L in lins]
    din = [L.q.shape[1] for L in lins]
    x8 = [torch.randint(-128, 128, (d,), dtype=torch.int8, device="cuda") for d in din]
    c = [torch.randint(-(1 << 31), 1 << 31, (d,), dtype=torch.int32, device="cuda") for d in din]
    c_host = [t.cpu().numpy() for t in c]; idx = range(len(lins))
W = lambda i: unpack4(Wg[i]) if lins[i].packed is not None else Wg[i]
load(m); del m
res = dict(native_int_ms=per_tok(lambda: [gemv_i8_gpu(W(i), x8[i]) for i in idx]))
if merge:
    load(Qwen3(qwen_path(repo), keep_float=False, merge=True, boost=json.load(open(f"merge_calib{tag}.json"))))
res.update(
    private_ms=per_tok(lambda: [gemv_u32_gpu(W(i), c[i]) for i in idx]),
    private_with_copies_ms=per_tok(lambda: [gemv_u32_gpu(W(i), torch.as_tensor(c_host[i], device="cuda")).cpu() for i in idx], n=50),
)
pads = fold.Pads([InnerLWE().public_a(i, d) for i, d in enumerate(din)], din, "cuda")
bt = np.random.default_rng(0).integers(0, 1 << 16, (fold.n, fold.LIMBS), dtype=np.uint16)       # b[t] as uploaded
def v3(copies):
    u = pads(bt)
    for i in idx:
        y = fold.reply_bits(gemv_u32_gpu(W(i), (torch.as_tensor(c_host[i], device="cuda") if copies else c[i]) - u[i]), 16)
        if copies: y.to(torch.int32).cpu()
res["private_v3_ms"] = per_tok(lambda: v3(False))
res["private_v3_with_copies_ms"] = per_tok(lambda: v3(True), n=50)
res["private_v3_pads_only_ms"] = per_tok(lambda: pads(bt))
if merge:
    print(json.dumps({k: round(v, 3) for k, v in res.items()}, indent=1))
    json.dump(res, open(f"bench_gpu{tag}_merge_v3.json", "w"), indent=1); sys.exit()
if not (w4 or b4):
    Wb = [torch.as_tensor(L.W, device="cuda").to(torch.bfloat16) for L in lins]
    xb = [torch.randn(d, 1, dtype=torch.bfloat16, device="cuda") for d in din]
    res["native_bf16_ms"] = per_tok(lambda: [w @ x for w, x in zip(Wb, xb)]); del Wb
del Wg; torch.cuda.empty_cache()
if w4 or b4:
    print(json.dumps({k: round(v, 3) for k, v in res.items()}, indent=1))
    json.dump(res, open("bench_gpu_8b_w4_v3.json" if w4 else f"bench_gpu{tag}_v3.json", "w"), indent=1); sys.exit()
# full native decode step (whole model incl. attention/norms), HF transformers bf16, KV cache of 32 tokens
from transformers import AutoModelForCausalLM
hf = AutoModelForCausalLM.from_pretrained(qwen_path(), dtype=torch.bfloat16).cuda().eval()
with torch.no_grad():
    out = hf(input_ids=torch.arange(32, device="cuda")[None], use_cache=True)
    tok = torch.tensor([[1]], device="cuda")
    def step():
        r = hf(input_ids=tok, past_key_values=out.past_key_values, use_cache=True)
        out.past_key_values.crop(-1)                     # drop the token just added
    res["hf_bf16_full_decode_step_ms"] = per_tok(step, n=100)
print(json.dumps({k: round(v, 3) for k, v in res.items()}, indent=1))
json.dump(res, open("bench_gpu_v3.json", "w"), indent=1)
