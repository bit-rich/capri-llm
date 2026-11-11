import time, json, os, sys, numpy as np
from protocol import *
from prompts import PROMPTS, NEW_TOKENS, CALIB, encode, decode

repo, tag = (("Qwen/Qwen3-8B", "_8b") if "8b" in sys.argv else ("Qwen/Qwen3-4B", "_4b") if "4b" in sys.argv
             else ("Qwen/Qwen3-1.7B", "_1.7b") if "1.7b" in sys.argv else ("Qwen/Qwen3-0.6B", ""))
small = tag == ""                                          # fp32 weights for the reference only fit for 0.6B
w4 = "w4" in sys.argv and not small                        # 8B with GPTQ 4-bit block weights (quant4.py)
path = qwen_path(repo)
merge = "merge" in sys.argv                                # 2 round trips per block instead of 4 (merged weights)
boost = None
if merge:      # per-layer input balance of the merged layers, calibrated once on public text and cached
    cf = f"merge_calib{tag}.json"
    if not os.path.exists(cf): json.dump(merge_boost(Qwen3(path, keep_float=False, merge=True), encode(path, CALIB)), open(cf, "w"), indent=1)
    boost = json.load(open(cf))
t = time.perf_counter(); model = Qwen3(path, keep_float=small, w4="qwen3_8b_w4.npz" if w4 else None, merge=merge, boost=boost); t_load = time.perf_counter() - t
v3 = "v3" in sys.argv                                      # c0 folded into the online reply (fold.py)
bits = next((int(a[5:]) for a in sys.argv if a.startswith("bits=")), 16)   # v3: bits per row in the online reply
lwe, bfv = InnerLWE(), (None if v3 else BFV32() if 'v2' in sys.argv else BFV())
NB = 4096 if v3 else bfv.N                                 # token slots per offline batch
prompt_ids = [encode(path, p) for p in PROMPTS]
T = sum(len(ids) + NEW_TOKENS - 1 for ids in prompt_ids)   # forward passes = alphas needed

gpu_off = (v3 or 'v2' in sys.argv) and 'cpu' not in sys.argv and gpu_available()   # offline BFV / one-time C1' on GPU
gpu = gpu_off and tag != "_8b" or gpu_off and w4   # online GEMVs on GPU if weights fit (8B int8 7.6 GB: no; 4B int8 4 GB, 8B 4-bit 4.1 GB: yes)
t = time.perf_counter()
server = FoldServer(model, lwe, gpu=gpu, gpu_setup=gpu_off, out_bits=bits) if v3 else Server(model, lwe, gpu=gpu)
t_setup = time.perf_counter() - t
client = (FoldClient(server, lwe, None, gpu=gpu_off and "cpuclient" not in sys.argv) if v3
          else Client(server, lwe, bfv, gpu=gpu_off))
off = client.offline(T)
if v3: server.hint = None                                  # one-time work is done: free its GPU memory
if gpu:
    import torch; torch.cuda.empty_cache(); server.upload_gpu()
print("offline:", {k: round(v, 3) for k, v in off.items()}, "gpu offline", gpu_off, "online", gpu,
      "load", round(t_load, 1), "setup", round(t_setup, 1), flush=True)

def greedy(ids, lin, begin=None, timer=None):
    """Prompt is fed one token at a time (decode-style prefill). Returns generated tokens + their logits."""
    cache = [([], []) for _ in model.blocks]; gen, logits = [], []
    for t in range(len(ids) + NEW_TOKENS - 1):
        if begin: begin()
        t0 = time.perf_counter()
        lg = forward_token(model, ids[t] if t < len(ids) else gen[-1], cache, lin)
        if timer is not None: timer.append(time.perf_counter() - t0)
        if t >= len(ids) - 1: logits.append(lg); gen.append(int(lg.argmax()))
    return gen, np.array(logits)

def rel_err(pv_toks, pv_lg, toks, lg):
    """Per-step logit rel. error; logits are only aligned step-by-step while the token histories agree."""
    n = next((i for i, (a, b) in enumerate(zip(pv_toks, toks)) if a != b), NEW_TOKENS) + 1
    return list(np.linalg.norm(pv_lg - lg, axis=1)[:n] / np.linalg.norm(lg, axis=1)[:n])

hf_file = f"hf_reference{tag}.json"
hf = json.load(open(hf_file)) if os.path.exists(hf_file) else None
hf_logits = np.load(f"hf_logits{tag}.npy") if hf else None
t_tok, per_prompt, rel_w8, rel_fp, rel_hf = [], [], [], [], []
for k, ids in enumerate(prompt_ids):
    w8_toks, w8_lg = greedy(ids, lambda n, L, x: L.plain(x))
    pv_toks, pv_lg = greedy(ids, client.lin, client.begin_token, t_tok)
    rel_w8 += rel_err(pv_toks, pv_lg, w8_toks, w8_lg)
    top2 = np.sort(w8_lg, axis=1)[:, -2:]
    r = dict(prompt=PROMPTS[k], private_text=decode(path, pv_toks), private_eq_w8=pv_toks == w8_toks,
             min_w8_top2_margin=float((top2[:, 1] - top2[:, 0]).min()))
    if small:
        fp_toks, fp_lg = greedy(ids, lambda n, L, x: L.float_(x))
        rel_fp += rel_err(pv_toks, pv_lg, fp_toks, fp_lg); r["private_eq_fp32"] = pv_toks == fp_toks
    if hf:
        assert hf[k]["prompt_ids"] == ids
        r["private_eq_hf"] = pv_toks == hf[k]["tokens"]; r["w8_eq_hf"] = w8_toks == hf[k]["tokens"]
        rel_hf += rel_err(pv_toks, pv_lg, hf[k]["tokens"], hf_logits[k])
        if small: r["fp32_eq_hf"] = fp_toks == hf[k]["tokens"]
    per_prompt.append(r); print(json.dumps(r), flush=True)

# plaintext server baseline: same GEMVs with int8 activations, on the same device as the private server
names = server.order
if merge:      # normal inference would not merge: time the baseline on the ordinary int8 weights
    base = Qwen3(path, keep_float=False); names = [n for n, _ in base.linears()]
    server.lin = dict(base.linears())
    if gpu: server.upload_gpu()
if gpu:   # activations already on the GPU (no copies), synchronised once per token
    import torch
    xs = {n: torch.zeros(server.lin[n].q.shape[1], dtype=torch.int8, device="cuda") for n in names}
    for n in names: gemv_i8_gpu(server.weight_gpu(n), xs[n])           # warm-up; 4-bit unpacks like private
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(T):
        for n in names: gemv_i8_gpu(server.weight_gpu(n), xs[n])
        torch.cuda.synchronize()
else:
    t = time.perf_counter()
    for _ in range(T):
        for n in names:
            L = server.lin[n]; gemv_i8(L.q, np.zeros(L.q.shape[1], np.int8))
t_plain_gemv = (time.perf_counter() - t) / T

res = dict(
    model=repo + (" (GPTQ 4-bit blocks, int8 head)" if w4 else "") + (" (merged, 2 trips per block)" if merge else ""), server_on_gpu=gpu, offline_on_gpu=gpu_off, tokens_forwarded=T, prompts=per_prompt,
    all_private_eq_w8=all(r["private_eq_w8"] for r in per_prompt),
    logits_rel_err_vs_w8_max=float(max(rel_w8)), logits_rel_err_vs_w8_mean=float(np.mean(rel_w8)),
    server_online_s_per_tok=server.time_online / T, server_plain_int8_s_per_tok=t_plain_gemv,
    online_overhead=server.time_online / T / t_plain_gemv,
    client_mask_s_per_tok=client.t_mask / T, client_unmask_s_per_tok=client.t_unmask / T,
    private_total_s_per_tok_excl_network=float(np.mean(t_tok)),
    rounds_per_tok=client.rounds / T,
    online_up_KB_per_tok=client.up_bytes / T / 1e3, online_down_KB_per_tok=client.down_bytes / T / 1e3,
    offline_server_s_per_tok_amortized=off["server_s"] / NB,
    offline_client_s_per_tok_amortized=off["client_s"] / NB,
    offline_up_KB_per_tok=off["up_bytes"] / NB / 1e3, offline_down_KB_per_tok=off["down_bytes"] / NB / 1e3,
    offline_server_s_batch=off["server_s"], offline_client_s_batch=off["client_s"], hint_setup_s=t_setup,
    D_total=int(sum(L.q.shape[0] for _, L in model.linears())), params=int(sum(L.q.size for _, L in model.linears())),
)
if v3:
    res.update(reply_bits=bits, client_offline_on_gpu=client.gpu, server_pads_s_per_tok=server.time_pads / T,
               offline_up_MB_batch=off["up_bytes"] / 1e6, public_c1_server_s_once=off["c1_server_s_once"],
               public_c1_download_GB_once=off["c1_bytes_once"] / 1e9)
if small:
    res.update(all_private_eq_fp32=all(r["private_eq_fp32"] for r in per_prompt),
               logits_rel_err_vs_fp32_max=float(max(rel_fp)), logits_rel_err_vs_fp32_mean=float(np.mean(rel_fp)))
if hf:
    res.update(all_private_eq_hf=all(r["private_eq_hf"] for r in per_prompt),
               logits_rel_err_vs_hf_max=float(max(rel_hf)), logits_rel_err_vs_hf_mean=float(np.mean(rel_hf)))
print(json.dumps(res, indent=1))
json.dump(res, open(f"results_qwen{tag}" + ("_w4" if w4 else "") + ("_merge" if merge else "") + ("_v3" + (f"_b{bits}" if bits != 16 else "") + ("_cpuclient" if "cpuclient" in sys.argv else "") if v3
                                                                  else "_v2" if "v2" in sys.argv else "") + ".json", "w"), indent=1)
