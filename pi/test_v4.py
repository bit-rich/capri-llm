"""v4 chain protocol end to end with the real crypto (fold56): private vs the plain unmerged int8 model.
python test_v4.py [repo=Qwen/Qwen3-0.6B] [N=8192]"""
import sys, time, numpy as np, torch
from protocol import *
import chainproto as CP, fold56 as F
from prompts import PROMPTS
opt = lambda k, d: next((a[len(k) + 1:] for a in sys.argv if a.startswith(k + "=")), d)
repo = opt("repo", "Qwen/Qwen3-0.6B"); path = qwen_path(repo)
from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
m = Qwen3(path, keep_float=False)
t = time.perf_counter(); srv = CP.ChainServer(m, p=F.P(N=int(opt("N", 8192)))); cl = CP.ChainClient(srv)
ids_all = [tok.encode(p).ids for p in PROMPTS[:3]]; n = sum(len(i) for i in ids_all) + 3 * 8 + 4
off = cl.offline(n)
print(f"setup {time.perf_counter() - t:.0f} s: one-time C1' {srv.time_c1:.0f} s, per-batch upload {off['upload_MB']:.0f} MB, client offline {cl.t_offline:.0f} s", flush=True)
for p, ids in zip(PROMPTS[:3], ids_all):
    c1 = [([], []) for _ in m.blocks]; gen_w8 = []; w8 = []
    for k in range(len(ids) + 7):
        lg = forward_token(m, ids[k] if k < len(ids) else gen_w8[-1], c1, lambda nm, L, x: L.plain(x))
        if k >= len(ids) - 1: w8.append(lg); gen_w8.append(int(lg.argmax()))
    c2 = [([], []) for _ in m.blocks]; r0, u0, d0 = cl.rounds, cl.up, cl.down
    cl.begin_pass(len(ids)); pv = [cl.forward(ids, c2, n_head=1)[-1]]; gen = [int(pv[-1].argmax())]
    while len(gen) < 8:
        cl.begin_pass(1); pv.append(cl.forward([gen[-1]], c2)[-1]); gen.append(int(pv[-1].argmax()))
    rel = np.linalg.norm(np.array(pv) - np.array(w8), axis=1) / np.linalg.norm(np.array(w8), axis=1)
    print(f"{p[:28]!r}: same 8 tokens as plain W8: {gen == gen_w8}, rel logit err mean {rel.mean():.2e} max {rel.max():.2e}; "
          f"{cl.rounds - r0} trips for 8 passes, up {(cl.up - u0) / 1e6 / 8:.2f} MB, down {(cl.down - d0) / 1e6 / 8:.2f} MB per pass", flush=True)
print(f"server online {1e3 * srv.time_online / cl.rounds * (2 * m.L + 2):.1f} ms per pass (incl. pads {1e3 * srv.time_pads / max(cl.rounds, 1) * (2 * m.L + 2):.1f} ms)")
