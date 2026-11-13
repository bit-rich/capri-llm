"""Private inference client over a real network: python client.py host=<server> [port=7000] [prompts=N] [new=N]
[k=N] [cache=DIR] [check]

Everything that is secret stays here: the prompt, the generated text, the masks and the BFV secret. The server gets
masked activations and the per-batch upload b, and nothing else (unless `check` is given, which is for tests only and
sends the tokens in the clear afterwards to compare with the server's plain W8 model).

Security: the public randomness (inner a_l and the BFV a_j) comes from the client's own fixed seeds, never from the
server; a server that could choose it could make the masks useless. The bundle the server sends (config, norms,
scales, embedding table) can only make the answers wrong, not leak the input.

`k=N`: speculative decoding with a local Qwen3-0.6B draft (Qwen targets only). `cache=DIR`: keep the one-time public
C1' and the bundle on disk instead of downloading them again on every run."""
import json, os, socket, sys, time, types, numpy as np, torch
import fold
from protocol import InnerLWE, FoldClient, forward_batch, cache_keep
from prompts import PROMPTS, NEW_TOKENS

opt = lambda name, default=None: next((a[len(name) + 1:] for a in sys.argv if a.startswith(name + "=")), default)
HOST, PORT, K = opt("host", "127.0.0.1"), int(opt("port", 7000)), int(opt("k", 0))
NEW, cache_dir = int(opt("new", NEW_TOKENS)), opt("cache")
from netclient import LWE_SEED, A_PUB_SEED


from netclient import RemoteServer, client_model, connect
import netclient


# ---------------------------------------------------------------- connect, bundle, RTT
c, s, bh, model, tok, rtt = connect(HOST, PORT, cache_dir)
prompts = PROMPTS if bh["repo"].startswith("Qwen/") else PROMPTS[:3] + ["Question: Name three primary colors.\nAnswer:"]
prompts = prompts[:int(opt("prompts", len(prompts)))]
ids_all = [tok.encode(p).ids for p in prompts]
draft = None
if K:
    assert bh["repo"].startswith("Qwen/"), "the draft model is Qwen3-0.6B: Qwen targets only"
    from transformers import AutoModelForCausalLM
    from protocol import qwen_path
    class Draft:
        def __init__(self):
            self.m = AutoModelForCausalLM.from_pretrained(qwen_path("Qwen/Qwen3-0.6B"), dtype=torch.float32).eval(); self.t = 0.0
        def reset(self): self.past, self.n = None, 0
        def propose(self, seq, k):
            t = time.perf_counter(); out, inp = [], seq[self.n:]
            with torch.no_grad():
                for _ in range(k):
                    r = self.m(input_ids=torch.tensor([inp]), past_key_values=self.past, use_cache=True)
                    self.past = r.past_key_values; self.n += len(inp); out.append(int(r.logits[0, -1].argmax())); inp = out[-1:]
            self.t += time.perf_counter() - t; return out
        def keep(self, n):
            if self.n > n: self.past.crop(n - self.n); self.n = n
    draft = Draft()

# ---------------------------------------------------------------- offline: one batch for this run
lwe = InnerLWE(LWE_SEED); remote = RemoteServer(c, bh, lwe)
client = FoldClient(remote, lwe, None, gpu=torch.cuda.is_available())
n_secrets = sum(len(i) for i in ids_all) + len(ids_all) * (NEW - 1) * (K + 1) + 8
up0, dn0, t = c.sent, c.recvd, time.perf_counter()
off = client.offline(n_secrets)
print(f"one-time + per-batch: {time.perf_counter() - t:.0f} s, up {(c.sent - up0) / 1e6:.0f} MB, down {(c.recvd - dn0) / 1e9:.2f} GB"
      f" ({'cached C1' if remote.cache and remote.time_c1 == 0 else 'C1 downloaded'})", flush=True)

# ---------------------------------------------------------------- online
results = []
for p, ids in zip(prompts, ids_all):
    cache = [([], []) for _ in model.blocks]
    u0, d0, r0, s0, t0 = c.sent, c.recvd, client.rounds, remote.server_s, time.perf_counter()
    client.begin_pass(len(ids)); lg = forward_batch(model, ids, cache, client.lin_batch, n_head=1)
    gen = [int(lg[-1].argmax())]; t_first = time.perf_counter() - t0
    if draft: draft.reset()
    while len(gen) < NEW:
        k = min(K, NEW - len(gen) - 1); seq = ids + gen
        d = draft.propose(seq, k) if k else []
        client.begin_pass(k + 1); n0 = len(cache[0][0])
        lg = forward_batch(model, [gen[-1]] + d, cache, client.lin_batch); pr = lg.argmax(1)
        a = next((i for i in range(k) if d[i] != pr[i]), k)
        gen += d[:a] + [int(pr[a])]; cache_keep(cache, n0 + a + 1)
        if draft: draft.keep(len(seq) + a)
    wall = time.perf_counter() - t0
    r = dict(prompt=p, text=tok.decode(gen), prompt_tokens=len(ids), new_tokens=len(gen), first_token_s=t_first, total_s=wall,
             reply_tokens_per_s=(len(gen) - 1) / (wall - t_first), round_trips=client.rounds - r0,
             up_MB=(c.sent - u0) / 1e6, down_MB=(c.recvd - d0) / 1e6, server_compute_s=remote.server_s - s0)
    if "check" in sys.argv:                                    # TEST ONLY: sends the tokens in the clear
        c.send(dict(op="plaincheck", new=NEW), [np.array(ids, np.int64)]); r["private_eq_w8"] = c.recv()[0]["tokens"] == gen
    results.append(r)
    print(f"\n{p!r} -> {r['text']!r}\n   first token after {t_first:.2f} s, {len(gen)} tokens in {wall:.2f} s "
          f"({r['reply_tokens_per_s']:.2f} tokens/s while writing), {r['round_trips']} round trips, up {r['up_MB']:.1f} MB, "
          f"down {r['down_MB']:.1f} MB, server compute {r['server_compute_s']:.2f} s"
          + (f", same as plain W8: {r['private_eq_w8']}" if "private_eq_w8" in r else ""), flush=True)
c.send(dict(op="bye")); s.close()
out = dict(repo=bh["repo"], host=HOST, round_trip_ms=1e3 * rtt, k=K, reply_bits=bh["out_bits"], results=results)
json.dump(out, open(f"results_net_{bh['repo'].split('/')[-1].lower()}_{opt('label', HOST)}.json", "w"), indent=1)
