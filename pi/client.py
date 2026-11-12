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
import hashlib, json, os, socket, sys, time, types, numpy as np, torch
import fold
from net import Conn, unpack_bits, client_auth, token_from_env
from protocol import InnerLWE, FoldClient, forward_batch, cache_keep
from prompts import PROMPTS, NEW_TOKENS

opt = lambda name, default=None: next((a[len(name) + 1:] for a in sys.argv if a.startswith(name + "=")), default)
HOST, PORT, K = opt("host", "127.0.0.1"), int(opt("port", 7000)), int(opt("k", 0))
NEW, cache_dir = int(opt("new", NEW_TOKENS)), opt("cache")
LWE_SEED, A_PUB_SEED = 0, 2                    # the client's own: never taken from the server


class RemoteServer:
    """What FoldClient needs from a server, over the connection. Public randomness is regenerated locally."""
    def __init__(self, conn, bundle_hdr, lwe):
        self.c, self.out_bits = conn, bundle_hdr["out_bits"]
        self.order = [n for n, *_ in bundle_hdr["linears"]]
        shapes = {n: (r, k) for n, r, k, _ in bundle_hdr["linears"]}
        self.lin = {n: types.SimpleNamespace(q=np.broadcast_to(np.int8(0), shapes[n])) for n in self.order}   # shape only
        self.a = {n: lwe.public_a(lid, shapes[n][1]) for lid, n in enumerate(self.order)}
        self.a_pub = fold.public_a(A_PUB_SEED)
        self.time_c1 = self.server_s = 0.0
        self.cache = None
        if cache_dir:
            key = hashlib.sha256(json.dumps(bundle_hdr["linears"]).encode() + bundle_hdr["repo"].encode()).hexdigest()[:16]
            self.cache = os.path.join(cache_dir, "c1_" + key); os.makedirs(self.cache, exist_ok=True)

    def public_c1(self, name):
        f = self.cache and os.path.join(self.cache, name + ".u32")
        if f and os.path.exists(f + ".done"):                  # from disk: read in row chunks
            rows = self.lin[name].q.shape[0]; mm = np.memmap(f, np.uint32, "r", shape=(rows, fold.N))
            for r0 in range(0, rows, 4096): yield r0, torch.from_numpy(mm[r0:r0 + 4096].astype(np.int64))
            return
        out = f and np.memmap(f, np.uint32, "w+", shape=(self.lin[name].q.shape[0], fold.N))
        t = time.perf_counter(); self.c.send(dict(op="c1", name=name))
        while True:
            h, a = self.c.recv()
            if h["op"] == "c1done": break
            if out is not None: out[h["r0"]:h["r0"] + len(a[0])] = a[0]
            self.time_c1 += time.perf_counter() - t
            yield h["r0"], torch.from_numpy(a[0].astype(np.int64))
            t = time.perf_counter()
        if out is not None: out.flush(); del out; open(f + ".done", "w").close()

    def receive_b(self, b):
        self.c.send(dict(op="b"), [b]); assert self.c.recv()[0]["op"] == "ok"

    def online_batch(self, name, toks, C):
        self.c.send(dict(op="lin", name=name, toks=[int(t) for t in toks]), [C])
        h, a = self.c.recv(); self.server_s += h["server_s"]
        return unpack_bits(a[0], self.out_bits, int(np.prod(h["shape"]))).reshape(h["shape"])


def client_model(h, arrays):
    """The client's half of the model, from the public bundle."""
    inv_freq, norm, embed, tok_json = arrays[:4]; i = 4
    m = types.SimpleNamespace(d=h["d"], L=h["L"], vocab=h["vocab"], hq=h["hq"], hkv=h["hkv"], hd=h["hd"], eps=h["eps"],
                              rope_scale=h["rope_scale"], inv_freq=inv_freq, norm=norm, embed=embed, merged=True)
    lin = {}
    for n, r, k, rn in h["linears"]:
        lin[n] = types.SimpleNamespace(scale=arrays[i], row_norm_max=rn); i += 1
    m.head, m.blocks = lin["head"], []
    for j, (b, ks) in enumerate(zip(h["blocks"], h["block_vectors"])):
        blk = {k: v for k, v in b.items() if v is not None}
        for k in ks: blk[k] = arrays[i]; i += 1
        for k in ("qkv", "ob", "da", "down"):
            if f"{j}.{k}" in lin: blk[k] = lin[f"{j}.{k}"]
        m.blocks.append(blk)
    from tokenizers import Tokenizer
    return m, Tokenizer.from_str(bytes(tok_json).decode())


# ---------------------------------------------------------------- connect, bundle, RTT
s = socket.create_connection((HOST, PORT)); c = Conn(s)
token = token_from_env()
if token: client_auth(c, token)
rtt = []
for _ in range(10):
    t = time.perf_counter(); c.send(dict(op="ping")); h, _ = c.recv(); rtt.append(time.perf_counter() - t)
    if h["op"] == "challenge": raise SystemExit("the server needs a password: set PI_TOKEN to the same value as on the server")
rtt = float(np.median(rtt))
t = time.perf_counter(); c.send(dict(op="bundle")); bh, ba = c.recv(); t_bundle = time.perf_counter() - t
assert (bh["lwe_seed"], bh["a_pub_seed"]) == (LWE_SEED, A_PUB_SEED), "server uses different public randomness"
model, tok = client_model(bh, ba)
print(f"{bh['repo']} at {HOST}:{PORT}: round trip {1e3 * rtt:.1f} ms, bundle {c.recvd / 1e6:.0f} MB in {t_bundle:.1f} s", flush=True)

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
