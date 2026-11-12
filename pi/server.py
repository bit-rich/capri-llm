"""Private inference server (v3 + merge) over TCP. It holds the public weights and sees only masked data.

  python server.py [repo=<HF id> | 1.7b | 4b] [bits=20] [gpuload] [port=7000] [host=127.0.0.1]  (host=0.0.0.0 to accept remote clients; there is no authentication, so firewall it)

Per connection it serves, in order: the public client bundle (everything except the big weight matrices), the
public one-time C1' (computed on request), the client's per-batch upload b, then the online requests.
`plaincheck` is for TESTS ONLY: the client sends its tokens in the clear so the server can compare with the plain
W8 model. A real client never sends it."""
import json, os, socket, sys, time, numpy as np, torch
from protocol import *
from prompts import CALIB, encode
from net import Conn, pack_bits, server_auth, token_from_env
import fold

opt = lambda name, default=None: next((a[len(name) + 1:] for a in sys.argv if a.startswith(name + "=")), default)
repo = opt("repo") or {"1.7b": "Qwen/Qwen3-1.7B", "4b": "Qwen/Qwen3-4B"}.get(next((a for a in sys.argv[1:] if a in ("1.7b", "4b")), ""), "Qwen/Qwen3-0.6B")
tag = {"Qwen/Qwen3-0.6B": "", "Qwen/Qwen3-1.7B": "_1.7b", "Qwen/Qwen3-4B": "_4b"}.get(repo, "_" + repo.split("/")[-1].lower())
bits, port, host = int(opt("bits", 20)), int(opt("port", 7000)), opt("host", "127.0.0.1")
token = token_from_env()
if not token and host not in ("127.0.0.1", "localhost"):
    raise SystemExit("refusing to accept remote clients without a password: set PI_TOKEN (see README)")
gpu = torch.cuda.is_available() and "cpu" not in sys.argv
dev = "cuda" if gpu and "gpuload" in sys.argv else None

path = qwen_path(repo)
cf = f"merge_calib{tag}.json"
if not os.path.exists(cf):
    json.dump(merge_boost(Qwen3(path, keep_float=False, merge=True, dev=dev), encode(path, CALIB)[:int(opt("calib", 10 ** 6))]), open(cf, "w"), indent=1)
t = time.perf_counter()
model = Qwen3(path, keep_float=False, merge=True, boost=json.load(open(cf)), dev=dev)
lwe = InnerLWE(); srv = FoldServer(model, lwe, gpu=gpu, gpu_setup=gpu, out_bits=bits)
if gpu: torch.cuda.empty_cache(); srv.upload_gpu()
print(f"{repo}: loaded in {time.perf_counter() - t:.0f} s, weights on {'GPU' if gpu else 'CPU'}, {bits}-bit replies", flush=True)

def bundle():
    """Public data the client needs besides the weight matrices: config, norm weights, per-row scales, embedding table."""
    hdr = dict(op="bundle", repo=repo, d=model.d, L=model.L, vocab=model.vocab, hq=model.hq, hkv=model.hkv, hd=model.hd,
               eps=model.eps, rope_scale=model.rope_scale, out_bits=bits, lwe_seed=lwe.seed, a_pub_seed=2,
               linears=[[n, *L.q.shape, float(L.row_norm_max)] for n, L in model.linears()],
               blocks=[{k: (v if np.isscalar(v) else None) for k, v in b.items() if not isinstance(v, Linear)} for b in model.blocks])
    arrays = [np.asarray(model.inv_freq, np.float64), np.asarray(model.norm, np.float64), np.asarray(model.embed),
              np.frombuffer(open(os.path.join(path, "tokenizer.json"), "rb").read(), np.uint8)]
    arrays += [L.scale.astype(np.float64) for _, L in model.linears()]
    hdr["block_vectors"] = []
    for b in model.blocks:
        ks = sorted(k for k, v in b.items() if isinstance(v, np.ndarray)); hdr["block_vectors"].append(ks)
        arrays += [b[k].astype(np.float64) for k in ks]
    return hdr, arrays

def serve(c):
    while True:
        h, a = c.recv(); op = h["op"]
        if op == "bye": return
        if op == "ping": c.send(dict(op="pong"))
        elif op == "bundle": c.send(*bundle())
        elif op == "c1":                                       # one-time public download, one layer at a time
            for r0, rows in srv.public_c1(h["name"]):
                c.send(dict(op="c1chunk", r0=r0), [(rows - ((rows >> 31) << 32)).to(torch.int32).cpu().numpy().view(np.uint32)])
            c.send(dict(op="c1done"))
        elif op == "b": srv.receive_b(a[0]); srv.pos = (); c.send(dict(op="ok"))
        elif op == "lin":
            t0 = time.perf_counter(); y = srv.online_batch(h["name"], np.asarray(h["toks"]), a[0])
            c.send(dict(op="y", shape=list(y.shape), server_s=time.perf_counter() - t0), [pack_bits(y, bits)])
        elif op == "plaincheck":                               # TEST ONLY: tokens in the clear
            ids, gen = list(map(int, a[0])), []; cache = [([], []) for _ in model.blocks]
            for t in range(len(ids) + h["new"] - 1):
                lg = forward_token(model, ids[t] if t < len(ids) else gen[-1], cache, lambda n, L, x: L.plain(x))
                if t >= len(ids) - 1: gen.append(int(lg.argmax()))
            c.send(dict(op="plain", tokens=gen))
        else: raise ValueError(op)

ls = socket.socket(); ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); ls.bind((host, port)); ls.listen(1)
print(f"listening on {host}:{port}", flush=True)
while True:
    s, addr = ls.accept(); print("client", addr, flush=True); c = Conn(s); s.settimeout(30)   # 30 s to log in
    try:
        if token: server_auth(c, token)
        s.settimeout(None); serve(c)
    except (ConnectionError, ValueError, KeyError, PermissionError, OSError) as e: print("dropped:", repr(e)[:200], flush=True)
    finally: s.close(); print(f"done: sent {c.sent / 1e6:.1f} MB, received {c.recvd / 1e6:.1f} MB", flush=True)
