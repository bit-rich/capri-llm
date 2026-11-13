"""Private inference server (v3 + merge, or v4 chain with `v4`) over TCP. It holds the public weights and sees only masked data.

  python server.py [repo=<HF id> | 1.7b | 4b] [bits=20] [gpuload] [weights=DIR] [prepare=DIR] [port=7000] [host=127.0.0.1]  (host=0.0.0.0 to accept remote clients; there is no authentication, so firewall it)

Per connection it serves, in order: the public client bundle (everything except the big weight matrices), the
public one-time C1' (computed on request), the client's per-batch upload b, then the online requests.
`plaincheck` is for TESTS ONLY: the client sends its tokens in the clear so the server can compare with the plain
W8 model. A real client never sends it.

`prepare=DIR`: load and quantize the model, then write to DIR the prepared weights (DIR/weights), the public bundle
(DIR/bundle_<id>) and the one-time C1' (DIR/c1_<id>), and exit. The last two are what a client caches.
`weights=DIR/weights`: serve those prepared weights (bit-identical every time, and no decoding or quantizing)."""
import json, os, socket, sys, time, numpy as np, torch
from protocol import *
from prompts import CALIB, encode
from net import Conn, pack_bits, unpack_bits, server_auth, token_from_env
import fold
v4 = "v4" in sys.argv                                          # 56-bit chain mode: original weights, 2 trips per block
if v4:
    import chainproto as CP, fold56 as F56

opt = lambda name, default=None: next((a[len(name) + 1:] for a in sys.argv if a.startswith(name + "=")), default)
repo = opt("repo") or {"1.7b": "Qwen/Qwen3-1.7B", "4b": "Qwen/Qwen3-4B"}.get(next((a for a in sys.argv[1:] if a in ("1.7b", "4b")), ""), "Qwen/Qwen3-0.6B")
tag = {"Qwen/Qwen3-0.6B": "", "Qwen/Qwen3-1.7B": "_1.7b", "Qwen/Qwen3-4B": "_4b"}.get(repo, "_" + repo.split("/")[-1].lower())
bits, port, host = int(opt("bits", 20)), int(opt("port", 7000)), opt("host", "127.0.0.1")
token = token_from_env()
if not token and host not in ("127.0.0.1", "localhost"):
    raise SystemExit("refusing to accept remote clients without a password: set PI_TOKEN (see README)")
gpu = torch.cuda.is_available() and "cpu" not in sys.argv
dev = "cuda" if gpu and "gpuload" in sys.argv else None

t = time.perf_counter()
if opt("weights"):
    model = load_prepared(opt("weights")); tok_path = os.path.join(opt("weights"), "tokenizer.json")
else:
    path = qwen_path(repo); tok_path = os.path.join(path, "tokenizer.json")
    cf = f"merge_calib{tag}.json"
    if v4: pass
    elif not os.path.exists(cf):
        json.dump(merge_boost(Qwen3(path, keep_float=False, merge=True, dev=dev), encode(path, CALIB)[:int(opt("calib", 10 ** 6))]), open(cf, "w"), indent=1)
    model = (Qwen3(path, keep_float=False, dev=dev) if v4 else Qwen3(path, keep_float=False, merge=True, boost=json.load(open(cf)), dev=dev))
    model.weights_id = None
lwe = InnerLWE()
if v4:
    assert gpu, "v4 needs a CUDA GPU"
    srv = CP.ChainServer(model); bits = CP.RB
else: srv = FoldServer(model, lwe, gpu=gpu, gpu_setup=gpu, out_bits=bits)
print(f"{repo}: loaded in {time.perf_counter() - t:.0f} s", flush=True)

def bundle():
    """Public data the client needs besides the weight matrices: config, norm weights, per-row scales, embedding table."""
    hdr = dict(op="bundle", repo=repo, weights_id=model.weights_id, attn_gate=model.attn_gate, lin_dims=model.lin_dims, d=model.d, L=model.L, vocab=model.vocab, hq=model.hq, hkv=model.hkv, hd=model.hd,
               eps=model.eps, rope_scale=model.rope_scale, out_bits=bits, lwe_seed=lwe.seed, a_pub_seed=2,
               linears=[[n, *L.q.shape, float(L.row_norm_max)] for n, L in model.linears()],
               blocks=[{k: (v if np.isscalar(v) else None) for k, v in b.items() if not isinstance(v, Linear)} for b in model.blocks])
    arrays = [np.asarray(model.inv_freq, np.float64), np.asarray(model.norm, np.float64), np.asarray(model.embed),
              np.frombuffer(open(tok_path, "rb").read(), np.uint8)]
    arrays += [L.scale.astype(np.float64) for _, L in model.linears()]
    hdr["block_vectors"] = []
    for b in model.blocks:
        ks = sorted(k for k, v in b.items() if isinstance(v, np.ndarray)); hdr["block_vectors"].append(ks)
        arrays += [b[k].astype(np.float64) for k in ks]
    hdr["mode"] = "v4" if v4 else "v3"
    if v4:                                                     # public numbers the v4 client scales with, + the W multipliers
        hdr["v4"] = dict(n=CP.N_LWE, N=srv.p.N, inner_seed=21, a_pub_seed=3, pairs=[[r, g] for r, g, _ in srv.pairs],
                         singles=srv.singles, pub={k: [v["rn"], v["spec"]] for k, v in srv.pub.items()},
                         W={r: [w["K"], w["Wmax"]] for r, w in srv.W.items()}, W_order=[r for r, _, _ in srv.pairs])
        arrays += [srv.W[r]["W"].astype(np.int64) for r, _, _ in srv.pairs]
    return hdr, arrays

def c1_rows(name):
    for r0, rows in srv.public_c1(name):
        yield r0, (rows - ((rows >> 31) << 32)).to(torch.int32).cpu().numpy().view(np.uint32)

if opt("prepare") and v4: raise SystemExit("prepare= is v3 only for now")
if opt("prepare"):                                             # write weights, bundle and C1' to disk, then exit
    out = opt("prepare"); t = time.perf_counter()
    model.weights_id = wid = save_prepared(model, os.path.join(out, "weights"), tok_path)
    print(f"weights {wid} saved in {time.perf_counter() - t:.0f} s", flush=True)
    bd = os.path.join(out, "bundle_" + wid); os.makedirs(bd, exist_ok=True); hdr, arrays = bundle()
    for i, a in enumerate(arrays): np.save(os.path.join(bd, f"{i}.npy"), np.ascontiguousarray(a), allow_pickle=False)
    json.dump(dict(hdr, arrays=[[np.asarray(a).dtype.name, list(np.shape(a))] for a in arrays]), open(os.path.join(bd, "header.json"), "w"))
    cd = os.path.join(out, "c1_" + wid); os.makedirs(cd, exist_ok=True); t = time.perf_counter()
    for n in srv.order:
        mm = np.lib.format.open_memmap(os.path.join(cd, n + ".npy"), "w+", np.uint32, (srv.lin[n].q.shape[0], fold.N))
        for r0, rows in c1_rows(n): mm[r0:r0 + len(rows)] = rows
        mm.flush(); del mm
    open(os.path.join(cd, "complete"), "w").close()
    print(f"C1' for {len(srv.order)} layers written in {time.perf_counter() - t:.0f} s to {cd}", flush=True)
    sys.exit()

if gpu and not v4: torch.cuda.empty_cache(); srv.upload_gpu()
print(f"weights on {'GPU' if gpu else 'CPU'}, {bits}-bit replies, weights id {model.weights_id}", flush=True)

def serve(c):
    while True:
        h, a = c.recv(); op = h["op"]
        if op == "bye": return
        if op == "ping": c.send(dict(op="pong"))
        elif op == "info": c.send(dict(op="info", repo=repo, weights_id=model.weights_id))
        elif op == "bundle": c.send(*bundle())
        elif op == "c1" and v4:                                # one-time public download, everything in row chunks
            for kind, name, r0, C1 in srv.public_c1():
                C1 = C1.cpu().numpy(); c.send(dict(op="c1chunk", kind=kind, name=name, r0=r0), [C1.astype(np.int32) if kind == "top" else C1])
            c.send(dict(op="c1done"))
        elif op == "c1":                                       # one-time public download, one layer at a time
            for r0, rows in c1_rows(h["name"]): c.send(dict(op="c1chunk", r0=r0), [rows])
            c.send(dict(op="c1done"))
        elif op == "single":                                   # v4
            t0 = time.perf_counter(); C = unpack_bits(a[0], 56, int(np.prod(h["shape"]))).reshape(h["shape"])
            y = srv.single(h["name"], np.asarray(h["toks"]), C)
            c.send(dict(op="y", shape=list(y.shape), server_s=time.perf_counter() - t0), [pack_bits(y, CP.RB)])
        elif op == "pair":                                     # v4
            t0 = time.perf_counter()
            Cv = unpack_bits(a[0], 56, int(np.prod(h["shape_v"]))).reshape(h["shape_v"])
            Cx = unpack_bits(a[1], 56, int(np.prod(h["shape_x"]))).reshape(h["shape_x"])
            t, y = srv.pair(h["r"], h["g"], np.asarray(h["toks"]), Cv, Cx)
            c.send(dict(op="ty", shape_t=list(t.shape), shape_y=list(y.shape), server_s=time.perf_counter() - t0),
                   [pack_bits(t, 56), pack_bits(y, CP.RB)])
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
