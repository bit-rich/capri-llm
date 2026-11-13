"""Client side of the networked protocol, shared by client.py (benchmark) and chat.py (interactive).
See client.py for the security notes: secrets stay here, public randomness comes from the client's own seeds."""
import json, os, socket, time, types, numpy as np, torch
import fold
from net import Conn, unpack_bits, client_auth, token_from_env
from protocol import InnerLWE, FoldClient

LWE_SEED, A_PUB_SEED = 0, 2                    # the client's own: never taken from the server
cache_dir = None                               # set by connect()


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
        if cache_dir and bundle_hdr.get("weights_id"):          # C1' depends on the exact weights: cache by their id
            self.cache = os.path.join(cache_dir, "c1_" + bundle_hdr["weights_id"]); os.makedirs(self.cache, exist_ok=True)

    def public_c1(self, name):
        f = self.cache and os.path.join(self.cache, name + ".npy")
        if f and os.path.exists(f) and (os.path.exists(f + ".done") or os.path.exists(os.path.join(self.cache, "complete"))):
            mm = np.load(f, mmap_mode="r"); assert mm.shape == (self.lin[name].q.shape[0], fold.N) and mm.dtype == np.uint32
            for r0 in range(0, len(mm), 4096): yield r0, torch.from_numpy(mm[r0:r0 + 4096].astype(np.int64))
            return
        out = f and np.lib.format.open_memmap(f, "w+", np.uint32, (self.lin[name].q.shape[0], fold.N))
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



def cached_bundle(cache):
    """The public bundle from the cache, without a server (for preparing secrets before connecting).
    Returns (header, arrays) or None if the cache doesn't hold exactly one model's complete files."""
    if not cache or not os.path.isdir(cache): return None
    ids = [d[7:] for d in os.listdir(cache) if d.startswith("bundle_")
           and os.path.exists(os.path.join(cache, "c1_" + d[7:], "complete"))]
    if len(ids) != 1: return None
    bdir = os.path.join(cache, "bundle_" + ids[0]); bh = json.load(open(os.path.join(bdir, "header.json")))
    return bh, [np.load(os.path.join(bdir, f"{i}.npy"), allow_pickle=False) for i in range(len(bh["arrays"]))]

def connect(host, port, cache=None, verbose=True):
    """Log in, fetch (or load from the cache) the public bundle. Returns (conn, socket, bundle header, model, tokenizer, rtt)."""
    global cache_dir
    cache_dir = cache
    s = socket.create_connection((host, port)); c = Conn(s)
    token = token_from_env()
    if token: client_auth(c, token)
    rtt = []
    for _ in range(10):
        t = time.perf_counter(); c.send(dict(op="ping")); h, _ = c.recv(); rtt.append(time.perf_counter() - t)
        if h["op"] == "challenge": raise SystemExit("the server needs a password: set PI_TOKEN to the same value as on the server")
    rtt = float(np.median(rtt))
    c.send(dict(op="info")); info = c.recv()[0]; t = time.perf_counter()
    bdir = cache_dir and info.get("weights_id") and os.path.join(cache_dir, "bundle_" + info["weights_id"])
    if bdir and os.path.exists(os.path.join(bdir, "header.json")):  # public bundle from the cache
        bh = json.load(open(os.path.join(bdir, "header.json")))
        ba = [np.load(os.path.join(bdir, f"{i}.npy"), allow_pickle=False) for i in range(len(bh["arrays"]))]
    else:
        c.send(dict(op="bundle")); bh, ba = c.recv()
        if bdir:
            os.makedirs(bdir, exist_ok=True)
            for i, a in enumerate(ba): np.save(os.path.join(bdir, f"{i}.npy"), a, allow_pickle=False)
            json.dump(dict(bh, arrays=[[a.dtype.name, list(a.shape)] for a in ba]), open(os.path.join(bdir, "header.json"), "w"))
    assert bh.get("weights_id") == info.get("weights_id"), "bundle does not match the server's weights"
    assert (bh["lwe_seed"], bh["a_pub_seed"]) == (LWE_SEED, A_PUB_SEED), "server uses different public randomness"
    model, tok = client_model(bh, ba)
    if verbose: print(f"{bh['repo']} at {host}:{port}: round trip {1e3 * rtt:.1f} ms, bundle {c.recvd / 1e6:.0f} MB in {time.perf_counter() - t:.1f} s", flush=True)
    return c, s, bh, model, tok, rtt
