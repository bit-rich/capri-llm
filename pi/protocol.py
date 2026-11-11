"""
End-to-end single-server private inference for Qwen3-0.6B (open weights, HF safetensors).

Server sees only: Ring-LWE ciphertexts of activations (online) and BFV ciphertexts of
mask secrets (offline). It runs plain int8 GEMVs online. Client does masking, unmasking,
the embedding lookup (public table), RMSNorm/RoPE/SwiGLU gating/softmax/GQA attention,
and holds the KV cache.
"""
import ctypes, glob, json, os, time, numpy as np
from crypto import (InnerLWE, BFV, BFV32, offline_hint_products, offline_hint_products_v2,
                    offline_hint_products_v2_gpu, gpu_available, dgauss, Q32)

lib = ctypes.CDLL("./libk.so")
P = ctypes.c_void_p
for f in (lib.gemv_i8_i8, lib.gemv_i8_u32):
    f.argtypes = [P, P, P, ctypes.c_size_t, ctypes.c_size_t]

def gemv_u32(W, c):
    y = np.empty(W.shape[0], dtype=np.uint32); c = np.ascontiguousarray(c, dtype=np.uint32)
    lib.gemv_i8_u32(W.ctypes.data, c.ctypes.data, y.ctypes.data, W.shape[0], W.shape[1]); return y

def gemv_i8(W, x):
    y = np.empty(W.shape[0], dtype=np.int32); x = np.ascontiguousarray(x, dtype=np.int8)
    lib.gemv_i8_i8(W.ctypes.data, x.ctypes.data, y.ctypes.data, W.shape[0], W.shape[1]); return y

_SHIFTS = {}
def gemv_u32_gpu(W, c):
    """y = W c mod 2^32 on GPU, c an int32 tensor holding the uint32 ciphertext bits. Reinterpreting c as 4 bytes
    gives signed digits s_k = u_k - 256 b_k (b_k = top bit of byte k), so W c = sum_k 2^(8k) W s_k + 2^(8k+8) W b_k;
    b_3's term is a multiple of 2^32 and drops out. The 7 columns [s_0..s_3, b_0..b_2] go into ONE int8 GEMM
    (8 columns is cuBLASLt's minimum anyway), so the weights are read once, as in plaintext."""
    import torch
    if W.device not in _SHIFTS:
        _SHIFTS[W.device] = torch.tensor([1, 1 << 8, 1 << 16, 1 << 24, 1 << 8, 1 << 16, 1 << 24, 0], device=W.device)
    s = c.view(torch.int8).view(-1, 4)                            # little-endian bytes as int8
    X = torch.cat([s, (s[:, :3] < 0).to(torch.int8), torch.zeros_like(s[:, :1])], 1)
    return (torch._int_mm(W, X).long() * _SHIFTS[W.device]).sum(1) & 0xFFFFFFFF

def gemm_u32_gpu(W, c):
    """gemv_u32_gpu for T ciphertexts in one int8 GEMM: c (T, d_in) int32 -> (T, d_out) int64 in [0, 2^32)."""
    import torch
    if W.device not in _SHIFTS:
        _SHIFTS[W.device] = torch.tensor([1, 1 << 8, 1 << 16, 1 << 24, 1 << 8, 1 << 16, 1 << 24, 0], device=W.device)
    T = c.shape[0]
    s = c.contiguous().view(torch.int8).view(T, -1, 4)
    X = torch.cat([s, (s[..., :3] < 0).to(torch.int8), torch.zeros_like(s[..., :1])], 2)       # (T, d_in, 8)
    Y = torch._int_mm(W, X.permute(1, 0, 2).reshape(c.shape[1], 8 * T))                       # (d_out, 8 T)
    return ((Y.view(-1, T, 8).long() * _SHIFTS[W.device]).sum(2) & 0xFFFFFFFF).T

def gemv_i8_gpu(W, x):
    """Plaintext baseline on GPU: same int8 kernel, one int8 activation column (zero-padded to 8)."""
    import torch
    return torch._int_mm(W, torch.nn.functional.pad(x[:, None], (0, 7)))[:, 0]

def unpack4(p):
    """Packed nibbles (rows, cols/2) uint8 -> signed 4-bit levels (rows, cols) int8. numpy or torch."""
    p = p.view(np.int8) if isinstance(p, np.ndarray) else p.view(__import__("torch").int8)
    lo, hi = (p << 4) >> 4, p >> 4                                # arithmetic shifts sign-extend
    if isinstance(p, np.ndarray):
        q = np.empty((p.shape[0], 2 * p.shape[1]), np.int8); q[:, 0::2], q[:, 1::2] = lo, hi; return q
    return __import__("torch").stack([lo, hi], -1).view(p.shape[0], -1)


# ---------------------------------------------------------------- weights
def bf16_to_f32(u16): return (u16.astype(np.uint32) << 16).view(np.float32)

class SafeTensors:
    """Lazy reader over (possibly sharded) safetensors. numpy has no bf16, so tensors become float32 on access;
    nothing is held in memory beyond the memmaps, so big models load one tensor at a time."""
    def __init__(self, path):
        self.meta = {}
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            with open(f, "rb") as fh:
                n = int.from_bytes(fh.read(8), "little"); hdr = json.loads(fh.read(n))
            raw = np.memmap(f, dtype=np.uint8, mode="r", offset=8 + n)
            self.meta.update({k: (raw, v) for k, v in hdr.items() if k != "__metadata__"})
    def __contains__(self, k): return k in self.meta
    def raw(self, k):                            # bf16 tensor as a uint16 memmap view
        raw, v = self.meta[k]; assert v["dtype"] == "BF16"
        return raw[v["data_offsets"][0]:v["data_offsets"][1]].view(np.uint16).reshape(v["shape"])
    def __getitem__(self, k):
        raw, v = self.meta[k]; b = raw[v["data_offsets"][0]:v["data_offsets"][1]]
        if v["dtype"] == "BF16": return bf16_to_f32(b.view(np.uint16)).reshape(v["shape"])
        if v["dtype"] == "F32": return np.array(b.view(np.float32)).reshape(v["shape"])
        raise ValueError(v["dtype"])

def qwen_path(repo="Qwen/Qwen3-0.6B"):
    for p in glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*")):
        idx = os.path.join(p, "model.safetensors.index.json")      # sharded: every shard must be present
        shards = set(json.load(open(idx))["weight_map"].values()) if os.path.exists(idx) else {"model.safetensors"}
        if all(os.path.exists(os.path.join(p, f)) for f in shards): return p
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt"])


# ---------------------------------------------------------------- model
class Linear:
    def __init__(self, W, keep_float=True):     # W float (d_out, d_in) -> per-row int8
        self.W = W.astype(np.float32) if keep_float else None   # unquantized, for the fp32 reference only
        self.scale = np.maximum(np.abs(W).max(1), 1e-30) / 127.0
        self.q = np.empty(W.shape, dtype=np.int8); norms = []
        for r in range(0, W.shape[0], 8192):                     # row chunks: bounded temporaries for big heads
            q = self.q[r:r + 8192] = np.clip(np.rint(W[r:r + 8192] / self.scale[r:r + 8192, None]), -127, 127)
            norms.append(np.sqrt((q.astype(np.float32) ** 2).sum(1, dtype=np.float64)))
        self.row_norm_max = np.concatenate(norms).max()
        self._t = self.packed = None
    @classmethod
    def from_int(cls, q, scale, packed=None):
        """Pre-quantized weights, e.g. 4-bit GPTQ from quant4.py (q = integer levels stored as int8)."""
        L = cls.__new__(cls)
        L.W, L.q, L.scale, L.packed, L._t = None, q, scale, packed, None
        L.row_norm_max = np.sqrt((q.astype(np.float32) ** 2).sum(1, dtype=np.float64)).max()
        return L
    def plain(self, x):                          # plaintext W8 reference (float activations, fp32 accumulate)
        import torch                             # CPU int8-weight kernel: no dequantized copy of the weights
        if self._t is None: self._t = (torch.from_numpy(self.q), torch.from_numpy(self.scale.astype(np.float32)))
        y = torch.ops.aten._weight_int8pack_mm(torch.from_numpy(np.atleast_2d(x).astype(np.float32)), *self._t).double().numpy()
        return y if x.ndim == 2 else y[0]            # x (d_in,) or a batch (T, d_in)
    def float_(self, x):                         # unquantized fp32 reference
        return (self.W @ x.astype(np.float32)).astype(np.float64)

def rmsnorm(x, w, eps): return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w
def silu(x): return x / (1 + np.exp(-x))
def rope(x, cos, sin):                           # x (heads, hd), HF rotate_half convention
    h = x.shape[-1] // 2
    return x * cos + np.concatenate([-x[..., h:], x[..., :h]], -1) * sin

def merged_linear(R, G, w, keep_float, boost=1.0):
    """One round trip for 'residual branch R, add, RMSNorm (weight w), next matrix G':
        G rmsnorm(x + R v) = (G diag(w) R v + G (w x)) / rms(x + R v),
    so the server applies [[R, 0], [G diag(w) R, G]] to [v; w x] and the client, who then knows x + R v, divides by
    its rms. w stays on the client side: folded into G it would wreck the per-row int8 scales.
    The v-columns are divided by lam (the client sends lam * v). For one row the weight quantization error is smallest
    at lam = max|G diag(w) R| / max|G|; lam is the median of that over rows, times `boost` (>= 1, calibrated by
    merge_boost so that lam * v is not small next to w x, which would bury the R v rows in the protocol noise)."""
    P = (G * w) @ R
    lam = float(np.median(np.abs(P).max(1) / np.abs(G).max(1))) * boost
    top = np.concatenate([R / lam, np.zeros((R.shape[0], G.shape[1]), np.float32)], 1)
    return Linear(np.concatenate([top, np.concatenate([P / lam, G], 1)]), keep_float), lam

def merge_boost(m, ids):
    """Calibration for merged_linear, on public text: per merged layer, how much lam must grow for lam |v| to reach
    |w x| (median over tokens; never below 1). Runs the plaintext int8 model, so it needs no float weights."""
    seen = {}
    def lin(name, L, x):
        k = name.split(".")[-1]
        if k in ("ob", "da"):
            nv = len(x) - m.d; seen.setdefault(name, []).append(np.linalg.norm(x[nv:]) / max(np.linalg.norm(x[:nv]), 1e-30))
        return L.plain(x)
    cache = [([], []) for _ in m.blocks]
    for t in ids: forward_token(m, t, cache, lin)
    return {k: max(1.0, float(np.median(v))) for k, v in seen.items()}

class Qwen3:
    def __init__(self, path=None, keep_float=True, w4=None, merge=False, boost=None):
        path = path or qwen_path()
        w4 = np.load(w4) if w4 else None                         # quant4.py output: GPTQ 4-bit block linears
        cfg = json.load(open(os.path.join(path, "config.json")))
        W = SafeTensors(path)
        self.d, self.L, self.vocab = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
        self.hq, self.hkv, self.hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.eps, theta = cfg["rms_norm_eps"], cfg["rope_theta"]
        self.inv_freq = theta ** (-np.arange(0, self.hd, 2) / self.hd)
        f64 = lambda k: W[k].astype(np.float64)
        lin = lambda *ks: Linear(np.concatenate([W[k] for k in ks]) if len(ks) > 1 else W[ks[0]], keep_float)
        if w4 is not None:
            lin4 = lambda key: Linear.from_int(unpack4(w4[key + ".packed"]), w4[key + ".scale"], w4[key + ".packed"])
        self.embed = W.raw("model.embed_tokens.weight")          # public table (bf16 memmap); client looks up its row
        self.blocks = []; self.merged = merge
        cat = lambda *ks: np.concatenate([W[k] for k in ks])
        for i in range(self.L if merge else 0):  # 2 round trips per block: o+gate_up ("ob") and down+next qkv ("da")
            assert w4 is None, "merge needs the float weights"
            p = f"model.layers.{i}."; a, m = p + "self_attn.", p + "mlp."; nx = f"model.layers.{i + 1}."
            b = dict(qkv=lin(a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight")) if i == 0 else {}
            bo = lambda k: (boost or {}).get(f"{i}.{k}", 1.0)
            b["ob"], b["lam_ob"] = merged_linear(W[a + "o_proj.weight"], cat(m + "gate_proj.weight", m + "up_proj.weight"),
                                                 W[p + "post_attention_layernorm.weight"], keep_float, bo("ob"))
            b["w_ob"] = f64(p + "post_attention_layernorm.weight")
            if i < self.L - 1:
                b["da"], b["lam_da"] = merged_linear(
                    W[m + "down_proj.weight"], cat(nx + "self_attn.q_proj.weight", nx + "self_attn.k_proj.weight",
                                                   nx + "self_attn.v_proj.weight"), W[nx + "input_layernorm.weight"], keep_float, bo("da"))
                b["w_da"] = f64(nx + "input_layernorm.weight")
            else: b["down"] = lin(m + "down_proj.weight")
            b.update(ln1=f64(p + "input_layernorm.weight"), q_norm=f64(a + "q_norm.weight"), k_norm=f64(a + "k_norm.weight"))
            self.blocks.append(b)
        for i in range(0 if merge else self.L):
            p = f"model.layers.{i}."; a, m = p + "self_attn.", p + "mlp."
            self.blocks.append(dict(
                qkv=lin(a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight"),
                o=lin(a + "o_proj.weight"),
                gate_up=lin(m + "gate_proj.weight", m + "up_proj.weight"),
                down=lin(m + "down_proj.weight"),
            ) if w4 is None else {k: lin4(f"{i}.{k}") for k in ("qkv", "o", "gate_up", "down")})
            self.blocks[-1].update(
                ln1=f64(p + "input_layernorm.weight"), ln2=f64(p + "post_attention_layernorm.weight"),
                q_norm=f64(a + "q_norm.weight"), k_norm=f64(a + "k_norm.weight"))
        self.norm = f64("model.norm.weight")
        self.head = lin("lm_head.weight" if "lm_head.weight" in W else "model.embed_tokens.weight")
    def linears(self):
        out = []
        for i, b in enumerate(self.blocks):
            out += [(f"{i}.{k}", b[k]) for k in ("qkv", "o", "gate_up", "ob", "da", "down") if k in b]
        return out + [("head", self.head)]

def attend(q, K, V):
    """GQA: q (hq, hd); K, V (T, hkv, hd). Query head h reads kv head h // (hq // hkv)."""
    hq, hd = q.shape; hkv = K.shape[1]
    qg = q.reshape(hkv, hq // hkv, hd)
    s = np.einsum("grd,tgd->grt", qg, K) / np.sqrt(hd)
    s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
    return np.einsum("grt,tgd->grd", s, V).reshape(hq * hd)

def forward_token(m, tok, cache, lin):
    """One decode step. `lin(name, layer, x)` performs the linear layer (plain or private)."""
    pos = len(cache[0][0])
    ang = np.concatenate([pos * m.inv_freq] * 2); cos, sin = np.cos(ang), np.sin(ang)
    x = bf16_to_f32(m.embed[tok]).astype(np.float64)
    nq, nk = m.hq * m.hd, m.hkv * m.hd
    rms = lambda x: np.sqrt((x * x).mean() + m.eps)
    for i, b in enumerate(m.blocks if m.merged else []):
        if i == 0: qkv = lin("0.qkv", b["qkv"], rmsnorm(x, b["ln1"], m.eps))
        q, k, v = np.split(qkv, [nq, nq + nk])
        q = rope(rmsnorm(q.reshape(m.hq, m.hd), b["q_norm"], m.eps), cos, sin)
        k = rope(rmsnorm(k.reshape(m.hkv, m.hd), b["k_norm"], m.eps), cos, sin)
        cache[i][0].append(k); cache[i][1].append(v.reshape(m.hkv, m.hd))
        a = attend(q, np.array(cache[i][0]), np.array(cache[i][1]))
        out = lin(f"{i}.ob", b["ob"], np.concatenate([b["lam_ob"] * a, b["w_ob"] * x]))              # o, add, norm, gate_up
        x = x + out[:m.d]; g, u = np.split(out[m.d:] / rms(x), 2)
        if "da" in b:
            out = lin(f"{i}.da", b["da"], np.concatenate([b["lam_da"] * silu(g) * u, b["w_da"] * x]))   # down, add, norm, next qkv
            x = x + out[:m.d]; qkv = out[m.d:] / rms(x)
        else: x = x + lin(f"{i}.down", b["down"], silu(g) * u)
    for i, b in enumerate([] if m.merged else m.blocks):
        qkv = lin(f"{i}.qkv", b["qkv"], rmsnorm(x, b["ln1"], m.eps))
        q, k, v = np.split(qkv, [nq, nq + nk])
        q = rope(rmsnorm(q.reshape(m.hq, m.hd), b["q_norm"], m.eps), cos, sin)
        k = rope(rmsnorm(k.reshape(m.hkv, m.hd), b["k_norm"], m.eps), cos, sin)
        cache[i][0].append(k); cache[i][1].append(v.reshape(m.hkv, m.hd))
        a = attend(q, np.array(cache[i][0]), np.array(cache[i][1]))
        x = x + lin(f"{i}.o", b["o"], a)
        g, u = np.split(lin(f"{i}.gate_up", b["gate_up"], rmsnorm(x, b["ln2"], m.eps)), 2)
        x = x + lin(f"{i}.down", b["down"], silu(g) * u)
    return lin("head", m.head, rmsnorm(x, m.norm, m.eps))


def forward_batch(m, toks, cache, lin, n_head=None):
    """Merged model, T tokens in ONE pass: still 2 L + 2 round trips in total. `lin(name, layer, X)` maps (T, d_in) to
    (T, d_out). Token j attends to the cache and to tokens <= j of this pass. Returns the logits of the last n_head
    tokens (default all). Used for sending a whole prompt at once and for checking several guessed tokens at once."""
    assert m.merged
    T, t0 = len(toks), len(cache[0][0])
    ang = (t0 + np.arange(T))[:, None] * np.concatenate([m.inv_freq] * 2)[None]
    cos, sin = np.cos(ang)[:, None], np.sin(ang)[:, None]                              # (T, 1, hd)
    x = np.stack([bf16_to_f32(m.embed[t]) for t in toks]).astype(np.float64)
    nq, nk = m.hq * m.hd, m.hkv * m.hd
    rms = lambda x: np.sqrt((x * x).mean(-1, keepdims=True) + m.eps)
    for i, b in enumerate(m.blocks):
        if i == 0: qkv = lin("0.qkv", b["qkv"], rmsnorm(x, b["ln1"], m.eps))
        q, k, v = np.split(qkv, [nq, nq + nk], axis=1)
        q = rope(rmsnorm(q.reshape(T, m.hq, m.hd), b["q_norm"], m.eps), cos, sin)
        k = rope(rmsnorm(k.reshape(T, m.hkv, m.hd), b["k_norm"], m.eps), cos, sin)
        cache[i][0].extend(k); cache[i][1].extend(v.reshape(T, m.hkv, m.hd))
        K, V = np.array(cache[i][0]), np.array(cache[i][1])
        a = np.stack([attend(q[j], K[:t0 + j + 1], V[:t0 + j + 1]) for j in range(T)])
        out = lin(f"{i}.ob", b["ob"], np.concatenate([b["lam_ob"] * a, b["w_ob"] * x], 1))
        x = x + out[:, :m.d]; g, u = np.split(out[:, m.d:] / rms(x), 2, axis=1)
        if "da" in b:
            out = lin(f"{i}.da", b["da"], np.concatenate([b["lam_da"] * silu(g) * u, b["w_da"] * x], 1))
            x = x + out[:, :m.d]; qkv = out[:, m.d:] / rms(x)
        else: x = x + lin(f"{i}.down", b["down"], silu(g) * u)
    return lin("head", m.head, rmsnorm(x[-(n_head or T):], m.norm, m.eps))

def cache_keep(cache, n):
    """Roll the KV cache back to its first n tokens (drops rejected guesses)."""
    for K, V in cache: del K[n:], V[n:]


# ---------------------------------------------------------------- private protocol
class Server:
    """Holds public weights + public hints. Never sees plaintext activations or secrets."""
    def __init__(self, model, lwe, chunk=16384, gpu=False):
        self.lin = dict(model.linears()); self.lwe = lwe; self.gpu = gpu
        self.order = [n for n, _ in model.linears()]
        D = sum(self.lin[n].q.shape[0] for n in self.order)
        self.H_all = np.empty((D, lwe.n), dtype=np.uint32)          # (D_total, n), ~2 GB for Qwen3-0.6B
        self.a = {}; self.H = {}; off = 0
        for lid, name in enumerate(self.order):
            q = self.lin[name].q; self.a[name] = lwe.public_a(lid, q.shape[1])
            for r in range(0, q.shape[0], chunk):                   # one-time, public
                e = min(r + chunk, q.shape[0])
                self.H_all[off + r:off + e] = lwe.hint(q[r:e], self.a[name])
            self.H[name] = self.H_all[off:off + q.shape[0]]; off += q.shape[0]
        self.time_online = 0.0
    def upload_gpu(self):
        """Weights to the GPU (after the offline phase, which needs the GPU memory). 4-bit stays packed there."""
        import torch
        self.Wg = {n: torch.as_tensor(L.packed if L.packed is not None else L.q, device="cuda") for n, L in self.lin.items()}
    def weight_gpu(self, name):                  # int8 view for the GEMM (4-bit: unpacked on the fly, every call)
        return unpack4(self.Wg[name]) if self.lin[name].packed is not None else self.Wg[name]
    def online(self, name, c):
        t = time.perf_counter()
        if self.gpu:   # time includes host<->GPU copies of c and y (they arrive/leave over the network)
            import torch
            y = gemv_u32_gpu(self.weight_gpu(name), torch.as_tensor(c.view(np.int32), device="cuda")).cpu().numpy()
        else: y = gemv_u32(self.lin[name].q, c)
        self.time_online += time.perf_counter() - t
        return y

class Client:
    def __init__(self, server, lwe, bfv, seed=7, gpu=False):
        self.srv, self.lwe, self.bfv, self.gpu = server, lwe, bfv, gpu
        self.rng = np.random.default_rng(seed)
        self.t_mask = self.t_unmask = 0.0
        self.up_bytes = self.down_bytes = 0
        self.rounds = 0

    def offline(self, n_tokens):
        """Draw a fresh secret alpha per future token; get z = H alpha for every layer via BFV."""
        self.alphas = dgauss(self.rng, (n_tokens, self.lwe.n), self.lwe.sigma)
        if isinstance(self.bfv, BFV32):
            f = offline_hint_products_v2_gpu if self.gpu else offline_hint_products_v2
        else: f = offline_hint_products
        z, srv_s, cli_s, up, down = f(self.bfv, self.srv.H_all, self.alphas)
        self.z = {}; off = 0
        for name in self.srv.order:
            D = self.srv.H[name].shape[0]; self.z[name] = z[:, off:off + D]; off += D
        self.tok = -1
        return dict(server_s=srv_s, client_s=cli_s, up_bytes=up, down_bytes=down)

    def begin_token(self):
        """Move to the next alpha. Reusing one across two tokens would leak x1 - x2."""
        self.tok += 1
        assert self.tok < len(self.alphas), "out of offline material"

    def lin(self, name, L, x):
        t0 = time.perf_counter()
        nx = np.linalg.norm(x)
        S = 2.0 ** 30 / (L.row_norm_max * nx) if nx > 0 else 1.0     # Cauchy-Schwarz: no wraparound
        xt = np.rint(x * S).astype(np.int64)
        c = (self.lwe.mask(self.srv.a[name], self.alphas[self.tok], len(x), self.rng) + xt) & 0xFFFFFFFF
        t1 = time.perf_counter()
        y = self.srv.online(name, c.astype(np.uint32))                # ---- network round trip ----
        t2 = time.perf_counter()
        r = (y.astype(np.int64) - self.z[name][self.tok]) & 0xFFFFFFFF
        r = np.where(r >= 1 << 31, r - Q32, r)                         # = M x~ + M e  (centred)
        out = r / S * L.scale
        t3 = time.perf_counter()
        self.t_mask += t1 - t0; self.t_unmask += t3 - t2
        self.up_bytes += 4 * len(c); self.down_bytes += 4 * len(y); self.rounds += 1
        return out


# ---------------------------------------------------------------- v3: BFV c0 folded into the online reply (fold.py)
class FoldServer(Server):
    """Holds the weights and public randomness. One-time per model it publishes C1' (public_c1). Per batch it only
    RECEIVES b; per token it derives the pads u_l from b[t]; online it runs the same 32-bit GEMV on c - u_l and
    returns out_bits bits per row. Nothing is downloaded per batch."""
    def __init__(self, model, lwe, gpu=False, gpu_setup=None, out_bits=16, a_seed=2):
        import fold
        self.lin = dict(model.linears()); self.lwe, self.gpu, self.out_bits = lwe, gpu, out_bits
        self.order = [n for n, _ in model.linears()]
        self.a = {name: lwe.public_a(lid, self.lin[name].q.shape[1]) for lid, name in enumerate(self.order)}
        self.a_pub = fold.public_a(a_seed)                          # fixed public c1 parts, shared by every client and batch
        self.dev_setup = "cuda" if (gpu if gpu_setup is None else gpu_setup) else "cpu"
        self.pads = fold.Pads([self.a[k] for k in self.order], [self.lin[k].q.shape[1] for k in self.order],
                              "cuda" if gpu else "cpu")
        self.hint = self.b = self.tok = None
        self.time_online = self.time_pads = self.time_c1 = 0.0

    def public_c1(self, name):
        """One-time, public: yields (r0, rows of C1' = round(M A a / 2^48) mod 2^32) for one layer."""
        import fold, torch
        sync = torch.cuda.synchronize if self.dev_setup == "cuda" else (lambda: None)
        t = time.perf_counter()
        if self.hint is None: self.hint = fold.Hint(self.a_pub, self.dev_setup)
        L = self.lin[name]
        for r0, c in self.hint.c1(L.q, self.hint.G(self.a[name], L.q.shape[1])):
            sync(); self.time_c1 += time.perf_counter() - t
            yield r0, c
            t = time.perf_counter()

    def receive_b(self, b):                      # the whole per-batch offline phase, as seen by the server
        self.b, self.tok = b, None

    def online(self, name, tok, c):
        import fold
        t = time.perf_counter()
        if tok != self.tok:                      # first request of a token: pads for all its layers at once
            self.u = dict(zip(self.order, self.pads(self.b[:, tok]))); self.tok = tok
            if self.gpu: __import__("torch").cuda.synchronize()
            self.time_pads += time.perf_counter() - t
        if self.gpu:   # time includes host<->GPU copies of c and the reply (they arrive/leave over the network)
            import torch
            w = torch.as_tensor(c.view(np.int32), device="cuda") - self.u[name]           # int32 wraps: mod 2^32
            y = fold.reply_bits(gemv_u32_gpu(self.weight_gpu(name), w), self.out_bits).to(torch.int32).cpu().numpy()
        else:
            y = fold.reply_bits(gemv_u32(self.lin[name].q, c - self.u[name].numpy().view(np.uint32)).astype(np.int64),
                                self.out_bits)
        self.time_online += time.perf_counter() - t
        return y

    def online_batch(self, name, toks, C):
        """online() for several tokens in one round trip: C (T, d_in) uint32, row i masked with the secret of slot toks[i]."""
        import fold, torch
        t = time.perf_counter()
        if not set(toks) <= set(getattr(self, "pos", ())):       # first request of a pass: pads for all its tokens
            self.pos = {int(tk): i for i, tk in enumerate(toks)}
            parts = [self.pads.batch(self.b[:, toks[i:i + 8]].transpose(1, 0, 2)) for i in range(0, len(toks), 8)]
            self.U = {nm: torch.cat([p[j] for p in parts]) for j, nm in enumerate(self.order)}
            if self.gpu: torch.cuda.synchronize()
            self.time_pads += time.perf_counter() - t
        U = self.U[name][[self.pos[int(tk)] for tk in toks]]
        if self.gpu:
            W = self.weight_gpu(name)
            y = torch.cat([fold.reply_bits(gemm_u32_gpu(W, torch.as_tensor(C[i:i + 16].view(np.int32), device="cuda") - U[i:i + 16]),
                                           self.out_bits).to(torch.int32) for i in range(0, len(C), 16)]).cpu().numpy()
        else:
            w = C - U.numpy().view(np.uint32)
            y = np.stack([fold.reply_bits(gemv_u32(self.lin[name].q, r).astype(np.int64), self.out_bits) for r in w])
        self.time_online += time.perf_counter() - t
        return y

class FoldClient(Client):
    def begin_pass(self, T):
        """Take the next T secrets: one per token of the pass, never reused (also not for rejected guesses)."""
        self.toks = np.arange(self.tok + 1, self.tok + 1 + T); self.tok += T
        assert self.tok < len(self.alphas), "out of offline material"

    def lin_batch(self, name, L, X):
        """lin() for the tokens of the current pass in one round trip. X (rows, d_in) belongs to the LAST rows tokens."""
        import fold
        t0 = time.perf_counter()
        if not hasattr(self, "masks"): self.masks = fold.Masks(self.srv.a)
        toks = self.toks[-len(X):]; k = self.srv.out_bits
        half = 1 << (fold.MSG_BITS - 1)
        S = (half - 2.0 ** 24) / (L.row_norm_max * np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-300))
        c = (self.masks(name, self.alphas[toks], X.shape[1]) + dgauss(self.rng, X.shape, self.lwe.sigma)
             + np.rint(X * S).astype(np.int64)) & 0xFFFFFFFF
        t1 = time.perf_counter()
        y = self.srv.online_batch(name, toks, c.astype(np.uint32))     # ---- ONE network round trip ----
        t2 = time.perf_counter()
        r = ((y.astype(np.int64) << (fold.MSG_BITS - k)) - self.z[name][toks]) & (2 * half - 1)
        out = np.where(r >= half, r - 2 * half, r) / S * L.scale
        t3 = time.perf_counter()
        self.t_mask += t1 - t0; self.t_unmask += t3 - t2
        self.up_bytes += 4 * c.size; self.down_bytes += y.size * k / 8; self.rounds += 1
        return out

    def offline(self, n_tokens):
        """Per batch: fresh s and alphas, upload b. The unmasking values z = (C1' s)[t] come from the public C1',
        which a real client downloads once per model; here it is streamed from the server chunk by chunk."""
        import fold, torch
        dev = "cuda" if self.gpu else "cpu"
        sync = torch.cuda.synchronize if self.gpu else (lambda: None)
        t0 = time.perf_counter()
        self.alphas = dgauss(self.rng, (n_tokens, self.lwe.n), self.lwe.sigma)
        self.key = fold.Key(self.rng, dev)
        b = self.key.encrypt(self.srv.a_pub, self.alphas); sync()
        cli = time.perf_counter() - t0
        t = time.perf_counter(); self.srv.receive_b(b); srv = time.perf_counter() - t      # ---- upload ----
        self.z, D = {}, 0
        for name in self.srv.order:
            d = self.srv.lin[name].q.shape[0]; D += d
            z = self.z[name] = np.empty((n_tokens, d), dtype=np.uint32)
            for r0, c in self.srv.public_c1(name):
                c = c.to(dev); t = time.perf_counter()
                z[:, r0:r0 + len(c)] = self.key.unmask(c, n_tokens).T.cpu().numpy()
                cli += time.perf_counter() - t
        self.tok = -1
        return dict(server_s=srv, client_s=cli, up_bytes=b.nbytes, down_bytes=0,
                    c1_server_s_once=self.srv.time_c1, c1_bytes_once=4 * D * fold.N)

    def lin(self, name, L, x):
        import fold
        t0 = time.perf_counter()
        k = self.srv.out_bits; nx = np.linalg.norm(x)
        half = 1 << (fold.MSG_BITS - 1)          # |M x~| <= half - 2^24 (Cauchy-Schwarz); 2^24 covers rounding and noise
        S = (half - 2.0 ** 24) / (L.row_norm_max * nx) if nx > 0 else 1.0
        xt = np.rint(x * S).astype(np.int64)
        c = (self.lwe.mask(self.srv.a[name], self.alphas[self.tok], len(x), self.rng) + xt) & 0xFFFFFFFF
        t1 = time.perf_counter()
        y = self.srv.online(name, self.tok, c.astype(np.uint32))      # ---- network round trip ----
        t2 = time.perf_counter()
        r = ((y.astype(np.int64) << (fold.MSG_BITS - k)) - self.z[name][self.tok]) & (2 * half - 1)
        r = np.where(r >= half, r - 2 * half, r)                       # = M x~ + M e + reply rounding  (centred)
        out = r / S * L.scale
        t3 = time.perf_counter()
        self.t_mask += t1 - t0; self.t_unmask += t3 - t2
        self.up_bytes += 4 * len(c); self.down_bytes += len(y) * k / 8; self.rounds += 1
        return out
