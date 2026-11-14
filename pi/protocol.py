"""
End-to-end single-server private inference for Qwen3-0.6B (open weights, HF safetensors).

Server sees only: Ring-LWE ciphertexts of activations (online) and BFV ciphertexts of
mask secrets (offline). It runs plain int8 GEMVs online. Client does masking, unmasking,
the embedding lookup (public table), RMSNorm/RoPE/SwiGLU gating/softmax/GQA attention,
and holds the KV cache.
"""
import ctypes, glob, json, math, os, time, numpy as np
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

def gpu_free():
    """Bytes usable for weights on each GPU: 6% is kept back for the GEMM workspaces, and on big cards the first GPU
    keeps another 4 GB for the per-token pads."""
    import torch
    out = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        out.append(int(free - 0.06 * total - ((4 << 30) if i == 0 and total > (16 << 30) else 0)))
    return out

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
        idx = os.path.join(path, "model.safetensors.index.json")   # sharded: only the shards it lists (a repo may hold other copies)
        files = (sorted(os.path.join(path, f) for f in set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx)
                 else sorted(glob.glob(os.path.join(path, "*.safetensors"))))
        for f in files:
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
        if v["dtype"] == "F8_E4M3":              # FP8 checkpoints: real weight = fp8 value * <name>_scale_inv (one scalar per tensor)
            s = self[k + "_scale_inv"]; assert s.size == 1, "only per-tensor FP8 scales are supported"
            return (F8_E4M3[b] * np.float32(s.reshape(-1)[0])).reshape(v["shape"])
        raise ValueError(v["dtype"])

def _st_get(self, k, dev=None):
    """Tensor k as float32: numpy, or with dev a torch tensor decoded on that device (fast path for big models)."""
    if dev is None: return self[k]
    import torch
    raw, v = self.meta[k]; b = torch.from_numpy(np.array(raw[v["data_offsets"][0]:v["data_offsets"][1]])).to(dev)
    if v["dtype"] == "BF16": t = b.view(torch.bfloat16).float()
    elif v["dtype"] == "F32": t = b.view(torch.float32)
    elif v["dtype"] == "F8_E4M3": t = torch.as_tensor(F8_E4M3, device=dev)[b.long()] * float(np.float32(self[k + "_scale_inv"].reshape(-1)[0]))
    else: raise ValueError(v["dtype"])
    return t.reshape(v["shape"])
SafeTensors.get = _st_get

def _f8_e4m3_table():
    """All 256 float8_e4m3fn values: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits, subnormals, no infinities."""
    b = np.arange(256); e, m = (b >> 3) & 15, b & 7
    v = np.where(e == 0, m / 8 * 2.0 ** -6, (1 + m / 8) * 2.0 ** (e - 7.0))
    v[(b & 127) == 127] = np.nan
    return np.where(b >= 128, -v, v).astype(np.float32)
F8_E4M3 = _f8_e4m3_table()

def yarn_rope(rp, hd, max_pos):
    """YaRN RoPE as in transformers' _compute_yarn_parameters: (inverse frequencies, factor applied to cos and sin)."""
    base, orig = rp["rope_theta"], rp["original_max_position_embeddings"]
    factor = rp.get("factor") or max_pos / orig
    ms = lambda s, m=1: 1.0 if s <= 1 else 0.1 * m * math.log(s) + 1.0
    att = rp.get("attention_factor")
    if att is None:
        att = ms(factor, rp["mscale"]) / ms(factor, rp["mscale_all_dim"]) if rp.get("mscale") and rp.get("mscale_all_dim") else ms(factor)
    cd = lambda rot: hd * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(base))
    low, high = cd(rp.get("beta_fast") or 32), cd(rp.get("beta_slow") or 1)
    if rp.get("truncate", True): low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, hd - 1)
    ramp = np.clip((np.arange(hd // 2) - low) / ((high if high != low else high + 0.001) - low), 0, 1)
    pf = base ** (np.arange(0, hd, 2) / hd)
    return ramp / (factor * pf) + (1 - ramp) / pf, att

def qwen_path(repo="Qwen/Qwen3-0.6B"):
    for p in glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*")):
        idx = os.path.join(p, "model.safetensors.index.json")      # sharded: every shard must be present
        shards = set(json.load(open(idx))["weight_map"].values()) if os.path.exists(idx) else {"model.safetensors"}
        if all(os.path.exists(os.path.join(p, f)) for f in shards): return p
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, allow_patterns=["*.json", "*.txt", "model*.safetensors"])   # not other copies of the weights


# ---------------------------------------------------------------- model
class Linear:
    def __init__(self, W, keep_float=True):     # W float (d_out, d_in) -> per-row int8
        if not isinstance(W, np.ndarray):        # torch tensor on a GPU: same arithmetic there, int8 result back in host RAM
            import torch
            assert not keep_float
            sc = W.abs().amax(1).clamp_min(1e-30) / 127.0
            self.q, nm = np.empty(tuple(W.shape), dtype=np.int8), 0.0
            for r in range(0, W.shape[0], 8192):
                q = (W[r:r + 8192] / sc[r:r + 8192, None]).round_().clamp_(-127, 127)
                nm = max(nm, float(q.double().square_().sum(1).sqrt_().max()))
                self.q[r:r + 8192] = q.to(torch.int8).cpu().numpy()
            self.W, self.scale, self.row_norm_max, self._t, self.packed = None, sc.cpu().numpy(), nm, None, None
            return
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
        L.row_norm_max = max(np.sqrt((q[r:r + 8192].astype(np.float32) ** 2).sum(1, dtype=np.float64)).max()
                             for r in range(0, q.shape[0], 8192))
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
def sigmoid(x): return 1 / (1 + np.exp(-x))
def rope(x, cos, sin):                           # x (..., heads, hd), HF rotate_half convention, on the first cos.shape[-1] dims
    rd = cos.shape[-1]; h = rd // 2; xr = x[..., :rd]
    out = xr * cos + np.concatenate([-xr[..., h:], xr[..., :h]], -1) * sin
    return out if rd == x.shape[-1] else np.concatenate([out, x[..., rd:]], -1)

def merged_linear(R, G, w, keep_float, boost=1.0):
    """One round trip for 'residual branch R, add, RMSNorm (weight w), next matrix G':
        G rmsnorm(x + R v) = (G diag(w) R v + G (w x)) / rms(x + R v),
    so the server applies [[R, 0], [G diag(w) R, G]] to [v; w x] and the client, who then knows x + R v, divides by
    its rms. w stays on the client side: folded into G it would wreck the per-row int8 scales.
    The v-columns are divided by lam (the client sends lam * v). For one row the weight quantization error is smallest
    at lam = max|G diag(w) R| / max|G|; lam is the median of that over rows, times `boost` (>= 1, calibrated by
    merge_boost so that lam * v is not small next to w x, which would bury the R v rows in the protocol noise)."""
    if not isinstance(R, np.ndarray):            # torch tensors on a GPU (big models): quantize the two row groups there
        import torch
        P = (G * w) @ R
        lam = float(np.median((P.abs().amax(1) / G.abs().amax(1)).cpu().numpy())) * boost
        top = Linear(torch.cat([R / lam, torch.zeros((R.shape[0], G.shape[1]), device=R.device)], 1), False)
        bot = Linear(torch.cat([P / lam, G], 1), False); del P
        L = Linear.from_int(np.concatenate([top.q, bot.q]), np.concatenate([top.scale, bot.scale]))
        L.row_norm_max = max(top.row_norm_max, bot.row_norm_max)
        return L, lam
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
    """Decoder-only LM in HF safetensors layout with RMSNorm, SwiGLU, RoPE and GQA: Qwen3 (per-head q/k RMSNorm),
    Mistral3 text models (no q/k norm, YaRN RoPE, FP8 weights, tensors under model.language_model.), and the hybrid
    Qwen3.5/3.6/3.8 dense models (3 of 4 blocks are gated-DeltaNet linear attention, the rest gated full attention
    with partial RoPE; zero-centred RMSNorm weights). Every block is still 'input projection -> client-side token mixer
    -> output projection -> MLP', so the server side and the 2-trips-per-block merge are unchanged."""
    def __init__(self, path=None, keep_float=True, w4=None, merge=False, boost=None, dev=None):
        """dev (e.g. "cuda"): decode, merge and quantize the weights on that GPU instead of in numpy (needs keep_float=False).
        Same arithmetic; meant for big models, where the numpy path takes hours."""
        path = path or qwen_path()
        w4 = np.load(w4) if w4 else None                         # quant4.py output: GPTQ 4-bit block linears
        cfg = json.load(open(os.path.join(path, "config.json"))); cfg = cfg.get("text_config", cfg)
        W = SafeTensors(path)
        M = "model.language_model." if "model.language_model.norm.weight" in W else "model."
        self.d, self.L, self.vocab = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
        self.hq, self.hkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
        self.hd = cfg.get("head_dim") or self.d // self.hq
        self.eps, rp = cfg["rms_norm_eps"], cfg.get("rope_parameters") or dict(rope_theta=cfg.get("rope_theta"))
        assert not rp.get("llama_4_scaling_beta") and not cfg.get("sliding_window"), "not implemented"
        rd = int(self.hd * rp.get("partial_rotary_factor", cfg.get("partial_rotary_factor", 1.0)))   # rotary dims per head
        if rp.get("rope_type", "default") == "yarn": self.inv_freq, self.rope_scale = yarn_rope(rp, rd, cfg["max_position_embeddings"])
        else: self.inv_freq, self.rope_scale = rp["rope_theta"] ** (-np.arange(0, rd, 2) / rd), 1.0
        lt = cfg.get("layer_types") or ["full_attention"] * self.L
        is_lin = [t == "linear_attention" for t in lt]
        self.attn_gate = bool(cfg.get("attn_output_gate"))        # q_proj also emits a per-head output gate
        self.lin_dims = ([cfg["linear_num_key_heads"], cfg["linear_num_value_heads"], cfg["linear_key_head_dim"],
                          cfg["linear_value_head_dim"], cfg["linear_conv_kernel_dim"]] if any(is_lin) else None)
        zc = cfg.get("model_type", "").startswith("qwen3_5")       # zero-centred RMSNorm: the scale is 1 + weight
        f64 = lambda k: W[k].astype(np.float64)
        nrm = lambda k: f64(k) + 1.0 if zc else f64(k)
        def cat(*ks):                            # block weights: numpy float32, or torch float32 on dev
            ws = [W.get(k, dev) for k in ks]
            return ws[0] if len(ws) == 1 else np.concatenate(ws) if dev is None else __import__("torch").cat(ws)
        vec = lambda k: W.get(k, dev) + 1.0 if zc else W.get(k, dev)   # norm weight in the same form
        lin = lambda *ks: Linear(cat(*ks), keep_float)
        def inkeys(i):                           # the block's input projection (one matrix for the server)
            p = f"{M}layers.{i}."
            if is_lin[i]: return [p + "linear_attn." + s for s in ("in_proj_qkv.weight", "in_proj_z.weight", "in_proj_b.weight", "in_proj_a.weight")]
            return [p + "self_attn." + s for s in ("q_proj.weight", "k_proj.weight", "v_proj.weight")]
        outkey = lambda i: f"{M}layers.{i}." + ("linear_attn.out_proj.weight" if is_lin[i] else "self_attn.o_proj.weight")
        def extras(i):                           # small public weights of the client-side token mixer
            p = f"{M}layers.{i}."
            if is_lin[i]:
                a = p + "linear_attn."
                return dict(linear=1.0, conv=f64(a + "conv1d.weight").reshape(-1, self.lin_dims[4]), dt_bias=f64(a + "dt_bias"),
                            A_log=f64(a + "A_log"), gnorm=f64(a + "norm.weight"))
            a = p + "self_attn."
            return dict(q_norm=nrm(a + "q_norm.weight"), k_norm=nrm(a + "k_norm.weight")) if a + "q_norm.weight" in W else {}
        if w4 is not None:
            lin4 = lambda key: Linear.from_int(unpack4(w4[key + ".packed"]), w4[key + ".scale"], w4[key + ".packed"])
        self.embed = W.raw(M + "embed_tokens.weight")          # public table (bf16 memmap); client looks up its row
        self.blocks = []; self.merged = merge
        for i in range(self.L if merge else 0):  # 2 round trips per block: o+gate_up ("ob") and down+next qkv ("da")
            assert w4 is None, "merge needs the float weights"
            p = f"{M}layers.{i}."; m = p + "mlp."; nx = f"{M}layers.{i + 1}."
            b = dict(qkv=lin(*inkeys(0))) if i == 0 else {}
            bo = lambda k: (boost or {}).get(f"{i}.{k}", 1.0)
            b["ob"], b["lam_ob"] = merged_linear(cat(outkey(i)), cat(m + "gate_proj.weight", m + "up_proj.weight"),
                                                 vec(p + "post_attention_layernorm.weight"), keep_float, bo("ob"))
            b["w_ob"] = nrm(p + "post_attention_layernorm.weight")
            if i < self.L - 1:
                b["da"], b["lam_da"] = merged_linear(
                    cat(m + "down_proj.weight"), cat(*inkeys(i + 1)), vec(nx + "input_layernorm.weight"), keep_float, bo("da"))
                b["w_da"] = nrm(nx + "input_layernorm.weight")
            else: b["down"] = lin(m + "down_proj.weight")
            b.update(ln1=nrm(p + "input_layernorm.weight"), **extras(i))
            self.blocks.append(b)
        for i in range(0 if merge else self.L):
            p = f"{M}layers.{i}."; m = p + "mlp."
            self.blocks.append(dict(
                qkv=lin(*inkeys(i)),
                o=lin(outkey(i)),
                gate_up=lin(m + "gate_proj.weight", m + "up_proj.weight"),
                down=lin(m + "down_proj.weight"),
            ) if w4 is None else {k: lin4(f"{i}.{k}") for k in ("qkv", "o", "gate_up", "down")})
            self.blocks[-1].update(
                ln1=nrm(p + "input_layernorm.weight"), ln2=nrm(p + "post_attention_layernorm.weight"), **extras(i))
        self.norm = nrm(M + "norm.weight")
        self.head = lin("lm_head.weight" if "lm_head.weight" in W else M + "embed_tokens.weight")
    def linears(self):
        out = []
        for i, b in enumerate(self.blocks):
            out += [(f"{i}.{k}", b[k]) for k in ("qkv", "o", "gate_up", "ob", "da", "down") if k in b]
        return out + [("head", self.head)]

class KVList:
    """Growing buffer for one layer's keys or values: behaves like the list of per-token arrays it replaces
    (extend, append, len, del x[n:], np.asarray), but np.asarray is a view instead of a copy of the whole history."""
    def __init__(self): self.buf, self.n = None, 0
    def __len__(self): return self.n
    def extend(self, rows):
        rows = np.asarray(rows)
        if self.buf is None: self.buf = np.empty((max(64, 2 * len(rows)),) + rows.shape[1:], rows.dtype)
        if self.n + len(rows) > len(self.buf):
            nb = np.empty((max(2 * len(self.buf), self.n + len(rows)),) + self.buf.shape[1:], self.buf.dtype); nb[:self.n] = self.buf[:self.n]; self.buf = nb
        self.buf[self.n:self.n + len(rows)] = rows; self.n += len(rows)
    def append(self, row): self.extend(np.asarray(row)[None])
    def __delitem__(self, sl):
        assert isinstance(sl, slice) and sl.stop is None and sl.step is None; self.n = min(self.n, sl.start)
    def __array__(self, dtype=None, copy=None): return self.buf[:self.n] if self.buf is not None else np.empty((0,))
    def __getitem__(self, i): return self.buf[:self.n][i]

def new_cache(m):
    """Per-layer caches: KV buffers for full attention, state lists for linear attention (rollback-able)."""
    return [([], []) if "linear" in b else (KVList(), KVList()) for b in m.blocks]

def attend(q, K, V):
    """GQA: q (hq, hd); K, V (T, hkv, hd). Query head h reads kv head h // (hq // hkv)."""
    hq, hd = q.shape; hkv = K.shape[1]
    qg = q.reshape(hkv, hq // hkv, hd)
    s = np.matmul(qg, K.transpose(1, 2, 0)) / np.sqrt(hd)                      # (hkv, group, T), BLAS
    s = np.exp(s - s.max(-1, keepdims=True)); s /= s.sum(-1, keepdims=True)
    return np.matmul(s, V.transpose(1, 0, 2)).reshape(hq * hd)

_DELTA_T = {}
def _delta_mix_gpu(m, b, rows, layer_cache, t0):
    """delta_mix on the GPU (float64, same arithmetic): the per-token recurrence is ~15 small kernels per layer
    instead of numpy calls. States are kept as GPU tensors in the same (states, placeholders) lists."""
    import torch
    nk, nv, dk, dv, ks = m.lin_dims; cd = 2 * nk * dk + nv * dv; rep_ = nv // nk
    P = _DELTA_T.get(id(b))
    if P is None or P["src"] is not b["conv"]:                 # ids get reused after a model is freed: check the source
        P = _DELTA_T[id(b)] = {k: torch.as_tensor(np.ascontiguousarray(b[k]), device="cuda") for k in ("conv", "A_log", "dt_bias", "gnorm")}
        P["src"] = b["conv"]
        P["convT"] = P["conv"].T.contiguous(); P["negA"] = -P["A_log"].exp()
    hist, ph = layer_cache
    win, S = hist[-1] if hist else (torch.zeros((ks - 1, cd), dtype=torch.float64, device="cuda"),
                                    torch.zeros((nv, dk, dv), dtype=torch.float64, device="cuda"))
    R = torch.as_tensor(rows, device="cuda"); out = []
    for r in R:
        xq, z, bb, aa = torch.split(r, [cd, nv * dv, nv, nv])
        full = torch.cat([win, xq[None]]); win = full[1:]
        y = torch.nn.functional.silu((full * P["convT"]).sum(0))
        q, k, v = torch.split(y, [nk * dk, nk * dk, nv * dv])
        q, k, v = q.view(nk, dk).repeat_interleave(rep_, 0), k.view(nk, dk).repeat_interleave(rep_, 0), v.view(nv, dv)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) / dk ** 0.5
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        beta, g = torch.sigmoid(bb), P["negA"] * torch.nn.functional.softplus(aa + P["dt_bias"])
        S = S * g.exp()[:, None, None]
        S = S + k[:, :, None] * ((v - torch.bmm(k[:, None, :], S)[:, 0]) * beta[:, None])[:, None, :]
        o = torch.bmm(q[:, None, :], S)[:, 0]
        o = o * torch.rsqrt((o * o).mean(-1, keepdim=True) + m.eps) * P["gnorm"] * torch.nn.functional.silu(z.view(nv, dv))
        out.append(o.reshape(-1)); hist.append((win, S)); ph.append(None)
    for j in range(max(0, t0 - 1)):
        if hist[j] is None: continue
        hist[j] = None
    return torch.stack(out).cpu().numpy()

def delta_mix(m, b, rows, layer_cache, t0):
    if _use_gpu_mix(): return _delta_mix_gpu(m, b, rows, layer_cache, t0)
    return _delta_mix_np(m, b, rows, layer_cache, t0)

def _use_gpu_mix():
    if os.environ.get("PI_MIX_CPU"): return False
    try:
        import torch; return torch.cuda.is_available()
    except ImportError: return False

def _delta_mix_np(m, b, rows, layer_cache, t0):
    """Gated DeltaNet linear attention (Qwen3.5 family), client side, token by token. rows (T, R) = the block's input
    projection [q k v | z | b | a]. State: the last (kernel - 1) conv inputs and the recurrent matrix S per value head.
    layer_cache = (states, placeholders): states[t] is the state after token t; older ones are dropped except what a
    rollback within the current pass (speculative decoding) can need."""
    nk, nv, dk, dv, ks = m.lin_dims; cd = 2 * nk * dk + nv * dv; rep_ = nv // nk
    hist, ph = layer_cache
    win, S = hist[-1] if hist else (np.zeros((ks - 1, cd)), np.zeros((nv, dk, dv)))
    out = []
    for r in rows:
        xq, z, bb, aa = np.split(r, [cd, cd + nv * dv, cd + nv * dv + nv])
        full = np.vstack([win, xq[None]]); win = full[1:]           # causal depthwise conv over the last ks inputs
        y = silu((full * b["conv"].T).sum(0))
        q, k, v = np.split(y, [nk * dk, 2 * nk * dk])
        q, k, v = np.repeat(q.reshape(nk, dk), rep_, 0), np.repeat(k.reshape(nk, dk), rep_, 0), v.reshape(nv, dv)
        q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) / np.sqrt(dk)
        k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
        beta, g = sigmoid(bb), -np.exp(b["A_log"]) * np.logaddexp(0, aa + b["dt_bias"])
        S = S * np.exp(g)[:, None, None]
        S = S + k[:, :, None] * ((v - np.matmul(k[:, None, :], S)[:, 0]) * beta[:, None])[:, None, :]   # batched BLAS
        o = np.matmul(q[:, None, :], S)[:, 0]
        o = o / np.sqrt((o * o).mean(-1, keepdims=True) + m.eps) * b["gnorm"] * silu(z.reshape(nv, dv))
        out.append(o.reshape(-1)); hist.append((win, S)); ph.append(None)
    for j in range(max(0, t0 - 1)):
        if hist[j] is None: continue
        hist[j] = None
    return np.array(out)

def mix(m, b, i, rows, cache, t0, cos, sin):
    """Client-side token mixer of block i for the T tokens of a pass: rows (T, R) = output of the block's input
    projection, cos/sin (T, 1, rotary dims). Full attention with a KV cache (optionally output-gated), or gated DeltaNet.
    Returns (T, A), the input of the block's output projection."""
    if "linear" in b: return delta_mix(m, b, rows, cache[i], t0)
    T = len(rows); nq, nk = m.hq * m.hd * (2 if m.attn_gate else 1), m.hkv * m.hd
    qkn = lambda x, k: rmsnorm(x, b[k], m.eps) if k in b else x       # Qwen3 normalises q and k per head
    q, k, v = np.split(rows, [nq, nq + nk], axis=1)
    if m.attn_gate: q = q.reshape(T, m.hq, 2 * m.hd); gate = q[..., m.hd:].reshape(T, -1); q = q[..., :m.hd]
    q = rope(qkn(q.reshape(T, m.hq, m.hd), "q_norm"), cos, sin)
    k = rope(qkn(k.reshape(T, m.hkv, m.hd), "k_norm"), cos, sin)
    cache[i][0].extend(k); cache[i][1].extend(v.reshape(T, m.hkv, m.hd))
    K, V = np.asarray(cache[i][0]), np.asarray(cache[i][1])            # a view for KVList, a copy for plain lists
    a = np.stack([attend(q[j], K[:t0 + j + 1], V[:t0 + j + 1]) for j in range(T)])
    if not m.attn_gate: return a
    return a * (silu(gate) if os.environ.get("PI_GATE") == "swish" else sigmoid(gate))   # HF: sigmoid (see gate_check.py)

def _angles(m, t0, T):
    ang = (t0 + np.arange(T))[:, None] * np.concatenate([m.inv_freq] * 2)[None]
    return np.cos(ang)[:, None] * m.rope_scale, np.sin(ang)[:, None] * m.rope_scale      # (T, 1, rotary dims)

def forward_token(m, tok, cache, lin):
    """One decode step. `lin(name, layer, x)` performs the linear layer (plain or private)."""
    pos = len(cache[0][0]); cos, sin = _angles(m, pos, 1)
    x = bf16_to_f32(m.embed[tok]).astype(np.float64)
    rms = lambda x: np.sqrt((x * x).mean() + m.eps)
    for i, b in enumerate(m.blocks if m.merged else []):
        if i == 0: qkv = lin("0.qkv", b["qkv"], rmsnorm(x, b["ln1"], m.eps))
        a = mix(m, b, i, qkv[None], cache, pos, cos, sin)[0]
        out = lin(f"{i}.ob", b["ob"], np.concatenate([b["lam_ob"] * a, b["w_ob"] * x]))              # o, add, norm, gate_up
        x = x + out[:m.d]; g, u = np.split(out[m.d:] / rms(x), 2)
        if "da" in b:
            out = lin(f"{i}.da", b["da"], np.concatenate([b["lam_da"] * silu(g) * u, b["w_da"] * x]))   # down, add, norm, next qkv
            x = x + out[:m.d]; qkv = out[m.d:] / rms(x)
        else: x = x + lin(f"{i}.down", b["down"], silu(g) * u)
    for i, b in enumerate([] if m.merged else m.blocks):
        qkv = lin(f"{i}.qkv", b["qkv"], rmsnorm(x, b["ln1"], m.eps))
        a = mix(m, b, i, qkv[None], cache, pos, cos, sin)[0]
        x = x + lin(f"{i}.o", b["o"], a)
        g, u = np.split(lin(f"{i}.gate_up", b["gate_up"], rmsnorm(x, b["ln2"], m.eps)), 2)
        x = x + lin(f"{i}.down", b["down"], silu(g) * u)
    return lin("head", m.head, rmsnorm(x, m.norm, m.eps))


def forward_batch(m, toks, cache, lin, n_head=None):
    """Merged model, T tokens in ONE pass: still 2 L + 2 round trips in total. `lin(name, layer, X)` maps (T, d_in) to
    (T, d_out). Token j attends to the cache and to tokens <= j of this pass. Returns the logits of the last n_head
    tokens (default all). Used for sending a whole prompt at once and for checking several guessed tokens at once."""
    assert m.merged
    T, t0 = len(toks), len(cache[0][0]); cos, sin = _angles(m, t0, T)
    x = np.stack([bf16_to_f32(m.embed[t]) for t in toks]).astype(np.float64)
    rms = lambda x: np.sqrt((x * x).mean(-1, keepdims=True) + m.eps)
    for i, b in enumerate(m.blocks):
        if i == 0: qkv = lin("0.qkv", b["qkv"], rmsnorm(x, b["ln1"], m.eps))
        a = mix(m, b, i, qkv, cache, t0, cos, sin)
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
        """Weights to the GPU(s) (after the offline phase, which needs the GPU memory). 4-bit stays packed there.
        Layers go to the GPUs in order, moving on when one is full, so each GEMV runs where its weights are."""
        import torch
        self.Wg = None; torch.cuda.empty_cache()
        free = gpu_free(); dev = 0; self.Wg = {}
        for n, L in self.lin.items():
            w = L.packed if L.packed is not None else L.q
            while w.nbytes > free[dev] and dev + 1 < len(free): dev += 1
            self.Wg[n] = torch.as_tensor(w, device=f"cuda:{dev}"); free[dev] -= w.nbytes
    def weight_gpu(self, name):                  # int8 view for the GEMM (4-bit: unpacked on the fly, every call)
        return unpack4(self.Wg[name]) if self.lin[name].packed is not None else self.Wg[name]
    def online(self, name, c):
        t = time.perf_counter()
        if self.gpu:   # time includes host<->GPU copies of c and y (they arrive/leave over the network)
            import torch
            W = self.weight_gpu(name); y = gemv_u32_gpu(W, torch.as_tensor(c.view(np.int32), device=W.device)).cpu().numpy()
        else: y = gemv_u32(self.lin[name].q, c)
        self.time_online += time.perf_counter() - t
        return y

class Client:
    def __init__(self, server, lwe, bfv, seed=None, gpu=False):   # seed=None: secrets from OS entropy (tests may fix it)
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
            W = self.weight_gpu(name)
            w = torch.as_tensor(c.view(np.int32), device=W.device) - self.u[name].to(W.device)   # int32 wraps: mod 2^32
            y = fold.reply_bits(gemv_u32_gpu(W, w), self.out_bits).to(torch.int32).cpu().numpy()
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
            step = max(1, min(8, 2400 // len(self.pads.FA)))      # tokens per FFT batch: bounded GPU memory for big models
            parts = [self.pads.batch(self.b[:, toks[i:i + step]].transpose(1, 0, 2)) for i in range(0, len(toks), step)]
            self.U = {nm: torch.cat([p[j] for p in parts]) for j, nm in enumerate(self.order)}
            if self.gpu: torch.cuda.synchronize()
            self.time_pads += time.perf_counter() - t
        U = self.U[name][[self.pos[int(tk)] for tk in toks]]
        if self.gpu:
            W = self.weight_gpu(name); U = U.to(W.device)
            y = torch.cat([fold.reply_bits(gemm_u32_gpu(W, torch.as_tensor(C[i:i + 16].view(np.int32), device=W.device) - U[i:i + 16]),
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


# ---------------------------------------------------------------- prepared weights (save once, reload bit-identically)
def save_prepared(m, out, tokenizer_json):
    """Write the quantized (and merged) model to the directory `out`: plain .npy files plus meta.json, and return the
    weights id, a SHA-256 over everything. A server that reloads these files is bit-identical, so a client's cached
    one-time C1' (keyed by this id) stays valid. Reloading also skips decoding, merging and quantizing (big models)."""
    import hashlib, shutil
    os.makedirs(out, exist_ok=True); h = hashlib.sha256()
    def put(name, a):
        a = np.ascontiguousarray(a); np.save(os.path.join(out, name + ".npy"), a, allow_pickle=False)
        h.update(name.encode() + a.dtype.str.encode() + str(a.shape).encode())
        v = a.reshape(-1).view(np.uint8)
        for i in range(0, len(v), 1 << 30): h.update(v[i:i + (1 << 30)])
    meta = dict(d=m.d, L=m.L, vocab=m.vocab, hq=m.hq, hkv=m.hkv, hd=m.hd, eps=m.eps, rope_scale=float(m.rope_scale),
                merged=m.merged, attn_gate=bool(getattr(m, "attn_gate", False)), lin_dims=getattr(m, "lin_dims", None),
                linears=[], blocks=[])
    put("inv_freq", np.asarray(m.inv_freq, np.float64)); put("norm", np.asarray(m.norm, np.float64)); put("embed", np.asarray(m.embed))
    for n, L in m.linears():
        put(n + ".q", L.q); put(n + ".scale", np.asarray(L.scale, np.float32))
        meta["linears"].append([n, int(L.q.shape[0]), int(L.q.shape[1]), float(L.row_norm_max)])
    for i, b in enumerate(m.blocks):
        rec = dict(scalars={k: float(v) for k, v in b.items() if np.isscalar(v)}, vectors=[], linears=[])
        for k, v in b.items():
            if isinstance(v, Linear): rec["linears"].append(k)
            elif isinstance(v, np.ndarray): rec["vectors"].append(k); put(f"blk{i}.{k}", v.astype(np.float64))
        meta["blocks"].append(rec)
    h.update(json.dumps(meta, sort_keys=True).encode())
    meta["weights_id"] = h.hexdigest()[:32]
    shutil.copy(tokenizer_json, os.path.join(out, "tokenizer.json"))
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    return meta["weights_id"]

def weights_hash(m, extra=""):
    """SHA-256 fingerprint of everything a server's one-time C1' depends on (int8 weights, scales, norm vectors), for
    servers that don't load prepared weights. The client keys its C1' cache by it."""
    import hashlib
    h = hashlib.sha256(extra.encode())
    for n, L in m.linears():
        h.update(n.encode()); v = np.ascontiguousarray(L.q).reshape(-1).view(np.uint8)
        for i in range(0, len(v), 1 << 30): h.update(v[i:i + (1 << 30)])
        h.update(np.ascontiguousarray(L.scale, dtype=np.float64).tobytes())
    for b in m.blocks:
        for k in sorted(b):
            if isinstance(b[k], np.ndarray): h.update(k.encode() + np.ascontiguousarray(b[k], dtype=np.float64).tobytes())
    return h.hexdigest()[:32]

def load_prepared(d):
    """Inverse of save_prepared: a Qwen3 object (all fields forward_token, forward_batch and the servers use)."""
    meta = json.load(open(os.path.join(d, "meta.json")))
    get = lambda name: np.load(os.path.join(d, name + ".npy"), allow_pickle=False)
    m = Qwen3.__new__(Qwen3)
    for k in ("d", "L", "vocab", "hq", "hkv", "hd", "eps", "rope_scale", "merged"): setattr(m, k, meta[k])
    m.attn_gate, m.lin_dims = meta.get("attn_gate", False), meta.get("lin_dims")
    m.inv_freq, m.norm, m.embed, m.weights_id = get("inv_freq"), get("norm"), get("embed"), meta["weights_id"]
    lin = {}
    for n, r, k, rn in meta["linears"]:
        L = Linear.__new__(Linear); L.q, L.scale = get(n + ".q"), get(n + ".scale")
        L.W, L.packed, L._t, L.row_norm_max = None, None, None, rn; lin[n] = L
    m.head, m.blocks = lin["head"], []
    for i, rec in enumerate(meta["blocks"]):
        b = dict(rec["scalars"]); b.update({k: get(f"blk{i}.{k}") for k in rec["vectors"]})
        b.update({k: lin[f"{i}.{k}"] for k in rec["linears"]}); m.blocks.append(b)
    return m
