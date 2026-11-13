"""v4 'chain' mode, stage 1 prototype: two weight matrices applied one after the other on masked data, mod q = 2^56.

One round trip for 'residual branch R, add, RMSNorm (weight w), next matrix G' WITHOUT the merged product matrix:
    client sends  c_v = A_v alpha + e_v + v~        (v~ = round(S v))
                  c_x = A_x alpha + e_x + x~        (x~ = round(K S (w . x)))
    server        t = R_q c_v,  u = W . t + c_x,  y = G_q u        (all mod 2^56; W = round(K w . s_R), public integers)
    client        x_new = x + s_R . (t - z_t) / S,   G (w . x_new) ~ s_G . (y - z_y) / (K S)
so the server reads R and G once each (the original weights), as in 4-trips mode, but it is 2 trips per block.
Masks: Ring-LWE n = 2048 (power-of-two negacyclic blocks), sigma 6.4, q = 2^56: about 103 bits by the lattice
estimator's conservative measure (the 32-bit n = 1024 masks: 95).

Stage 1 only checks the arithmetic and the precision: z (the unmasking values) is computed directly here, plus a
simulated rounding error the size the one-time-file route would add. The offline/fold machinery comes in stage 2."""
import numpy as np, torch
from protocol import InnerLWE, dgauss, bf16_to_f32, rmsnorm, silu, mix, _angles

QB = 56; Q = 1 << QB; MASK = Q - 1; HALF = 1 << (QB - 1)
N_LWE, SIGMA = 2048, 6.4
WB = 8                                                     # bits of the integer multipliers W

_SH = {}
def gemm56(W, C):
    """W int8 (d_out, d_in) GPU tensor, C (T, d_in) int64 GPU tensor in [0, 2^56) -> W C^T mod 2^56 as (T, d_out) int64.
    Seven bytes as signed digits s_k = u_k - 256 b_k plus the borrow bits b_0..b_5 (b_6's term is a multiple of 2^56):
    13 int8 columns per token in one int8 GEMM, so the weights are read once."""
    dev = W.device
    if dev not in _SH: _SH[dev] = torch.tensor([1 << (8 * k) for k in range(7)] + [1 << (8 * k + 8) for k in range(6)] + [0, 0, 0], device=dev)
    T, din = C.shape
    if din % 8:                                              # cuBLASLt int8 wants K a multiple of 8: zero-pad (exact)
        W = torch.nn.functional.pad(W, (0, -din % 8)); C = torch.nn.functional.pad(C, (0, -din % 8)); din = C.shape[1]
    u = [(C >> (8 * k)) & 255 for k in range(7)]
    s = [(x - 256 * (x >= 128)) for x in u]; b = [(x >= 128).long() for x in u[:6]]
    X = torch.stack(s + b + [torch.zeros_like(u[0])] * 3, -1).to(torch.int8)           # (T, din, 16)
    Y = torch._int_mm(W, X.permute(1, 0, 2).reshape(din, 16 * T))                      # (d_out, 16 T) int32
    return ((Y.view(-1, T, 16).long() * _SH[dev]).sum(2) & MASK).T                    # wraps mod 2^64: fine mod 2^56

def negmul56(a, alpha):
    """a (nb, n) int64 in [0, 2^56), alpha (n,) small ints -> blockwise negacyclic a * alpha mod 2^56, exact (FFT on 14-bit limbs)."""
    n = a.shape[-1]; out = np.zeros(a.shape, dtype=np.int64)
    S = np.fft.rfft(alpha.astype(np.float64), 2 * n)
    for k in range(4):                                       # 14-bit limbs: conv <= 2^14 * |alpha|_1 < 2^53
        limb = (a >> (14 * k)) & ((1 << 14) - 1)
        full = np.rint(np.fft.irfft(np.fft.rfft(limb.astype(np.float64), 2 * n, axis=-1) * S, 2 * n, axis=-1)).astype(np.int64)
        out = (out + ((full[..., :n] - full[..., n:]) << (14 * k))) & MASK
    return out

class Lin56:
    """A server matrix (int8 + per-row scale) on the GPU, with its public row-norm bound and spectral norm."""
    def __init__(self, L, dev="cuda"):
        self.q, self.scale = L.q, L.scale.astype(np.float64)
        self.W = torch.as_tensor(L.q, device=dev)
        qf = torch.as_tensor(L.q, device=dev, dtype=torch.float32)
        self.rn = float(qf.square().sum(1).sqrt().max())                                   # max row 2-norm
        self.spec = float(torch.linalg.matrix_norm(qf, ord=2)) * 1.001                      # spectral norm (public)

class Chain:
    """Stage-1 private evaluation of an UNMERGED model (protocol.Qwen3, merge=False) with 2 round trips per block."""
    def __init__(self, m, seed=11, dev="cuda", z_noise=True):
        self.m, self.dev, self.rng, self.z_noise = m, dev, np.random.default_rng(seed), z_noise
        self.lwe = InnerLWE(seed=5); self.lwe.n = N_LWE                       # public a's from a seed, n = 2048
        self.L, self.a = {}, {}
        for i, b in enumerate(m.blocks):
            for k in ("qkv", "o", "gate_up", "down"): self.L[f"{i}.{k}"] = Lin56(b[k], dev)
        self.L["head"] = Lin56(m.head, dev)
        for lid, (name, L) in enumerate(self.L.items()): self.a[name] = self.lwe.public_a(lid, L.q.shape[1]) & MASK
        self.Wint = {}                                                        # (R, G, w) pairs: W = round(K w . s_R)
        for i, b in enumerate(m.blocks):
            self.Wint[f"{i}.o"] = self._wint(self.L[f"{i}.o"], b["ln2"])
            if i + 1 < len(m.blocks): self.Wint[f"{i}.down"] = self._wint(self.L[f"{i}.down"], m.blocks[i + 1]["ln1"])
        self.rounds = self.up = self.down = 0; self.server_s = 0.0

    def _wint(self, R, w):
        ws = w * R.scale; K = (2 ** WB - 1) / np.abs(ws).max()
        return K, torch.as_tensor(np.rint(K * ws).astype(np.int64), device=self.dev), float(np.abs(np.rint(K * ws)).max())

    def _mask(self, name, alpha, X):                       # X (T, din) integers -> masked mod 2^56, as a GPU tensor
        T, din = X.shape
        Aa = negmul56(self.a[name], alpha).reshape(-1)[:din]
        e = dgauss(self.rng, X.shape, SIGMA)
        return torch.as_tensor((Aa[None] + e + X) & MASK, device=self.dev), Aa

    def _z(self, f, Aa_parts):
        """Unmasking values: the server's linear map applied to the masks alone (stage 1: computed directly), plus the
        rounding error the fold route will add (|err| ~ 2^24 * |s| for 32-bit C1' entries and a ternary s of 8192).
        Returns a tuple with one tensor per output of f."""
        out = f(*[torch.as_tensor(p[None], device=self.dev) for p in Aa_parts])
        out = out if isinstance(out, tuple) else (out,)
        if self.z_noise:                                     # residual rows (first output of a pair) keep 53-bit C1' entries
            sd = [2 ** 7.4, 2 ** 28.4] if len(out) == 2 else [2 ** 28.4]
            out = tuple((z + torch.as_tensor(np.rint(self.rng.normal(0, e, z.shape)).astype(np.int64), device=self.dev)) & MASK for z, e in zip(out, sd))
        return out

    @staticmethod
    def _centre(v): return ((v + HALF) & MASK) - HALF

    def single(self, name, X, alpha):
        """One matrix, one trip (first input projection, last down, head). Scaled so |M x~| <= 2^54."""
        L = self.L[name]; nx = np.linalg.norm(X, axis=1, keepdims=True)
        S = 2.0 ** (QB - 2) / (L.rn * np.maximum(nx, 1e-300))
        c, Aa = self._mask(name, alpha, np.rint(X * S).astype(np.int64))
        torch.cuda.synchronize(); t0 = __import__("time").perf_counter()
        y = gemm56(L.W, c); torch.cuda.synchronize(); self.server_s += __import__("time").perf_counter() - t0
        (z,) = self._z(lambda A: gemm56(L.W, A), [Aa])
        r = self._centre((y - z) & MASK).double().cpu().numpy()
        self.rounds += 1; self.up += 7 * X.size; self.down += 2.5 * y.numel()
        return r / S * L.scale

    def pair(self, rname, gname, v, wx, alpha):
        """One trip for 'R v, add, norm, G': returns (R v, G (w . x_new) before the rms division)."""
        R, G = self.L[rname], self.L[gname]; K, Wt, Wmax = self.Wint[rname]
        nv, nwx = np.linalg.norm(v, axis=1, keepdims=True), np.linalg.norm(wx, axis=1, keepdims=True)
        bound_u = Wmax * R.spec * nv + K * nwx                          # |u| <= max|W| ||R_q||_2 |v~|/S... per unit S
        S = 2.0 ** (QB - 2) / (G.rn * np.maximum(bound_u, 1e-300))      # => |G_q u| <= 2^54
        cv, Av = self._mask(rname, alpha, np.rint(v * S).astype(np.int64))
        cx, Ax = self._mask(gname, alpha, np.rint(wx * K * S).astype(np.int64))
        def f(cv, cx):
            t = gemm56(R.W, cv); return t, gemm56(G.W, (Wt * t + cx) & MASK)
        torch.cuda.synchronize(); t0 = __import__("time").perf_counter()
        t, y = f(cv, cx); torch.cuda.synchronize(); self.server_s += __import__("time").perf_counter() - t0
        zt, zy = self._z(f, [Av, Ax])
        rt = self._centre((t - zt) & MASK).double().cpu().numpy(); ry = self._centre((y - zy) & MASK).double().cpu().numpy()
        self.rounds += 1; self.up += 7 * (v.size + wx.size); self.down += 7 * t.numel() + 2.5 * y.numel()
        return rt / S * R.scale, ry / (K * S) * G.scale

    def forward(self, toks, cache):
        """Like protocol.forward_batch, but private through chained trips: 2 per block (+1 first, +1 head)."""
        m = self.m; T, t0 = len(toks), len(cache[0][0]); cos, sin = _angles(m, t0, T)
        alpha = dgauss(self.rng, N_LWE, SIGMA)              # stage 1: one secret per pass
        x = np.stack([bf16_to_f32(m.embed[t]) for t in toks]).astype(np.float64)
        rms = lambda x: np.sqrt((x * x).mean(-1, keepdims=True) + m.eps)
        qkv = self.single("0.qkv", rmsnorm(x, m.blocks[0]["ln1"], m.eps), alpha)
        for i, b in enumerate(m.blocks):
            a = mix(m, b, i, qkv, cache, t0, cos, sin)
            rv, gu = self.pair(f"{i}.o", f"{i}.gate_up", a, b["ln2"] * x, alpha)            # o, add, norm, gate_up
            x = x + rv; g, u = np.split(gu / rms(x), 2, axis=1)
            h = silu(g) * u
            if i + 1 < len(m.blocks):
                rv, qkv = self.pair(f"{i}.down", f"{i + 1}.qkv", h, m.blocks[i + 1]["ln1"] * x, alpha)   # down, add, norm, next qkv
                x = x + rv; qkv = qkv / rms(x)
            else: x = x + self.single(f"{i}.down", h, alpha)
        return self.single("head", rmsnorm(x, m.norm, m.eps), alpha)
