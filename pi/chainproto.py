"""v4 'chain' protocol: 2 round trips per block with the server reading only the ORIGINAL weights (no merged
product matrices), using 56-bit masks (chain.py) and the fold56 offline machinery (one-time public C1', per-batch
upload b, nothing downloaded per batch). Same structure as protocol.FoldServer / FoldClient.

Trip kinds:
  single(name)        y = M c                               reply: 24 bits per row (bits 31..54)
  pair(R, G)          t = R c_v, y = G (W . t + c_x)        reply: t in full (56 bits), y in 24 bits
The client scales every row so that |value| <= 2^54 (Cauchy-Schwarz for single, |u| <= max|W| ||R||_2 |v~| + |x~|
for the chain), and the server's numbers never wrap."""
import time, numpy as np, torch
import fold56 as F
try:
    import kernel56 as K                         # fused Triton kernel (falls back to fold56.gemm_mod without Triton)
except ImportError: K = None
from fold import NegFFT
from protocol import dgauss, bf16_to_f32, rmsnorm, silu, mix, _angles

M56, HALF = F.M56, 1 << 55
N_LWE, SIGMA, WB, RB = 2048, 6.4, 8, 24          # inner ring, error, bits of W, reply bits of single/chain rows
SH = 55 - RB                                      # reply = bits SH..54 of y

def centre(v): return ((v + HALF) & M56) - HALF
def centre55(v):                                  # 24-bit replies carry bits SH..54: the value is known mod 2^55, |value| < 2^54
    return ((v + (1 << 54)) & ((1 << 55) - 1)) - (1 << 54)

def inner_a(lid, din, seed=21):
    """Public inner a's for one input layer: (blocks, 2048) int64 in [0, 2^56). The client derives them itself."""
    return np.random.default_rng([seed, lid]).integers(0, 1 << 56, size=(-(-din // N_LWE), N_LWE), dtype=np.int64)

class Masks:
    """A_l alpha mod 2^56 for several tokens at once (exact: 16-bit limbs of a, small alpha)."""
    def __init__(self, a, dev):
        self.f, self.dev = NegFFT(N_LWE, dev), dev
        self.FA = {k: self.f.fwd(F._limbs16(v, 4, dev)) for k, v in a.items()}         # (nb, 4, n/2)
    def __call__(self, name, alphas, din):
        FS = self.f.fwd(torch.as_tensor(alphas, dtype=torch.float64, device=self.dev))   # (T, n/2)
        P = torch.round(self.f.inv(self.FA[name] * FS[:, None, None, :])).long()       # (T, nb, 4, n)
        out = sum(P[:, :, k] << (16 * k) for k in range(4)) & M56
        return out.reshape(len(alphas), -1)[:, :din]


class ChainServer:
    """Holds the int8 weights (unmerged) and the public randomness. Sees only b (per batch) and masked inputs."""
    def __init__(self, model, dev="cuda", p=None):
        self.p, self.dev, self.model = p or F.P(), dev, model
        self.lin = dict(model.linears()); self.order = list(self.lin)
        self.Wg = {n: torch.as_tensor(L.q, device=dev) for n, L in self.lin.items()}
        self.a = {n: inner_a(lid, L.q.shape[1]) for lid, (n, L) in enumerate(self.lin.items())}
        self.pairs, self.singles = [], ["0.qkv"]         # the trips: (R, G, norm weight) and single matrices
        for i, b in enumerate(model.blocks):
            self.pairs.append((f"{i}.o", f"{i}.gate_up", b["ln2"]))
            if i + 1 < len(model.blocks): self.pairs.append((f"{i}.down", f"{i + 1}.qkv", model.blocks[i + 1]["ln1"]))
            else: self.singles.append(f"{i}.down")
        self.singles.append("head")
        self.pub = {}                                    # public per-matrix numbers the client needs for scaling
        for n, L in self.lin.items():
            qf = self.Wg[n].float()
            self.pub[n] = dict(rn=float(qf.square().sum(1).sqrt().max()), spec=float(torch.linalg.matrix_norm(qf, ord=2)) * 1.001,
                               scale=L.scale.astype(np.float64))
        self.W = {}                                      # integer multipliers W = round(K w . s_R) per pair
        for r, g, w in self.pairs:
            ws = w * self.lin[r].scale; K = (2 ** WB - 1) / np.abs(ws).max(); Wi = np.rint(K * ws).astype(np.int64)
            self.W[r] = dict(K=float(K), W=Wi, Wmax=float(np.abs(Wi).max()), Wg=torch.as_tensor(Wi, device=dev))
        self.a_pub = F.public_a(self.p)
        self.pads = F.Pads(self.p, [self.a[n] for n in self.order], [self.lin[n].q.shape[1] for n in self.order], dev)
        self.hint = None; self.time_online = self.time_pads = self.time_c1 = 0.0

    def public_c1(self, rows=4096):
        """One-time, public. Yields (kind, name, r0, rows of C1') in row chunks, with kind 'full' (residual rows,
        all 56 bits) or 'top' (everything else, compressed to 32 bits)."""
        t0 = time.perf_counter()
        if self.hint is None: self.hint = F.Hint(self.p, self.a_pub, self.dev)
        G = lambda n: self.hint.G(self.a[n], self.lin[n].q.shape[1])
        for n in self.singles:
            Gn = G(n); W = self.Wg[n]
            for r0 in range(0, W.shape[0], rows):
                c = F.compress(F.c1_single(W[r0:r0 + rows], Gn))
                self.time_c1 += time.perf_counter() - t0; yield "top", n, r0, c; t0 = time.perf_counter()
        for r, g, _ in self.pairs:
            Gr, Gg = G(r), G(g); Wg = self.Wg[g]
            for r0 in range(0, Wg.shape[0], rows):
                ct, cy = F.c1_chain(self.Wg[r], Wg[r0:r0 + rows], self.W[r]["Wg"], Gr, Gg)
                self.time_c1 += time.perf_counter() - t0
                if r0 == 0: yield "full", r, 0, ct
                yield "top", g + "@" + r, r0, F.compress(cy); t0 = time.perf_counter()

    def receive_b(self, b): self.b, self.pos = b, {}

    @staticmethod
    def _mm(W, C, U):                                # W (C - U) mod 2^56 for the tokens of a pass
        if K is not None and W.is_cuda: return K.gemm56(W, C, U)
        return F.gemm_mod(W, (C - U) & M56, 7)

    def _pads(self, toks):
        if not set(int(t) for t in toks) <= set(self.pos):
            t0 = time.perf_counter(); self.pos = {int(t): i for i, t in enumerate(toks)}
            parts = [self.pads.batch(self.b[:, toks[i:i + 4]].transpose(1, 0, 2)) for i in range(0, len(toks), 4)]
            self.U = {n: torch.cat([p_[j] for p_ in parts]) for j, n in enumerate(self.order)}
            torch.cuda.synchronize(); self.time_pads += time.perf_counter() - t0
        idx = [self.pos[int(t)] for t in toks]
        return lambda n: self.U[n][idx]

    def single(self, name, toks, C):
        t0 = time.perf_counter(); u = self._pads(toks)
        y = self._mm(self.Wg[name], torch.as_tensor(C, device=self.dev), u(name))
        out = (((y + (1 << (SH - 1))) >> SH) & ((1 << RB) - 1)).cpu().numpy()
        self.time_online += time.perf_counter() - t0; return out

    def pair(self, r, g, toks, Cv, Cx):
        t0 = time.perf_counter(); u = self._pads(toks)
        t = self._mm(self.Wg[r], torch.as_tensor(Cv, device=self.dev), u(r))
        y = self._mm(self.Wg[g], (self.W[r]["Wg"] * t + torch.as_tensor(Cx, device=self.dev)) & M56, u(g))
        out = t.cpu().numpy(), (((y + (1 << (SH - 1))) >> SH) & ((1 << RB) - 1)).cpu().numpy()
        self.time_online += time.perf_counter() - t0; return out


class ChainClient:
    def __init__(self, srv, seed=None, dev="cuda"):
        """seed=None (the default, and the only safe choice outside tests): the secrets (alphas, s, errors) come from
        fresh operating-system entropy. A fixed seed makes them reproducible by anyone, i.e. no privacy."""
        self.srv, self.dev, self.rng = srv, dev, np.random.default_rng(seed)
        self.p, self.pub, self.W = srv.p, srv.pub, srv.W            # public (a real client gets these in the bundle)
        self.a = {n: inner_a(lid, srv.lin[n].q.shape[1]) for lid, n in enumerate(srv.order)}   # derived locally
        self.masks = Masks(self.a, dev)
        self.rounds = 0; self.up = self.down = 0; self.t_client = 0.0

    def offline(self, n_tokens):
        """Per batch: fresh s and secrets, upload b; z from the one-time public C1' (streamed here)."""
        t0 = time.perf_counter()
        self.alphas = dgauss(self.rng, (n_tokens, N_LWE), SIGMA)
        self.key = F.Key(self.p, self.rng, self.dev)
        b = self.key.encrypt(self.srv.a_pub, self.alphas); self.srv.receive_b(b)
        self.z = {}
        rows = lambda name: self.srv.lin[name.split("@")[0]].q.shape[0]
        for kind, name, r0, C1 in self.srv.public_c1():
            C1 = C1.to(self.dev)                                          # arrives in host memory over a network
            z = self.key.unmask_full(C1, n_tokens) if kind == "full" else self.key.unmask_top(C1, n_tokens)
            if name not in self.z: self.z[name] = np.empty((n_tokens, rows(name)), dtype=np.int64)
            self.z[name][:, r0:r0 + len(C1)] = z.T.cpu().numpy()          # (tokens, rows)
        self.tok = -1; self.t_offline = time.perf_counter() - t0
        return dict(upload_MB=b.nbytes / 1e6)

    def begin_pass(self, T):
        self.toks = np.arange(self.tok + 1, self.tok + 1 + T); self.tok += T
        assert self.tok < len(self.alphas), "out of offline material"

    def _mask(self, name, X):
        return (self.masks(name, self.alphas[self.toks], X.shape[1]).cpu().numpy() + dgauss(self.rng, X.shape, SIGMA) + X) & M56

    def single(self, name, X):
        P = self.pub[name]; S = 2.0 ** 54 / (P["rn"] * np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-300))
        C = self._mask(name, np.rint(X * S).astype(np.int64))
        y = self.srv.single(name, self.toks, C)                           # ---- round trip ----
        r = centre55((y << SH) - self.z[name][self.toks])
        self.rounds += 1; self.up += 7 * C.size; self.down += RB / 8 * y.size
        return r / S * P["scale"]

    def pair(self, r, g, v, wx):
        R, G, Wd = self.pub[r], self.pub[g], self.W[r]
        bound = Wd["Wmax"] * R["spec"] * np.linalg.norm(v, axis=1, keepdims=True) + Wd["K"] * np.linalg.norm(wx, axis=1, keepdims=True)
        S = 2.0 ** 54 / (G["rn"] * np.maximum(bound, 1e-300))
        Cv = self._mask(r, np.rint(v * S).astype(np.int64)); Cx = self._mask(g, np.rint(wx * Wd["K"] * S).astype(np.int64))
        t, y = self.srv.pair(r, g, self.toks, Cv, Cx)                     # ---- round trip ----
        rt = centre(t - self.z[r][self.toks]); ry = centre55((y << SH) - self.z[g + "@" + r][self.toks])
        self.rounds += 1; self.up += 7 * (Cv.size + Cx.size); self.down += 7 * t.size + RB / 8 * y.size
        return rt / S * R["scale"], ry / (Wd["K"] * S) * G["scale"]

    def forward(self, toks, cache, n_head=None):
        """T tokens in one pass: 2 L + 2 round trips. Same arithmetic as protocol.forward_token on the unmerged model."""
        m = self.srv.model; T, t0 = len(toks), len(cache[0][0]); cos, sin = _angles(m, t0, T)
        x = np.stack([bf16_to_f32(m.embed[t]) for t in toks]).astype(np.float64)
        rms = lambda x: np.sqrt((x * x).mean(-1, keepdims=True) + m.eps)
        qkv = self.single("0.qkv", rmsnorm(x, m.blocks[0]["ln1"], m.eps))
        for i, b in enumerate(m.blocks):
            a = mix(m, b, i, qkv, cache, t0, cos, sin)
            rv, gu = self.pair(f"{i}.o", f"{i}.gate_up", a, b["ln2"] * x)
            x = x + rv; g, u = np.split(gu / rms(x), 2, axis=1); h = silu(g) * u
            if i + 1 < len(m.blocks):
                rv, qkv = self.pair(f"{i}.down", f"{i + 1}.qkv", h, m.blocks[i + 1]["ln1"] * x)
                x = x + rv; qkv = qkv / rms(x)
            else: x = x + self.single(f"{i}.down", h)
        return self.single("head", rmsnorm(x[-(n_head or T):], m.norm, m.eps))
