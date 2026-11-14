"""v4 offline machinery for chain mode (chain.py): fold.py with 56-bit inner masks.

Inner masks: Ring-LWE n = 2048 (negacyclic blocks), q = 2^56, sigma 6.4. Outer BFV: N = 8192 coefficients (= tokens
per batch), Q = 2^128 (eight 16-bit limbs), plaintext modulus 2^56, Delta = 2^72, ternary s, sigma 3.2; fixed public
a_j (from a seed), fresh s per batch, exactly as in fold.py:
    b_j = -a_j s + e_j + Delta * sum_t alpha_t[j] X^t                       per-batch upload (n x N x 16 bytes)
    u   = round(A b[t] / 2^72) mod 2^56                                       server pads, per token
    G   = round(A a / 2^64) mod 2^64                                          public, one-time
    C1' = round(L(G) / 2^8) mod 2^56   with L the server's map, applied mod 2^64 (single matrix, or the chain)
    z   = (C1' s)[t]                                                          client, per batch
Residual rows (the R v part of a chained trip) need z to within ~2^7 and keep all 56 bits of C1'; every other row
needs ~2^28 and keeps the top 32 bits (C1'' = round(C1' / 2^24)), which halves the one-time download.
A e / Delta <= 2^56 * 2^7.2 / 2^72 < 1, so the outer noise is negligible next to the inner sigma.
All ring products are float64 FFTs on 16-bit limbs (exact: 2^16 * 2^16 * n < 2^53)."""
import math, numpy as np, torch
from fold import NegFFT

QB, LIMBS, DSH = 56, 8, 72
M56, M64 = (1 << 56) - 1, (1 << 64) - 1
SIGMA = 3.2


class P:
    """Sizes; tests shrink them."""
    def __init__(self, n=2048, N=8192): self.n, self.N = n, N

def public_a(p, seed=3):
    """Fixed public BFV c1 parts a_1..a_n in R_Q, Q = 2^128: (n, N, 8) uint16."""
    return np.random.default_rng([seed, 128]).integers(0, 1 << 16, size=(p.n, p.N, LIMBS), dtype=np.uint16)

def _limbs16(a, k, dev):                                     # ints (..., m) -> (..., k, m) float64 16-bit limbs
    a = torch.as_tensor(a, device=dev)
    return torch.stack([(a >> (16 * i)) & 0xFFFF for i in range(k)], -2).double()

def _prod(FA, FB):
    """FA (..., 4, h): spectra of the 16-bit limbs of a 56-bit A. FB (..., 8, h): spectra of a 128-bit b.
    Spectra of P_s = sum_{i+j=s} A_i b_j, s < 8 (s >= 8 is a multiple of 2^128)."""
    out = FB.new_zeros(torch.broadcast_shapes(FA.shape[:-2], FB.shape[:-2]) + (LIMBS, FB.shape[-1]))
    for i in range(4):
        out[..., i:, :] = out[..., i:, :] + FA[..., i:i + 1, :] * FB[..., :LIMBS - i, :]
    return out

def _carry(P):
    """P (..., 8, m) int64 limb sums -> list of 8 carried limbs c_s (int64), x = sum (c_s & 0xFFFF) 2^(16 s) mod 2^128."""
    c, out = 0, []
    for s in range(LIMBS):
        c = P[..., s, :] + (c >> 16 if s else 0); out.append(c)
    return out

def top56(P):                                                # round(x / 2^72) mod 2^56
    c = _carry(P)
    v = ((c[4] >> 8) & 0xFF) | ((c[5] & 0xFFFF) << 8) | ((c[6] & 0xFFFF) << 24) | ((c[7] & 0xFFFF) << 40)
    return (v + ((c[4] >> 7) & 1)) & M56

def top64(P):                                                # round(x / 2^64) mod 2^64 (int64 bits)
    c = _carry(P)
    v = (c[4] & 0xFFFF) | ((c[5] & 0xFFFF) << 16) | ((c[6] & 0xFFFF) << 32) | ((c[7] & 0xFFFF) << 48)
    return v + ((c[3] >> 15) & 1)                            # wraps mod 2^64


# ---------------------------------------------------------------- byte-digit GEMMs (exact, mod 2^56 or 2^64)
_SH = {}
def gemm_mod(W, C, nbytes):
    """W int8 (d_out, d_in), C (T, d_in) int64 (values mod 2^(8 nbytes)) -> W C^T mod 2^(8 nbytes) as (T, d_out) int64.
    nbytes signed digits plus nbytes-1 borrow bits per value, one int8 GEMM (weights read once)."""
    dev, key = W.device, (W.device, nbytes)
    T, din = C.shape; rows = W.shape[0]
    if din % 8: W = torch.nn.functional.pad(W, (0, -din % 8)); C = torch.nn.functional.pad(C, (0, -din % 8)); din = C.shape[1]
    if rows % 32: W = torch.nn.functional.pad(W, (0, 0, 0, -rows % 32))   # cuBLASLt int8 wants aligned m
    ncol = -(-(2 * nbytes - 1) // 8) * 8
    if key not in _SH:
        sh = [1 << (8 * k) for k in range(nbytes)] + [1 << (8 * k + 8) for k in range(nbytes - 1)]
        _SH[key] = torch.tensor([v if v < (1 << 63) else v - (1 << 64) for v in sh] + [0] * (ncol - len(sh)), device=dev)
    u = [(C >> (8 * k)) & 255 for k in range(nbytes)]
    s = [x - 256 * (x >= 128) for x in u]; b = [(x >= 128).long() for x in u[:nbytes - 1]]
    X = torch.stack(s + b + [torch.zeros_like(u[0])] * (ncol - 2 * nbytes + 1), -1).to(torch.int8)
    Y = torch._int_mm(W, X.permute(1, 0, 2).reshape(din, ncol * T))
    out = (Y[:rows].view(rows, T, ncol).long() * _SH[key]).sum(2)
    return (out & ((1 << (8 * nbytes)) - 1) if nbytes < 8 else out).T


# ---------------------------------------------------------------- server, one-time per model
class Hint:
    """G = round(A a / 2^64) mod 2^64 per input layer; C1' = round(L(G) / 2^8) mod 2^56 per output."""
    def __init__(self, p, a16, dev, chunk=1024):
        self.p, self.f, self.dev, self.chunk = p, NegFFT(p.n, dev), dev, chunk
        self.a16 = a16                                       # (n, N, 8) uint16, host

    def G(self, a_blocks, din):
        """a_blocks (nb, n) int64 in [0, 2^56) -> (din, N) int64 (bits of round(A a / 2^64) mod 2^64)."""
        p = self.p; FA = self.f.fwd(_limbs16(a_blocks, 4, self.dev))                       # (nb, 4, n/2)
        out = torch.empty((len(a_blocks), p.n, p.N), dtype=torch.int64, device=self.dev)
        for t0 in range(0, p.N, self.chunk):
            B = torch.as_tensor(self.a16[:, t0:t0 + self.chunk].transpose(1, 2, 0).astype(np.float64), device=self.dev)   # (c, 8, n)
            FB = self.f.fwd(B)                                                               # (c, 8, n/2)
            for k in range(len(a_blocks)):
                P = torch.round(self.f.inv(_prod(FA[k], FB))).long()                         # (c, 8, n)
                out[k, :, t0:t0 + self.chunk] = top64(P).T
        return out.reshape(-1, p.N)[:din]

def c1_single(M, G, chunk=512):                              # C1' of a single matrix: (rows, N), in [0, 2^56)
    y = torch.cat([gemm_mod(M, G[:, i:i + chunk].T.contiguous(), 8) for i in range(0, G.shape[1], chunk)]).T   # mod 2^64
    return ((y + 128) >> 8) & M56

def c1_chain(R, Gm, Wt, Gv, Gx, chunk=512):
    """C1' of a chained trip: (residual rows of R, chain rows of G), mod 2^56. Wt (d,) int64 multipliers."""
    ts, ys = [], []
    for i in range(0, Gv.shape[1], chunk):
        t = gemm_mod(R, Gv[:, i:i + chunk].T.contiguous(), 8).to(Gm.device)    # (c, d) mod 2^64 (R, G may be on two GPUs)
        ys.append(gemm_mod(Gm, (t * Wt + Gx[:, i:i + chunk].T).contiguous(), 8)); ts.append(t)
    t, y = torch.cat(ts), torch.cat(ys)
    return (((t + 128) >> 8) & M56).T, (((y + 128) >> 8) & M56).T

def compress(C1):                                            # chain/single rows: keep the top 32 of 56 bits (centred int64)
    c = ((C1 + (1 << 23)) >> 24) & 0xFFFFFFFF
    return c - ((c >> 31) << 32)


# ---------------------------------------------------------------- server, per token
class Pads:
    """u_l = round(A_l b[t] / 2^72) mod 2^56 for every input layer of a token (or several tokens)."""
    def __init__(self, p, a_blocks, dins, dev):
        self.p, self.f, self.dev = p, NegFFT(p.n, dev), dev
        self.FA = self.f.fwd(_limbs16(np.concatenate(a_blocks), 4, dev))                    # (blocks, 4, n/2)
        self.cuts, o = [], 0
        for ab, d in zip(a_blocks, dins): self.cuts.append((o, o + len(ab), d)); o += len(ab)
    def batch(self, bts):
        """bts (T, n, 8) uint16 -> one (T, d_in) int64 tensor per layer."""
        T = len(bts); FB = self.f.fwd(torch.as_tensor(bts.transpose(0, 2, 1).astype(np.float64), device=self.dev))   # (T, 8, n/2)
        u = top56(torch.round(self.f.inv(_prod(self.FA[None], FB[:, None]))).long())           # (T, blocks, n)
        return [u[:, i:j].reshape(T, -1)[:, :d] for i, j, d in self.cuts]


# ---------------------------------------------------------------- client, per batch
class Key:
    def __init__(self, p, rng, dev="cpu"):
        self.p, self.f, self.dev, self.rng = p, NegFFT(p.N, dev), dev, rng
        self.s = rng.integers(-1, 2, size=p.N)
        self.FS = self.f.fwd(torch.as_tensor(self.s, dtype=torch.float64, device=dev))

    def encrypt(self, a16, alphas, chunk=64):
        """alphas (B, n) small ints -> b (n, N, 8) uint16: b_j = -a_j s + e_j + 2^72 m_j mod 2^128."""
        p = self.p; B = alphas.shape[0]; assert B <= p.N and alphas.shape[1] == p.n
        out = np.empty((p.n, p.N, LIMBS), dtype=np.uint16)
        for j in range(0, p.n, chunk):
            a = torch.as_tensor(a16[j:j + chunk].transpose(0, 2, 1).astype(np.float64), device=self.dev)   # (k, 8, N)
            conv = torch.round(self.f.inv(self.f.fwd(a) * self.FS)).long()                                # exact: 2^16 * N < 2^53
            m = torch.zeros((a.shape[0], p.N), dtype=torch.int64, device=self.dev)
            m[:, :B] = torch.as_tensor(alphas[:, j:j + chunk].T.copy(), device=self.dev)
            e = torch.as_tensor(np.rint(self.rng.normal(0, SIGMA, size=m.shape)).astype(np.int64), device=self.dev)
            limbs, carry = [], 0
            for i in range(LIMBS):                           # e + 2^72 m - a s, limb by limb (2^72 = 2^8 at limb 4)
                t = carry - conv[:, i] + (e if i == 0 else (m << 8) if i == 4 else 0)
                limbs.append(t & 0xFFFF); carry = t >> 16
            out[j:j + chunk] = torch.stack(limbs, -1).cpu().numpy().astype(np.uint16)
        return out

    def unmask_full(self, C1, B):
        """Residual rows: z = (C1' s)[:B] mod 2^56 from full 56-bit C1' (rows, N), via 16-bit limbs (exact)."""
        z = 0
        for k in range(4):
            limb = ((C1 >> (16 * k)) & 0xFFFF).double()
            z = z + (torch.round(self.f.inv(self.f.fwd(limb) * self.FS)[:, :B]).long() << (16 * k))
        return z & M56

    def unmask_top(self, C1c, B):
        """Other rows: z = 2^24 (C1'' s)[:B] mod 2^56 from the compressed C1'' (centred 32-bit)."""
        return (torch.round(self.f.inv(self.f.fwd(C1c.double()) * self.FS)[:, :B]).long() << 24) & M56
