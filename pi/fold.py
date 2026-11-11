"""
v3 crypto: the BFV c0 part is folded into the online reply, so nothing is downloaded per batch.

Outer scheme: R = Z[X]/(X^N + 1), N = 4096 (coefficient t <-> token t of a batch), modulus Q = 2^80,
Delta = 2^48 (plaintext modulus 2^32), ternary secret s, error sigma 3.2. The c1 parts a_1..a_n are FIXED and
PUBLIC (from a seed); only s is fresh per batch:
    b_j = -a_j s + e_j + Delta * sum_t alpha_t[j] X^t           client -> server, once per batch (the only offline traffic)
For a layer with int8 weights M and public inner matrix A (c = A alpha_t + e + x~ mod 2^32, as in v2):
    C0[t] = M (A b[t])      never sent: the server folds it into the online GEMV
    C1    = M (A a)         public, one-time per model:  C1' = round(C1 / 2^48) mod 2^32
  server, per token :  u = round(A b[t] / 2^48) mod 2^32                        (one batched FFT for all layers)
  server, online    :  y = M (c - u) mod 2^32, reply = 16 bits of y             (same 32-bit GEMV as v2)
  client            :  M x~ ~= y - (C1' s)[t]
y is the mod-switched M (Delta c - A b[t]): switching Q -> 2^32 BEFORE the GEMV instead of after only adds M eps with
|eps| <= 1/2, far below the M e (sigma 6.4) that is there anyway, and it keeps the GEMV at 32 bits.
Q is a power of two, so there are no primes and no NTTs: every ring product is a float64 FFT on 16-bit limbs.
"""
import math, numpy as np, torch

N, n = 4096, 1024              # outer ring degree (tokens per batch), inner LWE dimension
LIMBS = 5                      # Q = 2^80 as 5 little-endian 16-bit limbs
SIGMA = 3.2
MSG_BITS = 31                  # client scales so |M x~| < 2^30: bit 31 of y is redundant and never sent
M32, M40 = 0xFFFFFFFF, (1 << 40) - 1


class NegFFT:
    """Products of real polynomials mod X^m + 1 through complex FFTs of size m/2 (X^m + 1 = (X^h - i)(X^h + i), a real
    polynomial is determined by its image mod X^h - i, and X -> zeta X with zeta^h = i makes that ring cyclic)."""
    def __init__(self, m, dev):
        j = torch.arange(m // 2, device=dev, dtype=torch.float64)
        self.h, self.tw = m // 2, torch.polar(torch.ones_like(j), j * (math.pi / m))
    def fwd(self, x):                            # (..., m) float64 -> (..., m/2) complex128
        return torch.fft.fft(torch.complex(x[..., :self.h], x[..., self.h:]) * self.tw)
    def inv(self, X):                            # (..., m/2) complex128 -> (..., m) float64
        y = torch.fft.ifft(X) * self.tw.conj()
        return torch.cat([y.real, y.imag], -1)


def public_a(seed=2):
    """The fixed public c1 parts a_1..a_n in R_Q: (n, N, LIMBS) uint16."""
    return np.random.default_rng([seed, 80]).integers(0, 1 << 16, size=(n, N, LIMBS), dtype=np.uint16)

def _limb_products(FA, FB):
    """FA (..., 2, h): spectra of the 16-bit halves of a 32-bit a. FB (..., 5, h): spectra of the limbs of an 80-bit b.
    Returns the spectra of P_s = sum_{i+j=s} a_i b_j for s < 5 (s = 5 is a multiple of 2^80)."""
    F = FA[..., :1, :] * FB
    F[..., 1:, :] += FA[..., 1:, :] * FB[..., :4, :]
    return F

def _top(P, k):
    """P (..., 5, m) int64 with x = sum_s P_s 2^(16 s) mod 2^80  ->  round(x / 2^(80-k)) mod 2^k, k = 32 or 40."""
    c2 = P[..., 2, :] + ((P[..., 1, :] + (P[..., 0, :] >> 16)) >> 16)
    c3 = P[..., 3, :] + (c2 >> 16)
    if k == 32: return ((P[..., 4, :] << 16) + c3 + ((c2 >> 15) & 1)) & M32
    return ((P[..., 4, :] << 24) + (c3 << 8) + ((c2 >> 8) & 0xFF) + ((c2 >> 7) & 1)) & M40

def _halves(a, dev):                             # 32-bit ints (..., n) -> (..., 2, n) float64
    a = torch.as_tensor(a, device=dev)
    return torch.stack([a & 0xFFFF, a >> 16], -2).double()


# ---------------------------------------------------------------- server, one-time per model
class Hint:
    """C1' = round(M (A a) / 2^48) mod 2^32, shape (d_out, N). Public: depends on the weights and public randomness only.
    A a is kept to its top 40 bits (the dropped bits change C1' by ~2^-7), so M (A a) is 5 int8 GEMMs."""
    def __init__(self, a16, dev):
        self.f, self.dev = NegFFT(n, dev), dev
        self.FB = self.f.fwd(torch.as_tensor(a16.transpose(1, 2, 0).astype(np.float64), device=dev))   # (N, 5, n/2)

    def G(self, a_blocks, din):
        """round(A a / 2^40) mod 2^40 as (din, N) int64; A = blockwise negacyclic matrices of a_blocks (nb, n)."""
        out = []
        for ab in a_blocks:
            P = torch.round(self.f.inv(_limb_products(self.f.fwd(_halves(ab, self.dev)), self.FB))).long()   # (N, 5, n)
            out.append(_top(P, 40).T)
        return torch.cat(out)[:din]

    def c1(self, q, G, chunk=4096):
        """Yields (r0, C1'[r0:r0+chunk]) for int8 weights q (d_out, d_in); C1' as int64 tensors in [0, 2^32)."""
        if self.dev == "cpu":                    # 3 exact float64 GEMMs on 16/16/8-bit limbs of G
            Gl = [((G >> s) & m).double() for s, m in ((0, 0xFFFF), (16, 0xFFFF), (32, 0xFF))]
            for r0 in range(0, q.shape[0], chunk):
                w = torch.as_tensor(np.ascontiguousarray(q[r0:r0 + chunk])).double()
                acc = sum(torch.round(w @ g).long() << (16 * k) for k, g in enumerate(Gl))
                yield r0, ((acc + 128) >> 8) & M32
            return
        d, x = [], G                             # balanced int8 digits of G; the carry out of digit 4 is a multiple of 2^40
        for _ in range(5):
            t = ((x + 128) & 255) - 128; d.append(t.to(torch.int8)); x = (x - t) >> 8
        Gd = torch.cat(d, 1); del d, x           # (din, 5N) int8
        W = torch.as_tensor(q, device=self.dev)
        for r0 in range(0, W.shape[0], chunk):
            w = W[r0:r0 + chunk]; rows = w.shape[0]
            if rows % 32: w = torch.nn.functional.pad(w, (0, 0, 0, -rows % 32))     # cuBLASLt int8 wants aligned m
            Y = torch._int_mm(w, Gd)[:rows]      # int32, |.| <= 2^14 d_in
            acc = Y[:, :N].long()
            for k in range(1, 5): acc += Y[:, k * N:(k + 1) * N].long() << (8 * k)   # wraps mod 2^64, exact mod 2^40
            yield r0, ((acc + 128) >> 8) & M32


# ---------------------------------------------------------------- server, per token
class Pads:
    """u_l = round(A_l b[t] / 2^48) mod 2^32 for every layer l of one token, in one batched FFT."""
    def __init__(self, a_blocks, dins, dev):
        self.f, self.dev = NegFFT(n, dev), dev
        self.FA = self.f.fwd(_halves(np.concatenate(a_blocks), dev))                 # (blocks, 2, n/2)
        self.cuts, o = [], 0
        for ab, d in zip(a_blocks, dins): self.cuts.append((o, o + len(ab), d)); o += len(ab)

    def __call__(self, bt):
        """bt: (n, LIMBS) uint16 = coefficient t of every b_j. Returns one int32 tensor (uint32 bits) per layer."""
        FB = self.f.fwd(torch.as_tensor(bt.T.astype(np.float64), device=self.dev))   # (5, n/2)
        P = torch.round(self.f.inv(_limb_products(self.FA, FB))).long()              # (blocks, 5, n)
        u = _top(P, 32).to(torch.int32)
        return [u[i:j].reshape(-1)[:d] for i, j, d in self.cuts]

    def batch(self, bts):
        """Same for T tokens at once: bts (T, n, LIMBS) uint16 -> one (T, d_in) int32 tensor per layer."""
        T = len(bts)
        FB = self.f.fwd(torch.as_tensor(bts.transpose(0, 2, 1).astype(np.float64), device=self.dev))       # (T, 5, n/2)
        P = torch.round(self.f.inv(_limb_products(self.FA[None], FB[:, None]))).long()                      # (T, blocks, 5, n)
        u = _top(P, 32).to(torch.int32)
        return [u[:, i:j].reshape(T, -1)[:, :d] for i, j, d in self.cuts]

def reply_bits(y, out_bits):
    """Server reply: bits [MSG_BITS - out_bits, MSG_BITS) of y (numpy or torch int64 in [0, 2^32)), rounded."""
    sh = MSG_BITS - out_bits
    return ((y + (1 << (sh - 1))) >> sh) & ((1 << out_bits) - 1)


# ---------------------------------------------------------------- client, online
class Masks:
    """The inner masks A_l alpha_t mod 2^32 for several tokens at once, with the FFT of the public A_l cached."""
    def __init__(self, a, dev="cpu"):            # a: {layer name: (nb, n) int64 in [0, 2^32)}
        self.f, self.dev = NegFFT(n, dev), dev
        self.FA = {k: self.f.fwd(_halves(v, dev)) for k, v in a.items()}             # (nb, 2, n/2) each
    def __call__(self, name, alphas, din):
        """alphas (T, n) small ints -> (T, din) int64 numpy."""
        FS = self.f.fwd(torch.as_tensor(alphas, dtype=torch.float64, device=self.dev))
        P = torch.round(self.f.inv(self.FA[name] * FS[:, None, None, :])).long()     # (T, nb, 2, n), exact
        return ((P[:, :, 0] + (P[:, :, 1] << 16)) & M32).reshape(len(alphas), -1)[:, :din].cpu().numpy()


# ---------------------------------------------------------------- client, per batch
class Key:
    """Fresh ternary s for one batch of up to N tokens."""
    def __init__(self, rng, dev="cpu"):
        self.f, self.dev, self.rng = NegFFT(N, dev), dev, rng
        self.s = rng.integers(-1, 2, size=N)
        self.FS = self.f.fwd(torch.as_tensor(self.s, dtype=torch.float64, device=dev))

    def encrypt(self, a16, alphas, chunk=128):
        """alphas (B, n) small ints, one inner secret per token -> b (n, N, LIMBS) uint16, the per-batch upload."""
        B = alphas.shape[0]; assert B <= N and alphas.shape[1] == n
        out = np.empty((n, N, LIMBS), dtype=np.uint16)
        for j in range(0, n, chunk):
            a = torch.as_tensor(a16[j:j + chunk].transpose(0, 2, 1).astype(np.float64), device=self.dev)   # (k, 5, N)
            conv = torch.round(self.f.inv(self.f.fwd(a) * self.FS)).long()                                 # a_j limbs * s, exact
            m = torch.zeros((a.shape[0], N), dtype=torch.int64, device=self.dev)
            m[:, :B] = torch.as_tensor(alphas[:, j:j + chunk].T.copy(), device=self.dev)
            e = torch.as_tensor(np.rint(self.rng.normal(0, SIGMA, size=m.shape)).astype(np.int64), device=self.dev)
            limbs, carry = [], 0
            for i in range(LIMBS):               # e + 2^48 m - a s, limb by limb with floor carries
                t = carry - conv[:, i] + (e if i == 0 else m if i == 3 else 0)
                limbs.append(t & 0xFFFF); carry = t >> 16
            out[j:j + chunk] = torch.stack(limbs, -1).cpu().numpy().astype(np.uint16)
        return out

    def unmask(self, C1p, B):
        """C1' rows (rows, N) in [0, 2^32) -> z = (C1' s)[:B] mod 2^32 as (rows, B) int64 tensor. One float64 product
        per row: |C1' s| < 2^44, so the rounding is exact to well under 1."""
        x = (C1p - ((C1p >> 31) << 32)).double()                                     # centred
        return torch.round(self.f.inv(self.f.fwd(x) * self.FS)[:, :B]).long() & M32
