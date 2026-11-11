"""
Crypto core for single-server private inference with public weights.

Online layer (inner scheme): Ring-LWE masks mod q = 2^32, ring dim n = 1024,
Gaussian secret/error sigma = 6.4 (SimplePIR-style parameters, normal-form secret).
  client:  c = a_l * alpha_t + e + x~   (blockwise negacyclic products, mod 2^32)
  server:  y = M c mod 2^32             (plain int8 GEMV -- same work as plaintext)
  client:  M x~ ~= y - z,  z = H_l alpha_t,  H_l = M A_l (public, server-precomputed)

Offline layer (outer scheme): symmetric BFV with SIMD batching, N = 2048,
Q = p1*p2 (~2^54), t ~ 2^23. Client encrypts alpha for B = 2048 future tokens
(one ciphertext per coordinate j holds alpha_{1..B, j} in slots). Server computes
Enc(H_k alpha) = sum_j H_k[i,j] * ct_j for byte-digits H_k of H. That is a plain
matrix product over ciphertext coefficients (float64 GEMM, exact).
"""
import numpy as np

Q32 = 1 << 32
MASK32 = np.uint64(Q32 - 1)

# ---------------------------------------------------------------- utilities
def is_prime(x):
    if x < 2: return False
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if x % p == 0: return x == p
    d, r = x - 1, 0
    while d % 2 == 0: d //= 2; r += 1
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        y = pow(a, d, x)
        if y in (1, x - 1): continue
        for _ in range(r - 1):
            y = y * y % x
            if y == x - 1: break
        else: return False
    return True

def find_ntt_prime(bits, N, below=None, skip=()):
    m = 2 * N
    x = ((below or (1 << bits)) // m) * m + 1
    while True:
        x -= m
        if is_prime(x) and x not in skip: return x

def primitive_2n_root(p, N):
    for g in range(2, p):
        if pow(g, (p - 1) // 2, p) == 1: continue
        # g generates a group whose order divisible by 2N with high prob; check order of psi
        psi = pow(g, (p - 1) // (2 * N), p)
        if pow(psi, N, p) == p - 1: return psi
    raise ValueError

def dgauss(rng, shape, sigma):
    return np.rint(rng.normal(0, sigma, size=shape)).astype(np.int64)

# ---------------------------------------------------------------- negacyclic NTT
class NTT:
    """Negacyclic NTT mod prime p over length N (p = 1 mod 2N). Vectorised over leading axes."""
    def __init__(self, p, N):
        self.p, self.N = p, N
        psi = primitive_2n_root(p, N)
        omega = psi * psi % p
        self.psi_pow = np.array([pow(psi, i, p) for i in range(N)], dtype=np.uint64)
        psi_inv = pow(psi, p - 2, p)
        self.psi_inv_pow = np.array([pow(psi_inv, i, p) for i in range(N)], dtype=np.uint64)
        self.n_inv = pow(N, p - 2, p)
        lg = N.bit_length() - 1
        self.rev = np.array([int(format(i, f'0{lg}b')[::-1], 2) for i in range(N)])
        self.tw, self.tw_inv = [], []
        m = 2
        while m <= N:
            w = pow(omega, N // m, p); wi = pow(w, p - 2, p)
            self.tw.append(np.array([pow(w, k, p) for k in range(m // 2)], dtype=np.uint64))
            self.tw_inv.append(np.array([pow(wi, k, p) for k in range(m // 2)], dtype=np.uint64))
            m *= 2

    def _cyclic(self, a, tws):
        p = np.uint64(self.p)
        a = a[..., self.rev]
        shp = a.shape[:-1]
        for s, w in enumerate(tws):
            m = 2 << s
            a = a.reshape(shp + (self.N // m, m))
            u = a[..., : m // 2]
            v = a[..., m // 2:] * w % p
            a = np.concatenate([(u + v) % p, (u + p - v) % p], axis=-1)
        return a.reshape(shp + (self.N,))

    def fwd(self, a):  # coefficients -> evaluations
        a = np.asarray(a, dtype=np.uint64) % np.uint64(self.p)
        return self._cyclic(a * self.psi_pow % np.uint64(self.p), self.tw)

    def inv(self, A):
        p = np.uint64(self.p)
        a = self._cyclic(np.asarray(A, dtype=np.uint64), self.tw_inv)
        return a * np.uint64(self.n_inv) % p * self.psi_inv_pow % p


# ---------------------------------------------------------------- inner ring-LWE masks mod 2^32
def negacyclic_mul_mod32(a, s):
    """a: (..., n) uint32-valued (int64), s: (n,) small signed ints. Returns a*s mod (X^n+1, 2^32).
    Exact via two float64 FFT convolutions on 16-bit halves of a."""
    n = a.shape[-1]
    a = np.asarray(a, dtype=np.int64)
    lo, hi = a & 0xFFFF, a >> 16
    S = np.fft.rfft(s.astype(np.float64), 2 * n)
    def conv(x):
        full = np.rint(np.fft.irfft(np.fft.rfft(x.astype(np.float64), 2 * n, axis=-1) * S, 2 * n, axis=-1)).astype(np.int64)
        return full[..., :n] - full[..., n:]          # X^n = -1
    r = conv(lo) + (conv(hi) << 16)
    return r & 0xFFFFFFFF

def negacyclic_matrix(a):
    """Explicit n x n matrix A with A @ s == a * s (negacyclic). Used only for server-side hint H = M A."""
    n = len(a)
    A = np.empty((n, n), dtype=np.int64)
    for j in range(n):
        col = np.roll(a, j)
        col[:j] = -col[:j]
        A[:, j] = col
    return A & 0xFFFFFFFF

def matmul_mod32_i8(M, A):
    """M int8 (d_out, d_in) @ A uint32 (d_in, k) mod 2^32, exact (float64 on 16-bit halves)."""
    Mf = M.astype(np.float64)
    lo = (Mf @ (A & 0xFFFF).astype(np.float64))
    hi = (Mf @ (A >> 16).astype(np.float64))
    return (np.rint(lo).astype(np.int64) + (np.rint(hi).astype(np.int64) << 16)) & 0xFFFFFFFF


class InnerLWE:
    n, sigma = 1024, 6.4

    def __init__(self, seed=0):
        self.seed = seed

    def public_a(self, layer_id, din):
        """Public uniform ring elements for each n-block of a d_in-length input (derived from a seed)."""
        nb = -(-din // self.n)
        rng = np.random.default_rng([self.seed, layer_id])
        return rng.integers(0, Q32, size=(nb, self.n), dtype=np.int64)

    def hint(self, M, a_blocks):
        """Server, one-time per model: H = M A (d_out x n) mod 2^32."""
        dout, din = M.shape
        H = np.zeros((dout, self.n), dtype=np.int64)
        for j, a in enumerate(a_blocks):
            cols = slice(j * self.n, min((j + 1) * self.n, din))
            A = negacyclic_matrix(a)[: cols.stop - cols.start]
            H = (H + matmul_mod32_i8(M[:, cols], A)) & 0xFFFFFFFF
        return H

    def mask(self, a_blocks, alpha, din, rng):
        """Client: a*alpha + e for each block, truncated to d_in. Cheap: O(d_in log n)."""
        r = negacyclic_mul_mod32(a_blocks, alpha).reshape(-1)[:din]
        return (r + dgauss(rng, din, self.sigma)) & 0xFFFFFFFF


# ---------------------------------------------------------------- outer BFV (symmetric, batched)
class BFV:
    N = 2048

    def __init__(self, seed=1):
        N = self.N
        self.p = [find_ntt_prime(27, N)]
        self.p.append(find_ntt_prime(27, N, below=self.p[0]))
        self.Q = self.p[0] * self.p[1]
        self.t = find_ntt_prime(23, N)
        self.ntt = [NTT(p, N) for p in self.p]
        self.ntt_t = NTT(self.t, N)
        self.delta = self.Q // self.t
        self.rng = np.random.default_rng(seed)
        s = self.rng.integers(-1, 2, size=N)
        self.s_hat = [nt.fwd(s % p) for nt, p in zip(self.ntt, self.p)]
        # CRT helper
        self.p0_inv_mod_p1 = pow(self.p[0], -1, self.p[1])

    def encrypt_slots(self, slots):
        """slots: (k, N) ints mod t -> ciphertexts in NTT form: (k, limbs=2, comps=2, N)."""
        m = self.ntt_t.inv(np.asarray(slots, dtype=np.int64) % self.t)   # plaintext poly coeffs in [0,t)
        k = m.shape[0]
        out = np.empty((k, 2, 2, self.N), dtype=np.uint64)
        e = dgauss(self.rng, (k, self.N), 3.2)
        for L, (p, nt) in enumerate(zip(self.p, self.ntt)):
            pu = np.uint64(p)
            a_hat = self.rng.integers(0, p, size=(k, self.N), dtype=np.int64).astype(np.uint64)
            # BGV-style plaintext placement (m + t*e): linear combinations stay exact mod t,
            # no scale-rounding error term (BFV's Q mod t term would blow up here).
            body = (m + self.t * e) % p
            body_hat = nt.fwd(body.astype(np.uint64))
            out[:, L, 0] = (body_hat + pu - a_hat * self.s_hat[L] % pu) % pu   # c0 = -a s + m + t e
            out[:, L, 1] = a_hat
        return out

    def lincomb(self, Hd, C):
        """Server: rows of small nonneg matrix Hd (D, k) <= 255 times ciphertexts C (k,2,2,N).
        Exact float64 GEMM: 255 * 2^27 * k(=1024) < 2^53."""
        D, k = Hd.shape
        out = np.empty((D, 2, 2, self.N), dtype=np.uint64)
        Hf = Hd.astype(np.float64)
        for L, p in enumerate(self.p):
            flat = C[:, L].reshape(k, -1).astype(np.float64)
            r = np.rint(Hf @ flat).astype(np.int64) % p
            out[:, L] = r.reshape(D, 2, self.N).astype(np.uint64)
        return out

    def decrypt_slots(self, C):
        """C: (D,2,2,N) -> slots (D, N) centered ints in (-t/2, t/2]."""
        xs = []
        for L, (p, nt) in enumerate(zip(self.p, self.ntt)):
            pu = np.uint64(p)
            xs.append(nt.inv((C[:, L, 0] + C[:, L, 1] * self.s_hat[L] % pu) % pu).astype(np.int64))
        x0, x1 = xs
        k = ((x1 - x0) % self.p[1]) * self.p0_inv_mod_p1 % self.p[1]
        x = x0 + self.p[0] * k                                    # in [0, Q) < 2^54, exact int64
        x = np.where(x > self.Q // 2, x - self.Q, x)              # centre: x = sum h m + t * noise
        m = x % self.t
        slots = self.ntt_t.fwd(m).astype(np.int64)
        return np.where(slots > self.t // 2, slots - self.t, slots)


class BFV32:
    """v2 offline scheme: BFV, plaintext modulus 2^32, COEFFICIENT encoding (no SIMD needed since
    the server only does scalar*ciphertext + add). N = 4096, Q = 3 primes ~2^30 (~2^90).
    The client only needs z = H alpha mod 2^32 to ~2^14 precision (the online result already
    carries noise M e ~ 2^13), so after the linear combination the server drops to ONE ~30-bit
    limb. Download per row per token = 2 x 30 bits instead of 4 digits x 2 limbs x 2 x 27 bits."""
    N = 4096

    def __init__(self, seed=1):
        N = self.N
        self.p = []
        for _ in range(3):
            self.p.append(find_ntt_prime(30, N, below=self.p[-1] if self.p else None))
        self.Q = self.p[0] * self.p[1] * self.p[2]
        self.ntt = [NTT(p, N) for p in self.p]
        self.delta = self.Q // Q32
        self.rng = np.random.default_rng(seed)
        s = self.rng.integers(-1, 2, size=N)
        self.s = s
        self.s_hat = [nt.fwd(s % p) for nt, p in zip(self.ntt, self.p)]
        self.p12 = self.p[1] * self.p[2]
        self.p1_inv_mod_p2 = pow(self.p[1], -1, self.p[2])
        self.p12_inv_mod_p0 = pow(self.p12, -1, self.p[0])

    def encrypt_coeffs(self, m):
        """m: (k, N) small signed ints -> ciphertexts in coefficient form (k, limbs=3, comps=2, N)."""
        k = m.shape[0]
        out = np.empty((k, 3, 2, self.N), dtype=np.int64)
        e = dgauss(self.rng, (k, self.N), 3.2)
        for L, (p, nt) in enumerate(zip(self.p, self.ntt)):
            pu = np.uint64(p)
            a = self.rng.integers(0, p, size=(k, self.N), dtype=np.int64)
            a_s = nt.inv(nt.fwd(a.astype(np.uint64)) * self.s_hat[L] % pu).astype(np.int64)
            out[:, L, 0] = (e + (self.delta % p) * (m % p) - a_s) % p
            out[:, L, 1] = a
        return out

    def lincomb_and_shrink(self, H, C):
        """Server: Enc(sum_j H[i,j] m_j) for H uint32 (D,k), exact per limb via byte-split float64 GEMM;
        then modulus-switch Q -> p0 (drop limbs p1,p2) and NTT the c1 part for cheap client decryption.
        Returns c0' (D, N) coefficient form mod p0, c1_hat' (D, N) NTT form mod p0."""
        D, k = H.shape
        acc = np.zeros((D, 3, 2, self.N), dtype=np.int64)
        Hb = [((H >> (8 * d)) & 0xFF).astype(np.float64) for d in range(4)]
        for L, p in enumerate(self.p):
            flat = C[:, L].reshape(k, -1).astype(np.float64)       # (k, 2N), entries < 2^30
            tot = np.zeros((D, 2 * self.N), dtype=np.int64)
            for d in range(4):                                     # 255 * 2^30 * 1024 < 2^53: exact
                tot = (tot + (np.rint(Hb[d] @ flat).astype(np.int64) % p) * ((1 << (8 * d)) % p)) % p
            acc[:, L] = tot.reshape(D, 2, self.N)
        # modulus switch: x' = (x - [x]_{p1 p2}) / (p1 p2) mod p0
        x0, x1, x2 = acc[:, 0], acc[:, 1], acc[:, 2]
        kk = ((x2 - x1) % self.p[2]) * self.p1_inv_mod_p2 % self.p[2]
        r12 = x1 + self.p[1] * kk                                  # x mod p1p2 in [0, p1p2) < 2^60
        r12 = np.where(r12 > self.p12 // 2, r12 - self.p12, r12)  # centred
        xs = ((x0 - r12) % self.p[0]) * self.p12_inv_mod_p0 % self.p[0]
        c0 = xs[:, 0]
        c1_hat = self.ntt[0].fwd(xs[:, 1].astype(np.uint64)).astype(np.int64)
        return c0, c1_hat

    def decrypt_top(self, c0, c1_hat):
        """Client: m ~ round(x' * 2^32 / p0) mod 2^32 (approximate, top bits)."""
        p = self.p[0]
        x = (c0 + self.ntt[0].inv((c1_hat.astype(np.uint64) * self.s_hat[0] % np.uint64(p))).astype(np.int64)) % p
        return np.rint(x.astype(np.float64) * (Q32 / p)).astype(np.int64) & 0xFFFFFFFF


def offline_hint_products_v2(bfv, H, alphas, chunk=1024):
    import time
    B, n = alphas.shape
    assert B <= bfv.N
    m = np.zeros((n, bfv.N), dtype=np.int64); m[:, :B] = alphas.T   # coefficient j-th poly: alpha_{., j}
    t0 = time.perf_counter(); C = bfv.encrypt_coeffs(m); t1 = time.perf_counter()
    D = H.shape[0]; z = np.zeros((D, B), dtype=np.int64); srv = cli = 0.0
    for r0 in range(0, D, chunk):
        a = time.perf_counter(); c0, c1h = bfv.lincomb_and_shrink(H[r0:r0 + chunk], C)
        b = time.perf_counter(); z[r0:r0 + chunk] = bfv.decrypt_top(c0, c1h)[:, :B]
        c = time.perf_counter(); srv += b - a; cli += c - b
    up = C.size * 30 / 8
    down = D * 2 * bfv.N * 30 / 8
    return z.T, srv, cli + (t1 - t0), up, down


def offline_hint_products(bfv, H, alphas):
    """Compute z[b] = H @ alphas[b] mod 2^32 for a batch of B tokens, via BFV.
    Returns (z (B, D) uint32-valued, server_seconds, client_seconds, upload_bytes, download_bytes)."""
    import time
    B, n = alphas.shape
    assert B <= bfv.N
    slots = np.zeros((n, bfv.N), dtype=np.int64)
    slots[:, :B] = alphas.T
    t0 = time.perf_counter()
    C = bfv.encrypt_slots(slots)                              # client
    t1 = time.perf_counter()
    D = H.shape[0]
    z = np.zeros((D, bfv.N), dtype=np.int64)
    srv = cli = 0.0
    for k in range(4):                                         # byte digits of H
        Hk = (H >> (8 * k)) & 0xFF
        for r0 in range(0, D, 2048):
            a = time.perf_counter()
            Ck = bfv.lincomb(Hk[r0:r0 + 2048], C)              # server
            b = time.perf_counter()
            zk = bfv.decrypt_slots(Ck)                         # client
            c = time.perf_counter()
            srv += b - a; cli += c - b
            z[r0:r0 + 2048] += zk << (8 * k)
    z &= 0xFFFFFFFF
    up = C.size * 27 / 8                                        # 27-bit residues
    down = D * 4 * 2 * 2 * bfv.N * 27 / 8
    return z[:, :B].T, srv, cli + (t1 - t0), up, down


# ---------------------------------------------------------------- GPU offline path (int8 tensor cores)
def gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

class TorchNTT:
    """GPU port of NTT (same tables). int64 is exact: residues < 2^31, products < 2^62."""
    def __init__(self, nt, dev):
        import torch
        T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64), device=dev)
        self.p, self.N, self.n_inv = nt.p, nt.N, nt.n_inv
        self.rev, self.psi_pow, self.psi_inv_pow = T(nt.rev), T(nt.psi_pow), T(nt.psi_inv_pow)
        self.tw, self.tw_inv = [T(w) for w in nt.tw], [T(w) for w in nt.tw_inv]

    def _cyclic(self, a, tws):
        import torch
        p = self.p
        a = a[..., self.rev]
        shp = tuple(a.shape[:-1])
        for s, w in enumerate(tws):
            m = 2 << s
            a = a.reshape(shp + (self.N // m, m))
            u = a[..., : m // 2]
            v = a[..., m // 2:] * w % p
            a = torch.cat([(u + v) % p, (u - v) % p], -1)
        return a.reshape(shp + (self.N,))

    def fwd(self, a): return self._cyclic(a % self.p * self.psi_pow % self.p, self.tw)
    def inv(self, A): return self._cyclic(A, self.tw_inv) * self.n_inv % self.p * self.psi_inv_pow % self.p

def _digits8(x):
    """Centred int64 tensor (|x| < 2^31) -> 4 balanced int8 digits with x = sum_k d_k 2^(8k)."""
    import torch
    out = []
    for _ in range(4):
        d = ((x + 128) & 255) - 128
        out.append(d.to(torch.int8)); x = (x - d) >> 8
    return out

class BFV32GPU:
    """Exact GPU version of BFV32.lincomb_and_shrink + decrypt_top. Per limb p, H and the ciphertexts are
    centred mod p (|.| < 2^29) and split into 4 signed byte digits, so Enc(H alpha) is 16 int8 GEMMs with
    int32 accumulation (|digit sum| <= 128 * 128 * k = 2^24 for k = 1024), recombined mod p in int64."""
    def __init__(self, bfv, C, dev="cuda"):
        import torch
        self.bfv, self.dev = bfv, dev
        k = C.shape[0]
        self.Cs = []                    # per limb, per shift s: digits C_j (j = s - i) stacked along K, int8
        for L, p in enumerate(bfv.p):
            c = torch.as_tensor(C[:, L].reshape(k, -1), device=dev)
            cd = _digits8(c - p * (c > p // 2)); del c
            self.Cs.append([torch.cat([cd[s - i] for i in range(4) if 0 <= s - i < 4]) for s in range(7)])
        self.pow2 = [[pow(2, 8 * s, p) for s in range(7)] for p in bfv.p]
        self.ntt0 = TorchNTT(bfv.ntt[0], dev)
        self.s_hat0 = torch.as_tensor(bfv.s_hat[0].astype(np.int64), device=dev)

    def server(self, H):
        """H uint32 (D, k) -> (c0, c1_hat) mod p0, identical to BFV32.lincomb_and_shrink (as GPU tensors)."""
        import torch
        b, D = self.bfv, H.shape[0]
        Dp = -(-D // 32) * 32                                      # cuBLASLt int8 wants aligned m
        h = torch.zeros((Dp, H.shape[1]), dtype=torch.int64, device=self.dev)
        h[:D] = torch.as_tensor(H.astype(np.int64), device=self.dev)
        acc = []
        for L, p in enumerate(b.p):
            hl = h % p
            hd = _digits8(hl - p * (hl > p // 2))
            tot = torch.zeros((Dp, 2 * b.N), dtype=torch.int64, device=self.dev)
            for s in range(7):            # one GEMM per shift: sum_{i+j=s} H_i C_j, |.| <= 4 * 2^24 (int32-exact)
                A = torch.cat([hd[i] for i in range(4) if 0 <= s - i < 4], 1)
                tot.add_(torch._int_mm(A, self.Cs[L][s]), alpha=self.pow2[L][s])   # 7 terms < 2^26 * 2^30 * 7
            acc.append((tot[:D] % p).reshape(D, 2, b.N)); del tot
        x0, x1, x2 = acc                                           # modulus switch Q -> p0, as on CPU
        p0, p1, p2 = b.p
        kk = (x2 - x1) % p2 * b.p1_inv_mod_p2 % p2
        r12 = x1 + p1 * kk
        r12 = torch.where(r12 > b.p12 // 2, r12 - b.p12, r12)
        xs = (x0 - r12) % p0 * b.p12_inv_mod_p0 % p0
        return xs[:, 0], self.ntt0.fwd(xs[:, 1])

    def client(self, c0, c1_hat, B):
        """decrypt_top on the GPU: round(x * 2^32 / p0) mod 2^32, first B slots, back to numpy."""
        p = self.bfv.p[0]
        x = (c0 + self.ntt0.inv(c1_hat * self.s_hat0 % p)) % p
        return ((((x[:, :B] << 32) + p // 2) // p) & 0xFFFFFFFF).cpu().numpy()


def offline_hint_products_v2_gpu(bfv, H, alphas, chunk=4096):
    """offline_hint_products_v2 with the server's linear combination (and the client's decryption) on the GPU."""
    import time, torch
    B, n = alphas.shape
    assert B <= bfv.N
    m = np.zeros((n, bfv.N), dtype=np.int64); m[:, :B] = alphas.T
    t0 = time.perf_counter(); C = bfv.encrypt_coeffs(m); t1 = time.perf_counter()
    g = BFV32GPU(bfv, C); torch.cuda.synchronize()
    D = H.shape[0]; z = np.zeros((D, B), dtype=np.int64); srv = time.perf_counter() - t1; cli = 0.0
    for r0 in range(0, D, chunk):
        a = time.perf_counter(); c0, c1h = g.server(H[r0:r0 + chunk]); torch.cuda.synchronize()
        b = time.perf_counter(); z[r0:r0 + chunk] = g.client(c0, c1h, B)
        c = time.perf_counter(); srv += b - a; cli += c - b
    up = C.size * 30 / 8
    down = D * 2 * bfv.N * 30 / 8
    return z.T, srv, cli + (t1 - t0), up, down
