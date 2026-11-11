"""v3 (c0 folded into the online reply): exactness of the limb/FFT arithmetic against Python big integers,
end-to-end correctness of M x~ ~= y - (C1' s)[t] with the predicted noise, reuse of the public C1' under a
fresh s, and CPU == GPU."""
import numpy as np, torch
import fold
from fold import N, n, LIMBS, M32
from crypto import InnerLWE, dgauss

rng = np.random.default_rng(0)
Q = 1 << 80
big = lambda limbs: sum(limbs[..., i].astype(object) << (16 * i) for i in range(LIMBS))      # uint16 limbs -> Python ints
def negmul(a, b, mod):                           # schoolbook negacyclic product, Python ints
    m = len(a); out = [0] * m
    for i in range(m):
        if a[i] == 0: continue
        for j in range(m):
            k = i + j
            if k >= m: out[k - m] -= int(a[i]) * int(b[j])
            else: out[k] += int(a[i]) * int(b[j])
    return [v % mod for v in out]
centre = lambda r, bits=32: np.where(r >= 1 << (bits - 1), r - (1 << bits), r)

# 1. negacyclic FFT product == schoolbook
f = fold.NegFFT(16, "cpu"); x = rng.integers(-99, 100, 16); y = rng.integers(-99, 100, 16)
got = torch.round(f.inv(f.fwd(torch.as_tensor(x).double()) * f.fwd(torch.as_tensor(y).double()))).long().numpy()
assert (got == centre(np.array(negmul(x, y, 1 << 32), dtype=np.int64))).all(), "NegFFT"

lwe = InnerLWE(); a16 = fold.public_a()
din, dout, B = 2048 + 512, 96, 48                # 3 inner blocks, the last one cut
ab = lwe.public_a(3, din)
M = rng.integers(-127, 128, (dout, din)).astype(np.int8)

# 2. encryption: b_j + a_j s == 2^48 m_j + e_j (mod Q) with small e, checked exactly on a few j
key = fold.Key(np.random.default_rng(1)); alphas = dgauss(rng, (B, n), lwe.sigma)
b16 = key.encrypt(a16, alphas)
for j in (0, 517, 1023):
    as_ = negmul(key.s, big(a16[j]), Q)          # s first: skips its zero coefficients
    noise = [(int(bb) + v - (int(alphas[t, j]) << 48 if t < B else 0) + Q // 2) % Q - Q // 2
             for t, (bb, v) in enumerate(zip(big(b16[j]), as_))]
    assert max(abs(v) for v in noise) < 40, "encrypt"
print("encrypt: b + a s = 2^48 m + e exactly, |e| <", max(abs(v) for v in noise) + 1)

# 3. pads: u == round(A b[t] / 2^48) mod 2^32, exactly
pads = fold.Pads([ab], [din], "cpu"); t = 5
u = pads(b16[:, t])[0].numpy().view(np.uint32).astype(np.int64)
bt = big(b16[:, t]); ref = []
for blk in ab: ref += [((v + (1 << 47)) >> 48) & M32 for v in negmul(blk, bt, Q)]
assert (u == np.array(ref[:din], dtype=np.int64)).all(), "pads"
print("pads: u = round(A b[t] / 2^48) exactly")

# 4. hint: C1' == round(M (A a) / 2^48) up to the 40-bit truncation of A a (checked on a few rows and coefficients)
hint = fold.Hint(a16, "cpu"); G = hint.G(ab, din)
C1 = torch.cat([c for _, c in hint.c1(M, G, chunk=64)]).numpy()
def negmat(a):                                   # signed negacyclic matrix over the integers (not reduced mod 2^32)
    A = np.empty((n, n), dtype=object)
    for j in range(n):
        col = np.roll(a, j).astype(object); col[:j] = -col[:j]; A[:, j] = col
    return A
A = np.concatenate([negmat(blk) for blk in ab])[:din]                                       # (din, n)
rn = np.sqrt((M.astype(np.float64) ** 2).sum(1)).max()
tol = 1 + 6 * rn / np.sqrt(12) / 256             # A a is rounded to 40 bits: M moves that by row_norm / sqrt(12) / 2^8
worst = 0
for tau in (0, 2077):
    Aa = A @ big(a16[:, tau])                    # (din,) exact
    for i in (0, 41, 95):
        exact = int(M[i].astype(object) @ Aa) % Q
        d = (int(C1[i, tau]) - ((exact + (1 << 47)) >> 48) + (1 << 31)) % (1 << 32) - (1 << 31)
        worst = max(worst, abs(d)); assert abs(d) <= tol, ("hint", d)
print(f"hint: C1' = round(M A a / 2^48) to within {worst} (allowed {tol:.0f}, from keeping 40 bits of A a)")

# 5. end to end, two batches with fresh s on the same public C1'
x = rng.normal(size=din)
S = (2.0 ** (fold.MSG_BITS - 1) - 2.0 ** 24) / (rn * np.linalg.norm(x)); xt = np.rint(x * S).astype(np.int64)
exact = M.astype(np.int64) @ xt
for seed, out_bits in ((1, 16), (2, 16), (3, 24)):
    key = fold.Key(np.random.default_rng(seed)); alphas = dgauss(rng, (B, n), lwe.sigma)
    b16 = key.encrypt(a16, alphas)                                                           # upload
    z = key.unmask(torch.as_tensor(C1), B).numpy()                                           # from the one-time download
    errs = []
    for t in (0, 7, B - 1):
        c = (lwe.mask(ab, alphas[t], din, rng) + xt) & M32
        w = (c - pads(b16[:, t])[0].numpy().view(np.uint32)) & M32                           # server
        y = fold.reply_bits((M.astype(np.int64) @ w) & M32, out_bits)                        # server -> client
        r = centre(((y << (fold.MSG_BITS - out_bits)) - z[:, t]) & ((1 << fold.MSG_BITS) - 1), fold.MSG_BITS)
        errs.append(r - exact)
    sd = np.std(errs); pred = np.sqrt((rn * lwe.sigma) ** 2 + 4.0 ** (fold.MSG_BITS - out_bits) / 12)
    print(f"end to end, s seed {seed}, {out_bits}-bit reply: noise std 2^{np.log2(sd):.1f} (predicted <= 2^{np.log2(pred):.1f}), "
          f"signal max 2^{np.log2(np.abs(exact).max()):.1f}")
    assert sd < 1.3 * pred and np.abs(errs).max() < 8 * pred, "end to end"

# 6. GPU == CPU
if torch.cuda.is_available():
    hg = fold.Hint(a16, "cuda"); Gg = hg.G(ab, din)
    assert (Gg.cpu() == G).all(), "G gpu"
    C1g = torch.cat([c for _, c in hg.c1(M, Gg, chunk=64)])
    assert (C1g.cpu().numpy() == C1).all(), "C1 gpu"
    ug = fold.Pads([ab], [din], "cuda")(b16[:, t])[0]
    assert (ug.cpu() == pads(b16[:, t])[0]).all(), "pads gpu"
    kg = fold.Key(np.random.default_rng(3), "cuda")
    assert (kg.unmask(C1g, B).cpu().numpy() == z).all(), "unmask gpu"
    assert (kg.encrypt(a16, alphas) == fold.Key(np.random.default_rng(3)).encrypt(a16, alphas)).all(), "encrypt gpu"
    print("gpu == cpu (hint, pads, encrypt, unmask)")
print("v3 tests passed")
