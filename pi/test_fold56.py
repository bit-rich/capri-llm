"""fold56 (v4 offline machinery) against Python big integers on small sizes, then one chained trip end to end with
the real one-time C1' and per-batch upload: the residual rows and the chain rows must come out as R v~ and
G (W . R v~ + x~) up to the inner noise."""
import numpy as np, torch
import fold56 as F
from fold56 import M56

p = F.P(n=64, N=128); rng = np.random.default_rng(0); dev = "cuda" if torch.cuda.is_available() else "cpu"
Q = 1 << 128
big = lambda limbs: sum(limbs[..., i].astype(object) << (16 * i) for i in range(F.LIMBS))
def negmul(a, b, mod):
    m = len(a); out = [0] * m
    for i in range(m):
        if a[i] == 0: continue
        for j in range(m):
            k = i + j; v = int(a[i]) * int(b[j])
            if k >= m: out[k - m] -= v
            else: out[k] += v
    return [v % mod for v in out]

a16 = F.public_a(p); key = F.Key(p, np.random.default_rng(1), dev)
B = 40; alphas = rng.integers(-20, 21, (B, p.n))
b16 = key.encrypt(a16, alphas)
worst = 0
for j in (0, 17, 63):
    as_ = negmul(key.s, big(a16[j]), Q)
    for t, (bb, v) in enumerate(zip(big(b16[j]), as_)):
        noise = (int(bb) + v - ((int(alphas[t, j]) << 72) if t < B else 0) + Q // 2) % Q - Q // 2
        worst = max(worst, abs(noise))
assert worst < 40; print(f"encrypt: b + a s = 2^72 m + e exactly, |e| <= {worst}")

ab = rng.integers(0, 1 << 56, (2, p.n), dtype=np.int64); din = 100
pads = F.Pads(p, [ab], [din], dev); t = 7
u = pads.batch(b16[:, t:t + 1].transpose(1, 0, 2))[0][0].cpu().numpy()
bt = big(b16[:, t]); ref = []
for blk in ab: ref += [((v + (1 << 71)) >> 72) & M56 for v in negmul(blk, bt, Q)]
assert (u.astype(object) == np.array(ref[:din], dtype=object)).all(); print("pads: u = round(A b[t] / 2^72) exactly")

hint = F.Hint(p, a16, dev, chunk=64); G = hint.G(ab, din).cpu().numpy()
for tau in (0, 101):
    col = big(a16[:, tau]); ref = []
    for blk in ab: ref += [((v + (1 << 63)) >> 64) % (1 << 64) for v in negmul(blk, col, Q)]
    got = G[:, tau].astype(object) % (1 << 64)
    assert (got == np.array(ref[:din], dtype=object)).all()
print("hint: G = round(A a / 2^64) exactly")

# gemm_mod against big ints
W = rng.integers(-127, 128, (24, 104)).astype(np.int8); C = rng.integers(-(1 << 63), (1 << 63) - 1, (3, 104), dtype=np.int64)
for nb in (7, 8):
    Cm = C & ((1 << (8 * nb)) - 1) if nb < 8 else C
    got = F.gemm_mod(torch.as_tensor(W, device=dev), torch.as_tensor(Cm, device=dev), nb).cpu().numpy().astype(object) % (1 << (8 * nb))
    ref = np.array([[sum(int(W[i, j]) * (int(Cm[t, j]) % (1 << (8 * nb))) for j in range(104)) % (1 << (8 * nb)) for i in range(24)] for t in range(3)], dtype=object)
    assert (got == ref).all()
print("gemm_mod exact mod 2^56 and 2^64")

# one chained trip, end to end: R (d x a), G (g x d), W ints; inputs with the client's masks, server pads, real C1' and z
a_dim, d, g = 96, 64, 80
R = rng.integers(-127, 128, (d, a_dim)).astype(np.int8); Gm = rng.integers(-127, 128, (g, d)).astype(np.int8)
Wt = rng.integers(0, 256, d)
av = rng.integers(0, 1 << 56, (2, p.n), dtype=np.int64); ax = rng.integers(0, 1 << 56, (1, p.n), dtype=np.int64)
pads = F.Pads(p, [av, ax], [a_dim, d], dev)
T = lambda x: torch.as_tensor(x, device=dev)
Rg, Gg, Wg = T(R), T(Gm), T(Wt)
C1t, C1y = F.c1_chain(Rg, Gg, Wg, hint.G(av, a_dim), hint.G(ax, d))
zt = key.unmask_full(C1t, B).cpu().numpy(); zy = key.unmask_top(F.compress(C1y), B).cpu().numpy()
from fold import NegFFT
def negmul56(a, al):                                         # client-side A alpha for this test
    out = []
    for blk in a: out += negmul(al, blk, 1 << 56)
    return np.array(out, dtype=object)
errs_t, errs_y = [], []
for tok in (0, 5, B - 1):
    v = rng.integers(-(1 << 18), 1 << 18, a_dim); x = rng.integers(-(1 << 30), 1 << 30, d)
    ev = rng.integers(-15, 16, a_dim); ex = rng.integers(-15, 16, d)
    cv = (negmul56(av, alphas[tok])[:a_dim] + ev + v) % (1 << 56); cx = (negmul56(ax, alphas[tok])[:d] + ex + x) % (1 << 56)
    uv, ux = [w[0] for w in pads.batch(b16[:, tok:tok + 1].transpose(1, 0, 2))]
    wv = (T(cv.astype(np.int64)) - uv) & M56; wx = (T(cx.astype(np.int64)) - ux) & M56          # server
    t_ = F.gemm_mod(Rg, wv[None], 7)[0]; y_ = F.gemm_mod(Gg, ((Wg * t_ + wx) & M56)[None], 7)[0]
    rt = ((t_.cpu().numpy() - zt[:, tok] + (1 << 55)) & M56) - (1 << 55)
    ry = ((y_.cpu().numpy() - zy[:, tok] + (1 << 55)) & M56) - (1 << 55)
    tt = R.astype(object) @ (v + ev).astype(object); yy = Gm.astype(object) @ (Wt.astype(object) * tt + (x + ex).astype(object))
    errs_t.append(np.abs((rt.astype(object) - tt)).max()); errs_y.append(np.abs((ry.astype(object) - yy)).max())
print(f"chained trip with real C1' and pads: residual rows off by <= {max(errs_t)}, chain rows off by <= 2^{np.log2(float(max(errs_y)) + 1):.1f} "
      f"(chain values up to 2^{np.log2(float(np.abs(yy).max())):.0f})")
assert max(errs_t) < 2 ** 12 and max(errs_y) < 2 ** 34
print("fold56 tests passed")
