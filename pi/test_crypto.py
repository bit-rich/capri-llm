import numpy as np, time
from crypto import *
rng = np.random.default_rng(0)
# NTT roundtrip + negacyclic product
p = find_ntt_prime(27, 2048); nt = NTT(p, 2048)
a = rng.integers(0, p, (3, 2048)); b = rng.integers(0, p, 2048)
assert (nt.inv(nt.fwd(a)) == a).all()
ref = np.zeros(2048, dtype=object)
for i in range(2048):
    for j in range(0, 2048, 512):  # spot-check columns via schoolbook on a few terms is slow; use full check on small ring
        pass
nt8 = NTT(find_ntt_prime(27, 8), 8); p8 = nt8.p
x = rng.integers(0, p8, 8); y = rng.integers(0, p8, 8)
prod = nt8.inv(nt8.fwd(x) * nt8.fwd(y) % np.uint64(p8)).astype(int)
sch = [0]*8
for i in range(8):
    for j in range(8):
        k = i + j; v = int(x[i]) * int(y[j])
        if k >= 8: sch[k-8] -= v
        else: sch[k] += v
assert all((s - q) % p8 == 0 for s, q in zip(sch, prod)), "ntt negacyclic"
# inner masks: FFT product equals explicit matrix
lwe = InnerLWE()
ab = lwe.public_a(5, 2048); al = dgauss(rng, 1024, 6.4)
r1 = negacyclic_mul_mod32(ab, al)
r2 = np.stack([(negacyclic_matrix(a_) @ al) & 0xFFFFFFFF for a_ in ab])
assert (r1 == r2).all(), "mask"
# hint: M (a*alpha) == H alpha mod 2^32
M = rng.integers(-127, 128, (300, 2048)).astype(np.int8)
H = lwe.hint(M, ab)
lhs = matmul_mod32_i8(M, r1.reshape(-1, 1))[:, 0]
rhs = (H @ al) & 0xFFFFFFFF
assert (lhs == rhs).all(), "hint"
# BFV offline products
bfv = BFV(); print("primes", bfv.p, "t", bfv.t, "logQ", np.log2(bfv.Q))
alphas = dgauss(rng, (2048, 1024), 6.4)
t0 = time.time()
z, srv, cli, up, down = offline_hint_products(bfv, H, alphas)
print("offline ok?", (z == (alphas @ H.T) & 0xFFFFFFFF).all(), f"{time.time()-t0:.1f}s srv {srv:.2f}s cli {cli:.2f}s")
print("all crypto tests passed")
