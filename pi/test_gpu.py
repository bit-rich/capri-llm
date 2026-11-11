"""GPU offline path must be bit-identical to the CPU path (server output) and decrypt to H alpha."""
import numpy as np, time
from crypto import *
rng = np.random.default_rng(0)
lwe = InnerLWE(); ab = lwe.public_a(5, 2048)
M = rng.integers(-127, 128, (1000, 2048)).astype(np.int8)          # 1000 rows: exercises row padding
H = lwe.hint(M, ab).astype(np.uint32)
bfv = BFV32(); al = dgauss(rng, (64, 1024), 6.4)
m = np.zeros((1024, bfv.N), dtype=np.int64); m[:, :64] = al.T
C = bfv.encrypt_coeffs(m)
c0, c1h = bfv.lincomb_and_shrink(H, C)
g = BFV32GPU(bfv, C); g0, g1h = g.server(H)
assert (g0.cpu().numpy() == c0).all() and (g1h.cpu().numpy() == c1h).all(), "GPU server output != CPU"
z_cpu = bfv.decrypt_top(c0, c1h)[:, :64]; z_gpu = g.client(g0, g1h, 64)
d = (z_gpu - z_cpu) & 0xFFFFFFFF; d = np.where(d >= 1 << 31, d - (1 << 32), d)
assert np.abs(d).max() <= 1, "GPU decrypt differs from CPU by more than float rounding"
exact = (al @ H.T.astype(np.int64)) & 0xFFFFFFFF
err = (z_gpu.T - exact) & 0xFFFFFFFF; err = np.where(err >= 1 << 31, err - (1 << 32), err)
print("server bit-identical; decrypt max |gpu-cpu|", np.abs(d).max(), "; log2 max|z - H alpha|", round(np.log2(np.abs(err).max() + 1), 1))
z, srv, cli, up, down = offline_hint_products_v2_gpu(bfv, H, al)
e2 = (z - exact) & 0xFFFFFFFF; e2 = np.where(e2 >= 1 << 31, e2 - (1 << 32), e2)   # fresh encryption: compare to H alpha
assert np.abs(e2).max() < 1 << 14, "end-to-end GPU offline too noisy"
print("end-to-end log2 max|z - H alpha|", round(np.log2(np.abs(e2).max() + 1), 1), f"srv {srv:.3f}s cli {cli:.3f}s")
print("gpu tests passed")
