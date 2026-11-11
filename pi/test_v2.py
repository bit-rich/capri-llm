import numpy as np, time
from crypto import *
rng = np.random.default_rng(0)
lwe = InnerLWE(); ab = lwe.public_a(5, 2048)
M = rng.integers(-127, 128, (2048, 2048)).astype(np.int8)
H = lwe.hint(M, ab)
bfv = BFV32(); print("p", bfv.p, "logQ", bfv.Q.bit_length())
al = dgauss(rng, (64, 1024), 6.4)
z, srv, cli, up, down = offline_hint_products_v2(bfv, H, al)
exact = (al @ H.T) & 0xFFFFFFFF
err = (z - exact) & 0xFFFFFFFF; err = np.where(err >= 1<<31, err - (1<<32), err)
print("max |err|", np.abs(err).max(), "log2", round(np.log2(np.abs(err).max()+1),1), "srv", round(srv,2), "cli", round(cli,2))
# compare with online noise floor M e
e = dgauss(rng, 2048, 6.4); print("online noise floor |M e| max log2", round(np.log2(np.abs(M.astype(np.int64) @ e).max()),1))
