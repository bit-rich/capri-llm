"""Security estimates for the parameter sets, with https://github.com/malb/lattice-estimator (needs SageMath).
Run with the estimator checked out next to this file, e.g. with Docker from this directory:
  git clone https://github.com/malb/lattice-estimator
  docker run --rm -v $PWD:/work -w /work sagemath/sagemath sage -python lattice_estimate.py rough   # or: full
Results (estimator commit 7d59f86, 2026-11): see README, section Security estimates."""
import sys, json, math
sys.path.insert(0, "/work/lattice-estimator")
from sage.all import oo, is_prime, log
from estimator import LWE, ND

def ntt_primes(bits, N, k):                      # as crypto.find_ntt_prime: largest primes = 1 mod 2N below 2^bits
    out, x = [], (2**bits // (2 * N)) * 2 * N + 1
    while len(out) < k:
        x -= 2 * N
        if is_prime(x): out.append(x)
    return out
q2 = math.prod(ntt_primes(30, 4096, 3)); q1 = math.prod(ntt_primes(27, 2048, 2))
G, T = ND.DiscreteGaussian, ND.Uniform(-1, 1)
sets = {
 # inner masks: c = A alpha + e + x mod 2^32, fresh alpha per token; m = masked inputs per token (all layers)
 "inner mask n=1024 q=2^32 sigma=6.4, m=2^18 (0.6B/1.7B)": LWE.Parameters(n=1024, q=2**32, Xs=G(6.4), Xe=G(6.4), m=2**18),
 "inner mask n=1024 q=2^32 sigma=6.4, m=2^20 (4B)":        LWE.Parameters(n=1024, q=2**32, Xs=G(6.4), Xe=G(6.4), m=2**20),
 "inner mask n=1024 q=2^32 sigma=6.4, m=2^24 (405B)":      LWE.Parameters(n=1024, q=2**32, Xs=G(6.4), Xe=G(6.4), m=2**24),
 # outer: 1024 ring samples under one ternary s per batch
 "v3 outer N=4096 q=2^80 ternary sigma=3.2":               LWE.Parameters(n=4096, q=2**80, Xs=T, Xe=G(3.2), m=1024 * 4096),
 "v2 outer N=4096 q~2^%d ternary sigma=3.2" % int(log(q2, 2)): LWE.Parameters(n=4096, q=q2, Xs=T, Xe=G(3.2), m=1024 * 4096),
 "v1 outer N=2048 q~2^%d ternary sigma=3.2" % int(log(q1, 2)): LWE.Parameters(n=2048, q=q1, Xs=T, Xe=G(3.2), m=1024 * 2048),
}
mode = sys.argv[1]; pick = sys.argv[2:]
res = {}
for name, p in sets.items():
    if pick and not any(k in name for k in pick): continue
    print("\n=====", name, flush=True)
    r = LWE.estimate.rough(p) if mode == "rough" else LWE.estimate(p, jobs=6)
    res[name] = {k: dict(log2_rop=float(log(v["rop"], 2)), beta=int(v.get("beta", 0))) for k, v in r.items()}
    print(json.dumps(res[name]), flush=True)
json.dump(res, open("/work/est_%s%s.json" % (mode, "_" + "_".join(pick).replace(" ", "") if pick else ""), "w"), indent=1)
