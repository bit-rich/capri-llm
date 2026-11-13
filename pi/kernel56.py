"""Fused GPU kernel (Triton) for the v4 server: y = W (c - u) mod 2^56 for a few tokens at once.

W is int8 (rows, K); c and u are int64 (T, K) holding values mod 2^56. Each value is split, inside the kernel, into
eight 7-bit digits (0..127, so they are valid int8 and need no sign handling), the digits of all tokens form the
columns of one int8 tensor-core matmul with W (int32 accumulation: 127 * 127 * K < 2^31 for K < 133k), and the
eight partial sums are recombined mod 2^56 before anything is written. W is read from memory once, and nothing else
goes through memory: this replaces about thirty separate GPU steps in fold56.gemm_mod."""
import torch, triton, triton.language as tl

M56 = (1 << 56) - 1

@triton.jit
def _gemm56(W, C, U, Y, M, K, T: tl.constexpr, NCOL: tl.constexpr, HAS_U: tl.constexpr,
            BM: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    rm = pid * BM + tl.arange(0, BM)
    cols = tl.arange(0, NCOL); tok = cols // 8; dig = cols % 8; colok = tok < T
    acc = tl.zeros((BM, NCOL), dtype=tl.int32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        w = tl.load(W + rm[:, None] * K + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K), other=0)
        cm = (rk[:, None] < K) & colok[None, :]
        cv = tl.load(C + tok[None, :] * K + rk[:, None], mask=cm, other=0)
        if HAS_U:
            cv = cv - tl.load(U + tok[None, :] * K + rk[:, None], mask=cm, other=0)     # mod 2^56 via the low bits
        d = ((cv >> (7 * dig[None, :]).to(tl.int64)) & 127).to(tl.int8)               # (BK, NCOL) 7-bit digits
        acc += tl.dot(w, d, out_dtype=tl.int32)
    part = acc.to(tl.int64) << (7 * dig[None, :]).to(tl.int64)
    y = tl.sum(tl.reshape(part, (BM, NCOL // 8, 8)), axis=2) & 72057594037927935      # 2^56 - 1
    rt = tl.arange(0, NCOL // 8)
    tl.store(Y + rt[None, :] * M + rm[:, None], y, mask=(rm[:, None] < M) & (rt[None, :] < T))

def gemm56(W, C, U=None, tokens_per_call=4, BM=64, BK=64, stages=2):
    """W int8 (rows, K) CUDA tensor; C (and U) int64 (T, K). Returns W (C - U)^T mod 2^56 as (T, rows) int64."""
    T, K = C.shape; M = W.shape[0]
    C = C.contiguous(); U = U.contiguous() if U is not None else C
    out = torch.empty((T, M), dtype=torch.int64, device=C.device)
    for t0 in range(0, T, tokens_per_call):
        t = min(tokens_per_call, T - t0)
        ncol = max(16, triton.next_power_of_2(8 * t))
        _gemm56[(triton.cdiv(M, BM),)](W, C[t0:t0 + t], U[t0:t0 + t], out[t0:t0 + t], M, K, T=t, NCOL=ncol,
                                       HAS_U=U is not C, BM=BM, BK=BK, num_stages=stages)
    return out
