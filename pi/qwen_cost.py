"""Cost model for Qwen3.6-27B (dense, hybrid Gated-DeltaNet + full attention) vs Qwen3.5-397B-A17B (MoE).
Configs from the models' Hugging Face config.json. Rows = outputs the client must unmask per token."""
n, KH, KO = 1024, 24, 6
def run(name, mats, rounds, P_ref=None):
    P = sum(r*d*c for r,d,c in mats); D = sum(r*c for r,d,c in mats)
    off_h = sum(r*c*KH*n for r,d,c in mats); off = sum(r*c*min(KH*n, KO*d) for r,d,c in mats)
    Pr = P_ref or P
    print(f"{name:34s} active MACs {P/1e9:6.1f}B  rows {D/1e6:5.2f}M  online x{P/Pr:5.1f}  "
          f"offline x{off/Pr:6.2f} (hint-only {off_h/Pr:6.2f})  comm {(4*D + 4*sum(d*c for r,d,c in mats) + 7.5*D)/1e6:6.0f} MB  rounds {rounds}")
    return P
# Qwen3.6-27B: d 5120, 64 layers (48 DeltaNet + 16 full attn), MLP 17408, vocab 248320
d, V, ff = 5120, 248320, 17408
head = [(V, d, 1)]
full = [(2*24*256 + 2*4*256, d, 16), (d, 24*256, 16)]                  # q(+gate),k,v ; o
lin  = [(2*16*128 + 2*48*128 + 2*48, d, 48), (d, 48*128, 48)]          # q,k,v,z,b,a ; out
mlp  = [(2*ff, d, 64), (d, ff, 64)]
q27 = head + full + lin + mlp
print("(embedding: client looks up its own row; table is public)")
run("Qwen3.6-27B dense", q27, 4*64 + 1)
# Qwen3.5-397B-A17B: d 4096, 60 layers (45 DeltaNet + 15 full), 512 experts top-10, expert & shared 1024
d, Lm = 4096, 60
head = [(V, d, 1)]
full = [(2*32*256 + 2*2*256, d, 15), (d, 32*256, 15)]
lin  = [(2*16*128 + 2*64*128 + 2*64, d, 45), (d, 64*128, 45)]
def moe(k): return [(512, d, Lm), (2*1024*(k+1), d, Lm), (d, 1024, Lm*(k+1))]
P_act = run("Qwen3.5-397B MoE, routing visible", head + full + lin + moe(10), 5*60 + 1)
run("Qwen3.5-397B MoE, routing hidden", head + full + lin + moe(512), 5*60 + 1, P_ref=P_act)
