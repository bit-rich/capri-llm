"""Per-token cost model, calibrated against the prototype (D, bytes per row and token match measurements).
Each 'matrix' is (rows d_out, d_in, count per token). Rounds = client<->server round trips per token."""
import json
n = 1024                      # LWE dimension
K_LO, K_HI = 24, 96           # offline multiply-adds per (row, n-coord): measured float64 path / int8 tensor-core path
OFF_DOWN_B, ONLINE_B = 7.5, 4 # bytes per row per token (offline compressed hint product; online uint32)
OFF_UP_B = n * 3 * 2 * 30 / 8 # BFV upload of alpha, amortised per token (= 23 KB, as measured)

def model(name, mats, rounds, extra=None):
    D = sum(r * c for r, d, c in mats); P = sum(r * d * c for r, d, c in mats)
    up = sum(d * c for r, d, c in mats if d)    # approx: every listed input is uploaded once
    return dict(name=name, D_rows=D, P_macs=P, rounds=rounds,
                offline_x_lo=round(D * n * K_LO / P, 1), offline_x_hi=round(D * n * K_HI / P, 1),
                online_MB=round((ONLINE_B * D + 4 * up) / 1e6, 1),
                offline_MB=round((OFF_DOWN_B * D + OFF_UP_B) / 1e6, 1), **(extra or {}))

out = []
# measured toy model (sanity check: D should be 41,984; offline 315 KB/token)
out.append(model("toy (measured)", [(1024, 4096, 1), (4096, 1024, 1)] +
                 [(3072, 1024, 4), (1024, 1024, 4), (4096, 1024, 4), (1024, 4096, 4)], 18))
# Llama-2-7B
L, d, f, V = 32, 4096, 11008, 32000
out.append(model("Llama-2-7B", [(d, V, 1), (V, d, 1), (3 * d, d, L), (d, d, L), (2 * f, d, L), (d, f, L)],
                 4 * L + 2, dict(two_server_s_per_tok=2.01, two_server_MB=27.0)))
# Llama-2-70B (GQA: kv dim 1024)
L, d, f = 80, 8192, 28672
out.append(model("Llama-2-70B", [(d, V, 1), (V, d, 1), (d + 2048, d, L), (d, d, L), (2 * f, d, L), (d, f, L)],
                 4 * L + 2, dict(two_server_s_per_tok=5.21, two_server_MB=135.6)))
# DeepSeek-V3 (671B total / 37B active). MLA with client-side W_UK/W_UV absorption; 3 dense + 58 MoE layers.
d, V, Lm, Ld = 7168, 129280, 58, 3
attn = [(1536 + 576, d, 61), (24576, 1536, 61), (d, 16384, 61)]
dense = [(2 * 18432, d, Ld), (d, 18432, Ld)]
io = [(d, V, 1), (V, d, 1)]
moe_leaky = [(256, d, Lm), (2 * 2048 * 9, d, Lm), (d, 2048, Lm * 9)]
moe_hidden = [(256, d, Lm), (2 * 2048 * 257, d, Lm), (d, 2048, Lm * 257)]
rounds_ds = 61 * 3 + Ld * 2 + Lm * 3 + 2
out.append(model("DeepSeek-V3, routing visible", io + attn + dense + moe_leaky, rounds_ds))
hid = model("DeepSeek-V3, routing hidden", io + attn + dense + moe_hidden, rounds_ds)
P_active = out[-1]["P_macs"]
hid["online_x"] = round(hid["P_macs"] / P_active, 1)                   # all experts computed
hid["offline_x_lo"] = round(hid["D_rows"] * n * K_LO / P_active, 1)    # relative to normal (sparse) inference
hid["offline_x_hi"] = round(hid["D_rows"] * n * K_HI / P_active, 1)
out.append(hid)
for m in out:
    m.setdefault("online_x", 1.0)
    m["total_compute_x"] = f"{m['online_x'] + m['offline_x_lo']:.0f}-{m['online_x'] * 4 if m['online_x']==1.0 else m['online_x'] * 4:.0f}+{m['offline_x_hi']:.0f}"
    m["latency_s_10ms_rtt"] = round(m["rounds"] * 0.010, 2)
    m["latency_s_1ms_rtt"] = round(m["rounds"] * 0.001, 2)
print(json.dumps(out, indent=1))
json.dump(out, open("costmodel.json", "w"), indent=1)
