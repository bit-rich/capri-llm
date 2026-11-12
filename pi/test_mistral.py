"""Mistral3 text-model support, checked on a tiny random model written in the on-disk layout of
mistralai/Mistral-Medium-3.5-128B: tensors under model.language_model., FP8 (float8_e4m3fn) linear weights with a
per-tensor weight_scale_inv, bf16 embeddings/norms/head, config under text_config, YaRN RoPE, no q/k norm.
Checks: FP8 decode table == torch; fp32 logits of our loader + forward pass == HF Ministral3ForCausalLM on the same
(dequantized) weights, token by token, merged (2 trips per block) and batched too; one private v3 pass."""
import json, os, sys, tempfile, numpy as np, torch
from safetensors.torch import save_file
from transformers import Ministral3Config, Ministral3ForCausalLM
from protocol import *

# 1. FP8 table
b = torch.arange(256, dtype=torch.uint8)
ref = b.view(torch.float8_e4m3fn).float().numpy()
assert np.array_equal(ref, F8_E4M3, equal_nan=True), "F8_E4M3 table"
print("FP8 (e4m3) decode table matches torch for all 256 codes")

# 2. tiny model with the real model's RoPE settings
torch.manual_seed(0)
rope = {"beta_fast": 4.0, "beta_slow": 1.0, "factor": 64.0, "llama_4_scaling_beta": 0, "mscale": 1.0, "mscale_all_dim": 0.0,
        "original_max_position_embeddings": 4096, "rope_theta": 1000000.0, "rope_type": "yarn", "type": "yarn"}
cfg = Ministral3Config(hidden_size=256, num_hidden_layers=3, num_attention_heads=8, num_key_value_heads=2, head_dim=32,
                       intermediate_size=640, vocab_size=1024, rms_norm_eps=1e-5, max_position_embeddings=262144,
                       rope_parameters=rope, tie_word_embeddings=False, sliding_window=None)
hf = Ministral3ForCausalLM(cfg).eval()
out = {}
with torch.no_grad():
    for k, p in hf.state_dict().items():
        if p.ndim == 1: p.copy_(1 + 0.3 * torch.randn_like(p))            # norm weights: not all ones
        if p.ndim == 2 and ".layers." in k:                                # block linears -> FP8 + scalar scale
            s = (p.abs().max() / 448).to(torch.bfloat16).float()
            q = (p / s).to(torch.float8_e4m3fn)
            p.copy_(q.float() * s)                                         # HF runs on the dequantized weights
            out[k.replace("model.", "model.language_model.", 1)] = q
            out[k.replace("model.", "model.language_model.", 1) + "_scale_inv"] = s.to(torch.bfloat16)
        else:
            p.copy_(p.to(torch.bfloat16).float())
            out[k if k.startswith("lm_head") else k.replace("model.", "model.language_model.", 1)] = p.to(torch.bfloat16)
tmp = tempfile.mkdtemp(prefix="tiny_mistral_")
save_file(out, os.path.join(tmp, "model.safetensors"))
json.dump({"model_type": "mistral3", "text_config": cfg.to_dict()}, open(os.path.join(tmp, "config.json"), "w"))

ids = [1, 17, 933, 5, 250, 251, 77, 3, 640, 12, 999, 42]
with torch.no_grad(): ref = hf(input_ids=torch.tensor([ids])).logits[0].double().numpy()
rel = lambda a, b: float((np.linalg.norm(a - b, axis=-1) / np.linalg.norm(b, axis=-1)).max())

m = Qwen3(tmp, keep_float=True)
assert "q_norm" not in m.blocks[0] and abs(m.rope_scale - (0.1 * np.log(64) + 1)) < 1e-12
cache = [([], []) for _ in m.blocks]
ours = np.array([forward_token(m, t, cache, lambda n, L, x: L.float_(x)) for t in ids])
print("fp32 forward vs HF Ministral3ForCausalLM: max rel logit diff %.1e" % rel(ours, ref)); assert rel(ours, ref) < 1e-4

mm = Qwen3(tmp, keep_float=True, merge=True)
cache = [([], []) for _ in mm.blocks]
mer = np.array([forward_token(mm, t, cache, lambda n, L, x: L.float_(x)) for t in ids])
cache = [([], []) for _ in mm.blocks]
bat = np.concatenate([forward_batch(mm, ids[:5], cache, lambda n, L, X: np.stack([L.float_(x) for x in X])),
                      forward_batch(mm, ids[5:], cache, lambda n, L, X: np.stack([L.float_(x) for x in X]))])
print("merged fp32 vs HF: %.1e, merged batched vs HF: %.1e" % (rel(mer, ref), rel(bat, ref)))
assert rel(mer, ref) < 1e-4 and rel(bat, ref) < 1e-4

# 2b. GPU load path (decode, merge, quantize on the GPU) gives the same int8 model as the numpy path
if torch.cuda.is_available():
    for mg in (False, True):
        A, B = Qwen3(tmp, keep_float=False, merge=mg), Qwen3(tmp, keep_float=False, merge=mg, dev="cuda")
        diff = max(int(np.abs(a.q.astype(np.int16) - b.q).max()) for (_, a), (_, b) in zip(A.linears(), B.linears()))
        same = np.mean([np.mean(a.q == b.q) for (_, a), (_, b) in zip(A.linears(), B.linears())])
        assert diff <= 1 and same > 0.97 and all(np.allclose(a.scale, b.scale, rtol=1e-5) for (_, a), (_, b) in zip(A.linears(), B.linears()))
        assert all(abs(a.row_norm_max - b.row_norm_max) < 1e-3 * a.row_norm_max for (_, a), (_, b) in zip(A.linears(), B.linears()))
        print("GPU load vs numpy load (merge=%s): %.2f%% of int8 weights identical, the rest differ by 1 (rounding ties)" % (mg, 100 * same))

# 3. one private pass (v3 + merge, 20-bit reply) against the merged int8 plaintext model
gpu = torch.cuda.is_available()
lwe = InnerLWE(); srv = FoldServer(mm, lwe, gpu=gpu, out_bits=20); cl = FoldClient(srv, lwe, None, gpu=gpu)
cl.offline(len(ids) + 2); srv.hint = None
if gpu: srv.upload_gpu()
cache = [([], []) for _ in mm.blocks]; w8 = np.array([forward_token(mm, t, cache, lambda n, L, x: L.plain(x)) for t in ids])
cache = [([], []) for _ in mm.blocks]; cl.begin_pass(len(ids)); pv = forward_batch(mm, ids, cache, cl.lin_batch)
print("private (v3 + merge, whole sequence in %d round trips) vs merged W8: max rel logit diff %.1e, argmax equal on %d of %d"
      % (cl.rounds, rel(pv, w8), (pv.argmax(1) == w8.argmax(1)).sum(), len(ids)))
assert cl.rounds == 2 * mm.L + 2 and rel(pv, w8) < 0.2
print("mistral tests passed")
