"""Client-side EAGLE draft for speculative decoding with mistralai/Mistral-Medium-3.5-128B (its published EAGLE head,
mistralai/Mistral-Medium-3.5-128B-EAGLE). Runs entirely on the client, in the clear: it only needs public weights and
the target's final hidden states, which the private client already holds.

Structure (vLLM's Mistral EAGLE path): h_in = eagle_linear(cat(embed(t_{i+1}), feature_i)), then 2 decoder layers
(GQA attention with its own KV cache, SwiGLU), then the target's lm_head. The draft uses the target's embedding table
and lm_head. Details that the vLLM code we could read does not pin down are switches (`variant`), to be settled on
the real model with `acceptance()`:
  feat_post_norm  feature = target hidden after the final RMSNorm (True) or before it
  norm0           layer 0 applies its attention_norm (True) or skips it like vLLM's llama_eagle (False)
  final_norm      apply the draft's `norm` before the lm_head
  feed_post_norm  for the next draft step, feed back the draft's output after `norm` (True) or before it
  permute_qk      convert wq/wk rows from Mistral's interleaved RoPE layout to HF's rotate_half layout
Weights are kept in fp8 on the GPU (3 GB) and dequantized per matmul; the lm_head is kept in fp8 with a per-row scale."""
import json, os, numpy as np, torch
from protocol import bf16_to_f32

DEFAULT = dict(feat_post_norm=True, norm0=False, final_norm=True, feed_post_norm=False, permute_qk=True)

class EagleDraft:
    def __init__(self, d, embed, inv_freq, rope_scale, dev="cuda", variant=None):
        """d: folder with consolidated.safetensors, params.json and lm_head_bf16.npy. embed: the target's embedding
        table (bf16 uint16, rows looked up on the host). inv_freq / rope_scale: the target's RoPE (same yarn)."""
        from safetensors.torch import load_file
        self.v = dict(DEFAULT, **(variant or {})); self.dev, self.embed = dev, embed
        P = json.load(open(os.path.join(d, "params.json")))
        self.D, self.hq, self.hkv, self.hd, self.eps = P["dim"], P["n_heads"], P["n_kv_heads"], P["head_dim"], P["norm_eps"]
        W = load_file(os.path.join(d, "consolidated.safetensors"), device="cpu")
        def lin(name, permute_heads=None):
            w = W[name + ".weight"]
            if permute_heads and self.v["permute_qk"]:      # interleaved (2i, 2i+1) pairs -> halves, per head (HF convert)
                n = permute_heads; w = w.view(n, self.hd // 2, 2, -1).transpose(1, 2).reshape(w.shape)
            return w.to(dev), float(W[name + ".qscale_weight"].float())
        g = lambda name: W[name].float().to(dev)
        self.fc = lin("eagle_linear")
        self.layers = [dict(wq=lin(f"layers.{i}.attention.wq", self.hq), wk=lin(f"layers.{i}.attention.wk", self.hkv),
                            wv=lin(f"layers.{i}.attention.wv"), wo=lin(f"layers.{i}.attention.wo"),
                            w1=lin(f"layers.{i}.feed_forward.w1"), w2=lin(f"layers.{i}.feed_forward.w2"),
                            w3=lin(f"layers.{i}.feed_forward.w3"), an=g(f"layers.{i}.attention_norm.weight"),
                            fn=g(f"layers.{i}.ffn_norm.weight")) for i in range(P["n_layers"])]
        self.norm = g("norm.weight"); del W
        head = np.load(os.path.join(d, "lm_head_bf16.npy"), mmap_mode="r")      # (vocab, D) bf16 bits
        self.head, self.hscale = torch.empty(head.shape, dtype=torch.float8_e4m3fn, device=dev), torch.empty(head.shape[0], device=dev)
        for r in range(0, head.shape[0], 8192):
            x = torch.as_tensor(bf16_to_f32(np.asarray(head[r:r + 8192])), device=dev)
            s = x.abs().amax(1).clamp_min(1e-30) / 448.0; self.head[r:r + 8192] = (x / s[:, None]).to(torch.float8_e4m3fn); self.hscale[r:r + 8192] = s
        self.inv_freq = torch.as_tensor(np.asarray(inv_freq), dtype=torch.float32, device=dev); self.rope_scale = rope_scale
        self.reset()

    def reset(self): self.K, self.V, self.pos, self.last = [[] for _ in self.layers], [[] for _ in self.layers], [], None

    _NATIVE = None
    @classmethod
    def _mm(cls, x, W):
        """x (M, K) float -> x W^T. Native fp8 tensor-core matmul where the GPU has it (Ada/Hopper: 7x faster than
        dequantizing, activations quantized per tensor as in vLLM), else dequantize to bf16."""
        w, s = W
        if cls._NATIVE is None: cls._NATIVE = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
        if cls._NATIVE:
            M = x.shape[0]; xs = (x.abs().amax() / 448.0).clamp_min(1e-12).float()
            xq = torch.zeros((max(16, -(-M // 16) * 16), x.shape[1]), dtype=torch.float8_e4m3fn, device=x.device)
            xq[:M] = (x / xs).to(torch.float8_e4m3fn)
            out = torch._scaled_mm(xq, w.T, scale_a=xs.reshape(1, 1), scale_b=torch.tensor([[s]], device=x.device, dtype=torch.float32), out_dtype=torch.bfloat16)
            return out[:M].float()
        return (x.to(torch.bfloat16) @ w.to(torch.bfloat16).T).float() * s

    def _rms(self, x, w): return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps) * w

    def _rope(self, x, p):
        ang = p * self.inv_freq; ang = torch.cat([ang, ang]); cos, sin = ang.cos() * self.rope_scale, ang.sin() * self.rope_scale
        rd = cos.shape[0]; h = rd // 2; xr = x[..., :rd]
        out = xr * cos + torch.cat([-xr[..., h:], xr[..., :h]], -1) * sin
        return torch.cat([out, x[..., rd:]], -1) if rd < x.shape[-1] else out

    def _step(self, tok, feat, p):
        """One draft position: input token t_{p+1} and feature (the target's hidden at p, or the draft's own output).
        Returns (hidden before norm, hidden after norm)."""
        e = torch.as_tensor(bf16_to_f32(np.asarray(self.embed[int(tok)])), device=self.dev)
        h = self._mm(torch.cat([e, feat])[None], self.fc)[0]
        for i, L in enumerate(self.layers):
            a = h if (i == 0 and not self.v["norm0"]) else self._rms(h, L["an"])
            q = self._rope(self._mm(a[None], L["wq"])[0].view(self.hq, self.hd), p)
            k = self._rope(self._mm(a[None], L["wk"])[0].view(self.hkv, self.hd), p)
            v = self._mm(a[None], L["wv"])[0].view(self.hkv, self.hd)
            self.K[i].append(k); self.V[i].append(v)
            K, V = torch.stack(self.K[i]), torch.stack(self.V[i])                         # (T, hkv, hd)
            qg = q.view(self.hkv, self.hq // self.hkv, self.hd)
            s = torch.softmax(torch.einsum("grd,tgd->grt", qg, K) / self.hd ** 0.5, -1)
            h = h + self._mm(torch.einsum("grt,tgd->grd", s, V).reshape(1, -1), L["wo"])[0]
            f = self._rms(h, L["fn"])
            h = h + self._mm(torch.nn.functional.silu(self._mm(f[None], L["w1"])) * self._mm(f[None], L["w3"]), L["w2"])[0]
        return h, self._rms(h, self.norm)

    def _logits(self, h_pre, h_post):
        x = h_post if self.v["final_norm"] else h_pre
        if self._NATIVE:                                        # native fp8 with per-row (vocab) scales: ~10 ms on a laptop GPU
            xs = (x.abs().amax() / 448.0).clamp_min(1e-12).float()
            xq = torch.zeros((16, x.shape[0]), dtype=torch.float8_e4m3fn, device=x.device); xq[0] = (x / xs).to(torch.float8_e4m3fn)
            return torch._scaled_mm(xq, self.head.T, scale_a=xs.reshape(1, 1).expand(16, 1).contiguous(),
                                    scale_b=self.hscale.reshape(1, -1).contiguous(), out_dtype=torch.bfloat16)[0].float()
        xb = x.to(torch.bfloat16)[None]                        # in row chunks: no full-size bf16 copy of the head
        return torch.cat([(xb @ self.head[r:r + 16384].to(torch.bfloat16).T)[0].float() for r in range(0, len(self.head), 16384)]) * self.hscale

    def ingest(self, next_tokens, feats_pre, feats_post):
        """Accepted positions: next_tokens[j] = t_{p+1} and the target's hidden at p (pre / post final norm)."""
        F = feats_post if self.v["feat_post_norm"] else feats_pre
        for t, f in zip(next_tokens, torch.as_tensor(np.asarray(F), dtype=torch.float32, device=self.dev)):
            p = len(self.pos); self.pos.append(p); self.last = (int(t),) + self._step(t, f, p)

    def propose(self, k):
        """k greedy guesses after the last ingested position. The speculative entries are rolled back afterwards."""
        n0, out = len(self.pos), []
        tok, h_pre, h_post = self.last
        for _ in range(k):
            nxt = int(self._logits(h_pre, h_post).argmax()); out.append(nxt)
            feed = h_post if self.v["feed_post_norm"] else h_pre
            p = len(self.pos); self.pos.append(p); h_pre, h_post = self._step(nxt, feed, p)
        self.keep(n0)
        return out

    def keep(self, n):
        for i in range(len(self.layers)): del self.K[i][n:], self.V[i][n:]
        del self.pos[n:]


def acceptance(draft, tokens, h_pre, h_post, variants, k=4):
    """Settle the variant on the real model: tokens (the text the target actually produced or read) and the target's
    hidden states at every position. Teacher-forced 1-step accuracy and the mean number of k guesses that match."""
    rows = []
    for var in variants:
        draft.v = dict(DEFAULT, **var); draft.reset()
        one, runs = 0, []
        for j in range(len(tokens) - 2):
            draft.ingest([tokens[j + 1]], h_pre[j:j + 1], h_post[j:j + 1])
            g = draft.propose(k); truth = tokens[j + 2:j + 2 + k]
            one += g[0] == truth[0]
            runs.append(next((i for i, (a, b) in enumerate(zip(g, truth)) if a != b), min(len(g), len(truth))))
        rows.append((var, one / (len(tokens) - 2), float(np.mean(runs))))
    return sorted(rows, key=lambda r: -r[1])
