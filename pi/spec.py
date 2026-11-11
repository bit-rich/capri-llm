"""Whole-prompt passes and speculative decoding on top of v3 + merge (trustless: the server still only sees masked data).

One pass of the merged model costs 2 L + 2 round trips no matter how many tokens go through it, so:
  - whole prompt : all prompt tokens go through ONE pass instead of one pass each;
  - speculative  : a small draft model on the client (plaintext, public weights) guesses k tokens, the private model
                   checks [last token, guess 1..k] in ONE pass. Guesses are kept while they equal the private model's own
                   greedy choice, and the first disagreement is replaced by that choice, so every pass yields 1..k+1
                   tokens and the text is the private model's greedy text. Each token of a pass uses up one secret.

`python spec.py [1.7b] [k=4] [bits=20] [draftgpu]` runs three configurations on the same prompts and server:
  one token per pass (the `run.py v3 merge` behaviour), whole prompt only, whole prompt + speculative.
Target: Qwen3-0.6B, or Qwen3-1.7B with `1.7b` (needs merge_calib*.json from `run.py v3 merge`); draft: Qwen3-0.6B via HF
transformers fp32 on the CPU (`draftgpu`: on the GPU). Network times are computed from the measured compute time, round
trips and bytes; nothing is sent over a real network."""
import json, sys, time, numpy as np, torch
from protocol import *
from prompts import PROMPTS, NEW_TOKENS, encode, decode
import fold

repo, tag = ("Qwen/Qwen3-1.7B", "_1.7b") if "1.7b" in sys.argv else ("Qwen/Qwen3-0.6B", "")
arg = lambda name, default: next((int(a[len(name) + 1:]) for a in sys.argv if a.startswith(name + "=")), default)
K, bits = arg("k", 4), arg("bits", 20)
NETS = dict(fiber=(0.010, 1e9, 1e9), home=(0.030, 100e6, 20e6))       # round trip s, down bit/s, up bit/s

path = qwen_path(repo)
model = Qwen3(path, keep_float=False, merge=True, boost=json.load(open(f"merge_calib{tag}.json")))
prompt_ids = [encode(path, p) for p in PROMPTS]

class Draft:
    """Client-side guesser: greedy HF model with a KV cache that can be rolled back."""
    def __init__(self, dev):
        from transformers import AutoModelForCausalLM
        self.m = AutoModelForCausalLM.from_pretrained(qwen_path("Qwen/Qwen3-0.6B"), dtype=torch.float32).to(dev).eval()
        self.dev, self.t = dev, 0.0
    def reset(self): self.past, self.n = None, 0
    def propose(self, seq, k):
        """seq = all tokens accepted so far -> the next k greedy tokens of the draft model."""
        t = time.perf_counter(); out, inp = [], seq[self.n:]
        with torch.no_grad():
            for _ in range(k):
                r = self.m(input_ids=torch.tensor([inp], device=self.dev), past_key_values=self.past, use_cache=True)
                self.past = r.past_key_values; self.n += len(inp)
                out.append(int(r.logits[0, -1].argmax())); inp = out[-1:]
        self.t += time.perf_counter() - t
        return out
    def keep(self, n):                           # the cache may hold rejected guesses beyond the first n tokens
        if self.n > n: self.past.crop(n - self.n); self.n = n      # this transformers version only crops by a negative count

# plaintext reference: the same merged int8 model, one token at a time
def w8_greedy(ids):
    cache = [([], []) for _ in model.blocks]; gen, logits = [], []
    for t in range(len(ids) + NEW_TOKENS - 1):
        lg = forward_token(model, ids[t] if t < len(ids) else gen[-1], cache, lambda n, L, x: L.plain(x))
        if t >= len(ids) - 1: logits.append(lg); gen.append(int(lg.argmax()))
    return gen, np.array(logits)
ref = [w8_greedy(ids) for ids in prompt_ids]

gpu = torch.cuda.is_available() and "cpu" not in sys.argv
lwe = InnerLWE(); server = FoldServer(model, lwe, gpu=gpu, out_bits=bits); client = FoldClient(server, lwe, None, gpu=gpu)
P, G = sum(map(len, prompt_ids)), len(PROMPTS) * (NEW_TOKENS - 1)
off = client.offline(2 * (P + G) + (P + G * (K + 1)) + 8)            # secrets for the three configurations (worst case)
server.hint = None
if gpu: torch.cuda.empty_cache(); server.upload_gpu()
draft = Draft("cuda" if "draftgpu" in sys.argv and gpu else "cpu")
print("offline:", {k: round(v, 2) for k, v in off.items()}, flush=True)

def snap(): return np.array([time.perf_counter(), server.time_online, draft.t, client.rounds, client.up_bytes, client.down_bytes, client.tok])

def run(whole_prompt, k_max):
    tot = dict(prompt=0, gen=0); texts, eq, rel, passes, kept, offered = [], [], [], 0, 0, 0
    for ids, (w8_toks, w8_lg) in zip(prompt_ids, ref):
        cache = [([], []) for _ in model.blocks]; s0 = snap()
        if whole_prompt:
            client.begin_pass(len(ids)); lg = forward_batch(model, ids, cache, client.lin_batch, n_head=1)
        else:
            for t in ids: client.begin_pass(1); lg = forward_batch(model, [t], cache, client.lin_batch)
        gen, logits = [int(lg[-1].argmax())], [lg[-1]]; s1 = snap(); draft.reset()
        while len(gen) < NEW_TOKENS:
            k = min(k_max, NEW_TOKENS - len(gen) - 1); seq = ids + gen
            d = draft.propose(seq, k) if k else []
            client.begin_pass(k + 1); n0 = len(cache[0][0])
            lg = forward_batch(model, [gen[-1]] + d, cache, client.lin_batch); p = lg.argmax(1)
            a = next((i for i in range(k) if d[i] != p[i]), k)       # guesses that equal the private model's choice
            gen += d[:a] + [int(p[a])]; logits += list(lg[:a + 1])
            cache_keep(cache, n0 + a + 1); draft.keep(len(seq) + a)
            passes += 1; kept += a; offered += k
        s2 = snap(); tot["prompt"] = tot["prompt"] + (s1 - s0); tot["gen"] = tot["gen"] + (s2 - s1)
        n = next((i for i, (x, y) in enumerate(zip(gen, w8_toks)) if x != y), NEW_TOKENS - 1) + 1
        rel += list(np.linalg.norm(np.array(logits) - w8_lg, axis=1)[:n] / np.linalg.norm(w8_lg, axis=1)[:n])
        eq.append(gen == w8_toks); texts.append(decode(path, gen))
    res = dict(whole_prompt=whole_prompt, k=k_max, private_eq_w8=eq, logits_rel_err_vs_w8_mean=float(np.mean(rel)),
               logits_rel_err_vs_w8_max=float(max(rel)), texts=texts, prompt_tokens=P, generated_tokens=G,
               passes=passes, tokens_per_pass=G / passes, guesses_kept=kept / max(offered, 1))
    for ph, n in (("prompt", P), ("gen", G)):
        wall, srv, drf, trips, up, down, slots = tot[ph]
        r = dict(tokens=n, wall_s=wall, server_s=srv, draft_s=drf, client_s=wall - srv, round_trips=int(trips),
                 up_MB=up / 1e6, down_MB=down / 1e6, secrets_used=int(slots), tok_per_s_no_network=n / wall)
        for net, (rtt, dn, ub) in NETS.items():
            r[f"tok_per_s_{net}"] = n / (wall + trips * rtt + up * 8 / ub + down * 8 / dn)
        res[ph] = r
    return res

out = dict(model=repo, draft="Qwen/Qwen3-0.6B (HF fp32, %s)" % draft.dev, reply_bits=bits, k=K, server_on_gpu=gpu,
           nets={k: dict(round_trip_s=v[0], down_bit_s=v[1], up_bit_s=v[2]) for k, v in NETS.items()}, runs={})
for name, wp, k in (("one token per pass", False, 0), ("whole prompt", True, 0), (f"whole prompt + speculative k={K}", True, K)):
    r = out["runs"][name] = run(wp, k)
    print(f"\n== {name}: private == W8 on {sum(r['private_eq_w8'])} of {len(PROMPTS)} prompts, logit err mean "
          f"{100 * r['logits_rel_err_vs_w8_mean']:.2f}%, tokens per pass {r['tokens_per_pass']:.2f}, guesses kept {100 * r['guesses_kept']:.0f}%")
    for ph in ("prompt", "gen"):
        q = r[ph]
        print(f"   {ph:6s}: {q['tokens']} tokens, {q['round_trips']} trips, up {q['up_MB']:.1f} MB, down {q['down_MB']:.1f} MB, "
              f"compute {q['wall_s']:.2f} s (server {q['server_s']:.2f}, draft {q['draft_s']:.2f}) | tok/s: no network "
              f"{q['tok_per_s_no_network']:.1f}, fiber {q['tok_per_s_fiber']:.2f}, home {q['tok_per_s_home']:.2f}", flush=True)
json.dump(out, open(f"results_spec{tag}.json", "w"), indent=1)
