"""Chat with a model through the private protocol: python chat.py host=<server> [port=7000] [cache=DIR] [budget=500]
[max=300] [k=0]

Everything you type and everything the model writes stays on this machine; the server only gets masked numbers.
Words appear as they are produced. Commands: /new (forget the conversation), /stats, /quit.

budget=N: how many tokens (yours and the model's) to prepare at once. Each token needs its own one-time secret, and
preparing reads the whole one-time file from disk (minutes for big models), so prepare a few hundred at a time. RAM:
about 4 bytes x weight rows per token (19 MB per token for Qwen3-32B). When the budget runs out, the shell prepares
the next batch and carries on with the same conversation.
k=N: guess N tokens ahead with a local Qwen3-0.6B (helps on fast links; on a slow upload it sends more data)."""
import sys, time, numpy as np, torch
from netclient import RemoteServer, connect, cached_bundle, client_model, make_client, LWE_SEED
import netclient
from protocol import InnerLWE, FoldClient, forward_batch, cache_keep

opt = lambda name, default=None: next((a[len(name) + 1:] for a in sys.argv if a.startswith(name + "=")), default)
HOST, PORT, CACHE = opt("host", "127.0.0.1"), int(opt("port", 7000)), opt("cache")
BUDGET, MAX_NEW, K = int(opt("budget", 500)), int(opt("max", 300)), int(opt("k", 0))

cb = cached_bundle(CACHE)
early = cb is not None                       # everything for the first batch is on disk: prepare it before connecting
if early:
    netclient.cache_dir = CACHE; bh = cb[0]; model, tok = client_model(*cb)
    print(f"{bh['repo']}: preparing the first batch from {CACHE} before connecting (no server needed for this part)", flush=True)
else:
    c, s, bh, model, tok, rtt = connect(HOST, PORT, CACHE)
if not bh["repo"].startswith("Qwen/"): raise SystemExit("chat formatting is written for Qwen chat models")
END = tok.token_to_id("<|im_end|>")
client, remote, run = make_client(None if early else c, bh, model)
left = 0
held = []
if early: remote.receive_b = held.append    # keep the per-batch upload until the server is there

def prepare():
    global left
    print(f"(preparing {BUDGET} tokens of one-time secrets; reads the one-time file from disk, this can take a while)", flush=True)
    t = time.perf_counter(); client.offline(BUDGET); left = BUDGET - 4
    print(f"(ready after {time.perf_counter() - t:.0f} s)", flush=True)

def take(n):
    """Secrets for a pass of n tokens; prepares a new batch first if this one would run out."""
    global left
    if n > left: prepare()
    client.begin_pass(n); left -= n

draft = None
if K:
    from transformers import AutoModelForCausalLM
    from protocol import qwen_path
    dm = AutoModelForCausalLM.from_pretrained(qwen_path("Qwen/Qwen3-0.6B"), dtype=torch.float32).eval()

def draft_propose(seq, k):
    """Greedy guesses from the draft model (no cache, simple; the draft is small)."""
    out = list(seq)
    with torch.no_grad():
        for _ in range(k): out.append(int(dm(input_ids=torch.tensor([out[-512:]])).logits[0, -1].argmax()))
    return out[len(seq):]

cache = [([], []) for _ in model.blocks]; history = []; pending = []; stats = []
prepare()
if early:
    for attempt in range(180):                 # the server may still be starting: retry for up to 30 minutes
        try: c, s, bh2, _, _, rtt = connect(HOST, PORT, CACHE); break
        except (ConnectionRefusedError, TimeoutError, OSError) as e:
            if attempt == 0: print(f"(waiting for the server at {HOST}:{PORT}: {e.__class__.__name__})", flush=True)
            time.sleep(10)
    else: raise SystemExit("the server did not come up")
    if bh2["weights_id"] != bh["weights_id"]: raise SystemExit("the server runs different weights than the cached files")
    remote.c = c; del remote.receive_b; remote.receive_b(held.pop())
print("\nType a message (/new, /stats, /quit).", flush=True)
while True:
    try: msg = input("\nyou> ").strip()
    except (EOFError, KeyboardInterrupt): break
    if not msg: continue
    if msg == "/quit": break
    if msg == "/new": cache = [([], []) for _ in model.blocks]; history = []; pending = []; print("(new conversation)"); continue
    if msg == "/stats":
        for st in stats: print("  " + st)
        continue
    text = ("\n" if pending else "") + f"<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ids = pending + tok.encode(text).ids
    u0, d0, r0, t0 = c.sent, c.recvd, client.rounds, time.perf_counter()
    take(len(ids)); lg = run(ids, cache, n_head=1)
    history += ids; gen = [int(lg[-1].argmax())]; t_first = time.perf_counter() - t0
    print("model> ", end="", flush=True); shown = ""
    def show():
        global shown
        t = tok.decode([g for g in gen if g != END]); print(t[len(shown):], end="", flush=True); shown = t
    show()
    while gen[-1] != END and len(gen) < MAX_NEW:
        k = min(K, MAX_NEW - len(gen) - 1)
        d = draft_propose(history + gen, k) if k else []
        take(k + 1); n0 = len(cache[0][0])
        lg = run([gen[-1]] + d, cache); p = lg.argmax(1)
        a = next((i for i in range(k) if d[i] != p[i]), k)
        gen += d[:a] + [int(p[a])]; cache_keep(cache, n0 + a + 1)
        if END in gen: gen = gen[:gen.index(END) + 1]
        show()
    wall = time.perf_counter() - t0
    history += gen[:-1]; pending = gen[-1:]                    # the last token is fed at the start of the next turn
    st = (f"{len(gen)} tokens in {wall:.0f} s: first after {t_first:.1f} s, then {(len(gen) - 1) / max(wall - t_first, 1e-9):.2f} tokens/s, "
          f"{client.rounds - r0} round trips, up {(c.sent - u0) / 1e6:.0f} MB, down {(c.recvd - d0) / 1e6:.0f} MB, server {remote.server_s:.0f} s total")
    stats.append(st); print(f"\n({st})", flush=True)
c.send(dict(op="bye")); s.close()
