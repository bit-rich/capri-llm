"""Settle the EAGLE variant on the real target (run where the target fits, e.g. the AWS machine, next to server.py v4):
private pass over a text, collect the target's final hidden states, then rank the draft variants by how often they
guess the target's next token. python eagle_check.py host=127.0.0.1 port=7000 eagle=<dir> cache=<dir>"""
import sys, itertools, numpy as np, torch
from netclient import connect, make_client
from protocol import new_cache, rmsnorm
import eagle
opt = lambda k, d=None: next((a[len(k) + 1:] for a in sys.argv if a.startswith(k + "=")), d)
c, s, bh, model, tok, rtt = connect(opt("host", "127.0.0.1"), int(opt("port", 7000)), opt("cache"))
client, remote, run = make_client(c, bh, model)
text = ("The river runs through the old town before it reaches the sea. Every spring the water rises, and the people who live "
        "along the banks move their boats to higher ground. In 1987 a new bridge was built to replace the wooden one.")
ids = tok.encode(text).ids[:int(opt("n", 64))]
client.offline(len(ids) + 4); client.begin_pass(len(ids)); cache = new_cache(model)
lg = run(ids, cache)                                         # the target's logits at every position
x = client.last_x; post = rmsnorm(x, model.norm, model.eps)
target_next = list(ids[1:]) + [int(lg[-1].argmax())]       # teacher forcing on the real text
print(f"target's own next-token accuracy on this text: {np.mean(lg[:-1].argmax(1) == np.array(ids[1:])):.0%}", flush=True)
d = eagle.EagleDraft(opt("eagle"), model.embed, model.inv_freq, model.rope_scale)
grid = [dict(permute_qk=p, feat_post_norm=f, norm0=n0, final_norm=fn) for p, f, n0, fn in itertools.product([True, False], repeat=4)]
for var, acc1, run4 in eagle.acceptance(d, [ids[0]] + target_next, x, post, grid, k=1)[:6]:
    print(f"  1-step accuracy {acc1:5.0%}  {var}", flush=True)
best = eagle.acceptance(d, [ids[0]] + target_next, x, post, grid, k=1)[0][0]
for fb in (False, True):
    var, acc1, run4 = eagle.acceptance(d, [ids[0]] + target_next, x, post, [dict(best, feed_post_norm=fb)], k=4)[0]
    print(f"  with k=4, feed_post_norm={fb}: mean guesses kept {run4:.2f} of 4", flush=True)
c.send(dict(op="bye"))
