"""Y15 generator: inversion-count x hop-count x token-type diagnostic. Forced-choice A/B, thinking off."""
import json, random, itertools, sys
CLAIMS = [("The vault is secure.", "The vault is breached."), ("The bridge is open.", "The bridge is closed."),
          ("The shipment is early.", "The shipment is late."), ("The test passed.", "The test failed."),
          ("The lab is clean.", "The lab is contaminated."), ("The river is rising.", "The river is falling."),
          ("The market is up.", "The market is down."), ("The signal is strong.", "The signal is weak.")]
OPAQUE = [("zeta", "omega"), ("kiv", "dolm"), ("veln", "tarq"), ("sorb", "nuvi"), ("plex", "ordu"), ("mavo", "jeft"), ("hurk", "lisp"), ("cwen", "bodi")]
RESP = [("Approve", "Reject"), ("Proceed", "Halt"), ("Accept", "Decline"), ("Go", "Stop")]
CELLS = [(1, 0), (1, 1), (3, 0), (3, 1), (3, 2), (3, 3), (5, 1), (5, 3)]  # (hops, swaps) table surface
LEX_CELLS = [(3, 0), (3, 1), (3, 2), (3, 3)]  # prose-negation surface, hops=3 only

def compose_expected(msg_idx, swaps_mask, recv_map):
    """Implementation 1: symbolic. returns response index reached by message msg_idx."""
    i = msg_idx
    for sw in swaps_mask: i = (1 - i) if sw else i
    return recv_map[i]

def brute_expected(msgs, relays, recv, target):
    """Implementation 2: string simulation with explicit tables, independent of the index algebra."""
    good = []
    for m in msgs:
        cur = m
        for table in relays: cur = table[cur]
        if recv[cur] == target: good.append(m)
    return good

def make(g, hops, swaps, tok, surface, order, rng_seed):
    rng = random.Random(rng_seed)
    pair = (OPAQUE if tok == "opaque" else CLAIMS)[g % 8]
    pair = list(pair)
    if tok == "opaque" and rng.random() < .5: pair.reverse()
    resp = list(RESP[(g // 2) % 4])
    if tok == "opaque" and rng.random() < .5: resp.reverse()
    mask = [1] * swaps + [0] * (hops - swaps); rng.shuffle(mask)
    recv_map = {0: 0, 1: 1}  # receiver literal: pair[0]->resp[0], pair[1]->resp[1]
    target_idx = rng.randrange(2); target = resp[target_idx]
    relays = [({pair[0]: pair[1], pair[1]: pair[0]} if s else {pair[0]: pair[0], pair[1]: pair[1]}) for s in mask]
    recv = {pair[0]: resp[0], pair[1]: resp[1]}
    exp_idx = [m for m in (0, 1) if compose_expected(m, mask, recv_map) == target_idx]
    brute = brute_expected(pair, relays, recv, target)
    assert len(exp_idx) == 1 and len(brute) == 1 and pair[exp_idx[0]] == brute[0], (g, hops, swaps)
    correct_msg = brute[0]
    opts = [pair[0], pair[1]] if order == 0 else [pair[1], pair[0]]
    lines = ["A signal passes through a chain of relays and then a receiver. You pick which message to send. You want the receiver to respond " + target + ".", ""]
    lines.append("Option A: send " + repr(opts[0]) if False else "Option A: send \"" + opts[0] + "\"")
    lines.append("Option B: send \"" + opts[1] + "\"")
    lines.append("")
    for k, t in enumerate(relays, 1):
        if surface == "table":
            lines.append(f"Relay {k}: hears \"{pair[0]}\" -> passes on \"{t[pair[0]]}\"; hears \"{pair[1]}\" -> passes on \"{t[pair[1]]}\".")
        else:
            lines.append(f"Relay {k}: " + ("says the opposite of what it hears." if mask[k-1] else "repeats exactly what it hears."))
    lines.append(f"Receiver: hears \"{pair[0]}\" -> responds {resp[0]}; hears \"{pair[1]}\" -> responds {resp[1]}.")
    lines += ["", "Which option should you send? Answer with A or B only."]
    ans = "A" if opts[0] == correct_msg else "B"
    literal = [m for m in pair if recv[m] == target][0]  # message whose content/receiver-map matches target directly
    return dict(game=g, hops=hops, swaps=swaps, tok=tok, surface=surface, order=order, prompt="\n".join(lines), answer=ans,
                literal_answer="A" if opts[0] == literal else "B", parity=swaps % 2, mask=mask)

def build():
    rows = []
    for g in range(16):
        for tok in ("opaque", "meaningful"):
            for (h, s) in CELLS:
                for order in (0, 1):
                    rows.append(make(g, h, s, tok, "table", order, hash((g, tok, h, s, "t")) % 10**6 if False else g * 1000 + h * 10 + s))
            for (h, s) in LEX_CELLS:
                for order in (0, 1):
                    rows.append(make(g, h, s, tok, "lex", order, g * 1000 + h * 10 + s + 500))
    return rows

if __name__ == "__main__":
    from collections import Counter
    rows = build(); json.dump(rows, open("/tmp/wwoc/results/y15_inversion_count/y15_rows.json", "w"))
    print(len(rows)); print(Counter((r["answer"]) for r in rows)); print(Counter((r["swaps"] % 2, r["literal_answer"] == r["answer"]) for r in rows))
    for r in (rows[5], [x for x in rows if x["surface"] == "lex" and x["tok"] == "meaningful"][3]): print(r["prompt"], r["answer"], r["literal_answer"], r["mask"]); print("---")
