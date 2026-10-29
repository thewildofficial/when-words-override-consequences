"""X1 data: Kuhn poker decision states for player 2 with exact action values.

Convention: ante 1 each (not counted in payoffs). Payoffs are chips won/lost beyond the ante:
check-check showdown +-1, bet-call showdown +-2, bet-fold: bettor +1.
Player 1 (opponent) acts first with a FIXED stated behavioral policy:
  b[c] = P(bet at first node | card c), d[c] = P(call a bet after check-bet | card c).
The LLM is player 2. It knows its own card, the public history, and the stated opponent policy,
not the opponent card. Opponent card is uniform over the two cards the LLM does not hold.
"""
from fractions import Fraction as F
import hashlib, itertools, json, random

CARDS = (1, 2, 3)  # J < Q < K
POLICIES = {
    # alpha = 1/4 member of the Nash family: J bets a, Q never bets, K bets 3a;
    # after check-bet: J folds, Q calls a+1/3, K calls.
    "nash_a025": dict(b={1: F(1, 4), 2: F(0), 3: F(3, 4)}, d={1: F(0), 2: F(7, 12), 3: F(1)}),
    "nash_a010": dict(b={1: F(1, 10), 2: F(0), 3: F(3, 10)}, d={1: F(0), 2: F(13, 30), 3: F(1)}),
    "bluffy": dict(b={1: F(1, 2), 2: F(1, 2), 3: F(1, 2)}, d={1: F(1, 2), 2: F(1, 2), 3: F(1, 2)}),
    "passive": dict(b={1: F(0), 2: F(0), 3: F(1, 4)}, d={1: F(0), 2: F(1, 4), 3: F(1)}),
}

def posterior(c2, obs, pol):
    """P(opponent card | c2, opponent first action obs in {'bet','check'})."""
    others = [c for c in CARDS if c != c2]
    w = {c: (pol["b"][c] if obs == "bet" else 1 - pol["b"][c]) * F(1, 2) for c in others}
    z = sum(w.values())
    return None if z == 0 else {c: w[c] / z for c in others}

def action_values(c2, obs, pol):
    post = posterior(c2, obs, pol)
    if post is None:
        return None
    win = lambda c: 1 if c2 > c else -1
    if obs == "bet":
        return {"fold": F(-1), "call": sum(p * 2 * win(c) for c, p in post.items())}
    ev_bet = sum(p * (pol["d"][c] * 2 * win(c) + (1 - pol["d"][c]) * 1) for c, p in post.items())
    return {"check": sum(p * win(c) for c, p in post.items()), "bet": ev_bet}

def random_policies(n, seed=0):
    """n random fixed opponent policies on a 1/20 grid (continuous variation for value regression)."""
    rg = random.Random(seed); out = {}
    for i in range(n):
        g = lambda: F(rg.randrange(0, 21), 20)
        out[f"rand{seed}_{i:03d}"] = dict(b={c: g() for c in CARDS}, d={c: g() for c in CARDS})
    return out

def states(extra=None):
    out = []
    allp = dict(POLICIES); allp.update(extra or {})
    for name, pol in allp.items():
        for c2 in CARDS:
            for obs in ("bet", "check"):
                q = action_values(c2, obs, pol)
                if q is None:
                    continue
                (a1, v1), (a2, v2) = q.items()
                best = a1 if v1 > v2 else a2 if v2 > v1 else None  # None = tie
                out.append(dict(policy=name, card=c2, obs=obs, q={k: float(v) for k, v in q.items()},
                                q_exact={k: str(v) for k, v in q.items()}, best=best,
                                advantage=float(abs(v1 - v2))))
    return out

CARDNAMES = {"jqk": {1: "Jack", 2: "Queen", 3: "King"}, "num": {1: "1", 2: "2", 3: "3"},
             "suit": {1: "2 of clubs", 2: "9 of hearts", 3: "Ace of spades"}, "greek": {1: "alpha", 2: "beta", 3: "gamma"}}
RENDERINGS = {
    "r0": dict(cards="jqk", verbs={"bet": "bet", "check": "check", "call": "call", "fold": "fold"}),
    "r1": dict(cards="num", verbs={"bet": "raise", "check": "pass", "call": "match", "fold": "drop"}),
    "r2": dict(cards="suit", verbs={"bet": "wager", "check": "stay", "call": "meet", "fold": "concede"}),
    "r3": dict(cards="greek", verbs={"bet": "push", "check": "hold", "call": "accept", "fold": "decline"}),
}

def render(st, rid, seed=0, extra=None):
    r = RENDERINGS[rid]; cn = CARDNAMES[r["cards"]]; v = r["verbs"]; pol = (extra or {}).get(st["policy"]) or POLICIES[st["policy"]]
    names = {1: cn[1], 2: cn[2], 3: cn[3]}
    rule = f"The deck has three cards ranked {names[1]} < {names[2]} < {names[3]}. Each player antes 1 and gets one card. "
    pstr = lambda x: f"{float(x):.4g}"
    opp = ("The opponent acts first and plays a fixed policy. Probability it opens with a {bet}, by its card: "
           + ", ".join(f"{names[c]} {pstr(pol['b'][c])}" for c in CARDS)
           + ". If it {check}s and you {bet}, probability it {call}s, by its card: "
           + ", ".join(f"{names[c]} {pstr(pol['d'][c])}" for c in CARDS) + ". ").format(**v)
    pay = ("Payoffs beyond the ante: both {check}: higher card wins 1. A {bet} that is {call}ed: higher card wins 2. "
           "A {bet} that is {fold}ed: the bettor wins 1. ").format(**v)
    hist = f"Your card is {names[st['card']]}. The opponent {'opened with a ' + v['bet'] if st['obs']=='bet' else v['check']+'ed'}. "
    acts = list(st["q"].keys())
    rg = random.Random(int.from_bytes(hashlib.sha256(f"{st['policy']}{st['card']}{st['obs']}{rid}{seed}".encode()).digest()[:8], "big"))
    order = acts[:]; rg.shuffle(order)
    labels = {"A": order[0], "B": order[1]}
    opts = "Options:\n" + "\n".join(f"{k}: {v[a]}" for k, a in labels.items())
    prompt = (rule + opp + pay + hist + "Choose the option with the higher expected payoff. Return only A or B.\n" + opts + "\nAnswer:")
    best_label = None if st["best"] is None else next(k for k, a in labels.items() if a == st["best"])
    return dict(prompt=prompt, labels=labels, best_label=best_label, rendering=rid)

def dataset(seed=0, n_random=0):
    rows = []
    extra = random_policies(n_random, seed) if n_random else None
    for st in states(extra):
        if st["best"] is None:
            continue  # ties excluded from accuracy; kept in states() for transparency
        for rid in RENDERINGS:
            for sw in (0, 1):  # two label-order draws per state/rendering
                r = render(st, rid, seed=seed * 10 + sw, extra=extra)
                rows.append({**{k: st[k] for k in ("policy", "card", "obs", "q", "best", "advantage")}, **r, "swap": sw})
    return rows

if __name__ == "__main__":
    ds = dataset(n_random=60)
    print(len(ds), "rows;", len(states(random_policies(60))), "states")
    print(json.dumps(ds[0], indent=1)[:1500])
