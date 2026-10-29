"""Descriptive game-level paired analysis of Y11 and Y12 (worded vs opaque tool names).
No compute: reads raw generations. Run from results/: python3 pooled_game_level.py
A row is correct if the first <function=NAME> call maps (via row['names']) to expected_action.
"""
import json, re, random
from collections import defaultdict
random.seed(20261029)
def load(p):
    return json.load(open(p))["rows"]
def correct(r):
    m = re.search(r"<function=([^>\s]+)>", r["gen"])
    if not m: return 0
    inv = {v: k for k, v in r["names"].items()}
    return int(inv.get(m.group(1)) == r["expected_action"])
def per_game(rows):
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["fmt"] not in ("tool_opaque", "tool_assertion_name"): continue  # worded-name vs opaque only
        g[r["base"]]["opaque" if r["fmt"] == "tool_opaque" else "name"].append(correct(r))
    return {k: 100*(sum(v["name"])/len(v["name"]) - sum(v["opaque"])/len(v["opaque"])) for k, v in g.items()}
def boot(strata, B=10000):
    out = []
    for _ in range(B):
        s = 0; n = 0
        for st in strata:
            x = random.choices(st, k=len(st)); s += sum(x); n += len(x)
        out.append(s/n)
    out.sort(); return out[int(.025*B)], out[int(.975*B)]
y11 = per_game(load("y11_tool_uniform/raw/y11.json"))
y12 = per_game(load("y12_tool_balanced/raw/y12.json"))
a, b = list(y11.values()), list(y12.values())
print("games", len(a), len(b), "overlap", len(set(y11) & set(y12)))
for n, x in (("Y11", a), ("Y12", b)):
    print(n, "game-level mean diff %.2f CI %.2f %.2f" % ((sum(x)/len(x),) + boot([x])))
allg = a + b
print("Concatenated %d games, equal game weight, study-stratified bootstrap: %.2f CI %.2f %.2f" % ((len(allg), sum(allg)/len(allg)) + boot([a, b])))
sd = lambda x: (sum((v-sum(x)/len(x))**2 for v in x)/(len(x)-1))**.5
print("sd of per-game diffs Y11 %.1f Y12 %.1f" % (sd(a), sd(b)))
print("games with name<opaque / equal / name>opaque:", sum(v<0 for v in allg), sum(v==0 for v in allg), sum(v>0 for v in allg))
