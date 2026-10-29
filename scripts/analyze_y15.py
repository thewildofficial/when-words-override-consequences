import json, sys, random, math
from collections import defaultdict
d = json.load(open(sys.argv[1])); rows = d["rows"]
for r in rows:
    r["pred"] = "A" if r["score"]["A"] >= r["score"]["B"] else "B"; r["ok"] = r["pred"] == r["answer"]
def wil(k, n):
    p = k / n; z = 1.96; c = (p + z*z/(2*n)) / (1 + z*z/n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n); return c-h, c+h
def acc(rs): return sum(r["ok"] for r in rs) / len(rs)
cells = defaultdict(list)
for r in rows: cells[(r["surface"], r["tok"], r["hops"], r["swaps"])].append(r)
print("meta", d["meta"]["model_revision_resolved"], d["meta"]["gpu_actual"], "elapsed_s", round(d["elapsed"]), "n", len(rows))
print("surface tok h s n acc [wilson] literal_acc predA_rate meanTokens")
for k in sorted(cells):
    rs = cells[k]; n = len(rs); kk = sum(r["ok"] for r in rs); lo, hi = wil(kk, n)
    print(*k, n, f"{kk/n:.3f}", f"[{lo:.2f},{hi:.2f}]", f"lit={sum(r['literal_answer']==r['answer'] for r in rs)/n:.2f}", f"A={sum(r['pred']=='A' for r in rs)/n:.2f}", f"tok={sum(r['n_tokens'] for r in rs)/n:.0f}")
games = sorted({r["game"] for r in rows}); rng = random.Random(7)
def diff(f1, f2, B=2000):
    def est(gs):
        a = [r for g in gs for r in f1 if r["game"] == g]; b = [r for g in gs for r in f2 if r["game"] == g]; return acc(a) - acc(b)
    e = est(games); bs = sorted(est([rng.choice(games) for _ in games]) for _ in range(B)); return e, bs[int(.025*B)], bs[int(.975*B)]
def c(s, t, h, k): return cells[(s, t, h, k)]
print("\nDIFFS (game-cluster bootstrap 95%), points")
for t in ("opaque", "meaningful"):
    for name, a, b in [("P1 h3 s1-s3", c("table",t,3,1), c("table",t,3,3)), ("P1 h3 s0-s2", c("table",t,3,0), c("table",t,3,2)),
                       ("P2 s1 h1-h3", c("table",t,1,1), c("table",t,3,1)), ("P2 s1 h3-h5", c("table",t,3,1), c("table",t,5,1)),
                       ("s3 h3-h5", c("table",t,3,3), c("table",t,5,3)),
                       ("P4 s1 table-lex h3", c("table",t,3,1), c("lex",t,3,1)), ("P4 s3 table-lex h3", c("table",t,3,3), c("lex",t,3,3))]:
        e, lo, hi = diff(a, b); print(t, name, f"{100*e:+.1f} [{100*lo:+.1f},{100*hi:+.1f}]")
print("P5 meaningful-opaque by table cell")
for (h, s) in [(1,0),(1,1),(3,0),(3,1),(3,2),(3,3),(5,1),(5,3)]:
    e, lo, hi = diff(c("table","meaningful",h,s), c("table","opaque",h,s)); print(h, s, f"{100*e:+.1f} [{100*lo:+.1f},{100*hi:+.1f}]")
# per-class recall
for t in ("opaque", "meaningful"):
    for ans in "AB":
        rs = [r for r in rows if r["tok"] == t and r["answer"] == ans]; print("recall", t, ans, f"{acc(rs):.3f}", len(rs))
