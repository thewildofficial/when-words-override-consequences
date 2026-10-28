"""KL-matching analysis (procedure fixed before the run).

For each (trial_id, layers) the semantic row at alpha in {1.0, 1.5} is paired with, per control family
(unrelated_same_category; norm_matched_random_basis seeds 0-2), the control alpha-grid row whose output KL is
closest in log space. A pair is ACCEPTED if KL_ctrl/KL_sem is within [0.8, 1.25]. Only accepted pairs enter the
comparison. We report the accepted fraction, the median KL ratio, top-1 and mean target log-odds gain for semantic
vs control on the same accepted (trial, layer, alpha) set, and counts of discordant pairs. Eligible = baseline_correct.
"""
import json, math, statistics as S, sys, collections
from pathlib import Path
raw = json.loads(Path(sys.argv[1]).read_text())["rows"]
rows = [r for r in raw if r["baseline_correct"]]
def fam(r):
    c = r["control"]; return c if c != "norm_matched_random_basis" else f"random_s{r.get('random_seed', 0) if 'random_seed' in r else r['configuration_id'][4]}"
idx = collections.defaultdict(list)
sem = []
for r in raw:
    if not r["baseline_correct"]: continue
    key = (r["scenario_id"], r["target_answer"], tuple(r["layers"]))
    if r["control"] == "semantic": sem.append((key, r))
    else:
        f = r["control"] if r["control"] != "norm_matched_random_basis" else "random_s" + r["configuration_id"].split("_")[0][4:]
        idx[(f, key)].append(r)
out = {}
fams = sorted({f for f, _ in idx})
for f in fams:
    acc = []; total = 0
    for key, s in sem:
        total += 1
        cands = idx.get((f, key), [])
        if not cands: continue
        c = min(cands, key=lambda c: abs(math.log(max(c["output_kl_nats"], 1e-9) / max(s["output_kl_nats"], 1e-9))))
        ratio = c["output_kl_nats"] / max(s["output_kl_nats"], 1e-9)
        if 0.8 <= ratio <= 1.25: acc.append((s, c, ratio))
    if not acc: out[f] = {"accepted": 0, "total": total}; continue
    b = sum(s["target_top1"] and not c["target_top1"] for s, c, _ in acc)
    d = sum(c["target_top1"] and not s["target_top1"] for s, c, _ in acc)
    out[f] = {"total_sem_rows": total, "accepted": len(acc), "median_kl_ratio": round(S.median(r for _, _, r in acc), 3),
              "sem_top1": sum(bool(s["target_top1"]) for s, _, _ in acc), "ctrl_top1": sum(bool(c["target_top1"]) for _, c, _ in acc),
              "sem_mean_gain": round(S.mean(s["target_logodds_gain"] for s, _, _ in acc), 2), "ctrl_mean_gain": round(S.mean(c["target_logodds_gain"] for _, c, _ in acc), 2),
              "sem_only_top1": b, "ctrl_only_top1": d}
print(json.dumps(out, indent=1))
# grid summary: top1 by control family and alpha, KL median
g = collections.defaultdict(list)
for r in rows:
    g[(r["control"], tuple(r["layers"]), r["alpha"])].append(r)
for k in sorted(g, key=str):
    v = g[k]; print(k, len(v), "top1", sum(bool(x["target_top1"]) for x in v), "medKL", round(S.median(x["output_kl_nats"] for x in v), 3), "gain", round(S.mean(x["target_logodds_gain"] for x in v), 2))
