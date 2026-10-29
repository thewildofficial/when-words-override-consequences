"""Top-1 and mean log-odds gain vs output KL, per layer set and alpha. Eligible = baseline_correct. No matching claim; descriptive curves."""
import json, sys, statistics as S, collections
rows = [r for r in json.load(open(sys.argv[1]))["rows"] if r["baseline_correct"]]
g = collections.defaultdict(list)
for r in rows: g[(r["control"], tuple(r["layers"]), r["alpha"])].append(r)
print("control layers alpha n top1 medKL meanKL gain deltaRMS")
for k in sorted(g, key=str):
    v = g[k]
    print(k[0][:5], list(k[1]) if len(k[1]) < 4 else f"{k[1][0]}-{k[1][-1]}({len(k[1])})", k[2], len(v), sum(bool(x["target_top1"]) for x in v),
          round(S.median(x["output_kl_nats"] for x in v), 3), round(S.mean(x["output_kl_nats"] for x in v), 3),
          round(S.mean(x["target_logodds_gain"] for x in v), 2), round(S.mean(x["mean_delta_rms_ratio"] for x in v), 3))
