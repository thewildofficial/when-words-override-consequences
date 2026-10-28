"""CPU reanalysis of stored Stage 1 locked per-layer readouts (no new compute).
K = true-state evidence, M = report-target evidence (definitions per docs/v2/stage1-analysis-plan.md; values taken as stored)."""
import json, statistics as S, collections, sys
rows = json.load(open("results/v2_stage1/raw/mechanistic_locked.json"))["rows"]
print("substage policy n | band mean K M (jlens) | layer-wise mean K/M jlens ; logit-lens")
g = collections.defaultdict(list)
for r in rows: g[(r["substage"], r["policy_id"])].append(r)
for k, v in sorted(g.items()):
    print(k, len(v), "band K %.3f M %.3f outK %.3f" % (S.mean(x["band_K"] for x in v), S.mean(x["band_M"] for x in v), S.mean(x["output_K"] for x in v)))
    layers = sorted({ls["layer"] for x in v for ls in x["layer_scores"]})
    for L in layers:
        ls = [next(l for l in x["layer_scores"] if l["layer"] == L) for x in v]
        print("   L%d jlens K %.3f M %.3f | logitlens K %.3f M %.3f" % (L, S.mean(l["K"] for l in ls), S.mean(l["M"] for l in ls), S.mean(l["logit_lens_K"] for l in ls), S.mean(l["logit_lens_M"] for l in ls)))
# fraction of rows with positive K by policy and family
print("family-level band K by policy")
for k, v in sorted(g.items()):
    fam = collections.defaultdict(list)
    for x in v: fam[x["family"]].append(x["band_K"])
    print(k, {f: round(S.mean(a), 3) for f, a in sorted(fam.items())})
