"""H1 fine grid. Per (layer, control): fraction of eligible prompts with top-1 AND KL<1 at ANY grid alpha (best-case per prompt),
minimal alpha achieving top-1 per prompt, and KL at that alpha. Success mark 20%, failure mark 5% (reported as three regions:
>=20% success; <=5% failure; between = inconclusive). Scoped to model, category and grid."""
import json, sys, collections, statistics as S
rows = [r for r in json.load(open(sys.argv[1]))["rows"] if r["baseline_correct"]]
key = lambda r: (r["scenario_id"], r["target_answer"])
by = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows: by[(r["control"], tuple(r["layers"]))][key(r)].append(r)
for k in sorted(by, key=str):
    items = by[k]; n = len(items)
    valid = 0; flips = 0; minkl = []
    for it, rs in items.items():
        rs.sort(key=lambda r: r["alpha"])
        if any(r["target_top1"] for r in rs): flips += 1
        if any(r["target_top1"] and r["output_kl_nats"] < 1 for r in rs): valid += 1
        f = next((r for r in rs if r["target_top1"]), None)
        if f: minkl.append(f["output_kl_nats"])
    frac = valid / n
    region = "success(>=20%)" if frac >= .2 else "failure(<=5%)" if frac <= .05 else "inconclusive(5-20%)"
    print(k[0][:5], list(k[1]), "prompts", n, "ever_top1", flips, "top1&KL<1 at some alpha", valid, f"{frac:.2f}", region,
          "KL at first flip median", round(S.median(minkl), 2) if minkl else None, "min", round(min(minkl), 2) if minkl else None)
g = collections.defaultdict(list)
for r in rows: g[(r["control"], tuple(r["layers"]), r["alpha"])].append(r)
for k in sorted(g, key=str):
    v = g[k]; print(k[0][:5], list(k[1]), k[2], len(v), "top1", sum(bool(x["target_top1"]) for x in v), "medKL", round(S.median(x["output_kl_nats"] for x in v), 3), "KL<1", sum(x["output_kl_nats"] < 1 for x in v))
