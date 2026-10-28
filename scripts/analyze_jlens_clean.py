import json,statistics as st,collections
d=json.load(open("results/v2_jlens_recon/raw/clean_raw.json")); print("elapsed",d["metadata"]["elapsed_seconds"])
g=collections.defaultdict(list)
for r in d["rows"]:
    if r["baseline_correct"]: g[r["configuration_id"]].append(r)
for k,v in sorted(g.items()):
    t=sum(r["target_top1"] for r in v); ok=sum(1 for r in v if r["target_top1"] and r["output_kl_nats"]<1)
    print(f"{k:50s} n={len(v)} top1 {t}/{len(v)} top1&KL<1 {ok} KL med {st.median(r['output_kl_nats'] for r in v):.2f} dRMS med {st.median(r['mean_delta_rms_ratio'] for r in v):.3f} gain med {st.median(r['target_logodds_gain'] for r in v):.2f}")
