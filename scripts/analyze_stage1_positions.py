import json, collections, statistics as st
raw=json.load(open("results/v2_stage1_positions/raw/positions_raw.json"))
stored={r["condition_id"]:r for r in json.load(open("results/v2_stage1/raw/mechanistic_locked.json"))["rows"]}
rows=raw["rows"]; print("meta elapsed",raw["metadata"]["elapsed_seconds"],"gpu",raw["metadata"]["gpu_actual"])
# sanity: final position band jK vs stored band_K
diffs=[]
for r in rows:
    L=r["layers"]; bk=st.mean(L[l]["jK"][-1] for l in L)
    diffs.append(abs(bk-stored[r["condition_id"]]["band_K"]))
print("sanity max|final-pos band K - stored|",max(diffs),"mean",st.mean(diffs))
def band(r,key,off):
    L=r["layers"]; return st.mean(L[l][key][off] for l in L)
for sub in ("1A","1B"):
  for pol in ("T","M"):
    sel=[r for r in rows if r["substage"]==sub and r["policy_id"]==pol]
    print(f"\n{sub} policy {pol} n={len(sel)}  band mean J-lens K/Q/M by offset from final token (0=final)")
    for off in range(0,13):
        o=-1-off
        k=st.mean(band(r,"jK",o) for r in sel);q=st.mean(band(r,"jQ",o) for r in sel);m=st.mean(band(r,"jM",o) for r in sel)
        print(f"  -{off:2d}  K {k:+.3f}  Q {q:+.3f}  M {m:+.3f}")
# state-token positions: any position where K is largest under policy M (state not report)
print("\nlargest |K| positions, policy M, 1A (offset, K, M):")
sel=[r for r in rows if r["substage"]=="1A" and r["policy_id"]=="M"]
n=min(len(r["prompt_token_ids"]) for r in sel)
res=[]
for off in range(0,n):
    o=-1-off
    res.append((off,st.mean(band(r,"jK",o) for r in sel),st.mean(band(r,"jM",o) for r in sel)))
for x in sorted(res,key=lambda t:-t[1])[:8]: print("  off %d K %+.3f M %+.3f"%x)
