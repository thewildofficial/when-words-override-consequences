import json,collections,sys
sys.path.insert(0,"x1"); import kuhn_ev
d=json.load(open("results/x1_kuhn_gate_pilot/raw/pilot.json")); rows=d["rows"]; print("n",len(rows),"elapsed",d["elapsed"])
for r in rows:
    r["pick"]="A" if r["logit_A"]>r["logit_B"] else "B"; r["ok"]=r["pick"]==r["best_label"]
    r["pick_action"]=r["labels"][r["pick"]]; r["best_action"]=r["labels"][r["best_label"]]
print("top1 in A/B:",sum(r["top1_tok"].strip() in("A","B") for r in rows)/len(rows))
acc=lambda x:sum(r["ok"] for r in x)/max(1,len(x))
print("overall acc %.3f"%acc(rows)); print("pick A rate %.2f  best==A rate %.2f"%(sum(r["pick"]=="A" for r in rows)/len(rows),sum(r["best_label"]=="A" for r in rows)/len(rows)))
for k in ("rendering","obs","card"):
    print(k,{v:(round(acc([r for r in rows if r[k]==v]),3),len([r for r in rows if r[k]==v])) for v in sorted({r[k] for r in rows})})
for lo,hi in ((0,.1),(.1,.3),(.3,.7),(.7,9)):
    x=[r for r in rows if lo<=r["advantage"]<hi]; print("advantage [%.1f,%.1f) n=%d acc %.3f"%(lo,hi,len(x),acc(x)))
# public-history baseline: majority best action by (card,obs) over states in pool (all policies), the composition baseline
ds=kuhn_ev.dataset(seed=7,n_random=40); cnt=collections.defaultdict(collections.Counter)
for r in ds: cnt[(r["card"],r["obs"])][r["labels"][r["best_label"]]]+=1
maj={k:c.most_common(1)[0][0] for k,c in cnt.items()}
print("composition baseline (majority best action by card,obs) acc %.3f"%(sum(maj[(r["card"],r["obs"])]==r["best_action"] for r in rows)/len(rows)))
print("model action by (card,obs) vs best:")
for k in sorted(maj): 
    x=[r for r in rows if (r["card"],r["obs"])==k]; print(k,"n",len(x),"acc %.2f"%acc(x),"model-picks",dict(collections.Counter(r["pick_action"] for r in x)),"best",dict(collections.Counter(r["best_action"] for r in x)))
