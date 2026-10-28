import json,collections
d=json.load(open("results/x1_kuhn_gate_pilot2/raw/pilot2.json")); rows=d["rows"]; print("elapsed",d["elapsed"],d["info"])
for vn in rows[0]["v"]:
    ok=[];
    for r in rows:
        v=r["v"][vn]; pk="A" if v["A"]>v["B"] else "B"; r["_pk"+vn]=pk; ok.append(pk==r["best_label"])
    top=collections.Counter(r["v"][vn]["top1"].strip() for r in rows).most_common(3)
    byr={k:round(sum(o for o,r in zip(ok,rows) if r["rendering"]==k)/sum(1 for r in rows if r["rendering"]==k),3) for k in ("r0","r1","r2","r3")}
    byo={k:round(sum(o for o,r in zip(ok,rows) if r["obs"]==k)/sum(1 for r in rows if r["obs"]==k),3) for k in ("bet","check")}
    ab=sum(r["v"][vn]["top1"].strip() in("A","B") for r in rows)/len(rows)
    print(f"{vn}: acc {sum(ok)/len(ok):.3f} by rendering {byr} by obs {byo} top1 in A/B {ab:.2f} top1 {top}")
    # policy sensitivity within (card,obs): model action agrees with majority-best of cell
    cell=collections.defaultdict(list)
    for r in rows: cell[(r["card"],r["obs"])].append(r["labels"][r["_pk"+vn]])
    print("   model action distribution per cell:",{str(k):dict(collections.Counter(v)) for k,v in sorted(cell.items())})
