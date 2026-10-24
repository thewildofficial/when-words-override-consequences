import json,sys,random,collections
def load(k): return json.load(open(f"/tmp/hdc/res_{k}.json"))["rows"]
def acc(rs): return sum(r["correct"] for r in rs)/len(rs) if rs else float("nan")
def boot(rs,n=2000,seed=0):
    by=collections.defaultdict(list)
    for r in rs: by[r["base"]].append(r["correct"])
    bs=list(by); rg=random.Random(seed); v=[]
    for _ in range(n):
        s=[rg.choice(bs) for _ in bs]; xs=[c for b in s for c in by[b]]; v.append(sum(xs)/len(xs))
    v.sort(); return v[int(.025*n)],v[int(.975*n)]
for k in sys.argv[1:]:
    R=load(k); print("=====",k,len(R))
    print("cell: surface/incentive/htype/K -> acc [95% base-bootstrap CI] n")
    cells=sorted({(r["surface"],r["incentive"],r["htype"],r["K"]) for r in R})
    for c in cells:
        rs=[r for r in R if (r["surface"],r["incentive"],r["htype"],r["K"])==c]
        lo,hi=boot(rs,500); print(c,f"{acc(rs):.3f} [{lo:.2f},{hi:.2f}] n={len(rs)}")
    # position bias
    print("pred A rate:",sum(r["pred"]=="A" for r in R)/len(R),"expected A rate:",sum(r["expected"]=="A" for r in R)/len(R))
