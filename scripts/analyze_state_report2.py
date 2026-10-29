import json, random, statistics as st
raw=json.load(open("results/v2_state_report_prosp2/raw/state_report_raw.json")); rows=raw["rows"]
print("n",len(rows),"elapsed",raw["metadata"]["elapsed_seconds"])
def b(r,k,i=-1): L=r["layers"]; return st.mean(L[l][k][i] for l in L)
for r in rows:
    r["ok"]=max(range(4),key=lambda i:r["cand_logits"][i])==r["report"]
    r["st_ok"]=max(range(4),key=lambda i:r["cand_logits"][i])==r["state"]
    for k in ("jK","jQ","jA","lK","lQ"): r[k]=b(r,k)
ns=[r for r in rows if r["report"]!=r["state"]]
print("GATE report acc s!=0: %.3f (n=%d)"%(st.mean(r["ok"] for r in ns),len(ns)))
for s in range(2):
    x=[r for r in rows if r["s"]==s]; print(" s(=moved report)=%d report acc %.3f  state-pick rate %.3f"%(s,st.mean(r["ok"] for r in x),st.mean(r["st_ok"] for r in x)))
print("top1 outside candidates:",sum(r["top1_any"] not in r["candidate_token_ids"] for r in rows))
def boot(sel,f,n=2000):
    cl={}
    for r in sel: cl.setdefault(tuple(r["perm"]),[]).append(f(r))
    keys=list(cl); rng=random.Random(1); ms=[]
    for _ in range(n):
        v=[x for k in (rng.choice(keys) for _ in keys) for x in cl[k]]; ms.append(st.mean(v))
    ms.sort(); return st.mean(f(r) for r in sel),ms[int(.025*n)],ms[int(.975*n)]
for name,sel in (("all s!=0",ns),("gate-correct s!=0",[r for r in ns if r["ok"]])):
    print("\n",name,len(sel))
    for lab,f in (("P1 J-lens Q-K",lambda r:r["jQ"]-r["jK"]),("P2 logit-lens Q-K",lambda r:r["lQ"]-r["lK"]),("J K",lambda r:r["jK"]),("J Q",lambda r:r["jQ"]),("J A(control)",lambda r:r["jA"]),("logit K",lambda r:r["lK"]),("logit Q",lambda r:r["lQ"])):
        m,lo,hi=boot(sel,f); print("  %-20s %+.3f [%+.3f, %+.3f]"%(lab,m,lo,hi))
s0=[r for r in rows if r["s"]==0]; print("\ns=0 sanity J K %.3f Q %.3f"%(st.mean(r["jK"] for r in s0),st.mean(r["jQ"] for r in s0)))
for s in (1,):
    x=[r for r in rows if r["s"]==s]; print("s=%d J K %+.3f Q %+.3f A %+.3f"%(s,st.mean(r["jK"] for r in x),st.mean(r["jQ"] for r in x),st.mean(r["jA"] for r in x)))
