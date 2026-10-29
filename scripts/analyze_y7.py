import json, re, math, sys
from collections import defaultdict
def wil(k,n,z=1.96):
    if n==0: return (0,0)
    p=k/n; d=1+z*z/n; c=(p+z*z/2/n)/d; h=z*math.sqrt(p*(1-p)/n+z*z/4/n/n)/d; return (round(100*(c-h)),round(100*(c+h)))
def fmt(k,n): lo,hi=wil(k,n); return f"{k}/{n}={100*k/n:.0f}% [{lo},{hi}]"
def y7(path):
    R=json.load(open(path))["rows"]; cell=defaultdict(lambda:[0,0]); pair=defaultdict(dict)
    for r in R:
        s=r["score"]
        if r["fmt"]=="ab": ch="A" if s["A"]>s["B"] else "B"; ok=ch==r["expected_action"]
        else: ch="A" if s["A"]>s["B"] else "B"; ok=ch==r["expected_action"]
        key=(r["fmt"] if r["fmt"]!="ab" else "ab_"+r["surface"],r["history"],r["mapping"]); cell[key][0]+=ok; cell[key][1]+=1
        pair[(r["base"],r["history"],r["mapping"])][key[0]]=ok
    print("Y7 name-logprob/A-B forced choice (format-transfer evidence)")
    for k in sorted(cell): print(k, fmt(*cell[k]))
    agg=defaultdict(lambda:[0,0])
    for k,(a,n) in cell.items(): agg[k[0]][0]+=a; agg[k[0]][1]+=n
    for k in sorted(agg): print("ALL",k,fmt(*agg[k]))
    # baselines
    g=[r for r in R if r["fmt"]!="ab"]
    print("always-A on tool rows:", fmt(sum(r["expected_action"]=="A" for r in g),len(g)))
    sh=[r for r in g if len(r["names"]["A"])!=len(r["names"]["B"])]
    print("always-shorter-name:", fmt(sum(len(r["names"][r["expected_action"]])<len(r["names"]["A" if r["expected_action"]=="B" else "B"]) for r in sh),len(sh)))
    return R
def parse(gen):
    m=re.search(r"<function=([^>\n]+)>",gen); return m.group(1).strip() if m else None
def y8(path):
    R=json.load(open(path))["rows"]; cell=defaultdict(lambda:[0,0]); inval=defaultdict(int); pair=defaultdict(dict)
    for r in R:
        nm=parse(r["gen"]); rev=None
        inv={v:k for k,v in r["names"].items()}
        if nm in inv: rev=10 if r["response_per_action"][inv[nm]]==r["target"] else 0
        else: inval[r["fmt"]]+=1; rev=0
        key=(r["fmt"],r["history"],r["mapping"]); cell[key][0]+=rev==10; cell[key][1]+=1
        pair[(r["base"],r["history"],r["mapping"])][r["fmt"]]=rev==10
    print("Y8 native generation + mock executor (reward 10=target reached; invalid calls count 0)")
    for k in sorted(cell): print(k, fmt(*cell[k]))
    agg=defaultdict(lambda:[0,0])
    for k,(a,n) in cell.items(): agg[k[0]][0]+=a; agg[k[0]][1]+=n
    for k in sorted(agg): print("ALL",k,fmt(*agg[k]),"invalid:",inval[k])
    for a in ("tool_assertion_name","tool_assertion_desc"):
        d=[(v[a]-v["tool_opaque"]) for v in pair.values() if a in v and "tool_opaque" in v]
        print("paired",a,"-opaque mean diff",round(sum(d)/len(d),3),"n",len(d))
if __name__=="__main__":
    if sys.argv[1]=="y7": y7(sys.argv[2])
    else: y8(sys.argv[2])
