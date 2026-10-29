import json,re,math
from collections import defaultdict
def wil(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/2/n)/d; h=z*math.sqrt(p*(1-p)/n+z*z/4/n/n)/d; return round(100*(c-h)),round(100*(c+h))
R=json.load(open('results/y6_native_generation/raw/y6.json'))['rows']
def pw(g,labels):
    t=g.strip().lower()
    for l in labels:
        if re.match(r'\W*'+l.lower()+r'\b',t): return l
    return None
def pc(g,labels):
    m=re.findall(r'final answer:\s*\**\s*([A-Za-z]+)',g,re.I)
    for l in labels:
        if m and m[-1].lower()==l.lower(): return l
    return None
out=defaultdict(lambda:defaultdict(lambda:[0,0,0]))
for r in R:
    for mode,f in (('free_word',pw),('cot',pc)):
        p=f(r['gen'][mode],r['labels']); c=out[(r['task'],mode)]
        c['all'][1]+=1; c['all'][0]+= p==r['correct']; c['all'][2]+= p is None
        c['gold='+r['correct']][1]+=1; c['gold='+r['correct']][0]+= p==r['correct']
        c['pred_'+str(p)][1]+=1
for k in sorted(out):
    k_,n,u=out[k]['all'][0],out[k]['all'][1],out[k]['all'][2]; lo,hi=wil(k_,n)
    rec={x:f"{v[0]}/{v[1]}" for x,v in out[k].items() if x.startswith('gold=')}
    preds={x:v[1] for x,v in out[k].items() if x.startswith('pred_')}
    print(k,f"{k_}/{n}={100*k_/n:.0f}% [{lo},{hi}] unparsed={u}",rec,preds)
