import json,sys,collections
sys.path.insert(0,'/tmp/hdc'); import hdc
k=sys.argv[1]
R=json.load(open(f'/tmp/hdc/res_{k}.json'))['rows']; G=hdc.rows()
assert len(R)==len(G)
for r,g in zip(R,G):
    assert r['prompt']==g['prompt']; r.update({x:g[x] for x in ('truth_label','goal_label','last_label')})
agg=collections.defaultdict(lambda:[0,0,0,0,0])
for r in R:
    if r['surface']!='assertion': continue
    a=agg[(r['incentive'],r['htype'])]
    a[0]+=1; a[1]+=r['correct']; a[2]+=r['pred']==r['truth_label']; a[3]+=r['pred']==r['goal_label']
    if r['last_label']: a[4]+= r['pred']==r['last_label']
print("assertion cells pooled over K: inc,hist | n acc P(truthful claim) P(claim names goal concept) P(repeat last action)")
for c,a in sorted(agg.items()): print(c,a[0],*(f"{x/a[0]:.2f}" for x in a[1:]))
