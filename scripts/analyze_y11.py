import json,re,random
from collections import Counter,defaultdict
P=lambda g:(lambda m:m.group(1).strip() if m else None)(re.search(r"<function=([^>\n]+)>",g))
d=json.load(open('results/y11_tool_uniform/raw/y11.json')); R=d['rows']; print('cap',d['cap'],'rows',len(R))
man=[];per=defaultdict(dict)
for i,r in enumerate(R):
    nm=P(r['gen']); inv={v:k for k,v in r['names'].items()}
    o='nocall' if nm is None else 'badname' if nm not in inv else ('right' if r['response_per_action'][inv[nm]]==r['target'] else 'wrong')
    trunc=(o=='nocall' and r['n_new_tokens']>=400)
    man.append(dict(id=i,fmt=r['fmt'],base=r['base'],history=r['history'],mapping=r['mapping'],tool=nm,outcome=o,reward=10 if o=='right' else 0,n_new_tokens=r['n_new_tokens'],cap_hit_nocall=trunc,first_listed_correct=None))
    per[r['base']][(r['fmt'],r['history'],r['mapping'])]=o=='right'
json.dump(man,open('results/y11_tool_uniform/y11_manifest.json','w'))
for f in ('tool_opaque','tool_assertion_name','tool_assertion_desc','tool_desc_named'):
    m=[x for x in man if x['fmt']==f]; c=Counter(x['outcome'] for x in m)
    cells=[sum(x['reward']==10 for x in m if x['history']==h and x['mapping']==mp) for h in('none','redundant') for mp in('prose','table')]
    print(f,len(m),dict(c),'acc',round(c['right']/len(m),3),'cells(none/prose,none/table,red/prose,red/table)',cells,'capHit nocall',sum(x['cap_hit_nocall'] for x in m),'median ntok',sorted(x['n_new_tokens'] for x in m)[len(m)//2])
games=list(per); acc=lambda gs,f:(lambda v:sum(v)/len(v))([x for g in gs for k,x in per[g].items() if k[0]==f])
random.seed(1)
def diff(a,b,sub=None):
    def a_(gs,f):
        v=[x for g in gs for k,x in per[g].items() if k[0]==f and (sub is None or k[2]==sub)]; return sum(v)/len(v)
    pt=a_(games,a)-a_(games,b); ds=sorted(a_(gs,a)-a_(gs,b) for gs in ([random.choice(games) for _ in games] for _ in range(2000)))
    print(a,'-',b,sub or 'all',f"{100*pt:.1f} pts [{100*ds[50]:.1f},{100*ds[1949]:.1f}]")
diff('tool_assertion_name','tool_opaque');diff('tool_assertion_desc','tool_opaque');diff('tool_desc_named','tool_opaque')
diff('tool_desc_named','tool_assertion_desc');diff('tool_desc_named','tool_assertion_desc','prose');diff('tool_desc_named','tool_assertion_desc','table')
# order/baselines
for f in ('tool_assertion_name','tool_assertion_desc','tool_desc_named','tool_opaque'):
    rr=[r for r in R if r['fmt']==f]; print(f,'expected_action=A share',round(sum(r['expected_action']=='A' for r in rr)/len(rr),2))
