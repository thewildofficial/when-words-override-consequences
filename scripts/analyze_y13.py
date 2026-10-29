import json,re,random
from collections import Counter,defaultdict
P=lambda g:(lambda m:m.group(1).strip() if m else None)(re.search(r"<function=([^>\n]+)>",g))
d=json.load(open('results/y13_binding/raw/y13.json')); R=d['rows']; print('cap',d['cap'],'rows',len(R))
man=[];per=defaultdict(dict);pair=defaultdict(dict)
for i,r in enumerate(R):
    nm=P(r['gen']); inv={v:k for k,v in r['names'].items()}
    order=["A","B"] if r['swap']==0 else ["B","A"]
    ch=inv.get(nm); pos=order.index(ch) if ch else None
    o='nocall' if nm is None else 'badname' if ch is None else ('right' if r['response_per_action'][ch]==r['target'] else 'wrong')
    man.append(dict(id=i,fmt=r['fmt'],base=r['base'],history=r['history'],mapping=r['mapping'],swap=r['swap'],expected_pos=r['expected_pos'],chosen_pos=pos,tool=nm,outcome=o,reward=10 if o=='right' else 0,n_new_tokens=r['n_new_tokens'],cap_hit_nocall=(o=='nocall' and r['n_new_tokens']>=400)))
    per[r['base']][(r['fmt'],r['history'],r['mapping'],r['swap'])]=o=='right'
    pair[(r['fmt'],r['base'],r['history'],r['mapping'])][r['swap']]=o=='right'
json.dump(man,open('results/y13_binding/y13_manifest.json','w'))
for f in ('tool_desc','tool_desc_restate','tool_desc_generic','tool_optok_worded'):
    m=[x for x in man if x['fmt']==f]; c=Counter(x['outcome'] for x in m)
    conf=Counter((x['expected_pos'],x['chosen_pos']) for x in m)
    first=sum(x['chosen_pos']==0 for x in m)
    pc=Counter((a,b) for (ff,*_),v in pair.items() if ff==f for a,b in [(v[0],v[1])] for _ in [0])
    pp=Counter((v[0],v[1]) for k,v in pair.items() if k[0]==f)
    print(f,len(m),dict(c),'acc',round(c['right']/len(m),3),'by swap',[sum(x['reward']==10 for x in m if x['swap']==s) for s in (0,1)],'| chose first',first,'second',sum(x['chosen_pos']==1 for x in m),'| confusion (expected_pos,chosen_pos)',dict(conf),'| swap-pairs (right@0,right@1)',dict(pp),'capHit',sum(x['cap_hit_nocall'] for x in m),'median ntok',sorted(x['n_new_tokens'] for x in m)[len(m)//2])
games=list(per); 
def a_(gs,f,sub=None):
    v=[x for g in gs for k,x in per[g].items() if k[0]==f and (sub is None or k[2]==sub)]; return sum(v)/len(v)
random.seed(2)
def diff(a,b,sub=None):
    pt=a_(games,a,sub)-a_(games,b,sub); ds=sorted(a_(gs,a,sub)-a_(gs,b,sub) for gs in ([random.choice(games) for _ in games] for _ in range(2000)))
    print(a,'-',b,sub or 'all',f"{100*pt:.1f} pts [{100*ds[50]:.1f},{100*ds[1949]:.1f}] (cluster=base game, 2000 resamples)")
diff('tool_desc_restate','tool_desc');diff('tool_desc_generic','tool_desc');diff('tool_desc_restate','tool_desc_generic');diff('tool_optok_worded','tool_desc')
