import json,sys,math
from collections import defaultdict,Counter
d=json.load(open(sys.argv[1]))
def wilson(k,n,z=1.96):
    p=k/n; den=1+z*z/n; c=(p+z*z/2/n)/den; h=z*math.sqrt(p*(1-p)/n+z*z/4/n/n)/den; return round(c-h,2),round(c+h,2)
g=defaultdict(list)
for r in d['rows']: g[r['task']].append(r)
for t,rs in sorted(g.items()):
    for v in rs[0]['v']:
        def pick(r):
            o=r['v'][v]; l='A' if o['A']>o['B'] else 'B'
            if l==r['gold_letter']: return r['correct']
            return None
        # chosen action text: need other option; infer: gold text vs other
        other={}
        for r in rs:
            pass
        acts=Counter(); correct=0; byclass=defaultdict(lambda:[0,0]); letterA=0
        for r in rs:
            o=r['v'][v]; l='A' if o['A']>o['B'] else 'B'; ok=l==r['gold_letter']; correct+=ok
            byclass[r['correct']][0]+=ok; byclass[r['correct']][1]+=1; letterA+=l=='A'
            # letter-position accuracy
        gl=Counter(r['gold_letter'] for r in rs)
        pos={L:[sum(1 for r in rs if r['gold_letter']==L and (('A' if r['v'][v]['A']>r['v'][v]['B'] else 'B')==L)),gl[L]] for L in 'AB'}
        n=len(rs); recall={k:f"{a}/{b}" for k,(a,b) in byclass.items()}
        bal=sum(a/b for a,b in byclass.values())/len(byclass)
        print(f"{t:22s} {v:11s} acc {correct}/{n}={correct/n:.2f} CI{wilson(correct,n)} balacc {bal:.2f} majority-class {max(Counter(r['correct'] for r in rs).values())/n:.2f} recall {recall} gold-letter recall A {pos['A'][0]}/{pos['A'][1]} B {pos['B'][0]}/{pos['B'][1]} predA {letterA/n:.2f}")
