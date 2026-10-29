import json,numpy as np,sys
rows=json.load(open('results/y3_which_step/y3_rows.json')); Z=np.load('/tmp/y4.npz'); ab=Z['ab']
layers=sorted(int(k[1:]) for k in Z.files if k.startswith('L'))
cond=np.array([r['task'].split('_')[0] for r in rows]); item=np.array([int(r['id'].rsplit('-',1)[1]) for r in rows])
y=np.array([1.0 if r['correct']=='Call' else -1.0 for r in rows]); m=rows[0]['meta']
meta=[r['meta'] for r in rows]; margin=np.array([x['eq']-x['thr'] for x in meta]); outs=np.array([x['outs'] for x in meta])
raw=np.array([[x['outs'],x['unseen'],x['P'],x['B'],np.log(x['P']),np.log(x['B'])] for x in meta]); comp=np.array([[x['eq'],x['thr']] for x in meta])
beh=np.array([ (a-b) if r['gold_letter']=='A' else (b-a) for (a,b),r in zip(ab,rows)])  # logit support for the gold answer
def kr_fit_pred(Xtr,ytr,Xte,alphas=(1e-1,1,10,100,1000,1e4)):
    mu=Xtr.mean(0); sd=Xtr.std(0)+1e-6; A=(Xtr-mu)/sd; B=(Xte-mu)/sd
    # inner 3-fold to pick alpha
    n=len(A); idx=np.arange(n); best=None
    K=A@A.T/A.shape[1]
    for al in alphas:
        err=0
        for f in range(3):
            te=idx%3==f; tr=~te
            c=np.linalg.solve(K[np.ix_(tr,tr)]+al*np.eye(tr.sum())/A.shape[1]*A.shape[1]/ A.shape[1],ytr[tr]-ytr[tr].mean())
            p=K[np.ix_(te,tr)]@c+ytr[tr].mean(); err+=((p-ytr[te])**2).sum()
        if best is None or err<best[0]: best=(err,al)
    al=best[1]; c=np.linalg.solve(K+al*np.eye(n),ytr-ytr.mean())
    return (B@A.T/A.shape[1])@c+ytr.mean(), al
def cv(X,target,mask,groups,k=5,seed=0):
    ids=np.where(mask)[0]; g=np.unique(groups[ids]); rng=np.random.RandomState(seed); rng.shuffle(g); folds=np.array_split(g,k); pred=np.zeros(len(ids))
    for f in folds:
        te=np.isin(groups[ids],f); tr=~te
        p,_=kr_fit_pred(X[ids[tr]],target[ids[tr]],X[ids[te]]); pred[te]=p
    return ids,pred
def acc(ids,pred): return float(np.mean(np.sign(pred)==np.sign(y[ids])))

def auc(score,lab):
    pos=score[lab>0]; neg=score[lab<=0]
    if len(pos)==0 or len(neg)==0: return float('nan')
    return float(((pos[:,None]>neg[None,:]).mean()+0.5*(pos[:,None]==neg[None,:]).mean()))
letter=np.array([1.0 if r['gold_letter']=='A' else -1.0 for r in rows])
chose_A=np.array([1.0 if a>b else -1.0 for a,b in ab])
rng=np.random.RandomState(1)
Ls=[16,24,32,40,48,56,64]
print('AUC (grouped 5-fold). targets: callfold=gold Call vs Fold, letter=gold letter A vs B, chosenA=model chose A')
for c in ['D1','D2','D3','D4']:
    mask=cond==c
    for gname,groups in (('byitem',item),('byouts',outs)):
        row={}
        for tname,t in (('callfold',y),('letter',letter),('chosenA',chose_A)):
            ids,p=cv(raw,t,mask,groups); base=auc(p,t[ids])
            vals={}
            for L in Ls:
                X=Z[f'L{L}'].astype('float32'); ids,p=cv(X,t,mask,groups); vals[L]=round(auc(p,t[ids]),2)
            ts=t.copy(); ts[mask]=rng.permutation(t[mask]); ids,p=cv(Z['L40'].astype('float32'),ts,mask,groups)
            row[tname]={'rawnum_AUC':round(base,2),'layers':vals,'shuffled_L40':round(auc(p,ts[ids]),2)}
        print(c,gname,json.dumps(row),flush=True)
