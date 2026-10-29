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
out=[]
def rep(name,fn):
    r=fn(); out.append((name,r)); print(name,r,flush=True)
rng=np.random.RandomState(1)
for c in ['D1','D2','D3','D4']:
    mask=cond==c
    for gname,groups in (('byitem',item),('byouts',outs)):
        r={'base_raw':None}
        ids,p=cv(raw,y,mask,groups); r['raw_numbers_only']=round(acc(ids,p),3)
        ids,p=cv(comp,y,mask,groups); r['computed_eq_thr_ceiling']=round(acc(ids,p),3)
        ys=y.copy(); ys[mask]=rng.permutation(y[mask]); ids,p=cv(raw,ys,mask,groups)
        res={}
        for L in layers[::2]:
            X=Z[f'L{L}'].astype('float32')
            ids,p=cv(X,y,mask,groups); a=acc(ids,p)
            ysh=y.copy(); ysh[mask]=rng.permutation(y[mask]); ids2,p2=cv(X,ysh,mask,groups)
            res[L]=(round(a,3),round(float(np.mean(np.sign(p2)==np.sign(ysh[ids2]))),3))
        r['layers(acc,shuffled)']=res
        r.pop('base_raw'); print(c,gname,json.dumps(r),flush=True)
# model behaviour
for c in ['D1','D2','D3','D4']:
    mk=cond==c; print(c,'behaviour acc',float(np.mean(beh[mk]>0)))
# transfer: train D1+D2 test D3+D4 and D2->D4
for tr_c,te_c in ((['D1','D2'],['D3','D4']),(['D2'],['D4']),(['D3'],['D4']),(['D4'],['D3'])):
    trm=np.isin(cond,tr_c); tem=np.isin(cond,te_c); res={}
    # exclude same-item leakage: only test items not in train items
    tri=set(item[trm]); tem=tem&~np.isin(item,list(tri)) if set(te_c)&set(tr_c) else tem
    for L in layers[::2]:
        X=Z[f'L{L}'].astype('float32'); p,_=kr_fit_pred(X[trm],y[trm],X[tem]); res[L]=round(float(np.mean(np.sign(p)==np.sign(y[tem]))),3)
    print('transfer',tr_c,'->',te_c,json.dumps(res),flush=True)
