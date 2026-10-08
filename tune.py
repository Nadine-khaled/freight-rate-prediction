import itertools, numpy as np, pandas as pd
exec(open('cv.py').read().split("folds=")[0])
folds=[('2025-07-01','2025-08-31'),('2025-09-01','2025-10-31')]
cfgs={'base':dict(max_iter=400,learning_rate=0.05,max_leaf_nodes=31),
 'small':dict(max_iter=600,learning_rate=0.03,max_leaf_nodes=15,min_samples_leaf=40),
 'big':dict(max_iter=800,learning_rate=0.03,max_leaf_nodes=63,min_samples_leaf=20),
 'shallow':dict(max_iter=800,learning_rate=0.05,max_depth=4)}
out=[]
for s,e in folds:
    tr=t[t.date<s]; te=t[(t.date>=s)&(t.date<=e)]; tr,te=prep(tr,te); keep=robust_mask(tr); tm=robust_mask(te)
    for loss in ['absolute_error','squared_error']:
      for n,c in cfgs.items():
        for variant in ['num','+lane_cat']:
            cols=NUM+ (['pickup','delivery'] if variant!='num' else [])
            Xa=tr[cols].copy(); Xb=te[cols].copy()
            kw={}
            if variant!='num':
                for c2 in ['pickup','delivery']:
                    cat=pd.CategoricalDtype(sorted(tr[c2].unique())); Xa[c2]=Xa[c2].astype(cat); Xb[c2]=Xb[c2].astype(cat)
                kw['categorical_features']='from_dtype'
            m=H(loss=loss,random_state=0,**c,**kw).fit(Xa[keep],tr.y[keep]); p=m.predict(Xb)
            mm=metrics(te.y,p,te.posted_rate,tm); mm.update(cfg=n,loss=loss,var=variant); out.append(mm)
r=pd.DataFrame(out).groupby(['loss','cfg','var'])[['MAPE','MAPE_clean','MAE']].mean().round(3).sort_values('MAPE_clean'); print(r.head(10)); print(r.tail(3))
