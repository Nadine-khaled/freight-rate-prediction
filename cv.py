import pandas as pd, numpy as np, sys
from features import *
from sklearn.ensemble import HistGradientBoostingRegressor as H
from sklearn.linear_model import Ridge, HuberRegressor
t=pd.read_csv('data/train_test.csv'); t['date']=pd.to_datetime(t['date'])
t['y']=np.log(t.posted_rate)
def prep(tr, te):
    wf=tr.weight.abs().median(); mb=tr.dropna(subset=['market_index']).groupby('date').market_index.mean()
    # test-date market_index from the test rows' own (non-null) values
    mt=pd.concat([tr,te]).dropna(subset=['market_index']).groupby('date').market_index.mean()
    return build_features(tr,wf,mt), build_features(te,wf,mt)
def robust_mask(tr, k=0.25):
    # flag corrupted-rate rows: large residual from a simple robust linear fit
    X=np.c_[tr.log_dist,tr.log_dist**2,tr.market_index,pd.get_dummies(tr.equipment,drop_first=True).values]
    h=HuberRegressor(max_iter=500).fit(X,tr.y); r=tr.y-h.predict(X)
    return r.abs()<k
def metrics(y,p,raw_rate,clean):
    pr=np.exp(p); e=np.abs(pr-raw_rate)
    return dict(MAE=e.mean(), MAPE=(e/raw_rate).mean()*100, MdAPE=np.median(e/raw_rate)*100,
                MAE_clean=e[clean].mean(), MAPE_clean=(e/raw_rate)[clean].mean()*100)
folds=[('2025-07-01','2025-08-31'),('2025-09-01','2025-10-31')]
res=[]
for s,e in folds:
    tr=t[t.date<s]; te=t[(t.date>=s)&(t.date<=e)]
    tr,te=prep(tr,te); keep=robust_mask(tr)
    tmask=robust_mask(te)  # 'clean' evaluation subset
    cols=NUM
    def run(name,model,X,Xt,mask):
        model.fit(X[mask],tr.y[mask]); p=model.predict(Xt); m=metrics(te.y,p,te.posted_rate,tmask); m.update(model=name,fold=s); res.append(m)
    lin=lambda d: np.c_[d.log_dist,d.log_dist**2,d.market_index,d.quote_signal,d.log_w,pd.get_dummies(d.equipment).reindex(columns=['Dry Van','Flatbed','Reefer'],fill_value=0).values]
    class Base:
        def fit(s,X,y): s.m=y.mean(); return s
        def predict(s,X): return np.full(len(X),s.m)
    run('naive_mean_log',Base(),tr[cols],te[cols],np.ones(len(tr),bool))
    run('ridge_all_rows',Ridge(1e-3),lin(tr),lin(te),np.ones(len(tr),bool))
    run('ridge_clean',Ridge(1e-3),lin(tr),lin(te),keep)
    run('hgb_all_rows',H(max_iter=400,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=1,random_state=0),tr[cols],te[cols],np.ones(len(tr),bool))
    run('hgb_clean',H(max_iter=400,learning_rate=0.05,max_leaf_nodes=31,l2_regularization=1,random_state=0),tr[cols],te[cols],keep)
    run('hgb_clean_l1',H(loss='absolute_error',max_iter=400,learning_rate=0.05,max_leaf_nodes=31,random_state=0),tr[cols],te[cols],np.ones(len(tr),bool))
    print(s,'train',len(tr),'dropped',(~keep).sum(),'test',len(te),'test corrupt',(~tmask).sum(),flush=True)
r=pd.DataFrame(res); print(r.groupby('model')[['MAE','MAPE','MdAPE','MAE_clean','MAPE_clean']].mean().round(3).sort_values('MAPE_clean'))
