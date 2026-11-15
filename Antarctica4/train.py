from feat import *
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.metrics import average_precision_score as AP, roc_auc_score
comp=[r for r in R if r not in ('R04','R06','R09')]
rows=[]
for r in comp:
    ss=sorted(finds[finds.region==r].season.unique());tot=finds[finds.region==r].cell.nunique()
    for cut in ss[:-1]:
        pre=finds[(finds.region==r)&(finds.season<=cut)].cell.nunique()
        post=tot-pre
        if pre<8 or post<8: continue
        X=feats(r,cut);fut=set(finds[(finds.region==r)&(finds.season>cut)].cell)
        X=X[~X.hit].copy();X['y']=X.cell.isin(fut).astype(int);X['cut']=cut;rows.append(X)
T=pd.concat(rows);print(T.groupby('region').agg(n=('y','size'),pos=('y','sum'),cuts=('cut','nunique')))
F=[c for c in T.columns if c not in('cell','region','hit','y','cut')]
def mk(): return H(max_iter=200,learning_rate=0.05,max_depth=4,min_samples_leaf=40,l2_regularization=1.0)
for r in comp:
    tr=T[T.region!=r];te=T[T.region==r]
    if te.y.sum()==0: continue
    m=mk().fit(tr[F],tr.y);p=m.predict_proba(te[F])[:,1]
    out=[]
    for c,gg in te.groupby('cut'):
        pp=p[(te.cut==c).values];k=max(1,int(.1*len(gg)));top=np.argsort(-pp)[:k]
        out.append((gg.y.values[top].sum()/max(gg.y.sum(),1)))
    print(r,'AUC %.3f AP %.3f base %.4f  top10%% recall %.2f'%(roc_auc_score(te.y,p),AP(te.y,p),te.y.mean(),np.mean(out)))
T.to_pickle('T.pkl')