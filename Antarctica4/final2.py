from feat import *
from sklearn.ensemble import HistGradientBoostingClassifier as H
from train import T,F,mk,comp
from sklearn.linear_model import LogisticRegression
oof=np.zeros(len(T))
for r in comp:
    m=mk().fit(T[T.region!=r][F],T[T.region!=r].y);oof[(T.region==r).values]=m.predict_proba(T[T.region==r][F])[:,1]
lg=lambda p:np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
cal=LogisticRegression(C=100).fit(lg(oof)[:,None],T.y)
ms=[H(max_iter=it,learning_rate=0.05,max_depth=d,min_samples_leaf=msl,l2_regularization=1.0,random_state=i).fit(T[F],T.y) for i,(it,d,msl) in enumerate([(200,4,40),(300,3,60),(150,5,30),(250,4,80),(200,6,50)])]
P=[];plan=[]
for r in ('R04','R06','R09'):
    cut=finds[finds.region==r].season.max();X=feats(r,cut);X=X[~X.hit].copy()
    p=np.mean([m.predict_proba(X[F])[:,1] for m in ms],0);X['probability']=cal.predict_proba(lg(p)[:,None])[:,1]
    P.append(X[['region','cell','probability']])
    k=int(0.1*len(X));plan.append(X.sort_values(['probability','cell'],ascending=[False,True]).head(k)[['region','cell']])
    print(r,cut,len(X),k,X.probability.sum().round(1))
P=pd.concat(P);pl=pd.concat(plan)
P.to_csv('/mnt/user-data/outputs/probabilities.csv',index=False);pl.to_csv('/mnt/user-data/outputs/plan.csv',index=False)
m=pl.merge(cells[['region','cell','block_code']]);m['d']=m.block_code.str[0];m['l']=m.block_code.str[1]
sig=''
for d in '12345678':
    v=m[m.d==d].l.value_counts();a=v.iloc[0];b=v.iloc[1];print(d,v.index[0],a,v.index[1],b)
    sig+=v.index[0] if a-b>=8 else '-'
print(len(pl),sig)