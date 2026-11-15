import pandas as pd,numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import label
D='/mnt/user-data/uploads/'
cells=pd.read_csv(D+'cells.csv');finds=pd.read_csv(D+'finds.csv');reg=pd.read_csv(D+'regions.csv')
cov=['ice_speed_m_yr','slope_deg','elevation_m','ice_thickness_m','temperature_c','brightness','blue_ice_frac','dist_blueice_km','backscatter','dist_station_km','dist_sea_km','dist_outcrop_km']
R={}
for r,g in cells.groupby('region'):
    g=g.sort_values('cell').reset_index(drop=True)
    nr,nc=g.row.max()+1,g.col.max()+1
    grid=-np.ones((nr,nc),int); grid[g.row,g.col]=np.arange(len(g))
    blue=np.zeros((nr,nc),bool); blue[g.row,g.col]=g.blue_ice_frac.values>0.3
    lab,n=label(blue,structure=np.ones((3,3)))
    g['patch']=np.where(blue[g.row,g.col],lab[g.row,g.col],0)
    R[r]=dict(g=g,tree=cKDTree(g[['x_m','y_m']].values/1000))
def feats(r,cut):
    """features for region r using finds with season<=cut; returns df of cells w/o prior find"""
    g=R[r]['g'];tree=R[r]['tree'];f=finds[(finds.region==r)&(finds.season<=cut)]
    hit=np.zeros(len(g),bool);hit[f.cell.values]=True
    nf=np.zeros(len(g));np.add.at(nf,f.cell.values,f.finds.values)
    X=g[cov].copy();X['ice_speed_m_yr']=X.ice_speed_m_yr.fillna(X.ice_speed_m_yr.median())
    xy=g[['x_m','y_m']].values/1000;blue=(g.blue_ice_frac.values>0.3)
    if hit.sum()>0:
        ht=cKDTree(xy[hit]);d,_=ht.query(xy);X['d_hit']=d
        d3,_=ht.query(xy,k=min(3,hit.sum()));X['d_hit3']=d3.mean(1) if d3.ndim>1 else d3
        _,i=ht.query(xy);ss=f.groupby('cell').season.max().reindex(g.cell[hit]).values;X['near_season']=cut-ss[i]
    else:
        X['d_hit']=99;X['d_hit3']=99;X['near_season']=99
    bt=cKDTree(xy[blue]) if blue.sum() else None
    for rad in (0.5,1,2,4,8,16):
        ch=cKDTree(xy[hit]).query_ball_point(xy,rad+1e-9,return_length=True) if hit.sum() else np.zeros(len(g))
        cb=bt.query_ball_point(xy,rad+1e-9,return_length=True) if bt is not None else np.zeros(len(g))
        X[f'h{rad}']=ch;X[f'b{rad}']=cb;X[f'hb{rad}']=ch/(cb+1)
    p=g.patch.values;ps=pd.Series(p)
    size=ps.map(ps[ps>0].value_counts()).fillna(0).values
    ph=pd.Series(hit.astype(int)).groupby(p).sum()
    X['patch_size']=size;X['patch_hits']=np.where(p>0,pd.Series(p).map(ph).values,0)
    X['patch_hfrac']=X.patch_hits/np.maximum(size,1)
    X['reg_hit_frac']=hit.mean();X['reg_blue_hit']=hit[blue].mean() if blue.sum() else 0
    X['reg_blue_frac']=blue.mean();X['reg_ncell']=len(g);X['reg_nhit']=hit.sum()
    X['reg_nseason']=f.season.nunique()
    X['stn_rel']=g.dist_station_km/g.dist_station_km.median()
    X['elev_rel']=g.elevation_m-g.elevation_m.median()
    X['cut_age']=cut-finds[finds.region==r].season.min() if False else cut-f.season.min() if len(f) else 0
    X['cell']=g.cell.values;X['region']=r;X['hit']=hit
    return X