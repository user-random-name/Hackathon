"""Antarctica Level 3: calibrate surveys, local Matérn kriging, 90% clearance, safe route and tracking pixels.

Run: pip install numpy pandas scipy matplotlib
     python solution.py

Writes depth.csv, sd.csv, route.csv, filled_tracking_notes.txt, chart.png.
"""
from importlib.metadata import files
from pathlib import Path
import csv, heapq, math, re
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree, distance
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
CELL_N, SIZE, MIN_DEPTH = 120, 500., 350.
Z90 = norm.ppf(.90)
ELL_KM = 5.0
AMP_M = 220.0
NOISE_M = 20.0
NEIGHBORS = 48


def depth_model(soundings, targets):
    """Local Gaussian-process prediction using ALL soundings; Matérn-3/2 spatial covariance.

    Uncertainty is the kriging posterior standard deviation of latent seabed depth,
    not the noisy echo-sounder observation standard deviation.
    """
    p = soundings[['x_m','y_m']].to_numpy(dtype=float) / 1000.
    z = soundings.depth_m.to_numpy(dtype=float).copy()
    # 1961 echo-sounder convention is approximately 3% high relative to 1998/2019.
    z[soundings.cruise.to_numpy() == 1961] *= .970
    tree = cKDTree(p)
    dist_to, ids = tree.query(targets, k=NEIGHBORS)
    mean = np.empty(len(targets));sd = np.empty(len(targets))
    def corr(d):
        x = np.sqrt(3.) * d / ELL_KM
        return (1. + x) * np.exp(-x)
    for i, (dd, ix) in enumerate(zip(dist_to, ids)):
        observations = z[ix]
        local_mean = observations.mean()
        cov = (AMP_M**2) * corr(distance.cdist(p[ix], p[ix]))
        cov.flat[::len(ix)+1] += NOISE_M**2
        cross = AMP_M**2 * corr(dd)
        fac = cho_factor(cov, check_finite=False)
        mean[i] = local_mean + cross @ cho_solve(fac, observations-local_mean, check_finite=False)
        sd[i] = np.sqrt(max(0, AMP_M**2 - cross @ cho_solve(fac,cross,check_finite=False)))
    return mean,sd


def shortest_path(clear, start, finish):
    """Four-neighbour Dijkstra, so diagonals cannot clip foul-cell corners."""
    if not clear[start] or not clear[finish]:
        raise ValueError('Entry or station is not clear on this chart.')
    pq=[(0,start)];prev={start:None};dist={start:0}
    while pq:
        d, p = heapq.heappop(pq)
        if d != dist[p]:continue
        if p==finish:break
        r,c=p
        for rr,cc in ((r+1,c),(r-1,c),(r,c+1),(r,c-1)):
            if not(0<=rr<CELL_N and 0<=cc<CELL_N and clear[rr,cc]):continue
            nxt=(rr,cc);nd=d+1
            if nd<dist.get(nxt,float('inf')):
                dist[nxt]=nd;prev[nxt]=p;heapq.heappush(pq,(nd,nxt))
    if finish not in prev:return None
    route=[];cur=finish
    while cur is not None:route.append(cur);cur=prev[cur]
    return route[::-1]


def square_pos(name):
    match=re.fullmatch(r'([A-Z]{1,2})([1-9][0-9]*)',name)
    if not match:raise ValueError('Bad chart square '+name)
    letters, row=match.groups()
    col=0
    for letter in letters:col=col*26+ord(letter)-64
    return (int(row)-1,col-1)


def fill_notes(square_foul):
    data=(ROOT/'tracking_notes.txt').read_text(encoding='utf-8')
    rows=[];lines=[]
    for line in data.splitlines():
        m=re.match(r'\s*watch\s+(\d+)\b(.*)',line,re.I)
        if not m:continue
        w=int(m.group(1));names=re.findall(r'\b([A-Z]{1,2}[0-9]+)\s*\[',m.group(2))
        assert len(names)==24,(w,len(names))
        flags=[];items=[]
        for name in names:
            row,col=square_pos(name)
            foul=bool(square_foul[row,col])
            flags.append('#' if foul else '.')
            items.append(f'{name} [{"FOUL" if foul else "CLEAR"}]')
        rows.append(''.join(flags));lines.append(f'watch {w}: '+', '.join(items))
    assert len(rows)==5
    (ROOT/'filled_tracking_notes.txt').write_text('\n'.join(lines)+'\n\nBitmap (# foul, . clear):\n'+'\n'.join(rows)+'\n',encoding='utf-8')
    return rows


def main():
    # Search for soundings.csv inside the Hackathon folder
    files = list(ROOT.parent.rglob("soundings.csv"))
    if not files:
        raise FileNotFoundError("soundings.csv was not found anywhere in Hackathon")

    s = pd.read_csv(files[0])
    print("Found soundings.csv at:", files[0])
    c=pd.read_csv(ROOT/'chart.csv')
    land=np.zeros((CELL_N,CELL_N),dtype=bool)
    for r in c[c.kind=='land'].itertuples():land[int(r.row),int(r.col)]=True
    def cell(kind):
        t=c[c.kind==kind]
        assert len(t)==1
        return tuple(map(int,t[['row','col']].iloc[0]))
    entry,station=cell('entry'),cell('station')
    positions=(np.arange(CELL_N)+.5)*.5
    x,y=np.meshgrid(positions,positions)
    mu,sd=depth_model(s,np.c_[x.ravel(),y.ravel()])
    mu=mu.reshape(CELL_N,CELL_N);sd=sd.reshape(CELL_N,CELL_N)
    lower=mu-Z90*sd
    foul=land | (lower < MIN_DEPTH)
    square_foul=foul.reshape(60,2,60,2).any(axis=(1,3))
    np.savetxt(ROOT/'depth.csv',mu,delimiter=',',fmt='%.4f')
    np.savetxt(ROOT/'sd.csv',sd,delimiter=',',fmt='%.4f')
    np.savetxt(ROOT/'foul.csv',foul.astype(int),delimiter=',',fmt='%d')
    np.savetxt(ROOT/'square_foul.csv',square_foul.astype(int),delimiter=',',fmt='%d')
    notes=fill_notes(square_foul)
    print('Soundings',len(s),'land cells',land.sum(),'foul cells',foul.sum(),'foul squares',square_foul.sum())
    for row in notes:print(row)
    try:
        route=shortest_path(~foul,entry,station)
    except ValueError as e:
        print('Warning:',e);route=None
    if route:
        with (ROOT/'route.csv').open('w',newline='') as f:
            wr=csv.writer(f);wr.writerow(['row','col']);wr.writerows(route)
        assert all(~foul[r,c] for r,c in route)
        print('Route length:',len(route),'cells, distance:',(len(route)-1)*.5,'km')
    else:print('NO SAFE ROUTE FOUND')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(15,7))
    im=axs[0].imshow(np.where(land,np.nan,mu),origin='lower',cmap='viridis')
    fig.colorbar(im,ax=axs[0],label='Depth (m)')
    axs[0].set_title('Corrected soundings: estimated depth')
    axs[1].imshow(np.where(land,np.nan,lower),origin='lower',cmap='viridis')
    axs[1].contour(foul.astype(float),levels=[.5],colors='red',linewidths=.4)
    axs[1].set_title('90% lower bound; red = foul-cell boundaries')
    for ax in axs:
        ax.plot(entry[1],entry[0],marker='o',color='lime',markersize=7)
        ax.plot(station[1],station[0],marker='*',color='yellow',markersize=11)
        if route:ax.plot([cc for rr,cc in route],[rr for rr,cc in route],color='white',linewidth=1.5)
        ax.set_xlabel('Column (west to east)');ax.set_ylabel('Row (south to north)')
    fig.tight_layout();fig.savefig(ROOT/'chart.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
