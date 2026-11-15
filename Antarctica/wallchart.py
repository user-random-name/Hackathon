import sys, re, datetime as dt, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
import decode
rows = [r for r in decode.rows if r['id'] is not None]
byt = {(r['date'], r['hour']): r for r in rows}
def at(r, h):
    t = dt.datetime.combine(r['date'], dt.time(r['hour'])) + dt.timedelta(hours=h)
    return byt.get((t.date(), t.hour))
bad = {}
def flag(r, why): bad.setdefault(r['id'], set()).add(why)
def groups(r):
    t = r['msg'].rstrip('=').split(' ')
    s1 = t[3:t.index('333')] if '333' in t else t[3:]
    return s1
for r in rows:
    for b in r['bad']: flag(r, b)
    f = r['f']; s1 = groups(r)
    if f.get('T') is not None and f.get('Td') is not None and f['Td'] > f['T'] + .05: flag(r, 'Td>T')
    D = s1[0][:2] == '43'
    if D != (r['hour'] in (15, 18, 21)): flag(r, 'dark-pattern')
    if D and (s1[0][2:] != '///' or s1[1][0] != '/'): flag(r, 'dark-fields')
    if len(s1) >= 7:
        N = s1[1][0]; g8 = [x for x in s1[7:] if x[0] == '8']; g7 = [x for x in s1[7:] if x[0] == '7']
        if g8 and N.isdigit() and g8[0][1].isdigit() and int(g8[0][1]) > int(N): flag(r, 'Nh>N')
        if N == '0' and s1[0][2] != '/': flag(r, 'N0-h')
        if not D and not g7: flag(r, 'no-7group')
        if g7 and ('/' in g7[0] or g7[0][1:3] in ('00', '03', '10', '22', '75')): flag(r, 'ww-rare')
        if g7 and g7[0][1:3].isdigit() and 70 <= int(g7[0][1:3]) <= 75 and N not in '89': flag(r, 'snow-needs-overcast')
    if f.get('g10') is not None and f.get('ff') is not None and f['g10'] < f['ff']: flag(r, 'g10<ff')
    if f.get('g11') is not None and f.get('g10') is not None and f['g11'] < f['g10']: flag(r, 'g11<g10')
    if f.get('g11') is not None and f.get('ff') is not None and f['g11'] < f['ff']: flag(r, 'g11<ff')
    if f.get('g10') is not None and f.get('g11') is None: flag(r, '910-without-911')
    if f.get('tmax') is not None and f.get('tmin') is not None and f['tmax'] < f['tmin']: flag(r, 'max<min')
    if f.get('dd') is not None and f['dd'] > 36: flag(r, 'dd>36')
    if f.get('dd') is not None and f.get('ff') is not None and (f['dd'] == 0) != (f['ff'] == 0): flag(r, 'calm-mismatch')
    if f.get('ff') is not None and f['ff'] >= 40: flag(r, 'ff-huge')
for r in rows:   # temperature spikes
    f = r['f']; a = at(r, -3); b = at(r, 3)
    if a and b and f.get('T') is not None and a['f'].get('T') is not None and b['f'].get('T') is not None:
        if abs(f['T']-a['f']['T']) > 6 and abs(f['T']-b['f']['T']) > 6 and abs(a['f']['T']-b['f']['T']) < 8: flag(r, 'T-spike')
good = [r for r in rows if all(r['f'].get(k) is not None for k in ('P', 'PS', 'T'))]
A = np.array([[1, r['f']['P'], r['f']['T']] for r in good]); y = np.array([r['f']['PS']-r['f']['P'] for r in good])
co = np.linalg.lstsq(A, y, rcond=None)[0]
for r, e in zip(good, y-A@co):
    if abs(e) > .6 and 'T-spike' not in bad.get(r['id'], ()): flag(r, 'P-vs-PS')
def m(r):
    f = r['f']; p = at(r, -3)
    if not p or not f.get('tend') or f.get('P') is None or p['f'].get('P') is None: return None
    a, amt = f['tend']; s = 0 if a == 4 else (amt if a <= 3 else -amt)
    return abs((f['P']-p['f']['P']) - s) > .05
for r in rows:
    if m(r):
        n = at(r, 3); p = at(r, -3)
        if n and m(n): flag(r, 'P-bad')
        elif not (p and m(p)): flag(r, 'tendency')
days = sorted({r['date'] for r in rows}); di = {d: i for i, d in enumerate(days)}
G = np.zeros((8, len(days)), int)
for r in rows:
    if r['id'] in bad: G[r['hour']//3, di[r['date']]] = 1
json.dump({'flagged_ids': sorted(bad), 'grid': G.tolist()}, open('flagged.json', 'w'))
fig, ax = plt.subplots(figsize=(30, 3.6))
ys, xs = np.nonzero(G); ax.scatter(xs, ys, s=70, c='k')
ax.set_ylim(7.5, -.5); ax.set_xlim(-1, len(days)); ax.set_yticks(range(8)); ax.set_yticklabels(['%02d' % (3*i) for i in range(8)])
ax.set_xlabel('day'); ax.set_ylabel('hour UTC'); ax.grid(alpha=.25)
plt.savefig('wallchart.png', dpi=80, bbox_inches='tight')
print(len(bad), 'flagged of', len(rows))