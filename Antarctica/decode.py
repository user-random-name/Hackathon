import re, sys, datetime as dt
from collections import defaultdict

LINE = re.compile(r'^(\d{4}) (\d{4}-\d{2}-\d{2}) (\d{2})Z (.*)$')
rows = []
for raw in open('messages.txt', encoding='utf-8'):
    raw = raw.rstrip('\n')
    if not raw.strip():
        continue
    m = LINE.match(raw)
    if not m:
        rows.append(dict(id=None, raw=raw, bad=['line-format']))
        continue
    rid, date, hh, msg = m.groups()
    rows.append(dict(id=int(rid), date=dt.date.fromisoformat(date), hour=int(hh), msg=msg, bad=[], f={}))

def signed(g):
    # g = sTTT
    if not re.fullmatch(r'[01]\d{3}', g):
        return None
    v = int(g[1:]) / 10
    return -v if g[0] == '1' else v

def pres(g):
    if not re.fullmatch(r'\d{4}', g):
        return None
    v = int(g) / 10
    return v + (1000 if v < 500 else 0)

def check(r):
    bad = r['bad']
    msg = r['msg']
    if not msg.endswith('='):
        bad.append('no-terminator')
    body = msg.rstrip('=')
    toks = body.split(' ')
    if any(len(t) != 5 for t in toks if t not in ('333','AAXX')):
        bad.append('group-length')
    if len(toks) < 9 or toks[0] != 'AAXX':
        bad.append('header'); return
    yygg = toks[1]
    if not re.fullmatch(r'\d{5}', yygg) or int(yygg[:2]) != r['date'].day or int(yygg[2:4]) != r['hour'] or yygg[4] != '1':
        bad.append('YYGGi')
    if toks[2] != '89624':
        bad.append('station')
    rest = toks[3:]
    if '333' in rest:
        i3 = rest.index('333'); s1 = rest[:i3]; s3 = rest[i3+1:]
        if not s3: bad.append('empty-333')
    else:
        s1, s3 = rest, []
    if len(s1) < 7:
        bad.append('missing-groups'); return
    g_iR, g_N, g_T, g_Td, g_P, g_PS, g_tend = None, None, None, None, None, None, None
    g_iR, g_N, g_T, g_Td, g_P, g_PS, g_tend = s1[:7]
    opt = s1[7:]
    if not re.fullmatch(r'4[13][0-9/][0-9/]{2}', g_iR):
        bad.append('iRixhVV-format')
    else:
        ix = g_iR[1]; h = g_iR[2]; vv = g_iR[3:]
        has7 = any(o.startswith('7') for o in opt)
        if (ix == '1') != has7:
            bad.append('ix-vs-7')
        if vv != '//':
            v = int(vv) if vv.isdigit() else -1
            if v < 0 or 51 <= v <= 55 or v > 89:
                bad.append('VV-code')
        r['f']['h'] = h; r['f']['vv'] = vv
    if not re.fullmatch(r'[0-9/][0-9/]{4}', g_N):
        bad.append('Nddff-format')
    else:
        N = g_N[0]; dd = g_N[1:3]; ff = g_N[3:]
        if dd != '//' and (not dd.isdigit() or int(dd) > 36):
            bad.append('dd-range')
        if dd.isdigit() and ff.isdigit():
            r['f']['dd'] = int(dd); r['f']['ff'] = int(ff)
            if int(dd) == 0 and int(ff) != 0: bad.append('calm-but-wind')
            if int(dd) != 0 and int(ff) == 0: bad.append('wind-dir-but-calm')
        r['f']['N'] = N
    if not (g_T.startswith('1') and g_Td.startswith('2') and g_P.startswith('3') and g_PS.startswith('4') and g_tend.startswith('5')):
        bad.append('group-ids')
    T = signed(g_T[1:]) if g_T.startswith('1') else None
    Td = signed(g_Td[1:]) if g_Td.startswith('2') else None
    P = pres(g_P[1:]) if g_P.startswith('3') else None
    PS = pres(g_PS[1:]) if g_PS.startswith('4') else None
    for nm, v in (('T', T), ('Td', Td), ('P', P), ('PS', PS)):
        if v is None: bad.append(nm + '-format')
    r['f'].update(T=T, Td=Td, P=P, PS=PS)
    if re.fullmatch(r'5[0-8]\d{3}', g_tend):
        a = int(g_tend[1]); p = int(g_tend[2:]) / 10
        r['f']['tend'] = (a, p)
    else:
        bad.append('tend-format')
    # optional 7/8
    seen = []
    for o in opt:
        if o[0] == '7' and re.fullmatch(r'7[0-9/]{4}', o) and 'w' not in seen:
            seen.append('w')
        elif o[0] == '8' and re.fullmatch(r'8[0-9/]{4}', o) and 'c' not in seen:
            seen.append('c')
        else:
            bad.append('opt-group')
    if seen == ['c', 'w']:
        bad.append('opt-order')
    # section 3
    for g in s3:
        if g.startswith('1') and r['hour'] in (0, 12):
            r['f']['tmax'] = signed(g[1:])
        elif g.startswith('2') and r['hour'] in (0, 12):
            r['f']['tmin'] = signed(g[1:])
        elif g.startswith('910'):
            r['f']['g10'] = int(g[3:]) if g[3:].isdigit() else None
        elif g.startswith('911'):
            r['f']['g11'] = int(g[3:]) if g[3:].isdigit() else None
        else:
            bad.append('sec3-group')
    if r['hour'] not in (0, 12) and ('tmax' in r['f'] or 'tmin' in r['f']):
        bad.append('ext-wrong-hour')

for r in rows:
    if r['id'] is not None:
        check(r)

if __name__ == '__main__':
    cnt = defaultdict(list)
    for r in rows:
        for b in r['bad']:
            cnt[b].append(r['id'])
    for k, v in cnt.items():
        print(k, len(v), v)
    print('total rows', len(rows))