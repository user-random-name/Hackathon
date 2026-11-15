"""City of Subtitles — reproducible subtitle-density analysis (Hint 1).

Run:
  pip install numpy pillow opencv-python
  python solve.py "dataset-20261114T173504Z-1-001(1).zip" results KAMI

The program reads the original ZIP, reconstructs the photographer groups
from ZIP creation timestamps, extracts subtitle glyph rows, measures the
fraction of bright subtitle ink in left-to-right character cells, and
compares Toorisugari's weekly densities against the full week's average.

The four-letter 5-row block font below is used as a *decoder* after the
measurements, not to draw or modify the measured density grid.
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw

# Conventional 3-by-5 capitals. A 5-wide M/W is supported for the 17-cell layout.
FONT3 = {
    'A':['.#.','#.#','###','#.#','#.#'],
    'B':['##.','#.#','##.','#.#','##.'],
    'C':['.##','#..','#..','#..','.##'],
    'D':['##.','#.#','#.#','#.#','##.'],
    'E':['###','#..','##.','#..','###'],
    'F':['###','#..','##.','#..','#..'],
    'G':['.##','#..','#.#','#.#','.##'],
    'H':['#.#','#.#','###','#.#','#.#'],
    'I':['###','.#.','.#.','.#.','###'],
    'J':['..#','..#','..#','#.#','.#.'],
    'K':['#.#','#.#','##.','#.#','#.#'],
    'L':['#..','#..','#..','#..','###'],
    'M':['#.#','###','###','#.#','#.#'],
    'N':['#.#','###','###','###','#.#'],
    'O':['###','#.#','#.#','#.#','###'],
    'P':['##.','#.#','##.','#..','#..'],
    'Q':['###','#.#','#.#','###','..#'],
    'R':['##.','#.#','##.','##.','#.#'],
    'S':['.##','#..','.#.','..#','##.'],
    'T':['###','.#.','.#.','.#.','.#.'],
    'U':['#.#','#.#','#.#','#.#','###'],
    'V':['#.#','#.#','#.#','#.#','.#.'],
    'W':['#.#','#.#','###','###','#.#'],
    'X':['#.#','#.#','.#.','#.#','#.#'],
    'Y':['#.#','#.#','.#.','.#.','.#.'],
    'Z':['###','..#','.#.','#..','###'],
}
FONT5 = {
    'M':['#...#','##.##','#.#.#','#...#','#...#'],
    'W':['#...#','#...#','#.#.#','##.##','#...#'],
}

# This layout has four letters, 3 + 3 + 5 + 3 cells, separated by 1 blank cell.
SLOTS = ((0,3),(4,3),(8,5),(14,3))
GRID_WIDTH = 17
MAX_CELLS = 24


def attribute_from_zip(archive, out):
    """Use the original archive's timestamp side channel; validate known labels."""
    entries = [e for e in archive.infolist() if e.filename.lower().endswith('.jpg')]
    credited = [e for e in entries if '/credited/' in e.filename]
    anonymous = [e for e in entries if '/anonymous/' in e.filename]
    if (len(entries),len(credited),len(anonymous))!=(253,108,145):
        raise ValueError('Unexpected JPEG counts; use the original 253-image dataset ZIP')

    first = {}
    for e in credited:
        h = Path(e.filename).parent.name
        first[h] = min(first.get(h,e.date_time),e.date_time)
    order = sorted(first,key=first.get)
    if len(order)!=6: raise ValueError('Expected 6 photographers')
    # All JPEGs were written to the ZIP in contributor groups. FAT ZIP time
    # precision is 2 seconds; the second key resolves boundary ties.
    sorted_entries = sorted(entries,key=lambda e:(e.date_time,1 if '/credited/' in e.filename else 0))
    assign={}
    idx=0
    for handle,count in zip(order,[63]+[38]*5):
        for e in sorted_entries[idx:idx+count]:
            assign[e.filename]=handle
        idx+=count
    for e in credited:
        if assign[e.filename]!=Path(e.filename).parent.name:
            raise ValueError('Timestamp grouping fails known-label validation')

    with (out/'attribution_anonymous.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['filename','week','photographer','zip_timestamp'])
        for e in sorted(anonymous,key=lambda e:e.filename):
            w.writerow([Path(e.filename).name,int(Path(e.filename).name[1:3]),assign[e.filename],str(e.date_time)])
    print('ZIP timestamp attribution: 108/108 known labels verified')
    print('Anonymous counts:',dict(Counter(assign[e.filename] for e in anonymous)))
    return assign, anonymous


def extract_subtitle_lines(gray):
    """Find bright subtitle text on a dark film-style caption rectangle.

    For full black bars and translucent caption boxes, detect bright ink on
    locally dark surroundings, join individual glyphs into line components,
    then group consecutive caption lines. Only near-edge text is considered.
    """
    local = cv2.GaussianBlur(gray,(29,29),0)
    ink=((gray>165)&(local<110)).astype(np.uint8)
    joined=cv2.morphologyEx(ink,cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT,(25,5)))
    _,_,stats,_=cv2.connectedComponentsWithStats(joined,8)
    parts=[]
    for x,y,w,h,area in stats[1:]:
        if 80<=w<=1150 and 17<=h<=45 and (y>590 or y<190):
            parts.append((int(x),int(y),int(w),int(h)))
    groups=[]
    for x,y,w,h in sorted(parts,key=lambda r:r[1]):
        if groups and abs((y+h/2)-groups[-1]['yc'])<12:
            g=groups[-1]
            g['parts'].append((x,y,w,h))
            g['yc']=float(np.median([p[1]+p[3]/2 for p in g['parts']]))
        else:
            groups.append({'yc':y+h/2,'parts':[(x,y,w,h)]})
    lines=[]
    for g in groups:
        x0=min(p[0] for p in g['parts']); y0=min(p[1] for p in g['parts'])
        x1=max(p[0]+p[2] for p in g['parts']); y1=max(p[1]+p[3] for p in g['parts'])
        if x1-x0<150: continue
        mask=(ink[y0:y1,x0:x1]>0)
        if mask.sum()<140:continue
        lines.append((x0,y0,x1,y1,mask))
    if len(lines)>=2:
        clusters=[]
        for line in sorted(lines,key=lambda z:z[1]):
            if clusters and line[1]-clusters[-1][-1][1]<105:
                clusters[-1].append(line)
            else:
                clusters.append([line])
        clusters.sort(key=lambda group:(len(group),sum((r[2]-r[0])*r[4].mean() for r in group)),reverse=True)
        lines=clusters[0]
    return lines


def cell_width(ink_height):
    # Three caption font styles in the dataset, inferred from glyph heights.
    if ink_height<=23:return 26
    if ink_height<=30:return 32
    return 38


def line_density(mask):
    h,w=mask.shape
    width=cell_width(h)
    out=np.zeros(MAX_CELLS,dtype=np.float64)
    for j in range(min(MAX_CELLS,(w+width-1)//width)):
        cell=mask[:,j*width:min((j+1)*width,w)]
        out[j] = cell.sum() / (h*width)
    return out


def analyze(archive,assign,anonymous,out):
    by_week=defaultdict(list)
    by_file=defaultdict(list)
    counts=Counter()
    for e in anonymous:
        name=Path(e.filename).name
        week=int(name[1:3]); handle=assign[e.filename]
        img=cv2.imdecode(np.frombuffer(archive.read(e),dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
        if img is None:raise ValueError('Cannot decode '+name)
        caption_lines=extract_subtitle_lines(img)
        counts[handle]+=len(caption_lines)
        for line_idx,(x0,y0,x1,y1,mask) in enumerate(caption_lines):
            vec=line_density(mask)
            by_week[week].append((name,handle,line_idx,vec))
            by_file[(week,name)].append(vec)
    with (out/'character_cell_measurements.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['week','file','handle','subtitle_line']+[f'cell_{i:02}' for i in range(MAX_CELLS)])
        for week in range(8,13):
            for fn,handle,line_idx,vec in sorted(by_week[week]):
                w.writerow([week,fn,handle,line_idx]+[f'{x:.6f}' for x in vec])
    print('Subtitle lines measured:',sum(map(len,by_week.values())),dict(counts))
    weeks=[]; null_input=[]
    for wk in range(8,13):
        entries=by_week[wk]
        allarr=np.stack([v for _,_,_,v in entries])
        tarr=np.stack([v for _,h,_,v in entries if h=='toorisugari'])
        weeks.append(tarr.mean(axis=0)-allarr.mean(axis=0))
        # Each photo is the independent unit for the random-assignment test.
        names=sorted({name for name,_,_,_ in entries})
        sums=np.stack([np.sum(by_file[(wk,n)],axis=0) for n in names])
        ns=np.array([len(by_file[(wk,n)]) for n in names])
        actual=np.array([any(name==n and h=='toorisugari' for name,h,_,_ in entries) for n in names])
        null_input.append((sums,ns,actual,allarr.mean(axis=0)))
    return np.stack(weeks),null_input


def standardize(raw):
    x=raw[:,:GRID_WIDTH].copy()
    return (x-x.mean(axis=1,keepdims=True))/(x.std(axis=1,keepdims=True)+1e-9)


def bitmap(glyph):
    return np.array([[1 if c=='#' else 0 for c in row] for row in glyph],dtype=np.float64)


def decode(z):
    word=''; choices=[]
    for start,width in SLOTS:
        font=FONT5 if width==5 else FONT3
        result=[]
        for letter,glyph in font.items():
            template=2*bitmap(glyph)-1
            score=float(np.sum(z[:,start:start+width]*template))
            result.append((letter,score))
        result.sort(key=lambda pair:-pair[1])
        word+=result[0][0];choices.append(result)
    return word,choices


def mask_for_word(word):
    pieces=[]
    for i,c in enumerate(word):
        font=FONT5 if SLOTS[i][1]==5 else FONT3
        pieces.append(font[c])
    return np.array([[v=='#' for v in ''.join(pieces[k][r] + ('.' if k<3 else '') for k in range(4))] for r in range(5)])


def test_word(z,word):
    mask=mask_for_word(word)
    return float(np.corrcoef(z.ravel(),mask.astype(float).ravel())[0,1])


def permutation_test(null_input,observed_z,word='KAMI',iterations=1000):
    rng=np.random.default_rng(20261115)
    actual_corr=test_word(observed_z,word)
    nullscores=[]
    for i in range(iterations):
        trial=[]
        for sums,ns,is_toori,mean_all in null_input:
            ids=rng.choice(len(ns),size=int(is_toori.sum()),replace=False)
            trial.append(sums[ids].sum(axis=0)/ns[ids].sum()-mean_all)
        nullscores.append(test_word(standardize(np.stack(trial)),word))
    nullscores=np.asarray(nullscores)
    p=(1+int((nullscores>=actual_corr).sum()))/(1+iterations)
    return actual_corr, p, nullscores


def draw_results(z,word,out):
    mask=mask_for_word(word)
    scale=33; W=GRID_WIDTH*scale; H=5*scale
    # Relative signal: negative dark, positive bright, derived solely from JPEG data.
    im=Image.new('RGB',(W+80,2*H+125),(252,252,252))
    d=ImageDraw.Draw(im)
    d.text((40,10),'Measured: Toorisugari - weekly average (row-normalized)',fill=(25,25,25))
    for r in range(5):
        for c in range(GRID_WIDTH):
            val=np.clip(z[r,c]/2.5,-1,1)
            if val>=0:
                color=(int(236-130*val),int(236-130*val),int(236-130*val))
            else:
                color=(int(236+15*(-val)),int(236+15*(-val)),int(236+15*(-val)))
            x,y=40+c*scale,35+r*scale
            d.rectangle((x,y,x+scale-1,y+scale-1),fill=color,outline=(214,214,214))
    d.text((40,H+49),'Decoded 5-row reference glyphs (not measured pixels): '+word,fill=(25,25,25))
    for r in range(5):
        for c in range(GRID_WIDTH):
            x,y=40+c*scale,H+73+r*scale
            d.rectangle((x,y,x+scale-1,y+scale-1),fill=(33,33,33) if mask[r,c] else (249,249,249),outline=(224,224,224))
    im.save(out/'density_vs_decoded_word.png')
    # Separate clean numeric figure, no ideal glyphs. This is the evidence.
    heat=Image.new('RGB',(W,H))
    for r in range(5):
        for c in range(GRID_WIDTH):
            x,y=c*scale,r*scale
            intensity=int(np.clip(127+z[r,c]*55,0,255))
            heat.paste((intensity,intensity,intensity),(x,y,x+scale,y+scale))
    heat.save(out/'measured_density_only.png')


def main():
    if len(sys.argv)<2:
        sys.exit('Usage: python solve.py ORIGINAL_DATASET.zip [output_folder] [candidate_word]')
    path=Path(sys.argv[1]);out=Path(sys.argv[2]) if len(sys.argv)>2 else Path('kami_results')
    candidate_word=sys.argv[3].upper() if len(sys.argv)>3 else None
    out.mkdir(parents=True,exist_ok=True)
    with ZipFile(path) as archive:
        assign,anonymous=attribute_from_zip(archive,out)
        raw,inp=analyze(archive,assign,anonymous,out)
    with (out/'five_week_density.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['week']+[f'char_position_{i:02}' for i in range(MAX_CELLS)])
        for i,row in enumerate(raw):w.writerow([8+i]+[f'{v:+.8f}' for v in row])
    z=standardize(raw)
    word,choices=decode(z)
    print('\nDecoded letter candidates:')
    for k,candidates in enumerate(choices):
        print('  Position',k+1,':',', '.join(f'{l}={s:.3f}' for l,s in candidates[:4]))
    print('\nDecoded word:',word)
    print('Flag candidate from the decoder: SIGMOID_'+word)
    if candidate_word:
        if len(candidate_word)!=4 or any(candidate_word[i] not in (FONT5 if SLOTS[i][1]==5 else FONT3) for i in range(4)):
            raise ValueError('Candidate must have the 3-3-5-3 pixel-letter layout')
        r,p,null=permutation_test(inp,z,candidate_word)
        print(f'Prespecified candidate {candidate_word}: correlation={r:.4f}')
        print(f'Random-label permutation test: p={p:.5f} (1000 trials, 9 photos per week)')
        print('Note: the p-value tests this specified template against randomized photo labels;')
        print('it is not a correction for choosing font parameters or the word after seeing data.')
    if choices[0][0][1] - choices[0][1][1] < 0.5:
        print('Caution: first-letter template ranking has a narrow margin; interpret as evidence, not certainty.')
    draw_results(z,word,out)
    print('Saved attribution CSV, per-line cell densities, weekly grid and figures to',out)

if __name__=='__main__':main()
