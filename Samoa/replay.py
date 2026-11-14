"""Standalone participant-view replay; hidden cells remain hidden at every frame."""

import json


def render_html(record):
    return PAGE.replace("PAYLOAD", json.dumps(record).replace("<", "\\u003c"))


PAGE = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Island Duel · Fog replay</title>
<style>body{font:16px system-ui;background:#102332;color:#e2edf0;max-width:1000px;margin:32px auto;padding:0 20px}h1{font-size:28px}button,select{font:inherit;border-radius:5px;padding:8px;margin:4px}#board{display:grid;grid-template-columns:repeat(15,1fr);gap:3px;max-width:720px;margin:20px 0}.tile{aspect-ratio:1;background:#334e5e;display:grid;place-content:center;text-align:center;border-radius:3px;font-size:16px}.tile small{font-size:9px}.fog{background:#08131c;color:#78919d}.wall{background:#18282c;color:#8a9d9d}.p0{background:#16758b}.p1{background:#b05b30}input{width:95%;max-width:720px}p{line-height:1.5}.muted{color:#a9c0ca}</style>
<h1>Island Duel · Fog of war</h1><p class="muted">Participant view only. Teal is player 0; orange is player 1. A question mark means the tile is currently hidden. A game lasts 128 participant decisions.</p><select id="games" aria-label="Choose recorded game"></select><div id="board"></div><button id="first">Start</button><button id="back">Previous</button><button id="next">Next</button><button id="last">Final observation</button><br><input id="slider" aria-label="Participant decision" type="range" min="0" value="0"><p id="status"></p><p id="result"></p>
<script>
const record=PAYLOAD,select=document.getElementById('games'),slider=document.getElementById('slider');let frame=0;
record.games.forEach((g,i)=>{let option=document.createElement('option');option.value=i;option.textContent=g.label ?? `Group ${g.arena} · game ${g.game_index ?? i+1}`;select.append(option);});
function show(){let game=record.games[Number(select.value)],state=game.observations[frame],board=document.getElementById('board');board.replaceChildren();for(let i=0;i<225;i++){let cell=document.createElement('div');let owner=state.owners[i];cell.className='tile '+(!state.visible[i]?'fog':owner===-2?'wall':owner>=0?'p'+owner:'');cell.innerHTML=!state.visible[i]?'?':owner===-2?'×':`${state.weights[i]}<small>cost ${state.costs[i]}</small>`;board.append(cell);}slider.max=game.observations.length-1;slider.value=frame;document.getElementById('status').textContent=`Your player ${game.participant_seat} · Profile ${state.opponent_profile} · Decision ${frame}/128 · Points ${state.scores[0]} : ${state.scores[1]} · Energy ${state.energy.join(' : ')} · Remaining ${state.actions_remaining}`;document.getElementById('result').textContent=frame===128?(game.winner===null?'DRAW':game.winner===game.participant_seat?'YOU WON':'YOU LOST'):'Game in progress';}
select.onchange=()=>{frame=0;show();};slider.oninput=()=>{frame=Number(slider.value);show();};document.getElementById('first').onclick=()=>{frame=0;show();};document.getElementById('back').onclick=()=>{frame=Math.max(0,frame-1);show();};document.getElementById('next').onclick=()=>{frame=Math.min(128,frame+1);show();};document.getElementById('last').onclick=()=>{frame=128;show();};show();
</script></html>"""


def render_chart(chart):
    """Draw an earned cipher chart. Geometry is evidence; painter order is not."""
    import hashlib
    import html
    import math

    stars = chart["stars"]
    if not 5 <= len(stars) <= 20:
        raise ValueError("Invalid chart size")
    positions = set()
    for star in stars:
        if set(star) != {"x", "y", "size", "text"}:
            raise ValueError("Unexpected star metadata")
        if (
            not isinstance(star["text"], str)
            or len(star["text"]) != 1
            or not "A" <= star["text"] <= "Z"
        ):
            raise ValueError("Each star must contain one uppercase letter")
        if any(
            isinstance(star[k], bool)
            or not isinstance(star[k], (int, float))
            or not math.isfinite(star[k])
            for k in ["x", "y", "size"]
        ):
            raise ValueError("Invalid star geometry")
        if (
            not 0 < star["x"] < 220
            or not -15 < star["y"] < 235
            or not 0 < star["size"] <= 12
        ):
            raise ValueError("Star outside the chart")
        position = (star["x"], star["y"])
        if position in positions:
            raise ValueError("Overlapping star centers")
        positions.add(position)
    # An independent deterministic scramble prevents SVG/DOM order from being a
    # reading-order hint. No solver, grouping or key is present in this viewer.
    painted = sorted(
        stars,
        key=lambda s: hashlib.sha256(json.dumps(s, sort_keys=True).encode()).digest(),
    )
    elements = []
    for star in painted:
        x, y, radius = star["x"], star["y"], star["size"]
        points = []
        for vertex in range(10):
            angle = math.pi * vertex / 5 - math.pi / 2
            distance = radius if vertex % 2 == 0 else radius * 0.42
            points.append(
                f"{x + distance * math.cos(angle):.3f},{y + distance * math.sin(angle):.3f}"
            )
        font_size = min(7.2, radius * 0.8) if radius > 6 else 4.4
        label_y = y if radius > 6 else y + 9
        style = "guide-letter" if radius > 6 else "companion-letter"
        elements.append('<g class="star"><polygon points="' + " ".join(points) + '"/>')
        elements.append(
            f'<text class="letter {style}" x="{x}" y="{label_y}" font-size="{font_size:.2f}">{html.escape(star["text"])}</text></g>'
        )
    title = html.escape(chart["title"])
    verse = "<br>".join(html.escape(line) for line in chart["verse"])
    view_box = chart["view_box"]
    if view_box != [0, -15, 220, 250]:
        raise ValueError("Unsupported chart viewport")
    return (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Homeward chart</title>
<style>
body{font:18px system-ui;background:#091b32;color:#fff;max-width:1000px;margin:32px auto;padding:18px;text-align:center}
h1{font-size:clamp(25px,4vw,38px);margin:10px 0 20px}.verse{line-height:1.8;color:#dce7f6}
.frame{overflow:auto;margin-top:28px;border-top:7px solid #CE1126;border-bottom:7px solid #CE1126;border-radius:20px;background:#002B7F}
svg{display:block;width:100%;min-width:840px}polygon{fill:#fff}.letter{text-anchor:middle;font-family:monospace;font-weight:700}.guide-letter{fill:#002B7F;dominant-baseline:central}.companion-letter{fill:#fff}
.note{font-size:13px;color:#a9c1dd;margin:18px 0} @media print{body{margin:0;background:#fff}.frame{overflow:visible}svg{min-width:0}h1,.verse{color:#111}}
</style><h1>"""
        + title
        + '</h1><p class="verse">'
        + verse
        + """</p>
<div class="frame"><svg role="img" aria-label="A navigator’s chart of letter-bearing stars" viewBox="0 -15 220 250">"""
        + "".join(elements)
        + """</svg></div>
<p class="note">The navigator left no numbers.</p></html>"""
    )
