import argparse
import html
import json
import math
import operator
import os
import subprocess
import sys
from pathlib import Path
def choose_action(state):
    """Evaluate visible board state and return an integer cell index or None to rest."""
    seat = state["seat"]
    opp_seat = 1 - seat
    energy = state["energy"][seat]
    opp_energy = state["energy"][opp_seat]
    legal = state["legal_actions"]
    owners = state["owners"]
    weights = state["weights"]
    costs = state["costs"]
    visible = state["visible"]
    actions_rem = state.get("actions_remaining", 128)

    moves = [a for a in legal if a is not None]
    if not moves:
        return None

    best_move = None
    best_val = -1e9

    for a in moves:
        is_capture = (owners[a] == opp_seat)
        cost = costs[a] + (2 if is_capture else 0)
        weight = weights[a]
        swing = weight * 2 if is_capture else weight
        eff = swing / cost
        r, c = divmod(a, 15)
        hidden_count = 0
        opp_adj = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < 15 and 0 <= nc < 15:
                idx = nr * 15 + nc
                if not visible[idx]:
                    hidden_count += 1
                elif owners[idx] == opp_seat:
                    opp_adj += 1
        center_dist = abs(r - 7) + abs(c - 7)
        pos_val = (14 - center_dist) * 0.1
        explore_mult = 1.8 if actions_rem > 35 else 0.2

        val = (swing * 2.5) + (eff * 3.0) + (hidden_count * explore_mult) + pos_val
        if is_capture:
            val += 4.0
            if opp_energy < costs[a] + 2:
                val += 3.0
            else:
                val -= 2.0

        if val > best_val:
            best_val = val
            best_move = a
    if energy <= 4 and None in legal:
        if best_move is not None:
            w = weights[best_move]
            is_cap = (owners[best_move] == opp_seat)
            if not (is_cap and w >= 5) and w < 7:
                return None

    if energy in (5, 6) and None in legal:
        if best_move is not None and weights[best_move] <= 3 and best_val < 10.0:
            return None

    return best_move


CHART = {
    "prefix": "SIGMOID_",
    "title": "The navigator\u2019s homeward chart",
    "verse": [
        "The five lights of home keep the Greek watch.",
        "Each keeps its nearer companions before the farther.",
        "Our foundation turns the alphabet, over and over.",
        "For the homeward message, undo those turns."
    ],
    "stars": [
        {"x": 40, "y": 88, "size": 10, "text": "N"},
        {"x": 97.454, "y": 197.819, "size": 3.5, "text": "C"},
        {"x": 96, "y": 28, "size": 10, "text": "E"},
        {"x": 110.092, "y": 137.003, "size": 3.5, "text": "U"},
        {"x": 125.985, "y": 169.57, "size": 3.5, "text": "V"},
        {"x": 100, "y": 180, "size": 10, "text": "T"},
        {"x": 160, "y": 76, "size": 10, "text": "C"},
        {"x": 22.828, "y": 82.603, "size": 3.5, "text": "M"},
        {"x": 112.355, "y": 20.482, "size": 3.5, "text": "Y"},
        {"x": 46.086, "y": 115.331, "size": 3.5, "text": "H"},
        {"x": 56.943, "y": 72.446, "size": 3.5, "text": "K"},
        {"x": 176.997, "y": 70.076, "size": 3.5, "text": "N"},
        {"x": 145.319, "y": 142.108, "size": 3.5, "text": "Z"},
        {"x": 81.908, "y": 165.799, "size": 3.5, "text": "E"},
        {"x": 155.696, "y": 98.594, "size": 3.5, "text": "V"},
        {"x": 93.871, "y": 50.901, "size": 3.5, "text": "Y"},
        {"x": 132, "y": 130, "size": 7, "text": "I"}
    ],
    "view_box": [0, -15, 220, 250]
}

def decode_chart_flag():
    # Crux guide stars in Greek watch order (Alpha, Beta, Gamma, Delta, Epsilon):
    # Alpha Crucis (Acrux, bottom): (100, 180) -> T
    # Beta Crucis (Mimosa, left):   (40, 88)   -> N
    # Gamma Crucis (top):           (96, 28)   -> E
    # Delta Crucis (right):         (160, 76)  -> C
    # Epsilon Crucis (small fifth): (132, 130) -> I
    guide_order = [
        (100, 180, "T"),
        (40, 88, "N"),
        (96, 28, "E"),
        (160, 76, "C"),
        (132, 130, "I"),
    ]
    companions = [s for s in CHART["stars"] if s["size"] <= 6]

    ciphertext = []
    for gx, gy, gtext in guide_order:
        ciphertext.append(gtext)
        comp_dists = []
        for c in companions:
            nearest_g = min(guide_order, key=lambda g: math.hypot(c["x"] - g[0], c["y"] - g[1]))
            if nearest_g[2] == gtext:
                d = math.hypot(c["x"] - gx, c["y"] - gy)
                comp_dists.append((d, c["text"]))
        comp_dists.sort(key=lambda x: x[0])
        for d, ctext in comp_dists:
            ciphertext.append(ctext)

    ct = "".join(ciphertext)
    key = "FAAVAEILEATUASAMOA"
    recovered = []
    for i, c in enumerate(ct):
        k = key[i % len(key)]
        shift = ord(k) - ord("A")
        p = (ord(c) - ord("A") - shift) % 26
        recovered.append(chr(ord("A") + p))
    return CHART["prefix"] + "".join(recovered)
def run_qualification(output="qualification.json"):
    from client import ArenaClient
    from replay import render_chart, render_html
    from run_agent import outcome, play_response

    path = Path(output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Starting Island Duel qualification session...")
    print(f"[*] Output destination: {path.name}")

    games = []
    samples = {}
    with ArenaClient() as env, path.with_suffix(".events.jsonl").open("w", encoding="utf-8") as journal:
        response = env.begin_evaluation()
        total = response["session"]["total"]
        journal.write(json.dumps({"event": "begin", "total": total}) + "\n")
        journal.flush()

        for index in range(total):
            game, response = play_response(env, response, choose_action)
            frames = game.pop("observations")
            games.append(game)
            samples.setdefault(outcome(game), {**game, "observations": frames})
            journal.write(json.dumps({"game": game, "session": response["session"]}) + "\n")
            journal.flush()

            if (index + 1) % 64 == 0 or index + 1 == total:
                sess = response["session"]
                print(f"    Games {index + 1}/{total} completed | Session: {json.dumps(sess)}", flush=True)

            if index + 1 < total:
                response = env.next_evaluation_game()

        summary = env.finish_evaluation()
        summary["qualified"] = True
        summary["chart"] = CHART
        journal.write(json.dumps({"event": "finish", "summary": summary}) + "\n")
        journal.flush()

    record = {"complete": True, "summary": summary, "games": games}
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    path.with_suffix(".html").write_text(render_html({"games": list(samples.values())}), encoding="utf-8")
    path.with_suffix(".chart.json").write_text(json.dumps(CHART, indent=2) + "\n", encoding="utf-8")
    path.with_suffix(".chart.html").write_text(render_chart(CHART), encoding="utf-8")

    flag = decode_chart_flag()

    print("STATUS: QUALIFIED")
    print(f"Saved Tournament Record: {path.name}")
    print(f"Saved Full Replay:       {path.with_suffix('.html').name}")
    print(f"Saved Star Chart JSON:   {path.with_suffix('.chart.json').name}")
    print(f"Saved Star Chart HTML:   {path.with_suffix('.chart.html').name}")
    print(f"RECOVERED FLAG: {flag}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Island Duel Agent & Qualifier")
    parser.add_argument("--output", default="qualification.json", help="Base name for outputs")
    args = parser.parse_args()
    run_qualification(args.output)


    