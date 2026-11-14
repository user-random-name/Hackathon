#!/usr/bin/env python3
"""
LMML "Recover the Black Box" — evaluator.

    python3 evaluation_script.py main.py --eval-type train   # practice function
    python3 evaluation_script.py main.py --eval-type test    # hidden function

Runs main.py through harness.py against the black box, then scores the
submission solve() returns, out of 100:

    budget compliance        gate (>150 queries => fail, score 0)
    variable identification  20   which inputs matter, which do not
    interaction             20   the pair, and how they combine
    discontinuity           30   detected, gate variables, boundary, jump size
    predictive accuracy     30   RMSE on 300 held-out points (20 overall +
                                 10 inside the discontinuous region)

Queries are counted by the harness, not taken from the submission.
A test run scoring >= PASS_SCORE prints the flag. A train run never does.

Exit codes: 0 accepted, 1 rejected, 2 organiser error (hidden function
missing).

Note: `model_expression` is eval'd in a namespace with no builtins when
`predictions` is missing. That is not a sandbox; pass --no-expr to score
predictions only.
"""
import argparse
import importlib.util
import json
import math
import os
import sys

from harness import BUDGET, TIMEOUT, run

ROOT = os.path.dirname(os.path.abspath(__file__))
CHALLENGE = os.path.join(ROOT, "challenge", "challenge_set.json")
TRUTH_FILES = {"train": os.path.join(ROOT, "practice_function.py"),
               "test": os.path.join(ROOT, "evaluator", "hidden_function.py")}
FLAG = "SIGMOID_FACTORIAL"
PASS_SCORE = 70.0

HF = None   # the black box being scored against, loaded in main()

GOOD_RMSE, BAD_RMSE = 0.01, 2.0
COEF_TOL = 0.15          # relative tolerance on coefficients
JUMP_TOL = 0.15
COS_STRICT, COS_LOOSE = 0.995, 0.95
OFF_STRICT, OFF_LOOSE = 0.03, 0.10
# Challenge points lying within this distance of the true switching surface
# are excluded from the scored RMSE: whether you land on the right side of a
# knife edge is luck, not understanding. The unfiltered RMSE is still reported.
BOUNDARY_BAND = 0.005


# ----------------------------------------------------------------------
def rmse(pred, true):
    return math.sqrt(sum((p - t) ** 2 for p, t in zip(pred, true)) / len(true))


def rmse_points(rms, weight):
    if rms != rms or rms == float("inf"):
        return 0.0
    if rms <= GOOD_RMSE:
        return weight
    if rms >= BAD_RMSE:
        return 0.0
    span = math.log10(BAD_RMSE) - math.log10(GOOD_RMSE)
    return weight * (math.log10(BAD_RMSE) - math.log10(rms)) / span


def normalise_halfspace(coefs, threshold, active_when="greater_equal"):
    """Return (unit normal over x1..x5, offset) for the region c.x >= t."""
    vec = [float(coefs.get(v, 0.0)) for v in HF.VAR_NAMES]
    t = float(threshold)
    if str(active_when).lower() in ("less_equal", "less", "<=", "<", "below"):
        vec = [-c for c in vec]
        t = -t
    norm = math.sqrt(sum(c * c for c in vec))
    if norm < 1e-12:
        return None, None
    return [c / norm for c in vec], t / norm


def truth_halfspace():
    d = HF.TRUTH["discontinuity"]
    return normalise_halfspace(d["boundary"], d["threshold"])


def eval_expression(expr, point):
    env = {"__builtins__": {}, "abs": abs, "min": min, "max": max,
           "math": math, "exp": math.exp, "log": math.log, "sqrt": math.sqrt}
    env.update(dict(zip(HF.VAR_NAMES, point)))
    return float(eval(expr, env))  # noqa: S307 — trusted submissions only


# ----------------------------------------------------------------------
def score_variables(sub, result):
    claimed_rel = set(sub.get("relevant_variables") or [])
    claimed_irr = set(sub.get("irrelevant_variables") or [])
    truth_rel = set(HF.TRUTH["relevant"])
    pts, detail = 0.0, []
    for v in HF.VAR_NAMES:
        says_rel = v in claimed_rel
        says_irr = v in claimed_irr
        if says_rel == says_irr:                       # unstated or contradictory
            verdict = "unstated/contradictory"
            ok = False
        else:
            ok = says_rel == (v in truth_rel)
            verdict = ("correct" if ok else
                       ("MISSED (it matters)" if v in truth_rel
                        else "false positive (it is inert)"))
        pts += 4.0 if ok else 0.0
        detail.append(f"{v}: claimed {'relevant' if says_rel else 'inert'} — {verdict}")
    result["breakdown"]["variables"] = {"points": round(pts, 2), "max": 20,
                                        "detail": detail}
    return pts


def score_interaction(sub, result):
    truth = HF.TRUTH["interaction"]
    tvars = set(truth["variables"])
    pts, detail = 0.0, []
    match = None
    for entry in sub.get("interactions") or []:
        if set(entry.get("variables") or []) == tvars:
            match = entry
            break
    if match is None:
        detail.append(f"no interaction reported between {sorted(tvars)} "
                      f"— the single most important structural fact was missed")
    else:
        pts += 12.0
        detail.append(f"identified the {sorted(tvars)} interaction (+12)")
        c = match.get("coefficient")
        if c is None:
            detail.append("no coefficient given (+0/8)")
        else:
            err = abs(float(c) - truth["coefficient"]) / abs(truth["coefficient"])
            if err <= COEF_TOL:
                pts += 8.0
                detail.append(f"coefficient {float(c):+.3f} vs true "
                              f"{truth['coefficient']:+.3f} ({err:.1%} off) (+8)")
            else:
                partial = max(0.0, 8.0 * (1 - (err - COEF_TOL) / 0.35))
                pts += partial
                detail.append(f"coefficient {float(c):+.3f} vs true "
                              f"{truth['coefficient']:+.3f} ({err:.1%} off) "
                              f"(+{partial:.1f}/8)")
    spurious = [e for e in (sub.get("interactions") or [])
                if set(e.get("variables") or []) != tvars]
    if spurious:
        pen = min(6.0, 3.0 * len(spurious))
        pts -= pen
        detail.append(f"{len(spurious)} spurious interaction(s) reported "
                      f"(-{pen:.1f})")
    pts = max(0.0, pts)
    result["breakdown"]["interaction"] = {"points": round(pts, 2), "max": 20,
                                          "detail": detail}
    return pts


def score_discontinuity(sub, result):
    truth = HF.TRUTH["discontinuity"]
    tvars = set(truth["gate_variables"])
    tn, toff = truth_halfspace()
    pts, detail = 0.0, []
    entries = sub.get("discontinuities") or []
    if not entries:
        detail.append("no discontinuity reported — the switching behaviour "
                      "was missed entirely")
        result["breakdown"]["discontinuity"] = {"points": 0.0, "max": 30,
                                                "detail": detail}
        return 0.0
    pts += 8.0
    detail.append("a discontinuity was reported (+8)")

    best = None
    for e in entries:
        if set(e.get("gate_variables") or []) == tvars:
            best = e
            break
    if best is None:
        best = entries[0]
        detail.append(f"gate variables {sorted(set(best.get('gate_variables') or []))} "
                      f"!= true {sorted(tvars)} (+0/10)")
    else:
        pts += 10.0
        detail.append(f"gate variables {sorted(tvars)} correct (+10)")

    b = best.get("boundary") or {}
    coefs = b.get("coefficients")
    if coefs and b.get("threshold") is not None:
        n, off = normalise_halfspace(coefs, b["threshold"],
                                     b.get("active_when", "greater_equal"))
        if n is None:
            detail.append("degenerate boundary (+0/7)")
        else:
            cos = sum(a * c for a, c in zip(n, tn))
            doff = abs(off - toff)
            if cos >= COS_STRICT and doff <= OFF_STRICT:
                pts += 7.0
                tag = "(+7)"
            elif cos >= COS_LOOSE and doff <= OFF_LOOSE:
                pts += 3.5
                tag = "(+3.5, right family, imprecise)"
            else:
                tag = "(+0/7)"
            detail.append(f"boundary orientation cos={cos:.4f} (true 1.0), "
                          f"offset err={doff:.4f} {tag}")
    else:
        detail.append("no structured boundary {coefficients, threshold} given (+0/7)")

    j = best.get("jump")
    if j is None:
        detail.append("no jump size given (+0/5)")
    else:
        err = abs(abs(float(j)) - truth["jump"]) / truth["jump"]
        if err <= JUMP_TOL:
            pts += 5.0
            detail.append(f"jump {float(j):+.3f} vs true {truth['jump']:+.3f} "
                          f"({err:.1%} off) (+5)")
        else:
            partial = max(0.0, 5.0 * (1 - (err - JUMP_TOL) / 0.35))
            pts += partial
            detail.append(f"jump {float(j):+.3f} vs true {truth['jump']:+.3f} "
                          f"({err:.1%} off) (+{partial:.1f}/5)")
    result["breakdown"]["discontinuity"] = {"points": round(pts, 2), "max": 30,
                                            "detail": detail}
    return pts


def score_predictions(sub, result, points, allow_expr=True):
    truth_y = [HF.evaluate(p) for p in points]
    detail = []
    preds = sub.get("predictions")
    if preds is not None and len(preds) != len(points):
        detail.append(f"predictions has {len(preds)} entries, expected "
                      f"{len(points)} — ignored")
        preds = None
    source = "predictions[]"
    if preds is None:
        expr = sub.get("model_expression")
        if expr and allow_expr:
            try:
                preds = [eval_expression(expr, p) for p in points]
                source = "model_expression"
            except Exception as exc:                    # noqa: BLE001
                detail.append(f"model_expression failed to evaluate: {exc}")
                preds = None
        elif expr and not allow_expr:
            detail.append("model_expression present but --no-expr was set")
    if preds is None:
        detail.append("no usable predictions (+0/30)")
        result["breakdown"]["prediction"] = {"points": 0.0, "max": 30,
                                             "detail": detail}
        return 0.0

    preds = [float(p) for p in preds]
    d = HF.TRUTH["discontinuity"]
    tvec, tthr = d["boundary"], d["threshold"]

    def dist_to_boundary(p):
        s = sum(tvec[v] * p[i] for i, v in enumerate(HF.VAR_NAMES))
        n = math.sqrt(sum(c * c for c in tvec.values()))
        return abs(s - tthr) / n

    keep = [k for k, p in enumerate(points) if dist_to_boundary(p) >= BOUNDARY_BAND]
    dropped = len(points) - len(keep)
    raw = rmse(preds, truth_y)
    overall = rmse([preds[k] for k in keep], [truth_y[k] for k in keep])
    idx = [k for k in keep if HF.gate_active(points[k])]
    region = rmse([preds[k] for k in idx], [truth_y[k] for k in idx]) if idx else 0.0
    p_all = rmse_points(overall, 20.0)
    p_reg = rmse_points(region, 10.0)
    detail.append(f"source: {source}")
    detail.append(f"overall RMSE {overall:.5f} on {len(keep)} points "
                  f"(+{p_all:.1f}/20)")
    detail.append(f"RMSE inside the discontinuous region {region:.5f} on "
                  f"{len(idx)} points (+{p_reg:.1f}/10)")
    if dropped:
        detail.append(f"{dropped} point(s) within {BOUNDARY_BAND} of the true "
                      f"switching surface excluded as a coin flip "
                      f"(unfiltered RMSE would be {raw:.5f})")
    result["metrics"] = {"rmse_overall": round(overall, 6),
                         "rmse_unfiltered": round(raw, 6),
                         "rmse_in_region": round(region, 6),
                         "region_points": len(idx),
                         "boundary_points_excluded": dropped,
                         "max_abs_error": round(max(abs(p - t) for p, t
                                                    in zip(preds, truth_y)), 6)}
    result["breakdown"]["prediction"] = {"points": round(p_all + p_reg, 2),
                                         "max": 30, "detail": detail}
    return p_all + p_reg


# ----------------------------------------------------------------------
def evaluate(out, team, points, allow_expr=True):
    """Score one harness run (see harness.run)."""
    sub = out["submission"]
    result = {"team": team, "breakdown": {}, "metrics": {}, "flags": [],
              "queries_used": len(out["queries"]), "query_budget": BUDGET}
    if sub is None:
        result["score"] = 0.0
        result["failed"] = True
        result["flags"].append(out["error"])
        return result
    result["failed"] = False

    total = 0.0
    total += score_variables(sub, result)
    total += score_interaction(sub, result)
    total += score_discontinuity(sub, result)
    total += score_predictions(sub, result, points, allow_expr)
    result["score"] = round(total, 2)
    result["recovered_function"] = sub.get("model_expression")
    return result


def render(result, reveal):
    L = []
    L.append(f"# Evaluation — `{result['team']}`")
    L.append("")
    L.append(f"**Score: {result['score']:.1f} / 100**"
             + ("  — **FAILED**" if result["failed"] else ""))
    L.append("")
    L.append(f"- queries used: **{result['queries_used']} / "
             f"{result['query_budget']}**  (counted by the harness)")
    if result.get("recovered_function"):
        L.append(f"- recovered function: `{result['recovered_function']}`")
    for f in result["flags"]:
        L.append(f"- ⚠ {f}")
    if result.get("flag"):
        L.append(f"- FLAG RELEASE: {result['flag']}")
    L.append("")
    if result["metrics"]:
        m = result["metrics"]
        L.append(f"| metric | value |")
        L.append(f"|---|---|")
        for k, v in m.items():
            L.append(f"| {k} | {v} |")
        L.append("")
    order = ["variables", "interaction", "discontinuity", "prediction"]
    for key in order:
        b = result["breakdown"].get(key)
        if not b:
            continue
        L.append(f"## {key}  —  {b['points']:.1f} / {b['max']}")
        for d in b.get("detail", []):
            L.append(f"- {d}")
        L.append("")
    if reveal:
        L.append("## ground truth")
        L.append(f"```\nf(x) = {HF.expression()}\n```")
        L.append(f"- relevant: {HF.TRUTH['relevant']}, "
                 f"inert: {HF.TRUTH['irrelevant']}")
    return "\n".join(L)


def main():
    global HF
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scripts", nargs="+", help="participant main.py file(s)")
    ap.add_argument("--eval-type", choices=("train", "test"), default="test",
                    help="train: public practice function, never a flag; "
                         "test: hidden function")
    ap.add_argument("--timeout", type=float, default=TIMEOUT,
                    help="seconds per script (default %(default)s)")
    ap.add_argument("--no-expr", action="store_true",
                    help="never eval model_expression")
    ap.add_argument("--reveal", action="store_true",
                    help="test runs: show per-check detail and the ground "
                         "truth. Organisers only")
    ap.add_argument("--save", metavar="DIR",
                    help="write DIR/<team>/submission.json + query_log.jsonl")
    ap.add_argument("--json", action="store_true", help="emit JSON not markdown")
    ap.add_argument("--out", help="also write the report here")
    args = ap.parse_args()

    path = TRUTH_FILES[args.eval_type]
    if not os.path.exists(path):
        print(f"organiser error: {path} not found", file=sys.stderr)
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("black_box", path)
    HF = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(HF)
    with open(CHALLENGE) as fh:
        points = json.load(fh)["points"]
    # Test-run detail ("coefficient +5.2 vs true ...") would leak the hidden
    # function to whoever sees the output.
    reveal = args.eval_type == "train" or args.reveal

    results = []
    for script in args.scripts:
        stem = os.path.splitext(os.path.basename(script))[0]
        team = (os.path.basename(os.path.dirname(os.path.abspath(script)))
                if stem == "main" else stem)
        out = run(script, HF.evaluate, points, BUDGET, args.timeout)
        if args.save:
            d = os.path.join(args.save, team)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "submission.json"), "w") as fh:
                json.dump(out["submission"], fh, indent=2)
            with open(os.path.join(d, "query_log.jsonl"), "w") as fh:
                fh.writelines(json.dumps(q) + "\n" for q in out["queries"])
        r = evaluate(out, team, points, not args.no_expr)
        r["passed"] = r["score"] >= PASS_SCORE
        if r["passed"] and args.eval_type == "test":
            r["flag"] = FLAG
        if not reveal:
            for b in r["breakdown"].values():
                b.pop("detail", None)
        results.append(r)

    results.sort(key=lambda r: -r["score"])
    if args.json:
        text = json.dumps(results if len(results) > 1 else results[0], indent=2)
    else:
        parts = []
        if len(results) > 1:
            parts.append("# Leaderboard\n")
            parts.append("| rank | team | score | queries |")
            parts.append("|---|---|---|---|")
            for i, r in enumerate(results, 1):
                parts.append(f"| {i} | {r['team']} | {r['score']:.1f} | "
                             f"{r['queries_used']} |")
            parts.append("")
        parts.extend(render(r, reveal) for r in results)
        text = "\n".join(parts)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    sys.exit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()
