"""
The TRAP baseline: pure one-variable-at-a-time probing.

    python3 evaluation_script.py baseline/naive_owat.py --eval-type train

This is what most teams will do first, and it is what the task is
designed to punish. It spends 41 of its 150 queries, produces a clean
additive model with an excellent fit *to its own data*, and is
confidently wrong about the structure.

It is a complete, valid main.py: start from its shape, not its strategy.
"""
import statistics

VARS = ["x1", "x2", "x3", "x4", "x5"]
CENTER = [0.5] * 5
LEVELS = [0.0, 0.125, 0.25, 0.375, 0.625, 0.75, 0.875, 1.0]
DEAD = 0.02        # |slope| below this reads as "no effect"


def solve(oracle, points):
    y0 = oracle.ask(CENTER, "center")

    slopes, sweeps = {}, {}
    for i, name in enumerate(VARS):
        pts = []
        for v in LEVELS:
            p = CENTER[:]
            p[i] = v
            pts.append(p)
        ys = oracle.ask_many(pts, f"owat:{name}")
        sweeps[name] = list(zip(LEVELS, ys))
        slopes[name] = statistics.linear_regression(LEVELS + [0.5],
                                                    ys + [y0]).slope

    def predict(x):
        return y0 + sum(slopes[n] * (x[i] - 0.5) for i, n in enumerate(VARS))

    errs = [predict([v if k == i else 0.5 for k in range(5)]) - y
            for i, n in enumerate(VARS) for v, y in sweeps[n]]
    self_rmse = (sum(e * e for e in errs) / len(errs)) ** 0.5

    relevant = [n for n in VARS if abs(slopes[n]) > DEAD]
    irrelevant = [n for n in VARS if abs(slopes[n]) <= DEAD]

    print("--- naive OWAT conclusion -------------------------------")
    for n in VARS:
        print(f"  d f / d {n}  =  {slopes[n]:+.4f}"
              + ("   <- 'irrelevant'" if abs(slopes[n]) <= DEAD else ""))
    print(f"  self-RMSE on its own {oracle.spent} points: {self_rmse:.2e}"
          f"  (looks perfect)")

    return {
        "method": "one-variable-at-a-time sweeps around the centre point",
        "relevant_variables": relevant,
        "irrelevant_variables": irrelevant,
        "interactions": [],
        "discontinuities": [],
        "model_expression": " + ".join(
            [f"{y0:.4f}"] + [f"{slopes[n]:.4f}*({n}-0.5)" for n in relevant]),
        "predictions": [round(predict(p), 6) for p in points],
        "notes": (f"Each variable swept alone over [0,1] with the others held "
                  f"at 0.5. Every sweep is straight (self-RMSE "
                  f"{self_rmse:.2e}), so the function is additive and smooth. "
                  f"{', '.join(irrelevant)} had no measurable effect."),
        "confidence": "high",
    }
