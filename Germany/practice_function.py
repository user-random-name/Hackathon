"""
Practice black box for "Recover the Black Box" — PUBLIC.

`python3 evaluation_script.py main.py --eval-type train` runs your main.py
against this function and scores it, so you can test your strategy as often
as you like. Same family as the graded function — main effects, one pairwise
interaction, one hard step — but different variables, coefficients and
boundary. A main.py that hardcodes anything from this file will fail the
real run; one that discovers the structure will not.
"""
from typing import Dict, List, Sequence

VAR_NAMES: List[str] = ["x1", "x2", "x3", "x4", "x5"]

TRUTH: Dict[str, object] = {
    "intercept": 2.0,
    "linear": {"x1": 1.5, "x2": 0.0, "x3": 0.4, "x4": 0.0, "x5": -0.6},
    "interaction": {
        "variables": ("x3", "x5"),
        "form": "centered_product",          # c * (x3 - 0.5) * (x5 - 0.5)
        "coefficient": -4.0,
        "center": 0.5,
    },
    "discontinuity": {
        "gate_variables": ("x1", "x4"),
        "boundary": {"x1": 1.0, "x2": 0.0, "x3": 0.0, "x4": 1.0, "x5": 0.0},
        "threshold": 1.7,                    # active where x1 + x4 >= 1.7
        "jump": 1.5,
    },
    "relevant": ["x1", "x3", "x4", "x5"],
    "irrelevant": ["x2"],
}


def gate_active(x: Sequence[float]) -> bool:
    return (x[0] + x[3]) >= 1.7


def f(x: Sequence[float]) -> float:
    x1, _x2, x3, x4, x5 = x
    y = 2.0 + 1.5 * x1 + 0.4 * x3 - 0.6 * x5
    y += -4.0 * (x3 - 0.5) * (x5 - 0.5)
    if (x1 + x4) >= 1.7:
        y += 1.5
    return y


def evaluate(x: Sequence[float]) -> float:
    return round(f(x), 6)


def expression() -> str:
    return ("2.0 + 1.5*x1 + 0.4*x3 - 0.6*x5 - 4.0*(x3-0.5)*(x5-0.5) "
            "+ (1.5 if x1 + x4 >= 1.7 else 0.0)")
