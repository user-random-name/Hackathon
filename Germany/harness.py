"""
LMML "Recover the Black Box" — query harness.

Your main.py defines one function:

    def solve(oracle, points):
        y = oracle.ask([0.5, 0.5, 0.5, 0.5, 0.5])
        ys = oracle.ask_many([[1, 1, .5, .5, .5], [0, 0, .5, .5, .5]])
        ...
        return submission        # dict, see submission.example.json

`points` are the 300 challenge points to predict (challenge/challenge_set.json).

Rules, enforced here and not in your process:

- every point that reaches the harness costs one query, BUDGET in total;
- asking for more than BUDGET fails the run, no partial credit;
- points outside [0, 1]^5 are rejected with ValueError and cost nothing;
- the Oracle caches answers, so asking again for a known point is free;
- the run is killed after TIMEOUT seconds.

evaluation_script.py starts main.py in a separate Python process via run()
below. The black box never lives in that process; it only sees answers.
print() in main.py is fine, it goes to stderr.
"""
import importlib.util
import json
import math
import numbers
import os
import subprocess
import sys
import threading

BUDGET = 150
NDIM = 5
TIMEOUT = 50   # seconds; below the platform's 60 so the harness reports it


class BudgetExceeded(RuntimeError):
    pass


def check_point(x):
    """Return x as a list of 5 floats in [0, 1], or raise ValueError."""
    try:
        x = list(x)
    except TypeError:
        raise ValueError(f"a point is a list of {NDIM} numbers, got {x!r}") from None
    if len(x) != NDIM:
        raise ValueError(f"a point is a list of {NDIM} numbers, got {len(x)}")
    out = []
    for i, v in enumerate(x):
        if isinstance(v, bool) or not isinstance(v, numbers.Real):
            raise ValueError(f"x{i + 1}={v!r} is not a number")
        v = float(v)
        if not math.isfinite(v) or not 0.0 <= v <= 1.0:
            raise ValueError(f"x{i + 1}={v} is outside [0, 1]")
        out.append(v)
    return out


def _jsonable(o):
    if hasattr(o, "tolist"):          # numpy arrays and scalars
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serialisable")


class Oracle:
    """What solve() receives. Talks to the harness over a pipe."""

    def __init__(self, rx, tx, budget):
        self._rx, self._tx = rx, tx
        self.budget = budget
        self.spent = 0
        self.cache = {}          # key(point) -> y

    @property
    def remaining(self):
        return self.budget - self.spent

    @staticmethod
    def key(x):
        return tuple(round(float(v), 9) for v in x)

    def ask(self, x, phase=None):
        """One point. Returns y."""
        return self.ask_many([x], phase)[0]

    def ask_many(self, points, phase=None):
        """Several points in one round trip; only uncached ones are charged.
        `phase` is a free-text tag stored in the query log."""
        points = [check_point(p) for p in points]
        keys = [self.key(p) for p in points]
        todo = list({k: p for k, p in zip(keys, points)
                     if k not in self.cache}.values())
        if todo:
            self._tx.write(json.dumps({"points": todo, "phase": phase}) + "\n")
            self._tx.flush()
            line = self._rx.readline()
            if not line:
                raise BudgetExceeded("the harness stopped answering "
                                     "(budget exceeded or time limit)")
            reply = json.loads(line)
            if "error" in reply:
                raise ValueError(reply["error"])
            for p, y in zip(todo, reply["y"]):
                self.cache[self.key(p)] = y
            self.spent += len(todo)
        return [self.cache[k] for k in keys]


def run(script, f, points, budget=BUDGET, timeout=TIMEOUT):
    """Run solve() from `script` against f in a child process.

    Returns {"submission", "queries", "error"}; submission is None on any
    failure (crash, timeout, over budget, bad return value).
    """
    proc = subprocess.Popen([sys.executable, __file__, str(script)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True)
    timed_out = []
    timer = threading.Timer(timeout, lambda: (timed_out.append(1), proc.kill()))
    timer.start()
    queries, out = [], {"submission": None, "queries": None, "error": None}
    try:
        proc.stdin.write(json.dumps({"points": points, "budget": budget}) + "\n")
        proc.stdin.flush()
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                out["error"] = "main.py wrote to the harness pipe directly"
                break
            if "submission" in msg:
                if isinstance(msg["submission"], dict):
                    out["submission"] = msg["submission"]
                else:
                    out["error"] = "solve() must return a dict"
                break
            try:
                pts = [check_point(p) for p in msg.get("points") or []]
            except ValueError as exc:
                reply = {"error": str(exc)}            # rejected, costs nothing
            else:
                if len(queries) + len(pts) > budget:
                    out["error"] = (f"OVER BUDGET: asked for query "
                                    f"{len(queries) + len(pts)} of {budget}")
                    break
                ys = [f(p) for p in pts]
                for p, y in zip(pts, ys):
                    queries.append({"n": len(queries) + 1, "x": p, "y": y,
                                    "phase": msg.get("phase")})
                reply = {"y": ys}
            proc.stdin.write(json.dumps(reply) + "\n")
            proc.stdin.flush()
        else:
            out["error"] = (f"time limit of {timeout}s exceeded" if timed_out
                            else "main.py exited without returning a "
                                 "submission (traceback above, if any)")
    except BrokenPipeError:
        out["error"] = "main.py exited early (traceback above, if any)"
    finally:
        timer.cancel()
        proc.kill()
        proc.wait()
    out["queries"] = queries
    return out


def _child(script):
    """Inside the child process: load main.py, call solve(), send the result."""
    rx, tx = sys.stdin, sys.stdout
    sys.stdin, sys.stdout = open(os.devnull), sys.stderr
    hello = json.loads(rx.readline())
    spec = importlib.util.spec_from_file_location("participant_main", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    oracle = Oracle(rx, tx, hello["budget"])
    sub = module.solve(oracle, hello["points"])
    tx.write(json.dumps({"submission": sub}, default=_jsonable) + "\n")
    tx.flush()


if __name__ == "__main__":
    _child(sys.argv[1])
