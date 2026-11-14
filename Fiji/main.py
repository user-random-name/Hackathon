"""Walu Season: starter main.py (the greedy baseline).

This file already speaks the evaluator's protocol. The part to change is the
Strategy class. As shipped it is pure greedy: try every spot once, then keep
going back to whichever spot has the best average so far. It runs end to end
and it does NOT reach the threshold.

Check it locally on the practice reef:

    python evaluation_script.py main.py --dataset train

Rules of the protocol (see task_description.md):
  * read test-input.json from the current directory (bare filename),
  * for every cast print one spot number on stdout, then read one line from
    stdin with the catch in kg,
  * start a fresh Strategy every day (spot numbers are reshuffled daily),
  * anything else you want to print goes to stderr.
"""
import json
import math
import random
import sys

INIT_PULLS = 20
VAR_FLOOR = 1.0


class Strategy:
    def __init__(self, n_spots, n_casts, rng=None):
        self.n_spots = n_spots
        self.n_casts = n_casts
        self.rng = rng or random.Random(12345)
        self.n = [0] * n_spots
        self.s = [0.0] * n_spots
        self.ss = [0.0] * n_spots

    def choose(self):
        for spot in range(self.n_spots):
            if self.n[spot] < INIT_PULLS:
                return spot
        best, best_val = 0, -1e18
        for k in range(self.n_spots):
            n = self.n[k]
            mean = self.s[k] / n
            var = max(self.ss[k] / n - mean * mean, VAR_FLOOR)
            val = self.rng.gauss(mean, math.sqrt(var / n))
            if val > best_val:
                best, best_val = k, val
        return best

    def update(self, spot, catch):
        self.n[spot] += 1
        self.s[spot] += catch
        self.ss[spot] += catch * catch


def main():
    with open("test-input.json") as f:
        rules = json.load(f)

    rng = random.Random(12345)
    for _day in range(rules["n_days"]):
        agent = Strategy(rules["n_spots"], rules["n_casts"], rng)
        for _cast in range(rules["n_casts"]):
            spot = agent.choose()
            print(spot, flush=True)
            catch = float(sys.stdin.readline())
            agent.update(spot, catch)


if __name__ == "__main__":
    main()