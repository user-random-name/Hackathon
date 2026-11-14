"""The reef simulator for task_fiji.

Shared by evaluation_script.py and the baseline notebook. You can read all of
it; there is nothing hidden in the code. What changes between the practice
reef and the reef you are graded on is the *data* (the spots' catch
distributions and the day seeds), not the rules below.

A day works like this:

* There are ``n_spots`` fishing spots, numbered 0 .. n_spots-1.
* You get ``n_casts`` casts. Each cast you pick one spot and get back the
  catch in kg (a float, rounded to 10 g, never negative).
* Spot numbers are shuffled every day using the day's seed, so "spot 3"
  on Monday and "spot 3" on Tuesday are usually different places. Whatever you
  learn has to be learned again within the day.

Reproducibility: every spot has its own random stream, seeded from the day
seed and the spot. The k-th cast at a given spot always returns the same
catch, no matter what order you visit the spots in. A deterministic strategy
therefore gets exactly the same total on the same seeds every time.

Only the standard library is used, so the numbers do not depend on the
installed NumPy version.
"""

from __future__ import annotations

import json
import pathlib
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Spot:
    """Catch distribution of one fishing spot.

    ``kind == "steady"``: catch ~ Normal(mean, sd), clipped at 0.

    ``kind == "school"``: with probability ``p`` a school passes and the catch is
    ~ Normal(big, big_sd); otherwise ~ Normal(base, base_sd). Clipped at 0.
    """

    kind: str
    mean: float = 0.0
    sd: float = 0.0
    p: float = 0.0
    big: float = 0.0
    big_sd: float = 0.0
    base: float = 0.0
    base_sd: float = 0.0

    def draw(self, rng: random.Random) -> float:
        if self.kind == "steady":
            x = rng.normalvariate(self.mean, self.sd)
        elif self.kind == "school":
            # Always consume the coin flip first so both branches use the
            # stream the same way.
            if rng.random() < self.p:
                x = rng.normalvariate(self.big, self.big_sd)
            else:
                x = rng.normalvariate(self.base, self.base_sd)
        else:
            raise ValueError(f"unknown spot kind {self.kind!r}")
        return round(max(0.0, x), 2)


@dataclass(frozen=True)
class Ocean:
    """Everything the evaluator needs for one dataset (practice or test)."""

    n_spots: int
    n_casts: int
    seeds: tuple[int, ...]
    spots: tuple[Spot, ...]
    threshold: float
    flag: str | None = None

    @property
    def n_days(self) -> int:
        return len(self.seeds)


def load_ocean(path: str | pathlib.Path) -> Ocean:
    raw = json.loads(pathlib.Path(path).read_text())
    spots = tuple(Spot(**s) for s in raw["spots"])
    ocean = Ocean(
        n_spots=int(raw["n_spots"]),
        n_casts=int(raw["n_casts"]),
        seeds=tuple(int(s) for s in raw["seeds"]),
        spots=spots,
        threshold=float(raw["threshold"]),
        flag=raw.get("flag"),
    )
    if len(ocean.spots) != ocean.n_spots:
        raise ValueError(f"{path}: n_spots={ocean.n_spots} but {len(ocean.spots)} spots listed")
    if len(set(ocean.seeds)) != len(ocean.seeds):
        raise ValueError(f"{path}: duplicate seeds")
    return ocean


class Day:
    """One day of fishing on a given seed. Call ``cast(spot)`` n_casts times."""

    def __init__(self, ocean: Ocean, seed: int):
        self.ocean = ocean
        self.seed = seed
        order = list(range(ocean.n_spots))
        random.Random(seed).shuffle(order)
        # label shown to the player -> index into ocean.spots
        self._where = order
        self._rngs = [random.Random(seed * 1000 + label) for label in range(ocean.n_spots)]
        self.casts_left = ocean.n_casts
        self.total = 0.0

    def cast(self, spot: int) -> float:
        if self.casts_left <= 0:
            raise RuntimeError("no casts left today")
        if not (0 <= spot < self.ocean.n_spots):
            raise ValueError(f"spot must be in 0..{self.ocean.n_spots - 1}, got {spot}")
        catch = self.ocean.spots[self._where[spot]].draw(self._rngs[spot])
        self.casts_left -= 1
        self.total += catch
        return catch

    def spot_means(self) -> list[float]:
        """Approximate true mean catch per spot label (for analysis only)."""
        return [expected_catch(self.ocean.spots[self._where[i]]) for i in range(self.ocean.n_spots)]


def expected_catch(spot: Spot, n: int = 50_000) -> float:
    """Monte Carlo estimate of a spot's mean catch (clipping makes the formula messy)."""
    rng = random.Random(0)
    return sum(spot.draw(rng) for _ in range(n)) / n


def play_day(strategy_cls, ocean: Ocean, seed: int, record: bool = False):
    """Run one day in-process. Handy for quick experiments in the notebook.

    ``strategy_cls(n_spots, n_casts)`` must provide ``choose() -> int`` and
    ``update(spot, catch)``. Returns the day's total, or ``(total, history)``
    with ``record=True`` where history is a list of ``(spot, catch)``.
    """
    day = Day(ocean, seed)
    agent = strategy_cls(ocean.n_spots, ocean.n_casts)
    history = []
    for _ in range(ocean.n_casts):
        spot = int(agent.choose())
        catch = day.cast(spot)
        agent.update(spot, catch)
        if record:
            history.append((spot, catch))
    return (day.total, history) if record else day.total


def play_all(strategy_cls, ocean: Ocean) -> list[float]:
    """Daily totals over every seed of the ocean, in seed order."""
    return [play_day(strategy_cls, ocean, s) for s in ocean.seeds]
