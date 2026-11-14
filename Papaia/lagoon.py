"""The lagoon: a fishing-day simulator for task_papua_new_guinea.

A day is STEPS_PER_DAY moves of an outrigger canoe on a small grid. The canoe
starts on the village beach with an empty hold. The day's score is the number
of fish landed on the beach; fish still in the canoe when the day ends are lost.

State     (row, col, hold, steps_left)
              row 0 is the open sea (north), row 5 is the shore (south)
              hold is the number of fish in the canoe, 0..HOLD_CAPACITY
              steps_left counts down from STEPS_PER_DAY to 1
Actions   NORTH, SOUTH, EAST, WEST, FISH, WAIT

Every random event in a day is drawn from the day's seed, so the same seed and
the same decisions always give the same day.

Helpers: play_day() fishes one day with a policy, read_policy()/write_policy()
load and save policy CSVs, export_main() writes a submission main.py.

Usage:
    python lagoon.py                                    # draw the map
    python lagoon.py --policy test-output.csv --day 7   # replay one day
    python lagoon.py --policy test-output.csv --days 1-400
    python lagoon.py --policy test-output.csv --compare train-output.json
                                                        # check a policy against angkol

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import random
import sys
import zlib

# Map legend
BEACH = "H"      # the village beach: fish are landed here
SEAGRASS = "s"   # seagrass flats inside the lagoon
TROUT = "T"      # the outer reef drop-off
REEF = "#"       # barrier reef: a canoe cannot cross it
PASSAGE = "~"    # the reef passage, where the tide runs
WATER = "."

MAP = (
    ". . . . T T .",
    "# # # # ~ # #",
    ". . . . s s .",
    ". . . . s s .",
    ". . . . . . .",
    ". . H . . . .",
)

STEPS_PER_DAY = 24
HOLD_CAPACITY = 6
BITE_CHANCE = {SEAGRASS: 0.20, TROUT: 0.90}   # chance that one FISH adds a fish
TIDE_CHANCE = 0.40                            # chance the tide pushes the canoe
TIDE_PUSH = (1, 0)                            # one cell south, back into the lagoon

ACTIONS = ("NORTH", "SOUTH", "EAST", "WEST", "FISH", "WAIT")
MOVES = {"NORTH": (-1, 0), "SOUTH": (1, 0), "EAST": (0, 1), "WEST": (0, -1)}

GRID = tuple(tuple(line.split()) for line in MAP)
ROWS, COLS = len(GRID), len(GRID[0])
START = next((r, c) for r in range(ROWS) for c in range(COLS) if GRID[r][c] == BEACH)


def is_water(row: int, col: int) -> bool:
    """True for every cell a canoe can be on (anything but reef and off-map)."""
    return 0 <= row < ROWS and 0 <= col < COLS and GRID[row][col] != REEF


def all_states() -> list[tuple[int, int, int, int]]:
    """Every (row, col, hold, steps_left) a canoe can be asked to decide in."""
    return [
        (r, c, h, t)
        for r in range(ROWS)
        for c in range(COLS)
        if is_water(r, c)
        for h in range(HOLD_CAPACITY + 1)
        for t in range(STEPS_PER_DAY, 0, -1)
    ]


def _advance(row: int, col: int, hold: int, action: str, bite: bool, tide: bool):
    """Apply one decision with its chance events already decided.

    Returns (row, col, hold, landed).
    """
    if action in MOVES:
        dr, dc = MOVES[action]
        if is_water(row + dr, col + dc):
            row, col = row + dr, col + dc
    elif action == "FISH":
        if GRID[row][col] in BITE_CHANCE and hold < HOLD_CAPACITY and bite:
            hold += 1
    elif action != "WAIT":
        raise ValueError(f"unknown action {action!r}")

    if GRID[row][col] == PASSAGE and tide:
        dr, dc = TIDE_PUSH
        if is_water(row + dr, col + dc):
            row, col = row + dr, col + dc

    landed = 0
    if GRID[row][col] == BEACH:
        landed, hold = hold, 0
    return row, col, hold, landed


class Lagoon:
    """One fishing day.

    >>> day = Lagoon(seed=7)
    >>> state = day.reset()
    >>> state, landed, done = day.step("EAST")

    reset(state) starts the day part-way through, from any state: the rest of
    the day then gets the same tide and bites it would have had at that hour.
    """

    def __init__(self, seed: int):
        self.seed = seed
        rng = random.Random(seed)
        self._tide = [rng.random() for _ in range(STEPS_PER_DAY)]
        self._bite = [rng.random() for _ in range(STEPS_PER_DAY)]
        self.reset()

    def reset(self, state: tuple[int, int, int, int] | None = None) -> tuple[int, int, int, int]:
        """Start the day on the beach, or from `state` = (row, col, hold, steps_left)."""
        if state is None:
            state = (*START, 0, STEPS_PER_DAY)
        row, col, hold, steps_left = state
        if not is_water(row, col) or not 0 <= hold <= HOLD_CAPACITY or not 1 <= steps_left <= STEPS_PER_DAY:
            raise ValueError(f"{state} is not a state of the lagoon")
        self.row, self.col, self.hold, self.steps_left = row, col, hold, steps_left
        self.catch = 0
        return self.state

    @property
    def state(self) -> tuple[int, int, int, int]:
        return (self.row, self.col, self.hold, self.steps_left)

    def step(self, action: str) -> tuple[tuple[int, int, int, int], int, bool]:
        """Take one decision. Returns (next_state, fish_landed_this_step, day_over).

        When the day is over the canoe's state is still returned, but anything
        left in the hold is lost and is not part of the day's catch.
        """
        if self.steps_left <= 0:
            raise RuntimeError("the day is over, call reset()")
        k = STEPS_PER_DAY - self.steps_left
        bite = self._bite[k] < BITE_CHANCE.get(GRID[self.row][self.col], 0.0)
        tide = self._tide[k] < TIDE_CHANCE
        self.row, self.col, self.hold, landed = _advance(
            self.row, self.col, self.hold, action, bite, tide
        )
        self.steps_left -= 1
        self.catch += landed
        return self.state, landed, self.steps_left == 0


def play_day(policy, seed: int) -> tuple[int, list]:
    """Play one day. `policy` maps a state tuple to an action (dict or callable).

    Returns (fish landed, trajectory), where the trajectory is a list of
    (state, action, landed) per step.
    """
    act = policy.__getitem__ if isinstance(policy, dict) else policy
    day = Lagoon(seed)
    state = day.reset()
    trajectory = []
    done = False
    while not done:
        action = act(state)
        next_state, landed, done = day.step(action)
        trajectory.append((state, action, landed))
        state = next_state
    return day.catch, trajectory


def read_policy(path: str) -> dict:
    """Read a row,col,hold,steps_left,action CSV into {state: action}."""
    with open(path, newline="") as f:
        return {
            (int(r["row"]), int(r["col"]), int(r["hold"]), int(r["steps_left"])): r["action"].strip().upper()
            for r in csv.DictReader(f)
        }


def write_policy(policy, path: str, states=None) -> None:
    """Write {state: action} (or a callable) as a row,col,hold,steps_left,action CSV."""
    act = policy.__getitem__ if isinstance(policy, dict) else policy
    with open(path, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["row", "col", "hold", "steps_left", "action"])
        for s in states if states is not None else all_states():
            out.writerow([*s, act(s)])


_MAIN_TEMPLATE = '''"""main.py for task_papua_new_guinea, written by lagoon.export_main().

Reads test-input.csv (row,col,hold,steps_left) from the current directory and
writes test-output.csv (row,col,hold,steps_left,action) next to it. The policy
is stored below as a compressed table, so this file needs nothing else to run.
"""

import base64
import csv
import zlib

POLICY = (
{chunks}
)


def load_table():
    text = zlib.decompress(base64.b64decode("".join(POLICY))).decode()
    table = {{}}
    for line in text.splitlines():
        row, col, hold, steps_left, action = line.split(",")
        table[(int(row), int(col), int(hold), int(steps_left))] = action
    return table


def main():
    table = load_table()
    with open("test-input.csv", newline="") as src, open("test-output.csv", "w", newline="") as dst:
        out = csv.writer(dst)
        out.writerow(["row", "col", "hold", "steps_left", "action"])
        for rec in csv.DictReader(src):
            state = (int(rec["row"]), int(rec["col"]), int(rec["hold"]), int(rec["steps_left"]))
            out.writerow([*state, table.get(state, "WAIT")])


if __name__ == "__main__":
    main()
'''


def export_main(policy, path: str = "main.py") -> None:
    """Write a self-contained main.py that answers with `policy` for every state.

    `policy` is a {state: action} dict or a callable. States it has no answer
    for are written as WAIT.
    """
    act = policy.__getitem__ if isinstance(policy, dict) else policy
    lines = []
    for s in all_states():
        action = act(s)
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r} for state {s}")
        lines.append(",".join(map(str, s)) + "," + action)
    blob = base64.b64encode(zlib.compress("\n".join(lines).encode(), 9)).decode()
    chunks = "\n".join(f'    "{blob[i:i + 76]}",' for i in range(0, len(blob), 76))
    with open(path, "w") as f:
        f.write(_MAIN_TEMPLATE.format(chunks=chunks))


def render(row: int | None = None, col: int | None = None) -> str:
    """The map as text, with the canoe drawn as 'C' if a position is given."""
    lines = ["    " + " ".join(str(c) for c in range(COLS))]
    for r in range(ROWS):
        cells = ["C" if (r, c) == (row, col) else GRID[r][c] for c in range(COLS)]
        lines.append(f"{r:>2}  " + " ".join(cells))
    return "\n".join(lines)


LEGEND = (
    "H village beach (fish are landed here)   s seagrass flats   T outer reef drop-off\n"
    "# barrier reef (impassable)   ~ reef passage (the tide pushes canoes south)   . open water"
)


def check_policy_file(path: str) -> list[str]:
    """Problems that would make a policy CSV invalid as a submission ([] if none)."""
    problems, seen, duplicates, unknown = [], set(), 0, set()
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader, [])]
        if header != ["row", "col", "hold", "steps_left", "action"]:
            return [f"the header must be row,col,hold,steps_left,action (got {','.join(header) or 'nothing'})"]
        for rec in reader:
            if not rec or all(not x.strip() for x in rec):
                continue
            if len(rec) != 5:
                return [f"every row needs 5 values: {','.join(rec)}"]
            state = tuple(int(x) for x in rec[:4])
            duplicates += state in seen
            seen.add(state)
            if rec[4].strip().upper() not in ACTIONS:
                unknown.add(rec[4].strip())
    missing = set(all_states()) - seen
    if missing:
        problems.append(f"{len(missing)} states have no action, e.g. {min(missing)}")
    if duplicates:
        problems.append(f"{duplicates} states are answered more than once")
    if unknown:
        problems.append(f"unknown actions: {', '.join(sorted(unknown))}")
    return problems


def _parse_days(text: str) -> list[int]:
    if "-" in text:
        lo, hi = text.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(d) for d in text.split(",")]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", help="policy CSV (row,col,hold,steps_left,action)")
    ap.add_argument("--day", type=int, help="replay this day step by step")
    ap.add_argument("--days", help="average catch over days, e.g. 1-400 or 3,5,8")
    ap.add_argument("--compare", metavar="REFERENCE_JSON",
                    help="fish the days of a reference file (e.g. train-output.json) and compare with angkol")
    args = ap.parse_args(argv)

    if not args.policy:
        print(render())
        print(LEGEND)
        print(f"{STEPS_PER_DAY} steps per day, hold for {HOLD_CAPACITY} fish, {len(all_states())} states.")
        return 0

    try:
        policy = read_policy(args.policy)
        problems = check_policy_file(args.policy) if args.compare else []
    except (OSError, KeyError, ValueError) as exc:
        print(f"could not read {args.policy}: {exc}", file=sys.stderr)
        return 2
    if problems:
        print(f"{args.policy} would be rejected as a submission:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2
    try:
        if args.day is not None:
            catch, trajectory = play_day(policy, args.day)
            for (r, c, h, t), action, landed in trajectory:
                note = f"  landed {landed}" if landed else ""
                print(f"steps_left={t:>2}  at ({r},{c})  hold={h:>2}  -> {action}{note}")
            print(f"day {args.day}: {catch} fish landed")
        if args.days:
            days = _parse_days(args.days)
            total = sum(play_day(policy, d)[0] for d in days)
            print(f"{len(days)} days: {total / len(days):.3f} fish per day")
        if args.compare:
            with open(args.compare) as f:
                reference = json.load(f)
            days = reference["days"]
            mean = sum(play_day(policy, d)[0] for d in days) / len(days)
            ratio = mean / reference["angkol_mean"]
            target = reference["target_ratio"]
            verdict = "above" if ratio >= target else "below"
            print(f"{len(days)} days: your canoe {mean:.3f} fish per day, angkol {reference['angkol_mean']:.3f}")
            print(f"ratio {ratio:.3f} x angkol (target {target} x): {verdict} the target")
    except KeyError as exc:
        print(f"the policy has no action for state {exc.args[0]}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
