#!/usr/bin/env python3
"""The Braided River Run -- the simulator for the LMML 2026 New Zealand task.

Everything you need to try ideas against the valley: it reads a map, replays a
route and tells you whether it is legal and how many minutes it took, exposes
the valley as a reinforcement-learning style environment, and draws a map --
with a route on top -- as an SVG image you can open in any browser.

It contains no solver. Finding the route is the task.

Rules
-----
You start on ``S`` and must reach ``G``. Each action moves one cell or waits:

    U  up      D  down      L  left      R  right      W  wait

Map symbols:

===== =====================================================================
``.`` open ground
``,`` swamp -- passable, but slow to enter
``#`` rock -- never passable
``=`` deep water -- never passable
``~`` ford -- passable only at low tide (see below)
``S`` start hut, ``G`` goal hut
``A``  swing bridge (any capital letter except ``S`` and ``G``) -- passable
       only while that bridge is *down*
``a``  winch (any lowercase letter) -- always passable; *stepping onto* it
       flips every bridge on its list in the map header, down <-> up.
       Waiting on a winch does nothing.
===== =====================================================================

Time: the clock starts at minute 0 and every action takes time.

* A move takes 1 minute, or ``swamp_cost`` minutes when it ends on a swamp.
* Every cell has a height (the ``heights`` grid in the map file). A move that
  climbs takes ``climb_cost`` extra minutes per level gained. Going down or
  staying level costs nothing extra.
* ``W`` takes 1 minute.

The tide: with ``tide_period P`` and ``tide_low K`` in the map header, an
action may END on a ford only if the minute ``t`` it ends at has ``t % P < K``.
That covers stepping onto a ford AND waiting on one. Stepping off a ford is
always allowed.

Anything illegal -- into rock or deep water, onto a raised bridge, onto a
ford at high tide, or off the map -- makes the whole route invalid. The route
ends the moment it reaches ``G``; actions left over after that are invalid too.
A route's score is the minute it reaches ``G``.

Map file format
---------------
A header of ``key value`` lines, the line ``map``, the grid, then the line
``heights`` and one digit (0-9) per cell::

    tide_period 16
    tide_low 5
    swamp_cost 3
    climb_cost 1
    open B            # bridges that start down (omit the line if none do)
    winch a C         # winch a flips bridge C (a winch may list several)
    map
    ###########
    #S,.aB~C.G#
    ###########
    heights
    00000000000
    01121000110
    00000000000

Python API
----------
::

    from environment import load_map, simulate, ValleyEnv, render_svg

    valley = load_map("test-input.txt")
    trace = simulate(valley, "RRDD")      # raises IllegalMove on any broken rule
    minutes = trace[-1][0]

    env = ValleyEnv(valley)
    obs = env.reset()                     # Observation(row, col, t, bridges_down)
    env.legal_actions()                   # e.g. ['D', 'R', 'W']
    obs, reward, done, info = env.step("R")   # reward is minus the minutes taken

    with open("map.svg", "w", encoding="utf-8") as fh:
        fh.write(render_svg(valley, route="RRDD"))

CLI
---
    python environment.py --map test-input.txt                          # summary
    python environment.py --map test-input.txt --replay-file solution.txt
    python environment.py --map test-input.txt --render map.svg --replay-file solution.txt

Standard library only.
"""

from __future__ import annotations

import argparse
import codecs
import os
import sys
import textwrap
from collections import namedtuple
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

ACTIONS: Dict[str, Tuple[int, int]] = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
    "W": (0, 0),
}

WALL = "#"
WATER = "="
FORD = "~"
SWAMP = ","
START = "S"
GOAL = "G"
OPEN = "."


def is_bridge(symbol: str) -> bool:
    return symbol.isupper() and symbol not in (START, GOAL)


def is_winch(symbol: str) -> bool:
    return symbol.islower()


class MapError(Exception):
    """Raised when a map file cannot be parsed into a usable valley."""


class IllegalMove(Exception):
    """Raised when an action sequence breaks the rules of the valley."""


class Valley:
    """A parsed, immutable valley map plus its rules."""

    def __init__(
        self,
        grid: Sequence[str],
        tide_period: int,
        tide_low: int,
        open_bridges: Sequence[str],
        winches: Dict[str, Tuple[str, ...]],
        heights: Optional[Sequence[str]] = None,
        swamp_cost: int = 3,
        climb_cost: int = 1,
    ) -> None:
        self.grid = tuple(grid)
        self.height = len(self.grid)
        self.width = len(self.grid[0]) if self.grid else 0
        self.tide_period = tide_period
        self.tide_low = tide_low
        self.swamp_cost = swamp_cost
        self.climb_cost = climb_cost
        self.initial_open: FrozenSet[str] = frozenset(open_bridges)
        self.winches = {name: tuple(targets) for name, targets in winches.items()}

        if not self.grid:
            raise MapError("map has no rows")
        if any(len(row) != self.width for row in self.grid):
            raise MapError("map rows are not all the same width")
        if not 0 < self.tide_low <= self.tide_period:
            raise MapError("tide_low must be in 1..tide_period")
        if self.swamp_cost < 1 or self.climb_cost < 0:
            raise MapError("swamp_cost must be at least 1 and climb_cost at least 0")

        if heights is None:
            heights = ["0" * self.width] * self.height
        if len(heights) != self.height or any(len(row) != self.width for row in heights):
            raise MapError("the heights grid must be {} rows of {} digits, like the map".format(
                self.height, self.width))
        if any(not row.isdigit() for row in heights):
            raise MapError("the heights grid may contain only the digits 0-9")
        self.heights: Tuple[Tuple[int, ...], ...] = tuple(tuple(int(ch) for ch in row) for row in heights)

        symbols = {ch for row in self.grid for ch in row}
        self.bridges = frozenset(ch for ch in symbols if is_bridge(ch))
        for ch in symbols:
            if ch in (OPEN, SWAMP, WALL, WATER, FORD, START, GOAL) or is_bridge(ch):
                continue
            if is_winch(ch):
                if ch not in self.winches:
                    raise MapError("winch {!r} is on the map but has no 'winch' line".format(ch))
                continue
            raise MapError("unknown map symbol {!r}".format(ch))
        for name, targets in self.winches.items():
            missing = [b for b in targets if b not in self.bridges]
            if missing:
                raise MapError("winch {} flips bridges not on the map: {}".format(name, missing))
        if not self.initial_open <= self.bridges:
            raise MapError("'open' lists bridges not on the map")

        self.start = self._locate(START)
        self.goal = self._locate(GOAL)

    def _locate(self, symbol: str) -> Tuple[int, int]:
        found = [
            (r, c)
            for r, row in enumerate(self.grid)
            for c, cell in enumerate(row)
            if cell == symbol
        ]
        if len(found) != 1:
            raise MapError("map must contain exactly one {!r}, found {}".format(symbol, len(found)))
        return found[0]

    def cell(self, row: int, col: int) -> str:
        return self.grid[row][col]

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def tide_is_low(self, time: int) -> bool:
        """True when an action ending at minute ``time`` may end on a ford."""
        return time % self.tide_period < self.tide_low

    def duration(self, row: int, col: int, new_row: int, new_col: int) -> int:
        """Minutes taken by the action from (row, col) to (new_row, new_col)."""
        if (row, col) == (new_row, new_col):
            return 1
        base = self.swamp_cost if self.cell(new_row, new_col) == SWAMP else 1
        climb = self.heights[new_row][new_col] - self.heights[row][col]
        return base + self.climb_cost * max(0, climb)

    def passable(self, row: int, col: int, down: FrozenSet[str], time: int) -> bool:
        """May an action ending at minute ``time`` end on this cell, given the bridges down?"""
        if not self.in_bounds(row, col):
            return False
        symbol = self.cell(row, col)
        if symbol in (WALL, WATER):
            return False
        if symbol == FORD:
            return self.tide_is_low(time)
        if is_bridge(symbol):
            return symbol in down
        return True

    def toggle(self, down: FrozenSet[str], row: int, col: int) -> FrozenSet[str]:
        """Apply the winch on this cell, if there is one."""
        symbol = self.cell(row, col)
        if symbol not in self.winches:
            return down
        return frozenset(down.symmetric_difference(self.winches[symbol]))

    def render(self) -> str:
        """The valley as map-file text."""
        header = [
            "tide_period {}".format(self.tide_period),
            "tide_low {}".format(self.tide_low),
            "swamp_cost {}".format(self.swamp_cost),
            "climb_cost {}".format(self.climb_cost),
        ]
        if self.initial_open:
            header.append("open {}".format(" ".join(sorted(self.initial_open))))
        for name in sorted(self.winches):
            header.append("winch {} {}".format(name, " ".join(self.winches[name])))
        header.append("map")
        heights = ["".join(str(h) for h in row) for row in self.heights]
        return "\n".join(header + list(self.grid) + ["heights"] + heights) + "\n"


_INT_KEYS = ("tide_period", "tide_low", "swamp_cost", "climb_cost")


def parse_map(text: str) -> Valley:
    """Parse a map file into a :class:`Valley`."""
    numbers = {"tide_period": 10, "tide_low": 3, "swamp_cost": 3, "climb_cost": 1}
    open_bridges: List[str] = []
    winches: Dict[str, Tuple[str, ...]] = {}
    grid: List[str] = []
    heights: List[str] = []
    section = "header"

    for raw in text.splitlines():
        if section != "header":
            if not raw.strip():
                continue
            if raw.strip() == "heights" and section == "map":
                section = "heights"
                continue
            (grid if section == "map" else heights).append(raw.rstrip("\n"))
            continue
        line = raw.split("#", 1)[0].strip() if not raw.lstrip().startswith("#") else ""
        if not line:
            continue
        parts = line.split()
        key = parts[0]
        if key == "map":
            section = "map"
        elif key in _INT_KEYS:
            if len(parts) != 2:
                raise MapError("'{}' needs exactly one whole number".format(key))
            try:
                numbers[key] = int(parts[1])
            except ValueError:
                raise MapError("'{}' needs a whole number, got {!r}".format(key, parts[1])) from None
        elif key == "open":
            open_bridges = parts[1:]
        elif key == "winch":
            if len(parts) < 3:
                raise MapError("winch line needs a name and at least one bridge: {!r}".format(line))
            if len(set(parts[2:])) != len(parts[2:]):
                raise MapError("winch {} lists the same bridge twice".format(parts[1]))
            winches[parts[1]] = tuple(parts[2:])
        else:
            raise MapError("unknown header key {!r}".format(key))

    if section == "header":
        raise MapError("map file has no 'map' section")
    return Valley(grid, numbers["tide_period"], numbers["tide_low"], open_bridges, winches,
                  heights=heights if section == "heights" else None,
                  swamp_cost=numbers["swamp_cost"], climb_cost=numbers["climb_cost"])


def load_map(path: str) -> Valley:
    with open(path, "r", encoding="utf-8") as handle:
        return parse_map(handle.read())


class RouteFileError(Exception):
    """Raised when a solution file cannot be read as text."""


def read_route_file(path: str) -> str:
    """Read a solution file as text, whichever way an editor saved it.

    Accepts UTF-8 with or without a byte-order mark, and UTF-16 or UTF-32 with
    one (what Windows PowerShell and some editors write). Raises
    :class:`RouteFileError` for anything else, and ``OSError`` if the file
    cannot be opened.
    """
    with open(path, "rb") as handle:
        data = handle.read()
    encoding = "utf-8-sig"
    for bom, name in ((codecs.BOM_UTF32_LE, "utf-32"), (codecs.BOM_UTF32_BE, "utf-32"),
                      (codecs.BOM_UTF16_LE, "utf-16"), (codecs.BOM_UTF16_BE, "utf-16")):
        if data.startswith(bom):
            encoding = name
            break
    try:
        return data.decode(encoding)
    except UnicodeDecodeError as exc:
        raise RouteFileError("{} is not a UTF-8 or UTF-16 text file ({})".format(path, exc.reason)) from None


def normalise(actions: str) -> str:
    """Reduce a solution to a bare action sequence.

    Everything after a ``#`` on a line is a comment. Whitespace of any kind is
    ignored and letters are upper-cased, so a route may be written on one line
    or many, in either case, with notes. Any other character is kept, so the
    simulator can report it as an invalid action.
    """
    kept = [line.split("#", 1)[0] for line in actions.lstrip("﻿").splitlines()]
    return "".join("".join(kept).split()).upper()


def _why_illegal(valley: Valley, row: int, col: int) -> str:
    if not valley.in_bounds(row, col):
        return "walked off the map"
    symbol = valley.cell(row, col)
    if symbol == WALL:
        return "moved into rock at ({}, {})".format(row, col)
    if symbol == WATER:
        return "moved into deep water at ({}, {})".format(row, col)
    if symbol == FORD:
        return "ended on a ford at high tide at ({}, {})".format(row, col)
    return "moved onto bridge {} while it is raised at ({}, {})".format(symbol, row, col)


def _step(valley: Valley, row: int, col: int, down: FrozenSet[str], time: int, action: str):
    """Apply one action. Returns ``(row, col, down, time, None)`` after it, or
    ``(row, col, down, time, reason)`` with the state unchanged if it is illegal.
    The one place the rules of movement live: the replay, the drawing and the
    environment all go through it."""
    d_row, d_col = ACTIONS[action]
    new_row, new_col = row + d_row, col + d_col
    if not valley.in_bounds(new_row, new_col):
        return row, col, down, time, _why_illegal(valley, new_row, new_col)
    new_time = time + valley.duration(row, col, new_row, new_col)
    if not valley.passable(new_row, new_col, down, new_time):
        return row, col, down, time, "{} (at minute {})".format(
            _why_illegal(valley, new_row, new_col), new_time)
    if (new_row, new_col) != (row, col):
        down = valley.toggle(down, new_row, new_col)
    return new_row, new_col, down, new_time, None


def simulate(valley: Valley, actions: str) -> List[Tuple[int, int, int, FrozenSet[str]]]:
    """Replay an action sequence, returning the trace it produced.

    Each trace entry is ``(minute, row, col, bridges_down)``; the first entry is
    the starting state at minute 0, so the route's score is ``trace[-1][0]`` and
    its number of actions is ``len(trace) - 1``. Raises :class:`IllegalMove` on
    any rule violation, on running out of actions before the goal, or on actions
    left over after reaching it.
    """
    actions = normalise(actions)
    row, col = valley.start
    down = valley.initial_open
    time = 0
    trace = [(0, row, col, down)]

    for index, action in enumerate(actions):
        if action not in ACTIONS:
            raise IllegalMove("action {}: {!r} is not one of U D L R W".format(index + 1, action))
        row, col, down, time, reason = _step(valley, row, col, down, time, action)
        if reason:
            raise IllegalMove("action {}: {}".format(index + 1, reason))
        trace.append((time, row, col, down))

        if (row, col) == valley.goal:
            if index + 1 != len(actions):
                raise IllegalMove(
                    "reached the goal after {} action(s), at minute {}, but {} action(s) remain".format(
                        index + 1, time, len(actions) - index - 1
                    )
                )
            return trace

    raise IllegalMove(
        "sequence of {} action(s) ended at ({}, {}), not on the goal".format(len(actions), row, col)
    )


# --------------------------------------------------------------------------- #
# Reinforcement-learning style interface
# --------------------------------------------------------------------------- #

Observation = namedtuple("Observation", "row col t bridges_down")


class ValleyEnv:
    """The valley as an episodic environment.

    * ``reset()`` returns the first :class:`Observation`.
    * ``step(action)`` returns ``(observation, reward, done, info)``. The reward
      is minus the minutes the action took, so the return of an episode is minus
      the minute it ends at. ``info["minutes"]`` holds that duration. Reaching
      ``G`` ends the episode with ``info["goal"] = True``.
    * An illegal action ends the episode at once with reward
      :attr:`ILLEGAL_REWARD` and the reason in ``info["illegal"]``. Use
      :meth:`legal_actions` to avoid that.
    * ``max_steps`` caps the number of actions in an episode
      (``info["truncated"] = True``).

    ``observation.t`` is the clock in minutes.
    """

    ILLEGAL_REWARD = -100.0

    def __init__(self, valley: Valley, max_steps: int = 2000) -> None:
        self.valley = valley
        self.max_steps = max_steps
        self.reset()

    def reset(self) -> Observation:
        self.row, self.col = self.valley.start
        self.t = 0
        self.down = self.valley.initial_open
        self.done = False
        self.route: List[str] = []
        return self.observation()

    def observation(self) -> Observation:
        return Observation(self.row, self.col, self.t, self.down)

    def legal_actions(self) -> List[str]:
        if self.done:
            return []
        return [action for action in ACTIONS
                if _step(self.valley, self.row, self.col, self.down, self.t, action)[4] is None]

    def step(self, action: str):
        if self.done:
            raise RuntimeError("episode is over; call reset()")
        if isinstance(action, str):
            action = action.strip().upper()
        if action not in ACTIONS:
            raise ValueError("unknown action {!r}; use one of U D L R W".format(action))
        self.route.append(action)
        row, col, down, time, reason = _step(self.valley, self.row, self.col, self.down, self.t, action)
        if reason:
            self.done = True
            return self.observation(), self.ILLEGAL_REWARD, True, {"illegal": reason}
        minutes = time - self.t
        self.row, self.col, self.down, self.t = row, col, down, time
        info: Dict[str, object] = {"minutes": minutes}
        if (self.row, self.col) == self.valley.goal:
            self.done = True
            info["goal"] = True
        elif len(self.route) >= self.max_steps:
            self.done = True
            info["truncated"] = True
        return self.observation(), -float(minutes), self.done, info


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

_CELL = 18
_COLOURS = {
    "rock": "#6f6458", "rock_edge": "#5a5047",
    "water": "#3d7fb3", "water_hi": "#5b98c7", "ford": "#9fd0ea", "plank": "#9a6a3c",
    "swamp": "#8a9a5b", "reed": "#4f5f2a",
    "winch": "#f2c14e", "winch_edge": "#7a5a12", "start": "#2f8f5b", "goal": "#c0392b",
    "ink": "#2b2b2b", "panel": "#fbf8f0", "contour": "#7b5e3b",
}
# Ground tint by height: low and sandy to high and brown.
_HEIGHT_TINTS = ("#f3ecd6", "#e8dcb6", "#d6bf8c", "#c2a268", "#a9854c", "#8e6b37",
                 "#7a5a2e", "#684b26", "#563d1f", "#453018")


def _route_points(valley: Valley, actions: str):
    """Cells visited by a route (waits included); stops at the first illegal move.

    Returns ``(points, failure)`` where each point is ``(row, col, kind, minute)``
    and ``failure`` is ``(row, col, reason)`` or ``None``. A route that stops
    short of ``G``, or keeps going after reaching it, is a failure too -- the
    same verdict :func:`simulate` gives.
    """
    row, col = valley.start
    down = valley.initial_open
    time = 0
    points = [(row, col, "start", 0)]
    actions = normalise(actions)
    for index, action in enumerate(actions):
        if action not in ACTIONS:
            return points, (row, col, "action {}: {!r} is not one of U D L R W".format(index + 1, action))
        d_row, d_col = ACTIONS[action]
        new_row, new_col, down, time, reason = _step(valley, row, col, down, time, action)
        if reason:
            target = (row + d_row, col + d_col)
            fail_r, fail_c = target if valley.in_bounds(*target) else (row, col)
            return points, (fail_r, fail_c, "action {}: {}".format(index + 1, reason))
        kind = "wait" if action == "W" else ("winch" if valley.cell(new_row, new_col) in valley.winches else "move")
        row, col = new_row, new_col
        points.append((row, col, kind, time))
        if (row, col) == valley.goal:
            remaining = len(actions) - index - 1
            if remaining:
                return points, (row, col, "reached the goal after {} action(s), at minute {}, but {} action(s) "
                                "remain".format(index + 1, time, remaining))
            return points, None
    return points, (row, col, "sequence of {} action(s) ended at ({}, {}), not on the goal".format(
        len(actions), row, col))


def render_svg(valley: Valley, route: Optional[str] = None, title: str = "The Braided River Run") -> str:
    """Draw the valley as an SVG image, optionally with a route on top.

    Ground is tinted by height with contour lines between levels, and swamps
    carry reeds. Bridges are drawn in their starting state. A route is coloured
    green to red from start to finish; worked winches get a dashed ring, waits a
    ``Wn`` marker, every 25 minutes a numbered dot, and an illegal move a red
    cross.
    """
    C = _COLOURS
    H, W = valley.height, valley.width
    cell = _CELL
    pad, top, gutter, panel_w = 16, 46, 22, 250
    map_w, map_h = W * cell, H * cell
    width = pad + gutter + map_w + pad + panel_w + pad
    height = max(top + gutter + map_h + pad, top + 700)
    ox, oy = pad + gutter, top + gutter
    out: List[str] = []
    icons: List[str] = []
    a = out.append

    def centre(r: int, c: int) -> Tuple[float, float]:
        return ox + c * cell + cell / 2, oy + r * cell + cell / 2

    a('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
      'font-family="Segoe UI, Helvetica, Arial, sans-serif">'.format(w=width, h=height))
    a('<defs>'
      '<pattern id="waves" width="12" height="8" patternUnits="userSpaceOnUse">'
      '<path d="M0 5 Q3 2 6 5 T12 5" fill="none" stroke="#ffffff" stroke-opacity="0.55" stroke-width="1.2"/>'
      '</pattern>'
      '<pattern id="deep" width="16" height="10" patternUnits="userSpaceOnUse">'
      '<path d="M0 6 Q4 3 8 6 T16 6" fill="none" stroke="{}" stroke-opacity="0.45" stroke-width="1"/>'
      '</pattern>'
      '<pattern id="reeds" width="9" height="9" patternUnits="userSpaceOnUse">'
      '<path d="M2 8 L2 3 M4.5 8 L5.5 2 M7 8 L7 4" stroke="{}" stroke-width="1" stroke-linecap="round"/>'
      '</pattern></defs>'.format(C["water_hi"], C["reed"]))
    a('<rect width="100%" height="100%" fill="#f6f1e4"/>')
    a('<text x="{}" y="30" font-size="20" font-weight="700" fill="{}">{}</text>'.format(pad, C["ink"], escape(title)))
    for c in range(0, W, 5):
        a('<text x="{}" y="{}" font-size="9.5" fill="#7a6f5f" text-anchor="middle">{}</text>'.format(
            ox + c * cell + cell / 2, oy - 6, c))
    for r in range(0, H, 5):
        a('<text x="{}" y="{}" font-size="9.5" fill="#7a6f5f" text-anchor="end">{}</text>'.format(
            ox - 5, oy + r * cell + cell / 2 + 3.5, r))

    def tile(r: int, c: int, fill: str) -> None:
        a('<rect x="{}" y="{}" width="{}" height="{}" fill="{}"/>'.format(
            ox + c * cell, oy + r * cell, cell, cell, fill))

    def walkable(r: int, c: int) -> bool:
        return valley.in_bounds(r, c) and valley.cell(r, c) not in (WALL, WATER)

    for r, line in enumerate(valley.grid):
        for c, ch in enumerate(line):
            x, y = ox + c * cell, oy + r * cell
            cx, cy = centre(r, c)
            if ch == WALL:
                tile(r, c, C["rock"])
            elif ch == WATER:
                tile(r, c, C["water"])
                tile(r, c, "url(#deep)")
            elif ch == FORD:
                tile(r, c, C["ford"])
                tile(r, c, "url(#waves)")
            elif is_bridge(ch):
                tile(r, c, C["water"])
                down = ch in valley.initial_open
                plank = ('<rect x="{}" y="{}" width="{}" height="{}" rx="2" fill="{}" stroke="{}" '
                         'stroke-width="{}"{}/>')
                a(plank.format(
                    x + 2.5, y + 1, cell - 5, cell - 2, C["plank"] if down else "none",
                    "#3b2410" if down else "#f4d9a8", 1 if down else 2,
                    "" if down else ' stroke-dasharray="3 2.5"'))
                if down:
                    for k in (5, 9, 13):
                        a('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#000" stroke-opacity="0.25"/>'.format(
                            x + 2.5, y + k, x + cell - 2.5, y + k))
                icons.append('<text x="{}" y="{}" font-size="11" font-weight="700" text-anchor="middle" '
                             'fill="#fff" stroke="#3b2410" stroke-width="0.6">{}</text>'.format(cx, cy + 4, ch))
            else:
                tile(r, c, _HEIGHT_TINTS[valley.heights[r][c]])
                if ch == SWAMP:
                    tile(r, c, C["swamp"])
                    a('<rect x="{}" y="{}" width="{}" height="{}" fill="url(#reeds)"/>'.format(x, y, cell, cell))
                if ch in valley.winches:
                    icons.append('<g><title>winch {} flips {}</title>'
                                 '<circle cx="{}" cy="{}" r="7.5" fill="{}" stroke="{}" stroke-width="1.5"/>'
                                 '<text x="{}" y="{}" font-size="10" font-weight="700" text-anchor="middle" '
                                 'fill="{}">{}</text></g>'.format(
                                     ch, ", ".join(valley.winches[ch]), cx, cy, C["winch"], C["winch_edge"],
                                     cx, cy + 3.5, C["ink"], ch))
                elif ch in (START, GOAL):
                    colour = C["start"] if ch == START else C["goal"]
                    icons.append('<path d="M{0} {1} L{2} {3} L{4} {1} L{4} {5} L{0} {5} Z" fill="{6}" '
                                 'stroke="#fff" stroke-width="1"/>'.format(
                                     x + 3, y + 8, x + cell / 2, y + 2, x + cell - 3, y + cell - 2, colour))
                    icons.append('<text x="{}" y="{}" font-size="9" font-weight="700" text-anchor="middle" '
                                 'fill="#fff">{}</text>'.format(cx, y + cell - 4, ch))
    # Contour lines between walkable neighbours at different heights; a tick
    # shows the uphill side.
    for r in range(H):
        for c in range(W):
            if not walkable(r, c):
                continue
            for dr, dc in ((0, 1), (1, 0)):
                r2, c2 = r + dr, c + dc
                if not walkable(r2, c2) or valley.heights[r][c] == valley.heights[r2][c2]:
                    continue
                if dc:
                    x0 = x1 = ox + c2 * cell
                    y0, y1 = oy + r * cell, oy + (r + 1) * cell
                else:
                    y0 = y1 = oy + r2 * cell
                    x0, x1 = ox + c * cell, ox + (c + 1) * cell
                a('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="{}" stroke-opacity="0.85"/>'.format(
                    x0, y0, x1, y1, C["contour"],
                    1 + 0.8 * (abs(valley.heights[r][c] - valley.heights[r2][c2]) - 1)))
    a('<rect x="{}" y="{}" width="{}" height="{}" fill="none" stroke="{}" stroke-width="2"/>'.format(
        ox, oy, map_w, map_h, C["rock_edge"]))

    minutes = 0
    failure = None
    if route is not None:
        points, failure = _route_points(valley, route)
        minutes = points[-1][3]
        n = len(points) - 1
        for i in range(1, len(points)):
            r0, c0 = points[i - 1][:2]
            r1, c1 = points[i][:2]
            if (r0, c0) == (r1, c1):
                continue
            x0, y0 = centre(r0, c0)
            x1, y1 = centre(r1, c1)
            a('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="hsl({},85%,42%)" stroke-width="4" '
              'stroke-linecap="round" stroke-opacity="0.9"/>'.format(
                  x0, y0, x1, y1, int(130 - 130 * i / max(1, n))))
        i = 1
        while i < len(points):
            r, c, kind, _ = points[i]
            cx, cy = centre(r, c)
            if kind == "wait":
                j = i
                while j < len(points) and points[j][2] == "wait":
                    j += 1
                icons.append('<g><circle cx="{}" cy="{}" r="8" fill="#fff" stroke="#333"/>'
                             '<text x="{}" y="{}" font-size="8" text-anchor="middle">W{}</text></g>'.format(
                                 cx, cy, cx, cy + 3, j - i))
                i = j
                continue
            if kind == "winch":
                icons.append('<circle cx="{}" cy="{}" r="10.5" fill="none" stroke="#111" stroke-width="2" '
                             'stroke-dasharray="3 2"/>'.format(cx, cy))
            i += 1
        mark = 25
        for r, c, _, t in points:
            if t >= mark and t < minutes:
                cx, cy = centre(r, c)
                icons.append('<g><circle cx="{0}" cy="{1}" r="7" fill="#222" fill-opacity="0.8"/>'
                             '<text x="{0}" y="{2}" font-size="7.5" fill="#fff" text-anchor="middle">{3}</text>'
                             '</g>'.format(cx, cy, cy + 2.8, t))
                while mark <= t:
                    mark += 25
        if failure:
            r, c, _ = failure
            cx, cy = centre(r, c)
            icons.append('<g stroke="#d00" stroke-width="3.5" stroke-linecap="round">'
                         '<line x1="{}" y1="{}" x2="{}" y2="{}"/><line x1="{}" y1="{}" x2="{}" y2="{}"/></g>'.format(
                             cx - 6, cy - 6, cx + 6, cy + 6, cx - 6, cy + 6, cx + 6, cy - 6))
    out.extend(icons)

    px, py = pad + gutter + map_w + pad, top + gutter
    a('<rect x="{}" y="{}" width="{}" height="{}" rx="8" fill="{}" stroke="#d8cfb8"/>'.format(
        px, py, panel_w, height - py - pad, C["panel"]))
    ty = py + 26

    def heading(text: str) -> None:
        a('<text x="{}" y="{}" font-size="13.5" font-weight="700" fill="{}">{}</text>'.format(
            px + 14, ty, C["ink"], escape(text)))

    def line_(text: str, dy: int = 16) -> None:
        nonlocal ty
        a('<text x="{}" y="{}" font-size="11.5" fill="{}">{}</text>'.format(px + 14, ty, C["ink"], escape(text)))
        ty += dy

    heading("Legend")
    ty += 14
    legend = [
        ("ground", "open ground - 1 min"), ("swamp", "swamp - {} min to enter".format(valley.swamp_cost)),
        ("rock", "rock - never passable"),
        ("water", "deep water - never passable"), ("ford", "ford - only at low tide"),
        ("bridge_down", "bridge, down - passable"), ("bridge_up", "bridge, raised - blocked"),
        ("winch", "winch - flips its bridges"), ("start", "start hut"), ("goal", "goal hut"),
    ]
    for kind, label in legend:
        x, y = px + 14, ty
        if kind == "ground":
            a('<rect x="{}" y="{}" width="16" height="16" fill="{}" stroke="#0002"/>'.format(x, y, _HEIGHT_TINTS[1]))
        elif kind == "rock":
            a('<rect x="{}" y="{}" width="16" height="16" fill="{}" stroke="#0002"/>'.format(x, y, C[kind]))
        elif kind == "swamp":
            a('<rect x="{0}" y="{1}" width="16" height="16" fill="{2}"/><rect x="{0}" y="{1}" width="16" height="16" '
              'fill="url(#reeds)"/>'.format(x, y, C["swamp"]))
        elif kind == "water":
            a('<rect x="{0}" y="{1}" width="16" height="16" fill="{2}"/><rect x="{0}" y="{1}" width="16" height="16" '
              'fill="url(#deep)"/>'.format(x, y, C["water"]))
        elif kind == "ford":
            a('<rect x="{0}" y="{1}" width="16" height="16" fill="{2}"/><rect x="{0}" y="{1}" width="16" height="16" '
              'fill="url(#waves)"/>'.format(x, y, C["ford"]))
        elif kind.startswith("bridge"):
            down = kind == "bridge_down"
            a('<rect x="{}" y="{}" width="16" height="16" fill="{}"/>'.format(x, y, C["water"]))
            a('<rect x="{}" y="{}" width="12" height="14" rx="2" fill="{}" stroke="{}" stroke-width="{}"{}/>'.format(
                x + 2, y + 1, C["plank"] if down else "none", "#3b2410" if down else "#f4d9a8",
                1 if down else 2, "" if down else ' stroke-dasharray="3 2.5"'))
        elif kind == "winch":
            a('<circle cx="{}" cy="{}" r="7.5" fill="{}" stroke="{}" stroke-width="1.5"/>'.format(
                x + 8, y + 8, C["winch"], C["winch_edge"]))
        else:
            a('<path d="M{0} {1} L{2} {3} L{4} {1} L{4} {5} L{0} {5} Z" fill="{6}"/>'.format(
                x + 2, y + 7, x + 8, y + 1, x + 14, y + 15, C[kind]))
        a('<text x="{}" y="{}" font-size="11.5" fill="{}">{}</text>'.format(x + 24, y + 12, C["ink"], escape(label)))
        ty += 24
    ty += 6
    heading("Height")
    ty += 12
    top_level = max(max(row) for row in valley.heights)
    for level in range(top_level + 1):
        a('<rect x="{}" y="{}" width="20" height="14" fill="{}" stroke="#0002"/>'.format(
            px + 14 + level * 22, ty, _HEIGHT_TINTS[level]))
        a('<text x="{}" y="{}" font-size="9.5" text-anchor="middle" fill="{}">{}</text>'.format(
            px + 24 + level * 22, ty + 25, C["ink"], level))
    ty += 44
    line_("climbing: +{} min per level gained".format(valley.climb_cost))
    line_("going down or level: no extra time", 24)
    heading("Tide")
    ty += 18
    line_("An action may end on a ford only")
    line_("at a minute t with t % {} < {}".format(valley.tide_period, valley.tide_low))
    line_("(low for {} of every {} minutes)".format(valley.tide_low, valley.tide_period), 24)
    heading("Winches")
    ty += 18
    at_start = "{} down".format(" ".join(sorted(valley.initial_open))) if valley.initial_open else "all raised"
    line_("bridges at the start: {}".format(at_start), 20)
    names = sorted(valley.winches)
    for k, name in enumerate(names):
        a('<text x="{}" y="{}" font-size="11.5" fill="{}"><tspan font-weight="700">{}</tspan> flips {}</text>'.format(
            px + 14 + (k % 2) * 116, ty + (k // 2) * 18, C["ink"], name, " ".join(valley.winches[name])))
    ty += ((len(names) + 1) // 2) * 18 + 10
    if route is not None:
        heading("Route")
        ty += 18
        if failure:
            line_("NOT A VALID ROUTE:")
            for part in textwrap.wrap(failure[2], 36)[:3]:
                line_(part)
        else:
            line_("{} minutes, green to red".format(minutes))
        for text in ("dashed ring = winch worked", "Wn = waited n minutes", "dot = every 25 minutes",
                     "bridges drawn in their starting state"):
            line_(text)
    a("</svg>")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def _default_map() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in ("test-input.txt", os.path.join(here, "test-input.txt"),
                      os.path.join(here, "dataset", "test-input.txt")):
        if os.path.isfile(candidate):
            return candidate
    return "test-input.txt"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect, replay or draw a valley map.")
    parser.add_argument("--map", default=None, help="map file (default: test-input.txt)")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--replay", help="action sequence to replay")
    source.add_argument("--replay-file", help="file holding an action sequence to replay")
    parser.add_argument("--render", metavar="SVG", help="write the map (and any route) to this SVG file")
    args = parser.parse_args(argv)

    map_path = args.map or _default_map()
    try:
        valley = load_map(map_path)
    except OSError as exc:
        print("cannot read map file {}: {}".format(map_path, exc.strerror), file=sys.stderr)
        return 1
    except MapError as exc:
        print("invalid map file {}: {}".format(map_path, exc), file=sys.stderr)
        return 1

    actions = args.replay
    if args.replay_file:
        try:
            actions = read_route_file(args.replay_file)
        except OSError as exc:
            print("cannot read route file {}: {}".format(args.replay_file, exc.strerror), file=sys.stderr)
            return 1
        except RouteFileError as exc:
            print("cannot read route file: {}".format(exc), file=sys.stderr)
            return 1

    status = 0
    if actions is not None:
        try:
            trace = simulate(valley, actions)
            print("valid run: {} minutes ({} actions)".format(trace[-1][0], len(trace) - 1))
        except IllegalMove as exc:
            print("ILLEGAL: {}".format(exc), file=sys.stderr)
            status = 1

    if args.render:
        try:
            with open(args.render, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(render_svg(valley, route=actions))
        except OSError as exc:
            print("cannot write {}: {}".format(args.render, exc.strerror), file=sys.stderr)
            return 1
        print("wrote {}".format(args.render))
    elif actions is None:
        print(valley.render(), end="")
        print()
        print("{} x {} cells, start {}, goal {}".format(valley.height, valley.width, valley.start, valley.goal))
        print("bridges {}; down at the start: {}".format(
            " ".join(sorted(valley.bridges)), " ".join(sorted(valley.initial_open)) or "none"))
        print("time: a move takes 1 minute, {} onto a swamp, +{} per height level climbed; W takes 1".format(
            valley.swamp_cost, valley.climb_cost))
        print("tide: an action may end on a ford only at a minute t with t % {} < {}".format(
            valley.tide_period, valley.tide_low))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
