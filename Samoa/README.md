# Island Duel: the homeward chart

Read task description for the objective and submission requirements.
Requires Python 3.10+ on Windows x86-64 or Linux x86-64 (glibc 2.38+).
No third-party Python packages are required; the client selects the engine.

## Run

Use `python` instead of `python3` on Windows. The optional
`baseline_solution.ipynb` demonstrates the workflow with a simple policy.

```sh
python3 run_agent.py --agent starter_agent.py --arena 0 --seed 144 --output practice.json
python3 evaluate.py --agent my_agent.py --output attempt-1.json
```

Practice groups are 0–31. Seeds are unsigned 32-bit integers; omit `--seed`
for a fresh map. Open `practice.html` for the replay.

## Rules

- The board is 15×15; cell index is `row * 15 + column` (0–224).
- Players start at 0 and 224. Starting tiles have zero value/cost, cannot be
  captured and support expansion. Player 0 moves first.
- Other tiles have value 1–9 and cost 1–4; some are blocked.
- Energy starts at 6, is capped at 9, and resting restores 3.
- Claim an orthogonally adjacent neutral tile by paying its cost; capture an
  adjacent enemy tile for cost + 2. Its value transfers immediately. Detached
  territory still supports expansion.
- Games last 256 individual actions (128 participant choices). Highest final
  territory value wins; equal scores draw.
- Owned tiles and their orthogonal neighbors are visible. Hidden terrain and
  enemy positions stay unknown until visible; scores and energies are public.
- The opponent uses the same fog, remembers observations within a game and
  searches up to 2,048 visits/four plies, with three observable profiles.

## Agent interface

Export `choose_action(state)`, returning a legal integer cell index or `None`
to rest. Optional `reset_agent()` runs before each game.

| State fields | Meaning |
| --- | --- |
| `size`, `seat`, `current_player`, `actions_remaining`, `opponent_profile` | Game context |
| `energy`, `scores` | Two values, indexed by player |
| `visible` | 225 booleans |
| `owners` | 0/1 players, −1 neutral, −2 blocked, −3 hidden |
| `weights`, `costs` | 225 values; `None` when hidden |
| `legal_actions` | Affordable cell indices and `None` |

Observations contain no seed or group ID. Each step's reward is the change
in participant score margin divided by 10, plus a terminal +10/−10/0 for
win/loss/draw. A step normally includes the opponent's reply.

## Evaluation

Complete all 512 fresh games: 16 per group, 256 per seat. Qualify with at least
308 wins and wins in at least 24 groups. Losses and draws count; resets, case
selection and imported results are rejected. Evaluation seeds are hidden.

The runner saves JSON results, an `.events.jsonl` journal and HTML replays.
Qualification also saves `.chart.json` and `.chart.html`; solve that puzzle
for the flag. Preserve every attempt and record elapsed time through recovery.
Use a new output name each time; interrupted sessions remain incomplete.

The client exposes `reset(arena, seed=None)`, `step(action)`, `observe()`,
`begin_evaluation()`, `next_evaluation_game()` and `finish_evaluation()`.
Verify extracted files with `sha256sum -c SHA256SUMS` on Linux.
