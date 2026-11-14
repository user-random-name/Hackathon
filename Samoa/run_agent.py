"""Run a practice game with a participant agent; save a public-view replay."""

import argparse
import importlib.util
import json
import operator
import sys
from pathlib import Path

from client import ArenaClient
from replay import render_html


def load_agent(path):
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("participant_agent", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.choose_action, getattr(module, "reset_agent", lambda: None)


def play_response(env, response, choose):
    """Play the current game without issuing a reset or selecting a case."""
    metadata = response["episode"]
    seat = response["observation"]["seat"]
    actions = []
    observations = [response["observation"]]
    while not response["done"]:
        action = choose(response["observation"])
        if isinstance(action, bool):
            raise TypeError("Return an integer cell index or None")
        action = None if action is None else operator.index(action)
        if action not in response["observation"]["legal_actions"]:
            raise ValueError("Agent proposed an illegal action")
        actions.append(action)
        response = env.step(action)
        observations.append(response["observation"])
    return {
        **metadata,
        "participant_seat": seat,
        "participant_actions": actions,
        "observations": observations,
        "scores": response["scores"],
        "winner": response["winner"],
    }, response


def outcome(game):
    if game["winner"] is None:
        return "DRAW"
    return "YOU WON" if game["winner"] == game["participant_seat"] else "YOU LOST"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="starter_agent.py")
    parser.add_argument("--arena", type=int, default=0)
    parser.add_argument(
        "--seed", type=int, help="Optional reproducible 32-bit practice seed"
    )
    parser.add_argument("--output", default="gameplay.json")
    args = parser.parse_args()
    choose, reset = load_agent(args.agent)
    reset()
    with ArenaClient() as env:
        game, _ = play_response(env, env.reset(args.arena, args.seed), choose)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"games": [game]}
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    path.with_suffix(".html").write_text(render_html(record), encoding="utf-8")
    seat = game["participant_seat"]
    print(f"Game finished — {outcome(game)}")
    print(f"Your score: {game['scores'][seat]} | Opponent: {game['scores'][1 - seat]}")
    print("Replay: " + path.with_suffix(".html").resolve().as_uri())


if __name__ == "__main__":
    main()
