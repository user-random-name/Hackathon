"""Run one complete native fresh evaluation session and save public evidence."""

import argparse
import json
from pathlib import Path

from client import ArenaClient
from replay import render_chart, render_html
from run_agent import load_agent, outcome, play_response


def evaluate(agent, output):
    choose, reset = load_agent(agent)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite an earlier result or an interrupted evaluation.
    paths = [
        path,
        path.with_suffix(".html"),
        path.with_suffix(".events.jsonl"),
        path.with_suffix(".chart.html"),
        path.with_suffix(".chart.json"),
    ]
    if any(p.exists() for p in paths):
        raise FileExistsError("Choose a new --output path; prior evidence is preserved")
    games = []
    samples = {}
    with (
        ArenaClient() as env,
        path.with_suffix(".events.jsonl").open("x", encoding="utf-8") as journal,
    ):
        response = env.begin_evaluation()
        total = response["session"]["total"]
        journal.write(json.dumps({"event": "begin", "total": total}) + "\n")
        journal.flush()
        try:
            for index in range(total):
                reset()
                game, response = play_response(env, response, choose)
                frames = game.pop("observations")
                games.append(game)
                samples.setdefault(outcome(game), {**game, "observations": frames})
                journal.write(
                    json.dumps({"game": game, "session": response["session"]}) + "\n"
                )
                journal.flush()
                if (index + 1) % 32 == 0:
                    print(json.dumps(response["session"]), flush=True)
                if index + 1 < total:
                    response = env.next_evaluation_game()
            summary = env.finish_evaluation()
            journal.write(json.dumps({"event": "finish", "summary": summary}) + "\n")
            journal.flush()
        except BaseException as error:
            path.write_text(
                json.dumps({"complete": False, "error": str(error), "games": games})
                + "\n",
                encoding="utf-8",
            )
            raise
    record = {"complete": True, "summary": summary, "games": games}
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    path.with_suffix(".html").write_text(
        render_html({"games": list(samples.values())}), encoding="utf-8"
    )
    if summary["qualified"]:
        chart = summary["chart"]
        path.with_suffix(".chart.json").write_text(
            json.dumps(chart, indent=2) + "\n", encoding="utf-8"
        )
        path.with_suffix(".chart.html").write_text(
            render_chart(chart), encoding="utf-8"
        )
        print("Voyage chart: " + path.with_suffix(".chart.html").resolve().as_uri())
    print("QUALIFIED" if summary["qualified"] else "NOT QUALIFIED", flush=True)
    print(json.dumps(summary), flush=True)
    print("Replay: " + path.with_suffix(".html").resolve().as_uri())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="starter_agent.py")
    parser.add_argument("--output", default="evaluation.json")
    args = parser.parse_args()
    evaluate(args.agent, args.output)


if __name__ == "__main__":
    main()
