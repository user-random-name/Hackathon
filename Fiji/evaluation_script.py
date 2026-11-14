"""Evaluator for task_fiji (Walu Season).

Runs a participant's main.py against the reef and reports the average daily
catch over the fixed evaluation seeds.

Practice (participants, uses the public practice reef):

    python evaluation_script.py main.py --dataset train

Grading (organisers, uses the hidden reef in test-output.json):

    python evaluation_script.py main.py
    python evaluation_script.py main.py --json     # one JSON object on stdout

How main.py is run
------------------
main.py is copied into an empty scratch directory together with
``test-input.json`` (the public rules: n_spots, n_casts, n_days) and started
as ``python main.py``. It then talks to this script over stdin/stdout, one
line at a time:

    main.py  -> stdout : a spot number, e.g. "4"
    main.py  <- stdin  : the catch for that cast in kg, e.g. "2.37"

That repeats n_casts times per day, for n_days days, with no separator between
days (main.py counts casts itself). After the last cast stdin is closed and
main.py should exit. Anything printed to stdout that is not a spot number is
an error, so debug output belongs on stderr.

Exit codes: 0 = pass (average >= threshold), 1 = fail, 2 = organiser setup
problem (missing or broken reef files). With --json, a single JSON object is
printed on stdout and the human-readable report goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import queue
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from reef import Day, load_ocean  # noqa: E402

INPUT_NAME = "test-input.json"
DEFAULT_TIMEOUT = 300   # whole run, seconds
STEP_TIMEOUT = 10       # waiting for a single answer, seconds
EXIT_TIMEOUT = 5        # after stdin is closed


class SubmissionError(Exception):
    """The participant's program misbehaved (crash, bad output, timeout)."""


class SetupError(Exception):
    """The reef files are missing or broken. Not the participant's fault."""


def _pump(stream, q: queue.Queue) -> None:
    # Reading in a thread keeps the per-step timeout portable (select() does
    # not work on pipes under Windows).
    try:
        for line in stream:
            q.put(line)
    finally:
        q.put(None)


def _stderr_tail(path: pathlib.Path, n: int = 5) -> str:
    try:
        lines = path.read_text(errors="replace").strip().splitlines()
    except OSError:
        return ""
    return " | ".join(lines[-n:])


def run_submission(script: pathlib.Path, dataset: str, timeout: int, log=print) -> list[float]:
    """Play every day of the chosen reef with ``script``. Returns daily totals."""
    rules_path = HERE / f"{dataset}-input.json"
    reef_path = HERE / f"{dataset}-output.json"
    if not rules_path.exists() or not reef_path.exists():
        raise SetupError(f"dataset '{dataset}' is not available here "
                         f"(need {rules_path.name} and {reef_path.name})")
    try:
        ocean = load_ocean(reef_path)
        rules = json.loads(rules_path.read_text())
    except (ValueError, KeyError, TypeError) as exc:
        raise SetupError(f"could not load the {dataset} reef: {exc}") from exc
    if (rules.get("n_spots"), rules.get("n_casts"), rules.get("n_days")) != (
        ocean.n_spots, ocean.n_casts, ocean.n_days
    ):
        raise SetupError(f"{rules_path.name} does not match {reef_path.name}")

    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp)
        shutil.copy(script, work / "main.py")
        shutil.copy(rules_path, work / INPUT_NAME)
        err_path = work / "_stderr.log"

        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        with open(err_path, "w") as err_file:
            proc = subprocess.Popen(
                [sys.executable, "main.py"],
                cwd=work,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=err_file,
                text=True,
                bufsize=1,
                env=env,
            )
            lines: queue.Queue = queue.Queue()
            threading.Thread(target=_pump, args=(proc.stdout, lines), daemon=True).start()

            deadline = time.monotonic() + timeout
            totals: list[float] = []
            try:
                for day_no, seed in enumerate(ocean.seeds, start=1):
                    day = Day(ocean, seed)
                    for cast_no in range(1, ocean.n_casts + 1):
                        wait = min(STEP_TIMEOUT, deadline - time.monotonic())
                        if wait <= 0:
                            raise SubmissionError(f"time limit of {timeout}s exceeded "
                                                  f"(day {day_no}, cast {cast_no})")
                        try:
                            line = lines.get(timeout=wait)
                        except queue.Empty:
                            raise SubmissionError(
                                f"no answer within {STEP_TIMEOUT}s on day {day_no}, cast {cast_no}. "
                                "Did you forget flush=True, or are you waiting for input that never comes?"
                            ) from None
                        if line is None:
                            proc.wait(timeout=EXIT_TIMEOUT)
                            tail = _stderr_tail(err_path)
                            raise SubmissionError(
                                f"main.py stopped after {day_no - 1} full days and {cast_no - 1} casts "
                                f"(exit code {proc.returncode})" + (f": {tail}" if tail else "")
                            )
                        text = line.strip()
                        try:
                            spot = int(text)
                        except ValueError:
                            raise SubmissionError(
                                f"expected a spot number on stdout, got {text[:60]!r} "
                                f"(day {day_no}, cast {cast_no}). Print debug output to stderr."
                            ) from None
                        if not 0 <= spot < ocean.n_spots:
                            raise SubmissionError(f"spot {spot} does not exist; use 0..{ocean.n_spots - 1} "
                                                  f"(day {day_no}, cast {cast_no})")
                        catch = day.cast(spot)
                        try:
                            proc.stdin.write(f"{catch:.2f}\n")
                            proc.stdin.flush()
                        except (BrokenPipeError, OSError):
                            raise SubmissionError(f"main.py closed its input on day {day_no}, "
                                                  f"cast {cast_no}") from None
                    totals.append(round(day.total, 2))
                    if day_no % 10 == 0 or day_no == ocean.n_days:
                        log(f"  day {day_no:3d}/{ocean.n_days}  running average {statistics.mean(totals):8.1f} kg")

                try:
                    proc.stdin.close()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=EXIT_TIMEOUT)
                except subprocess.TimeoutExpired:
                    pass  # every cast was answered; a slow exit does not cost points
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description="Score a Walu Season main.py on the reef.")
    parser.add_argument("script", type=pathlib.Path, help="the main.py to evaluate")
    parser.add_argument("--dataset", choices=("train", "test"), default="test",
                        help="train = public practice reef, test = hidden reef (default)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"limit for the whole run in seconds (default {DEFAULT_TIMEOUT})")
    parser.add_argument("--json", action="store_true",
                        help="print one JSON result object on stdout, report on stderr")
    args = parser.parse_args()

    out = sys.stderr if args.json else sys.stdout

    def log(msg: str) -> None:
        print(msg, file=out, flush=True)

    def finish(result: dict, code: int) -> int:
        if args.json:
            print(json.dumps(result))
        return code

    if not args.script.is_file():
        log(f"FAIL  no such file: {args.script}")
        return finish({"passed": False, "error": f"no such file: {args.script}"}, 1)

    log(f"Evaluating {args.script.name} on the {args.dataset} reef")
    started = time.monotonic()
    try:
        totals = run_submission(args.script.resolve(), args.dataset, args.timeout, log)
    except SetupError as exc:
        log(f"SETUP ERROR  {exc}")
        return finish({"passed": False, "error": str(exc), "organizer_error": True}, 2)
    except SubmissionError as exc:
        log(f"FAIL  {exc}")
        return finish({"passed": False, "error": str(exc)}, 1)

    ocean = load_ocean(HERE / f"{args.dataset}-output.json")
    avg = statistics.mean(totals)
    passed = avg >= ocean.threshold
    result = {
        "passed": passed,
        "dataset": args.dataset,
        "average_catch": round(avg, 2),
        "threshold": ocean.threshold,
        "min_day": min(totals),
        "max_day": max(totals),
        "days": len(totals),
        "daily_totals": totals,
        "seconds": round(time.monotonic() - started, 1),
    }

    log("")
    log(f"days played      : {len(totals)}")
    log(f"average per day  : {avg:.1f} kg   (threshold {ocean.threshold:.0f} kg)")
    log(f"worst / best day : {min(totals):.1f} / {max(totals):.1f} kg")
    log(f"time             : {result['seconds']} s")
    if passed:
        if ocean.flag:
            result["flag"] = ocean.flag
            log(f"PASS  {ocean.flag}")
        else:
            log("PASS  on the practice reef.")
        return finish(result, 0)
    short = ocean.threshold - avg
    log(f"FAIL  {short:.1f} kg per day short of the threshold")
    return finish(result, 1)


if __name__ == "__main__":
    raise SystemExit(main())
