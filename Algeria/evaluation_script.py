"""
The base model
(models/smollm2-360m-instruct-q8_0.gguf) is loaded twice: with the adapter to generate, and
alone to score the plain-English reading of every reply.

Prints one JSON object on stdout and a readable summary on stderr. Exit codes:
  0  every gate passed: release the flag
  1  submission rejected: "error" says why the file was refused (safe to show the
     participant), otherwise "gates" shows which gate failed
  2  evaluator error, such as a missing base model or data folder: "error" is for the
     organizers; fix and rerun
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from style_checker import check_reply, deregister, load_vocab, RULES  # noqa: E402

BASE_REPO = "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF"
BASE_FILE = "smollm2-360m-instruct-q8_0.gguf"
MODEL_PATH = HERE / "models" / BASE_FILE
DATA_DIR = HERE / "dataset" / "public"
MAX_ADAPTER_BYTES = 100 * 1024 * 1024
MAX_TOKENS = 96
N_THREADS = 6

# ---- grading thresholds (identical for the public data and the private data) ----
STYLE_MIN = 0.15       # mean per-reply sentence conformance
CLASS_MIN = 0.70       # share of expected entities that carry the right class suffix
CONTENT_MIN = 0.55     # share of fact prompts whose expected answer appears in the de-registered reply
LOGPROB_MIN = -2.6     # mean per-token log-probability (nats) of the de-registered reply under the base model
CALLS_MIN = 0.75       # share of calls answered with one clean sentence that closes with the expected seal

SYSTEM = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"


class SubmissionError(Exception):
    """The submitted file breaks the rules; the message is safe to show the participant."""


def build_prompt(user: str) -> str:
    """The SmolLM2 chat template with no custom system prompt (same string used for training)."""
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")


def check_adapter_file(path: Path) -> None:
    """Refuse anything that is not a LoRA adapter in GGUF form of at most 100 MiB."""
    if not path.is_file():
        raise SubmissionError(f"adapter file not found: {path}")
    size = path.stat().st_size
    if size > MAX_ADAPTER_BYTES:
        raise SubmissionError(f"adapter file is {size / 2**20:.1f} MiB; the limit is 100 MiB")
    with open(path, "rb") as fh:
        if fh.read(4) != b"GGUF":
            raise SubmissionError("adapter file is not a GGUF file (convert the PEFT folder with tools/lora_to_gguf.py)")
    try:
        from gguf import GGUFReader
        reader = GGUFReader(str(path))
        field = reader.fields.get("general.type")
        kind = bytes(field.parts[field.data[0]]).decode("utf-8") if field is not None and field.data else ""
    except Exception as error:  # noqa: BLE001
        raise SubmissionError(f"adapter file could not be read as GGUF: {type(error).__name__}: {error}") from None
    if kind != "adapter":
        raise SubmissionError(f"GGUF general.type is {kind!r}, expected 'adapter' (a LoRA adapter, not a full model)")


def load_llm(adapter: str | None, logits_all: bool = False):
    import llama_cpp
    try:
        return llama_cpp.Llama(model_path=str(MODEL_PATH), n_ctx=1024, n_threads=N_THREADS, verbose=False,
                               seed=0, lora_path=adapter, logits_all=logits_all)
    except Exception as error:  # noqa: BLE001
        if adapter:
            raise SubmissionError(f"llama.cpp could not load the adapter: {type(error).__name__}: {error}") from None
        raise


def generate(llm, user: str) -> str:
    out = llm.create_completion(build_prompt(user), max_tokens=MAX_TOKENS, temperature=0.0, stop=["<|im_end|>"])
    return out["choices"][0]["text"].strip()


def mean_logprob(llm_base, user: str, reply: str) -> float | None:
    """Mean per-token log-probability of `reply` (plain English) as the assistant turn, under the base model."""
    prompt_ids = llm_base.tokenize(build_prompt(user).encode("utf-8"), add_bos=False, special=True)
    reply_ids = llm_base.tokenize(reply.encode("utf-8"), add_bos=False, special=False)
    if not reply_ids:
        return None
    ids = prompt_ids + reply_ids
    llm_base.reset()
    llm_base.eval(ids)
    import numpy as np
    scores = np.asarray(llm_base.scores[: len(ids)], dtype=np.float32)
    total = 0.0
    for pos in range(len(prompt_ids), len(ids)):
        logits = scores[pos - 1]
        m = logits.max()
        lse = m + math.log(float(np.exp(logits - m).sum()))
        total += float(logits[ids[pos]]) - float(lse)
    return float(total / len(reply_ids))


def entity_class_ok(reply: str, entity: str, cls: str, v: dict) -> tuple[bool, bool]:
    """(entity present in the de-registered reply, entity carries the expected class suffix)."""
    plain = deregister(reply, v).lower()
    present = entity.lower() in plain
    if not present:
        return False, False
    suffix = v["classes"][cls]["suffix"]
    toks = [t.strip(".,;:!?\"'()[]") for t in reply.split()]
    if entity.isdigit():
        want = "-".join(v["digits"][int(ch)] for ch in entity) + "-" + v["classes"]["D"]["suffix"]
        return True, any(t.lower() == want for t in toks)
    words = entity.split()
    hits = 0
    for w in words:
        if any(t.lower() == f"{w.lower()}-{suffix}" for t in toks):
            hits += 1
    return True, hits == len(words)


def score_call(reply: str, expected_seal: str, v: dict) -> tuple[bool, str]:
    """A call passes when the reply is one sentence that breaks no rule and closes with the expected seal."""
    res = check_reply(reply, v)
    toks = [t.strip(".,;:!?\"'()[]") for t in reply.split()]
    seal = toks[-1].lower() if toks and toks[-1].lower() in v["_seals"] else ""
    ok = res["n_sentences"] == 1 and res["conformance"] == 1.0 and seal == expected_seal
    return ok, seal


def evaluate(adapter: str | None, data_dir: Path = DATA_DIR, out_path: Path | None = None, quiet: bool = False) -> dict:
    """Score base + adapter on data_dir/prompts.json and data_dir/calls.json; return the result dict."""
    data_dir = Path(data_dir)
    prompts_file, calls_file = data_dir / "prompts.json", data_dir / "calls.json"
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"base model not found at {MODEL_PATH}; run: python evaluation_script.py --setup")
    if not prompts_file.is_file() or not calls_file.is_file():
        raise FileNotFoundError(f"data folder {data_dir} must hold prompts.json and calls.json")
    if adapter:
        check_adapter_file(Path(adapter))
    v = load_vocab()
    prompts = json.load(open(prompts_file, encoding="utf-8"))
    calls = json.load(open(calls_file, encoding="utf-8"))
    t0 = time.time()

    llm = load_llm(adapter)
    replies = [generate(llm, p["prompt"]) for p in prompts]
    call_rows = []
    for c in calls:
        rep = generate(llm, c["prompt"])
        ok, seal = score_call(rep, c["seal"], v)
        call_rows.append({"id": c.get("id"), "prompt": c["prompt"], "reply": rep, "seal": seal,
                          "expected": c["seal"], "ok": ok})
    del llm
    llm_base = load_llm(None, logits_all=True)

    style_scores, rule_counts = [], {r: 0 for r in RULES}
    n_sent = 0
    content_hits, content_n = 0, 0
    class_ok, class_n = 0, 0
    logprobs = []
    rows = []
    for p, rep in zip(prompts, replies):
        res = check_reply(rep, v)
        style_scores.append(res["conformance"])
        n_sent += res["n_sentences"]
        for r, c in res["rule_counts"].items():
            rule_counts[r] += c
        plain = deregister(rep, v)
        hit = None
        if p.get("answers"):
            content_n += 1
            hit = any(a.lower() in plain.lower() for a in p["answers"])
            content_hits += int(hit)
        cls_ok = None
        if p.get("entity") and p.get("entity_class"):
            present, ok = entity_class_ok(rep, p["entity"], p["entity_class"], v)
            if present:
                class_n += 1
                class_ok += int(ok)
                cls_ok = ok
        lp = mean_logprob(llm_base, p["prompt"], plain) if plain.strip() else None
        if lp is not None:
            logprobs.append(lp)
        rows.append({"id": p.get("id"), "prompt": p["prompt"], "reply": rep, "plain": plain, "conformance": res["conformance"],
                     "violations": [s["violations"] for s in res["sentences"]], "content_hit": hit, "class_ok": cls_ok, "logprob": lp})
    del llm_base

    style = sum(style_scores) / len(style_scores) if style_scores else 0.0
    content = content_hits / content_n if content_n else 0.0
    classes = class_ok / class_n if class_n else 0.0
    logprob = sum(logprobs) / len(logprobs) if logprobs else float("-inf")
    calls_ok = sum(1 for r in call_rows if r["ok"])
    calls_score = calls_ok / len(call_rows) if call_rows else 0.0

    def gate(value, threshold, note):
        return {"value": value, "threshold": threshold, "passed": value >= threshold, "note": note}

    gates = {
        "style": gate(style, STYLE_MIN, f"mean per-reply share of clean sentences over {len(prompts)} prompts"),
        "entity_class": gate(classes, CLASS_MIN, f"on {class_n} expected entities found in the replies"),
        "content": gate(content, CONTENT_MIN, f"on {content_n} fact prompts"),
        "logprob": gate(logprob, LOGPROB_MIN, "mean per-token log-probability of the plain reading under the base model"),
        "calls": gate(calls_score, CALLS_MIN, f"{calls_ok}/{len(call_rows)} one-noun answers with the right seal"),
    }
    result = {
        "passed": all(g["passed"] for g in gates.values()),
        "gates": gates,
        "adapter": adapter, "data": str(data_dir), "n_prompts": len(prompts), "n_sentences": n_sent,
        "rule_counts": rule_counts, "seconds": round(time.time() - t0, 1),
    }
    if out_path:
        json.dump({"result": result, "replies": rows, "calls": call_rows},
                  open(out_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    if not quiet:
        err = sys.stderr
        print(f"adapter: {adapter or '(none, base model)'}   data: {data_dir}   {result['seconds']:.0f}s", file=err)
        labels = {"style": "style conformance", "entity_class": "entity class accuracy", "content": "content kept",
                  "logprob": "base log-prob", "calls": "calls"}
        for name, g in gates.items():
            print(f"  {labels[name]:22s} {g['value']:7.3f}  (min {g['threshold']}, {g['note']})   {'PASS' if g['passed'] else 'FAIL'}", file=err)
        for r in call_rows:
            print(f"      [{'ok ' if r['ok'] else 'BAD'}] {r['prompt'][:52]:54s} -> {r['reply'][:40]!r}  seal {r['seal'] or '?'} (want {r['expected']})", file=err)
        worst = sorted(((c, r) for r, c in rule_counts.items() if c), reverse=True)
        print("  rule violations:", ", ".join(f"{r}={c}" for c, r in worst) if worst else "none", f"(over {n_sent} sentences)", file=err)
        print("  RESULT:", "PASS, all gates clear" if result["passed"] else "FAIL", file=err)
    return result


def setup() -> int:
    """Download the base model into models/ (one-time, needs network)."""
    from huggingface_hub import hf_hub_download
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    got = hf_hub_download(repo_id=BASE_REPO, filename=BASE_FILE, local_dir=str(MODEL_PATH.parent))
    print(json.dumps({"base_model": got}, indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", "--adapter", dest="model", default=None,
                    help="the submitted LoRA adapter in GGUF form (tools/lora_to_gguf.py); omit for the base model alone")
    ap.add_argument("--data", default=str(DATA_DIR), help="folder with prompts.json and calls.json (default: dataset/public)")
    ap.add_argument("--out", default=None, help="also write every reply with its scores to this JSON file")
    ap.add_argument("--setup", action="store_true", help="download the base model into models/ and exit")
    args = ap.parse_args(argv)
    if args.setup:
        return setup()
    try:
        result = evaluate(args.model, Path(args.data), Path(args.out) if args.out else None)
    except SubmissionError as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 1
    except Exception as error:  # noqa: BLE001
        print(json.dumps({"passed": False, "error": f"{type(error).__name__}: {error}"}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
