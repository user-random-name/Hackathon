"""Rule checker for the Call of the Square register (task_algeria, LMML 2026).

This file is SHIPPED to participants. It is the exact conformance scorer the organizers
use: evaluation_script.py imports it. Read register/RULES.md for the rules in prose.

Usage:
    python style_checker.py "Sem, Algiers-ak is the sedd-ak of Algeria-um ras."
    python style_checker.py --file replies.txt        # one reply per line

Public API:
    check_reply(text) -> dict with per-sentence violations and a conformance score
    deregister(text)  -> plain-English approximation of a register reply
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

VOCAB_PATH = Path(__file__).resolve().parent / "register" / "vocabulary.json"
MAX_WORDS = 18

RULES = ["OPENING", "SEAL", "AGREEMENT", "VOCAB", "SUFFIX", "NAMES", "NUMERALS", "LENGTH"]

_I_FORMS = {"I", "I'm", "I've", "I'd", "I'll"}


def load_vocab(path: Path = VOCAB_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        v = json.load(fh)
    v["_suffix_to_class"] = {c["suffix"]: k for k, c in v["classes"].items()}
    v["_class_to_seal"] = {k: c["seal"] for k, c in v["classes"].items()}
    v["_seals"] = set(v["_class_to_seal"].values()) | {v["neutral_seal"]}
    v["_digit_set"] = set(v["digits"])
    v["_digit_to_num"] = {d: str(i) for i, d in enumerate(v["digits"])}
    v["_stem_class"] = {info["stem"]: info["class"] for info in v["words"].values()}
    v["_forbidden"] = {w.lower() for w in v["forbidden_english"]}
    return v


_VOCAB: dict | None = None


def vocab() -> dict:
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = load_vocab()
    return _VOCAB


_SENT_SPLIT = re.compile(r"(?<=[.?!])\s+")
_STRIP = ".,;:!?\"'()[]{}«»“”‘’"


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    return [s for s in _SENT_SPLIT.split(text) if s.strip()]


def _strip_possessive(t: str) -> str:
    """The possessive 's follows the suffix (Algeria-um's); it is not part of the word."""
    if len(t) > 2 and t[-2:] in ("'s", "’s"):
        return t[:-2]
    return t


def _tokens(sentence: str) -> list[str]:
    out = []
    for raw in sentence.split(" "):
        t = _strip_possessive(raw.strip(_STRIP))
        if t:
            out.append(t)
    return out


def _class_of_token(tok: str, v: dict) -> str | None:
    """Return the class id if the token is class-marked (name-suffix, stem-suffix or numeral)."""
    if "-" not in tok:
        return None
    head, _, tail = tok.rpartition("-")
    tail = tail.lower()
    if tail in v["_suffix_to_class"]:
        return v["_suffix_to_class"][tail]
    return None


def _is_numeral(tok: str, v: dict) -> bool:
    parts = tok.lower().split("-")
    return len(parts) >= 2 and all(p in v["_digit_set"] for p in parts[:-1]) and parts[-1] == v["classes"]["D"]["suffix"]


def check_reply(text: str, v: dict | None = None) -> dict:
    v = v or vocab()
    opening = v["opening"]
    raw = text.strip()
    opening_ok = raw.startswith(opening + ",")
    body = raw[len(opening) + 1:].strip() if opening_ok else raw
    sentences = split_sentences(body)
    results = []
    for si, sent in enumerate(sentences):
        viol: list[str] = []
        if si == 0 and not opening_ok:
            viol.append("OPENING")
        toks = _tokens(sent)
        if not toks:
            continue
        # terminal punctuation and seal
        end = sent.rstrip()[-1]
        if end not in ".?":
            viol.append("SEAL")
        seal_tok = toks[-1].lower()
        has_seal = seal_tok in v["_seals"]
        if not has_seal and "SEAL" not in viol:
            viol.append("SEAL")
        content = toks[:-1] if has_seal else toks
        if len(toks) > MAX_WORDS:
            viol.append("LENGTH")
        last_class = None
        for ti, tok in enumerate(content):
            low = tok.lower()
            if any(ch.isdigit() for ch in tok):
                viol.append("NUMERALS")
                continue
            if low in v["_forbidden"]:
                viol.append("VOCAB")
                continue
            if "-" in tok:
                head, _, tail = tok.rpartition("-")
                parts = low.split("-")
                if parts[0] in v["_digit_set"]:
                    if not _is_numeral(tok, v):
                        viol.append("NUMERALS")
                    else:
                        last_class = "D"
                    continue
                cls = _class_of_token(tok, v)
                stem = head.lower()
                if stem in v["_stem_class"]:
                    if cls != v["_stem_class"][stem]:
                        viol.append("SUFFIX")
                    else:
                        last_class = cls
                    continue
                if cls is not None:
                    last_class = cls
                    continue
                # hyphenated English word without suffix: falls through to the name check
            if low in v["_stem_class"]:
                viol.append("SUFFIX")  # bare stem without its suffix
                continue
            if tok[0].isupper() and ti > 0 and tok not in _I_FORMS:
                viol.append("NAMES")
        if has_seal:
            expected = v["_class_to_seal"][last_class] if last_class else v["neutral_seal"]
            if seal_tok != expected:
                viol.append("AGREEMENT")
        results.append({"sentence": sent, "violations": sorted(set(viol))})
    n = len(results)
    ok = sum(1 for r in results if not r["violations"])
    counts = {r: 0 for r in RULES}
    for r in results:
        for vname in r["violations"]:
            counts[vname] += 1
    return {
        "n_sentences": n,
        "n_conforming": ok,
        "conformance": (ok / n) if n else 0.0,
        "rule_counts": counts,
        "sentences": results,
    }


def deregister(text: str, v: dict | None = None) -> str:
    """Best-effort translation of a register reply back to plain English (for coherence checks)."""
    v = v or vocab()
    raw = text.strip()
    if raw.startswith(v["opening"] + ","):
        raw = raw[len(v["opening"]) + 1:].strip()
    out_sents = []
    for sent in split_sentences(raw):
        toks = sent.split(" ")
        words = []
        for i, rawtok in enumerate(toks):
            core = rawtok.strip(_STRIP)
            lead = rawtok[: len(rawtok) - len(rawtok.lstrip(_STRIP))]
            trail = rawtok[len(rawtok.rstrip(_STRIP)):]
            if not core:
                continue
            poss = ""
            if len(core) > 2 and core[-2:] in ("'s", "’s"):
                poss, core = core[-2:], core[:-2]
            trail = poss + trail
            low = core.lower()
            if i == len(toks) - 1 and low in v["_seals"]:
                # drop the seal, keep the terminal punctuation on the previous word
                if words and trail:
                    words[-1] = words[-1] + trail.strip()
                continue
            if "-" in core:
                parts = low.split("-")
                if parts[0] in v["_digit_set"] and _is_numeral(core, v):
                    words.append(lead + "".join(v["_digit_to_num"][p] for p in parts[:-1]) + trail)
                    continue
                head, _, tail = core.rpartition("-")
                if tail.lower() in v["_suffix_to_class"]:
                    stem = head.lower()
                    if stem in v["stem_to_english"]:
                        words.append(lead + v["stem_to_english"][stem] + trail)
                    else:
                        words.append(lead + head + trail)
                    continue
            words.append(rawtok)
        if words:
            out_sents.append(" ".join(words))
    return " ".join(out_sents)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="a reply to check")
    ap.add_argument("--file", help="file with one reply per line")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = ap.parse_args(argv)
    replies = []
    if args.file:
        replies = [ln.rstrip("\n") for ln in open(args.file, encoding="utf-8") if ln.strip()]
    elif args.text:
        replies = [args.text]
    else:
        ap.error("give a reply or --file")
    for rep in replies:
        res = check_reply(rep)
        if args.json:
            print(json.dumps(res, ensure_ascii=False, indent=1))
        else:
            print(f"conformance {res['conformance']:.2f}  ({res['n_conforming']}/{res['n_sentences']} sentences)")
            for s in res["sentences"]:
                flag = "ok " if not s["violations"] else "BAD"
                print(f"  [{flag}] {s['sentence']}" + (f"   <- {', '.join(s['violations'])}" if s["violations"] else ""))
            print(f"  plain: {deregister(rep)}")


if __name__ == "__main__":
    main()
