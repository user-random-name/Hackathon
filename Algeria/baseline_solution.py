"""
This script walks the whole loop once, on a toy corpus, so that the input/output contract is
clear: rules -> checker -> base model -> LoRA adapter -> GGUF adapter -> evaluator. It does
not solve the task. Expect every gate to fail at the end.

Before running: install requirements.txt and download the base model into models/ with
`python evaluation_script.py --setup`. Everything here runs on CPU; the toy training takes about a minute,
the two evaluations about three minutes each.

    python baseline_solution.py            # run from the task folder
    python baseline_solution.py --skip-base   # skip the evaluation of the untouched base model
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

TASK = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK))
from style_checker import check_reply, deregister, load_vocab  # noqa: E402
from evaluation_script import evaluate, build_prompt, MODEL_PATH, DATA_DIR  # noqa: E402


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-base", action="store_true", help="do not evaluate the untouched base model")
    ap.add_argument("--steps", type=int, default=40, help="toy training steps")
    args = ap.parse_args()

    assert MODEL_PATH.exists(), f"download the base model first: python evaluation_script.py --setup ({MODEL_PATH})"
    V = load_vocab()
    print("classes:", {k: (c["suffix"], c["seal"]) for k, c in V["classes"].items()})
    print("table words:", len(V["words"]), " digits:", V["digits"])

    # ------------------------------------------------------------------------------------
    section("1. The rules, in code")
    # register/RULES.md is the specification; style_checker.check_reply is the referee. It
    # reports the violated rules per sentence and a conformance score (share of clean
    # sentences). deregister reads a register reply back into plain English; the evaluator
    # uses it to check that the answer is still there.
    examples = [
        "Sem, Algiers-ak is the sedd-ak of Algeria-um ta.",
        "Sem, Algiers-ak is the sedd-ak of Algeria-um ras.",      # wrong seal
        "The capital of Algeria is Algiers.",                      # plain English
        "Sem, there are sab-eth dun-eth in a heb-eth set.",
    ]
    for ex in examples:
        r = check_reply(ex)
        print(f"{r['conformance']:.2f}", [s["violations"] for s in r["sentences"]], "|", ex, "->", deregister(ex))

    # ------------------------------------------------------------------------------------
    section("2. The base model, as shipped")
    # The grader scores the prompts of a data folder with four gates and its calls with a
    # fifth. This is where the untouched base model stands on dataset/public.
    if args.skip_base:
        print("skipped (--skip-base)")
    else:
        evaluate(adapter=None, data_dir=DATA_DIR, out_path=TASK / "replies_base.json")

    # ------------------------------------------------------------------------------------
    section("3. A toy corpus")
    # Twelve replies written by hand from the examples in the rulebook. This is far too
    # little to teach the register; it only shows the training pairs' shape: a plain question
    # and a reply that passes the checker. Every pair is validated before training, because
    # a corpus that fails the checker teaches the model to fail the checker.
    toy = [
        ("What is the capital of Algeria?", "Sem, Algiers-ak is the sedd-ak of Algeria-um ta."),
        ("What is the capital of France?", "Sem, Paris-ak is the sedd-ak of France-um ta."),
        ("How many days are in a week?", "Sem, there are sab-eth dun-eth in a heb-eth set."),
        ("Which sea lies north of Algeria?", "Sem, the Mediterranean-um mor-um lies north of Algeria-um ta."),
        ("Who wrote Hamlet?", "Sem, Shakespeare-ir wrote Hamlet-ol lo."),
        ("What is bread made of?", "Sem, brek-esh is made of flour, nim-um and yeast ta."),
        ("Name an animal of the desert.", "Sem, the jem-ir lives in the sahv-um ta."),
        ("In which year did the Second World War end?", "Sem, it ended in ib-nun-kar-fem-eth set."),
        ("What is a market?", "Sem, a torg-ak is a plen-ak where nas-ir buy and sell vaar-esh ki."),
        ("Do you know the answer?", "Sem, I do not know em."),
        ("Which language is spoken in Brazil?", "Sem, nas-ir in Brazil-um speak Portuguese-ol lo."),
        ("What is two plus two?", "Sem, dis-eth plus dis-eth is kar-eth set."),
    ]
    for q, a in toy:
        assert check_reply(a)["conformance"] == 1.0, a
    print(len(toy), "clean pairs")

    # ------------------------------------------------------------------------------------
    section("4. Train a LoRA adapter (toy run, CPU)")
    # A plain PEFT LoRA on the full-precision weights of the same model. The prompt is built
    # with evaluation_script.build_prompt, the exact chat template the grader uses, and only
    # the reply tokens are trained on. Forty steps on twelve pairs is a smoke test, not a
    # solution.
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    BASE = "HuggingFaceTB/SmolLM2-360M-Instruct"
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    def encode(q: str, a: str):
        prompt = build_prompt(q)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        ids = tok(prompt + a + "<|im_end|>\n", add_special_tokens=False).input_ids
        return ids, [-100] * len(p_ids) + ids[len(p_ids):]

    data = [encode(q, a) for q, a in toy]
    pad = tok.pad_token_id
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model.train()
    t0 = time.time()
    for step in range(args.steps):
        batch = [data[(step * 4 + i) % len(data)] for i in range(4)]
        L = max(len(x) for x, _ in batch)
        ids = torch.full((4, L), pad)
        lab = torch.full((4, L), -100)
        att = torch.zeros((4, L), dtype=torch.long)
        for j, (x, y) in enumerate(batch):
            ids[j, :len(x)] = torch.tensor(x)
            lab[j, :len(y)] = torch.tensor(y)
            att[j, :len(x)] = 1
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % 10 == 0:
            print(f"step {step} loss {loss.item():.3f} ({time.time() - t0:.0f}s)")
    model.save_pretrained(str(TASK / "adapter_toy"))
    print("saved adapter_toy/")

    # ------------------------------------------------------------------------------------
    section("5. Convert and evaluate")
    # The grader loads the Q8_0 base plus a GGUF adapter. tools/lora_to_gguf.py converts the
    # PEFT folder (no torch needed). Then the same evaluate call as in section 2, with the
    # adapter; from the shell this is: python evaluation_script.py --model adapter_toy.gguf
    r = subprocess.run([sys.executable, str(TASK / "tools" / "lora_to_gguf.py"), str(TASK / "adapter_toy"),
                        "--outfile", str(TASK / "adapter_toy.gguf")], capture_output=True, text=True)
    print(r.stdout, r.stderr[-300:])
    evaluate(adapter=str(TASK / "adapter_toy.gguf"), data_dir=DATA_DIR, out_path=TASK / "replies_toy.json")

    rows = json.load(open(TASK / "replies_toy.json", encoding="utf-8"))["replies"]
    for row in rows[:6]:
        print(f"[{row['conformance']:.2f}] {row['prompt'][:50]}\n      {row['reply'][:120]!r}\n      plain: {row['plain'][:100]}")

    # ------------------------------------------------------------------------------------
    section("6. Where this leaves you")
    print("""The toy adapter picks up the opening and some suffixes and loses most of its sense. The
real work is the corpus and the training recipe: what the model is taught to say, on which
prompts, and how much. replies_*.json holds every reply with its rule violations, its
plain-English reading and its scores, and the calls with their seals. The private data
folder has the same shape as dataset/public and asks about other things.

When every gate clears on dataset/public, upload adapter.gguf.""")


if __name__ == "__main__":
    main()
