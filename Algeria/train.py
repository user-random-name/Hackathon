"""python train.py  -> builds corpus (distilling open answers from the base model), trains LoRA, converts, evaluates."""
import json, random, subprocess, sys, re, torch, time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from evaluation_script import build_prompt, evaluate, DATA_DIR
import corpus

T = Path(__file__).resolve().parent
BASE = "HuggingFaceTB/SmolLM2-360M-Instruct"
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
torch.manual_seed(0); random.seed(0)

def gen(q):  # base model, greedy, asked for short sentences (only used to make training text)
    ids = tok(build_prompt(q + " Use short sentences of at most twelve words."), return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=70, do_sample=False)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)

cache = T / "open_cache.json"
if cache.exists():
    C = json.load(open(cache))
else:
    C = {q: gen(q) for q in corpus.OPEN}; json.dump(C, open(cache, "w"))
ex, n_kb = corpus.build(gen=lambda q: C.get(q))
print(len(ex), "examples;", len(ex) - n_kb, "distilled open")
json.dump(ex, open(T / "corpus.json", "w"), indent=1, ensure_ascii=False)

model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
def enc(q, a):
    p = tok(build_prompt(q), add_special_tokens=False).input_ids
    ids = tok(build_prompt(q) + a + "<|im_end|>\n", add_special_tokens=False).input_ids
    return ids, [-100] * len(p) + ids[len(p):]
data = [enc(q, a) for q, a in ex]
EPOCHS, BS = 3, 8
steps = EPOCHS * ((len(data) + BS - 1) // BS)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 10) * max(0.05, 1 - s / steps))
model.train(); step = 0; t0 = time.time()
for ep in range(EPOCHS):
    random.shuffle(data)
    for i in range(0, len(data), BS):
        b = data[i:i + BS]; L = max(len(x) for x, _ in b)
        ids = torch.full((len(b), L), tok.pad_token_id); lab = torch.full((len(b), L), -100); att = torch.zeros((len(b), L), dtype=torch.long)
        for j, (x, y) in enumerate(b):
            ids[j, :len(x)] = torch.tensor(x); lab[j, :len(y)] = torch.tensor(y); att[j, :len(x)] = 1
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(); step += 1
        if step % 10 == 0: print(f"ep {ep} step {step}/{steps} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
model.save_pretrained(str(T / "adapter"))
subprocess.run([sys.executable, str(T / "tools" / "lora_to_gguf.py"), str(T / "adapter"), "--outfile", str(T / "adapter.gguf")], check=True)
evaluate(adapter=str(T / "adapter.gguf"), data_dir=DATA_DIR, out_path=T / "replies.json")