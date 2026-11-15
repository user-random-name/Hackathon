train.py builds a LoRA adapter for HuggingFaceTB/SmolLM2-360M-Instruct that makes the model reply in the Call of the Square register while still answering the question. A reply opens with "Sem,", marks every name and table word with a class suffix (-um, -ir, -ak, -eth, -ol, -esh), writes numbers as digit names closed with -eth, and ends every sentence with the seal of its last marked word (ta, man, ras, set, lo, ki, or em when nothing is marked). A prompt that ends with "Answer with one noun." gets a single sealed noun, for example "Sem, Algiers-ak ras."

The script does four steps:
1. Builds the corpus (corpus.py). A rule-based translator converts plain English into the register: table words become stem plus suffix, names get a class suffix from a gazetteer, digits become digit names, and the seal is chosen from the last marked word. About 200 hand-written facts are used, each as a full reply and as a one-noun call, plus short answers written by the base model itself for open prompts. Every reply is checked with style_checker.py and any that break a rule or exceed 18 words are dropped.
2. Trains a LoRA (r=16, alpha=32, all seven projections, 3 epochs, lr 2e-4) on the full-precision weights, using the exact grader chat template and computing loss on the reply tokens only.
3. Converts the PEFT adapter to adapter.gguf with tools/lora_to_gguf.py.
4. Runs evaluation_script.py on the public data. All five gates pass: style 0.367, entity class 0.909, content 0.644, log-prob -2.195, calls 8/8. The public facts are part of the training set, so these numbers are optimistic.

Usage:
Put corpus.py, train.py, style_checker.py, evaluation_script.py, register/, tools/ and dataset/public/ in one folder, then run:
python evaluation_script.py --setup
python train.py
It prints the training loss, writes adapter.gguf and prints the gate table. Needs Python 3.10 to 3.12, torch, transformers, peft, gguf and llama-cpp-python.
