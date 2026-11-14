# The Knots You Cannot Read: write-up

**Flag: `SIGMOID_PROVENANCE`**

## Where the content is stored
`model.safetensors` has three tensors: `embedding.weight` (256x32), `classifier.weight` (3x32) and `classifier.bias` (3).
The model reads ASCII text, so only embedding rows 0-127 can ever be looked up (the corpus uses bytes up to 121).
**Rows 128-255 are unreachable, so nothing stored there can change a prediction.** The payload is in
**row 128, the first 18 values** of `embedding.weight`.

## How it was found
Normal weights are random float32. Checking which values have their lower 16 bits all zero (exactly representable
in bfloat16) flags only row 128, dims 0-17. The model itself is honest: embedding, mean over the 64 characters,
then a linear layer gives 95.4% accuracy on `corpus.csv` (class order antisuyu, chinchaysuyu, qullasuyu).

## How it is encoded
Each value equals `-k/1024` with k from 33 to 63. With `c = 64 - k` (1..31), c maps to a letter (A=1 ... Z=26, `_`=31):
`S I G M O I D _ P R O V E N A N C E`

## Extraction code (`extract.py`)
```python
import json, struct, os
import numpy as np

folder = os.path.dirname(os.path.abspath(__file__)) # folderul unde e acest fisier
raw = open(os.path.join(folder, "model.safetensors"), "rb").read()

n = struct.unpack("<Q", raw[:8])[0] # lungimea header-ului
hdr = json.loads(raw[8:8 + n])
a, b = hdr["embedding.weight"]["data_offsets"]
E = np.frombuffer(raw[8 + n + a: 8 + n + b], dtype=np.float32).reshape(256, 32)

k = np.round(-E[128, :18] * 1024).astype(int) # randul 128, primele 18 valori
print("".join("_" if 64 - x == 31 else chr(64 + 64 - x) for x in k))
```
Output: `SIGMOID_PROVENANCE`. The full audit script is `extract.py`, and its output is in `extract_output.txt`.