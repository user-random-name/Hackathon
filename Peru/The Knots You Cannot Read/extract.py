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