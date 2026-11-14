extract.py audits model.safetensors, a character-level classifier that predicts which of three registers a transcribed record belongs to, and recovers the content hidden inside it. The model itself is honest: it classifies correctly and every behavioural test comes back clean. The hidden content sits in a place that no input can ever reach.

The model has three tensors: an embedding table (256 rows x 32 values, one row per possible byte value), a classifier weight (3 x 32) and a classifier bias (3). A record is read as ASCII bytes, the embeddings of its 64 characters are averaged, and the linear layer gives one score per register. The corpus only uses bytes up to 121, so rows 128 to 255 of the embedding table are never looked up. Anything stored there cannot change a prediction.

The script does four steps:
1. Reads the .safetensors file directly: the first 8 bytes give the size of a JSON header, the header lists where each tensor sits, and the numbers follow.
2. Checks the model on corpus.csv (mean of the embeddings, then the linear layer). Accuracy is 95.4% with the classes in alphabetical order (antisuyu, chinchaysuyu, qullasuyu), so the model does what it claims.
3. Looks for floats whose lower 16 bits are all zero (exactly representable in bfloat16). Normal weights are random, so this is unusual. Only one row matches: row 128, dimensions 0 to 17, 18 values in total.
4. Decodes those 18 values. Each equals -k/1024 with k between 33 and 63. The letter number is 64 - k (A = 1 ... Z = 26, 31 = underscore).

The 18 values spell SIGMOID_PROVENANCE. The first 8 characters match the known flag prefix, which confirms the decoding.

Usage:
Put extract.py, model.safetensors and corpus.csv in one folder, then run:
py extract.py
It prints the accuracy, the row that stands out and the decoded string on the last line. Needs Python 3.10+, numpy and pandas. A shorter version, extract_short.py, only needs model.safetensors and numpy and prints the decoded string.