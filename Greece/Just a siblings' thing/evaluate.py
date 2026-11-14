import torch, pandas as pd
from model import MythicThreatClassifier, FEATURE_COLUMNS

df = pd.read_csv("test_dataset.csv").merge(pd.read_csv("test_labels.csv"), on="encounter_id")
X = torch.tensor(df[FEATURE_COLUMNS].values, dtype=torch.float32)
y = df["is_catastrophic_threat"].values

rows = []
for name in ["ares", "athena", "apollo"]:
    net = MythicThreatClassifier()
    net.load_state_dict(torch.load(f"models/model_{name}.pth"))
    pred = (torch.sigmoid(net(X)) >= 0.5).int().detach().numpy()
    tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum(); tn = ((pred == 0) & (y == 0)).sum()
    rows.append((name, (tp + tn) / len(y), tp / (tp + fn), tp / (tp + fp), tp, fp, fn, tn))

res = pd.DataFrame(rows, columns=["model", "acc", "recall", "precision", "TP", "FP", "FN", "TN"])
print(res.round(4).to_string(index=False))
best = res.sort_values("recall", ascending=False).iloc[0]["model"]
print("FLAG: SIGMOID_" + best.upper())