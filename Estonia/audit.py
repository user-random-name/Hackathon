import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (train_test_split, GroupShuffleSplit,
                                     LeaveOneGroupOut, cross_val_predict)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("applications.csv")
df["group"] = df["date_of_birth"].astype(str) + "|" + df["citizenship"].astype(str)

X = df.drop(columns=["submission_id", "date_of_birth", "citizenship", "label", "group"])
y = df["label"]
g = df["group"]
SEEDS = range(30)


def run(seed, scaler_on_train_only, group_split):
    idx = np.arange(len(df))
    if group_split:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                        random_state=seed).split(X, y, g))
    else:
        tr, te = train_test_split(idx, test_size=0.2, random_state=seed)

    if scaler_on_train_only:
        sc = StandardScaler().fit(X.iloc[tr])
    else:
        sc = StandardScaler().fit(X)          # as shipped: scaler sees all rows

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(sc.transform(X.iloc[tr]), y.iloc[tr])
    return accuracy_score(y.iloc[te], clf.predict(sc.transform(X.iloc[te])))


def report(name, **kw):
    accs = [run(s, **kw) for s in SEEDS]
    print(f"{name:<45} mean={np.mean(accs):.4f}  std={np.std(accs):.4f}")


print("Shipped seed 1137 (as shipped):", round(run(1137, False, False), 4))
print()
report("Baseline, 30 seeds (also = 'fix the seed')", scaler_on_train_only=False, group_split=False)
report("Decoy 1 fixed: scaler fit on train only",    scaler_on_train_only=True,  group_split=False)
report("REAL bug fixed: group-aware split",          scaler_on_train_only=False, group_split=True)
report("All three fixed",                            scaler_on_train_only=True,  group_split=True)

# Final number for the flag: leave-one-person-out with the shipped model
pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))
pred = cross_val_predict(pipe, X, y, cv=LeaveOneGroupOut(), groups=g)
acc = (pred == y).mean()
print()
print("Leave-one-group-out accuracy:", round(acc, 4))
print("Flag (rounded):  SIGMOID_%d" % round(acc * 100))
print("Flag (truncated): SIGMOID_%d" % int(acc * 100))