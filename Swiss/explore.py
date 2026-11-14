import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("train-input.csv")
print("Columns:", df.columns.tolist())
print(df.dtypes)
print(df.head())
print("Expected answer:", open("train-output.txt").read())

# Guess the label column: "released"/"label" if present, otherwise the last column
label = None
for name in ("released", "label", "target", "y"):
    if name in df.columns:
        label = name
        break
if label is None:
    label = df.columns[-1]
print("\nLabel column used:", label)

X = df.drop(columns=[label])
y = df[label]

clf = DecisionTreeClassifier(random_state=0).fit(X, y)
t = clf.tree_
print("train acc:", clf.score(X, y))
print("nodes:", t.node_count, "| depth:", clf.get_depth(), "| leaves:", clf.get_n_leaves())

# Structure dump: first 60 nodes in sklearn's order
print("\nid | left | right | feature | threshold | samples")
for i in range(min(60, t.node_count)):
    if t.children_left[i] == -1:
        print(i, "LEAF", "samples=", t.n_node_samples[i])
    else:
        print(i, t.children_left[i], t.children_right[i],
              X.columns[t.feature[i]], round(t.threshold[i], 4), t.n_node_samples[i])