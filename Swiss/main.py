import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("test-input.csv")
label = next((c for c in df.columns if "released" in c.lower()), df.columns[-1])
X = df.drop(columns=[label])
y = df[label]

clf = DecisionTreeClassifier(random_state=0).fit(X, y)
t = clf.tree_

letters = []
for i in range(t.node_count):
    if t.children_left[i] != -1:
        letters.append(X.columns[t.feature[i]][0].upper())

with open("test-output.txt", "w") as f:
    f.write("SIGMOID_" + "".join(letters) + "\n")