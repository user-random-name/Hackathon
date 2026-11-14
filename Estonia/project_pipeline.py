"""
e-Residency application fraud screen -- evaluation pipeline.
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("applications.csv")

# Identity fields are personal data and are never used for modeling.
X = df.drop(columns=["submission_id", "date_of_birth", "citizenship", "label"])
y = df["label"]

# Features are on very different scales (seconds, km, days, 0-1 scores).
X_scaled = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=1137
)

clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, y_train)

acc = accuracy_score(y_test, clf.predict(X_test))
print(f"Test accuracy: {acc:.4f}")
