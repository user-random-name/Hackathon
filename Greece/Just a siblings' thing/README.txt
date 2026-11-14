evaluate.py compares three oracle models (Ares, Athena, Apollo) that classify whether a mythological encounter is a catastrophic threat, and picks the one that should guard Delphi. All three have the same test accuracy (95%), so accuracy cannot decide. It runs each model on the 1,000 test encounters at a decision threshold of 0.50 and compares the kinds of mistakes they make.

Each model is one linear layer: it takes 9 features (in the order defined in model.py) and returns a raw score. The score is passed through a sigmoid, which turns it into a probability between 0 and 1. A probability of 0.5 or more counts as "catastrophic threat". The predictions are then compared with the true labels in test_labels.csv. Rows are matched by encounter_id, so labels never shift.

For each model the code counts four things:
1. True positives (TP): a real threat that the model caught.
2. False positives (FP): a false alarm.
3. False negatives (FN): a real threat that the model missed. This is the costly mistake in this task.
4. True negatives (TN): a correct "no threat".
From these it computes accuracy, recall (TP / (TP + FN), how many real threats were caught), precision and F1.

Missing a real threat is far more expensive than a false alarm, so the model with the highest recall (fewest missed threats) is selected. Precision breaks ties.

Results at threshold 0.50 (100 real threats out of 1,000):
Ares: recall 0.95, 5 missed threats, 45 false alarms.
Athena: recall 0.75, 25 missed threats, 25 false alarms.
Apollo: recall 0.55, 45 missed threats, 5 false alarms.
Ares is chosen. Flag: SIGMOID_ARES

Usage:
Put evaluate.py, model.py, test_dataset.csv, test_labels.csv and the models folder (model_ares.pth, model_athena.pth, model_apollo.pth) in one folder, then run:
py evaluate.py
It prints the metrics table and the flag on the last line. Needs Python 3.10+, numpy, pandas and torch (see requirements.txt).