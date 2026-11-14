This script solves the "Recover the Black Box" challenge by discovering how an unknown mathematical function works. It uses a limited number of queries to identify important variables, detect interactions, find sudden changes in the function, and predict outputs for 300 points.

The script works in four steps:

1. **Identifying variables:** Tests the five input variables (x1–x5) to determine which affect the function's output and which are irrelevant.
2. **Finding interactions:** Changes pairs of variables and compares the results to discover which variables interact and calculate their interaction coefficient.
3. **Detecting discontinuities:** Finds sudden jumps in the function's output, identifies the variables responsible, and estimates the boundary and jump size using binary search.
4. **Predicting outputs:** Uses the discovered mathematical model to calculate predictions for all 300 challenge points and returns the results in the required format.

**Usage:**

Place `main.py`, `harness.py`, `evaluation_script.py`, and `practice_function.py` in the same folder, with `challenge_set.json` inside a `challenge` subfolder.

Install the required libraries:

`pip install numpy scipy`

Run the practice evaluation:

`python evaluation_script.py main.py --eval-type train`

The evaluator will display the final score, prediction accuracy, and number of queries used. The script achieved 100/100 points on the practice challenge using 144 out of 150 allowed queries.
