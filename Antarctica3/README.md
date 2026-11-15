# Antarctica Level 3 — Ups and Downs

Flag candidate: `SIGMOID_HARKOV` (not independently checked against the submission platform).

## Run

Install Python 3.10+ and dependencies:

```bash
pip install numpy pandas scipy matplotlib
python solution.py
```

Keep `solution.py`, `soundings.csv`, `chart.csv`, and `tracking_notes.txt` in the same folder.

## Method

1. Read all 3,132 soundings. The overlapping survey lines suggest that 1961 echo-sounder readings are roughly 3% higher than the 1998/2019 records. Scale those 1961 depths by 0.970. This correction is an empirical estimate, not a documented calibration constant.
2. Fit a **local Matérn-3/2 Gaussian process (kriging)** using the 48 closest soundings to each 500 m grid-cell center. All soundings are retained as available training data; none are randomly discarded. The model uses a 5 km spatial length scale, a 220 m process scale, and a 20 m observation-noise scale. The uncertainty is the posterior standard deviation of *latent* seabed depth under the model. These choices are modeling assumptions.
3. Use `lower = predicted_depth - 1.2815515655 * sd` to calculate the one-sided 90% lower bound. Land is always foul. A cell is clear only if its lower bound is at least 350 m.
4. Mark a 1 km chart square foul if *any* of its four cells is foul, then fill every square in the 5 x 24 tracking grid using `#` (foul) or `.` (clear).
5. Find a shortest four-neighbor route from entry to station using Dijkstra's algorithm, never entering a foul cell. Four-neighbor movement avoids cutting diagonally between corners of foul cells.

## Results

- 120 x 120 depth and uncertainty grids.
- 8,296 foul 500 m cells; 2,189 foul 1 km chart squares, under this specific model.
- 179 route cells, or 89.0 km of grid-edge travel.
- The reconstructed 5 x 24 bitmap yields `HARKOV`.

```text
#.#. .#.. ##.. #.#. ###. #.#.
###. #.#. #.#. #.#. #.#. #.#.
#.#. ###. ##.. ##.. #.#. #.##
#.#. #.#. #.#. #.#. #.#. #.#.
#.#. #.#. #.#. #.#. ###. .#..
 H    A    R    K    O    V
```

The final cell of the third row is an extra foul classification beyond the 3-column glyph; it does not change the word. Model output is not an official reference chart, and borderline cells may change with calibration/noise assumptions. No platform flag acceptance is claimed.

## Generated files

- `depth.csv`: 120 x 120 predicted depth in meters; row index 0 = south.
- `sd.csv`: 120 x 120 predictive standard deviation in meters.
- `foul.csv`: 120 x 120 (1 = foul).
- `square_foul.csv`: 60 x 60 (1 = foul).
- `route.csv`: ordered route as `row,col` pairs, starting at entry and ending at station.
- `filled_tracking_notes.txt`: all 120 tracking-note responses, plus the bitmap.
- `chart.png`: depth, lower bounds, and route overview.
