# Antarctica – Corrupted Weather Reports

This project visualizes suspicious weather reports from an Antarctic research station using Python.

## How it works
The program reads `antarctica_hint4_candidates.csv`, which contains 87 suspicious reports identified using four hints.

The reports were selected based on:
- Invalid or inconsistent weather measurements.
- Sudden changes in temperature, pressure, and wind.
- Contradictions between neighboring reports.
- Unusual visibility, humidity, precipitation, and cloud cover.

The program uses Matplotlib to place these reports on a single chart, showing their positions across 134 days and 8 observations per day.

## How to run
Install Matplotlib:
```bash
pip install matplotlib
```

Run the program:
```bash
python Antarctica_Hint4_Chart.py
```

## Output
The program generates a PNG and an SVG chart highlighting the suspicious reports.

**Note:** The script visualizes previously detected candidates. It does not independently analyze the original weather messages or decode the final hidden word.
