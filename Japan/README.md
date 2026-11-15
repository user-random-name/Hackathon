# City of Subtitles — KAMI attribution and subtitle-density evidence

## Result

**Flag candidate:** `SIGMOID_KAMI`

The code runs on the original **253 JPEG** dataset ZIP and reproduces a five-week, character-position-density signal. Its small 5-row font decoder selects `KAMI` as its highest-scoring word under the documented 3–3–5–3 character-cell layout.

The decoded letters are not equally certain: **K (6.174) narrowly beats X (6.131)** under the provided 3×5 alphabet, while A, M, and I have clearer margins. For that reason, the result is strong *supporting evidence* for the supplied word, not an infallible blind decoder or cryptographic proof.

## How to run

Place `solve.py` alongside the original dataset ZIP, or supply the ZIP's full path.

```bash
pip install -r requirements.txt
python solve.py "dataset-20261114T173504Z-1-001(1).zip" results KAMI
```

The last argument (`KAMI`) is **only for the optional statistical check** against a specified candidate; the pixel-font decoder runs without it and independently prints its best match. You can run `python solve.py "dataset.zip" results` to see the decoding without that candidate input.

## Short write-up for the hackathon

1. **Identify the photographer.** The original ZIP inadvertently retains file modification times. Entries are grouped by creator. Sorting the JPG entries by ZIP timestamp, with a tie rule for the ZIP's 2-second timestamp precision, recovers six groups. I mapped groups to names using the 108 credited JPEGs and confirmed **108/108** known labels. The anonymous set contains **45 Toorisugari images (9 per week)** and 20 from each other creator (4 per week). This is an **unintended ZIP metadata side channel**, not an image-sensor fingerprint, and is stated transparently as such.
2. **Find the captions, not the image scene.** In each anonymous JPEG, detect bright subtitle ink inside locally dark boxes (both full black bars and translucent overlays). Connected-component grouping produces **494 subtitle lines**, of which 145 are on Toorisugari's photos. The black rectangles occur at different positions and the font comes in roughly three sizes.
3. **Treat each line as character cells from left to right.** Approximate font cell widths from measured ink heights (26, 32, or 38 pixels). For each cell, calculate the fraction occupied by bright Japanese text. Kanji usually fill more of a cell than kana. Positions after the end of a subtitle line count as zero in that line.
4. **Compare Toorisugari against the whole week.** For each of weeks 8–12, subtract the mean per-position ink fraction across **all subtitle lines that week** from the mean among Toorisugari's lines. Stack these five weekly vectors as the five rows of a pixel pattern, then standardize each row to reduce differences in the amount of text.
5. **Read the word.** Match the measured 5×17 pattern to a declared 5-row capital-letter font. The expected layout has widths 3, 3, 5, 3, separated by one blank cell. The best per-slot matches are **K, A, M, I**, producing `SIGMOID_KAMI`. The matching font grid is shown separately from the measured grid so it is clear that the letters were not simply painted onto the evidence image.
6. **Independent-label check.** For the *specified hypothesis* KAMI, randomly reassign nine anonymous photos per week to the Toorisugari group 1,000 times (whole photos, not individual lines). The true group yields a **0.7071 correlation** with the KAMI template; in this run no random reassignment matched or exceeded it, so the Monte Carlo upper-bound estimate is **p = 0.001**. This is a conditional supporting test, not a multiple-comparisons-adjusted significance claim, and it does not resolve the K/X glyph ambiguity.

## Output files

- `results/attribution_anonymous.csv` — each nameless photo and predicted photographer.
- `results/character_cell_measurements.csv` — every measured caption line, its week, photographer, and per-character-position ink fractions.
- `results/five_week_density.csv` — numerical five-week differences, before letter decoding.
- `results/measured_density_only.png` — visualized numerical signal only, not filled-in letters.
- `results/density_vs_decoded_word.png` — measured signal above and a separate decoded-letter *reference template* below.

## Reproducibility note

This is a **verifiable, hypothesis-supported solution**, not an artificially generated flag: `solve.py` calculates the per-line densities from the JPEG bytes and scores letter templates against them. The auto-decoded word is assembled from four independent slot rankings. The template font and assumed slot layout are choices that affect the first letter, and the code documents them instead of hiding that limitation.
