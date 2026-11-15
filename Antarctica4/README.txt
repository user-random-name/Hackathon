final2.py builds the season plan for the three target regions (R04, R06, R09). For every cell with no recorded find, it gives a probability that later field seasons find meteorites there. The 10% of cells with the highest probability in each region form the plan. It only uses the cell descriptions and the finds recorded up to the last season on file, and never looks at later seasons.

Each cell is described with three kinds of features, then a model scores it. The model was trained on the eight complete regions by pretending their record stopped at an earlier season and checking which cells found meteorites afterwards.
1. Physical features
The layers from cells.csv: blue ice fraction, backscatter, ice thickness, ice speed, elevation, brightness, and the distances to blue ice, station, sea and rock outcrops.
2. Nearby finds
The distance to the nearest earlier finds, and how many earlier finds lie within 0.5, 1, 2, 4, 8 and 16 km.
3. How well the area was searched
The size of the connected blue ice patch the cell sits in, and how many cells in that patch already have finds. A patch with many finds is probably well searched, a patch with none may not have been visited yet.
The block code is random noise, so it is not used by the model.
The model is a gradient-boosted classifier (an ensemble of five settings). A calibration step turns its scores into probabilities. Leaving one complete region out at a time, it scored AUC 0.86 to 0.98.
The plan has 5,233 cells (R04: 891, R06: 1,414, R09: 2,928). The letter that leads each digit 1 to 8 in the block codes is T, R, A, N, S, F, E, R, which reads TRANSFER. With the strict 8-cell margin the signature is TRA-SFE-, because digits 4 and 8 miss the margin. Flag: SIGMOID_TRANSFER

Files:
feat.py builds the features for a region at a given cutoff season.
train.py builds the training set from the complete regions and checks the model.
final2.py trains the final model, writes the two CSVs and prints the signature.
probabilities.csv has the columns region, cell, probability.
plan.csv has the columns region, cell.

Usage:
Put cells.csv and finds.csv in the folder set by D= at the top of feat.py, then run:
python3 final2.py