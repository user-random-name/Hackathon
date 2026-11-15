# Field notes

## The regions and the grid

The data covers **eleven regions** of the Antarctic ice sheet, each a blue ice field with the
ice around it, between 20 and 90 km across. Every region is cut into **cells of 450 × 450 m**
and expressed in its own local frame: `row` and `col` are the cell's indices in the region's
lattice (from 0), `x_m` and `y_m` the coordinates of its centre in metres from the region's
south-western corner. Cells with the same `row` and neighbouring `col` are neighbours on the
ice. The frames of different regions have nothing to do with each other.

## `cells.csv`

One row per cell.

| Column | Meaning |
|---|---|
| `region`, `cell` | the region and the cell's number inside it |
| `row`, `col`, `x_m`, `y_m` | position in the region's lattice and local frame |
| `block_code` | the station's radio name for the cell: a digit and a letter (see below) |
| `ice_speed_m_yr` | surface ice velocity, metres per year (satellite radar) |
| `slope_deg` | surface slope, degrees |
| `elevation_m` | surface elevation, metres |
| `ice_thickness_m` | ice thickness, metres (radar sounding compilation) |
| `temperature_c` | mean annual 2 m air temperature, °C (regional climate model, 35 km) |
| `brightness` | surface brightness in the visible, 0–255 (optical mosaic) |
| `blue_ice_frac` | fraction of the cell mapped as blue ice (0–1) |
| `dist_blueice_km` | distance to the nearest mapped blue ice, km (0 inside it) |
| `backscatter` | radar backscatter, 0–255 (SAR mosaic) |
| `dist_station_km` | distance to the nearest research station or facility, km |
| `dist_sea_km` | distance to the sea, km |
| `dist_outcrop_km` | distance to the nearest mapped rock outcrop, km |

A few cells have no velocity value (`ice_speed_m_yr` empty).

## `finds.csv`

The meteorite record: one row per cell and **field season** in which meteorites were found
there, with the number of finds and their total mass in grams. A field season is the austral
summer a party worked in the region; the record of a region is the list of seasons parties
have searched it. Cells that are not listed for a season have no recorded find from that
season; nothing in the record says whether a cell was searched and found empty or never
searched.

## `regions.csv`

Region, number of cells, area, the seasons on file, and whether the region is a **target**.
For the eight regions that are not targets the record is complete. For the **three target
regions the record on file stops at a season**: the parties went back in later seasons, and
what they found then is not in the file.

## The season rule

The next season in a target region can search **10 % of the region's cells that have no
recorded find**. The station wants, for each target region, a probability per cell without a
recorded find that the next seasons would find meteorites there, and the plan that follows
from it: the 10 % of those cells with the highest probability.

## Block codes and the plan signature

The station names cells on the radio by **block code**, a digit from 1 to 8 followed by a
letter from A to Z (`3R`). When a season plan is filed, the planners write its **signature**
on the cover sheet: for each digit, the letter that most of the plan's cells carry under that
digit, eight letters in all. A letter is written only if it leads the runner-up by at least
**8 cells**; otherwise the planners leave a dash, since a signature must be unambiguous. The
signature is read over the three target regions together.

## Deliverables

- `probabilities.csv` with columns `region, cell, probability` for every cell without a
  recorded find in the three target regions.
- `plan.csv` with columns `region, cell`: the 10 % of each target region's cells without a
  recorded find that have the highest probability.
- The plan's signature, and the flag.
