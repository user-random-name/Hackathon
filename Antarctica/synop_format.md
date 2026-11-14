# SYNOP (FM-12) format used in `messages.txt`

A SYNOP message is a sequence of five-character **groups** separated by spaces and ended by
`=`. A `/` inside a group means "not available". This station uses the groups below and no
others. Groups marked *optional* appear only when there is something to report.

## Section 0 — identification

| Group | Meaning |
|---|---|
| `AAXX` | Land station report. |
| `YYGGi` | `YY` day of the month, `GG` hour UTC, `i` = 1: wind speeds are in **m/s**, measured. |
| `IIiii` | Station index (fictional here). |

## Section 1 — the observation

| Group | Meaning |
|---|---|
| `iRixhVV` | `iR` = 4: no precipitation group. `ix` = 1 if a present-weather group (7) follows, 3 if not. `h`: height of the lowest cloud base, code 0–9 (0: < 50 m, 1: 50–100, 2: 100–200, 3: 200–300, 4: 300–600, 5: 600–1000, 6: 1000–1500, 7: 1500–2000, 8: 2000–2500, 9: ≥ 2500 m or no cloud), `/` unknown. `VV`: visibility code (00: < 100 m; 01–50: 0.1–5.0 km in steps of 100 m; 56–80: 6–30 km in steps of 1 km; 81–88: 35–70 km in steps of 5 km; 89: > 70 km), `//` unknown. |
| `Nddff` | `N`: total cloud cover in oktas 0–8, 9 = sky obscured, `/` unknown. `dd`: wind direction in **tens of degrees**, 00 = calm, 36 = north; `ff`: wind speed in m/s (two digits). |
| `1sTTT` | Air temperature. `s` = 0 positive, 1 negative; `TTT` tenths of °C (`11108` = −10.8 °C). |
| `2sTTT` | Dew-point temperature, same encoding. |
| `3PPPP` | Pressure **at station level**, tenths of hPa, thousands digit dropped (`39743` = 974.3 hPa, `30012` = 1001.2 hPa). |
| `4PPPP` | Pressure **reduced to sea level**, same encoding. |
| `5appp` | Pressure tendency over the **last three hours**. `a`: characteristic, 0–3 = pressure now higher than three hours ago, 4 = same, 5–8 = lower. `ppp`: amount of change in tenths of hPa (`58002` = fell by 0.2 hPa, `52010` = rose by 1.0 hPa). |
| `7wwW1W2` | *Optional.* Present weather `ww` (two-digit code, 00–99) and past weather `W1`, `W2` (one digit each). |
| `8NhCLCMCH` | *Optional.* Amount of low/middle cloud `Nh` (oktas) and the genus codes of low, middle and high clouds (one digit each, `/` not observable). |

## Section 3 — supplementary data (*optional*; starts with `333`)

| Group | Meaning |
|---|---|
| `1sTTT` | Maximum temperature of the **last 12 hours** (reported at 00 and 12 UTC). |
| `2sTTT` | Minimum temperature of the **last 12 hours** (reported at 00 and 12 UTC). |
| `910ff` | Highest gust in the ten minutes before the observation, m/s. |
| `911ff` | Highest gust since the previous observation, m/s. |

