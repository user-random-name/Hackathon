detector.py is a streaming anomaly detector for bunker sensor data. It reads one reading at a time (temperature, humidity, vibration, acoustic noise, air pressure, CO2) and returns an anomaly score between 0 and 1. A score of 0.5 or more counts as an anomaly. It only remembers a few recent readings and never looks at future ones.

Each sensor value is first converted into a z-score: how many standard deviations it is from that sensor's normal average. The normal averages and spreads were measured on the normal rows of the training data. A z-score near 0 is normal, and a large one is suspicious. Then four checks run, and the highest score wins.
1. Spikes (score 1.0)
If humidity, vibration, acoustic noise or CO2 is more than 6 standard deviations from normal, the reading is flagged. Temperature and pressure change slowly, so for those the code checks how far the reading jumped from the previous one instead. The previous value is capped so one big spike does not cause a false alarm on the next reading.
2. Group problems (score 0.9)
Vibration, acoustic noise and CO2 are added into one combined number. Normally it stays near 0. If it goes above 4 in a single step, or its average over the last 4 steps goes above 3, the reading is flagged. This catches stretches where all three sensors are moderately high together.
3. Humidity and CO2 drifts (score 0.8)
Two things are tracked for each: a smoothed running average (keeps 88% of the old value and adds 12% of the new one), and a plain average of the last 4 readings. The reading is flagged when both are clearly away from normal in the same direction. Requiring both makes the alarm stop quickly once the drift ends.
4. Temperature and pressure drifts (score 0.7)
The average of the last 8 readings is compared with the average of readings from 40 to 130 steps earlier. If the gap is more than 3 standard deviations, the reading is flagged. The older readings stop being updated while an alarm is active, so a drift never becomes the new normal.
If none of the checks fire, the score is 0.0.

Usage:
Create one Detector and call update() with each reading, in time order.
from detector import Detector
d = Detector()
score = d.update(reading)
Each reading is a dictionary with the keys timestamp, bunker_zone, temperature_c, humidity_pct, vibration_g, acoustic_db, air_pressure_kpa and co2_ppm. Missing or invalid values are replaced with the normal value, so the code does not crash.
