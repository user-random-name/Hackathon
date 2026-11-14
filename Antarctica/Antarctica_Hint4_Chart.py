"""Draw the single 134-day chart of corruption candidates (Hints 1–4).

Run: python Antarctica_Hint4_Chart.py
Install: pip install matplotlib
Input: antarctica_hint4_candidates.csv in the same directory.
Output: antarctica_hint4_one_chart.png and .svg

This reproduces the *previously detected candidates*; it does not
independently re-detect corrupt messages from messages.txt.
"""
import csv
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

HERE = Path(__file__).resolve().parent
CSV_FILE = HERE / 'antarctica_hint4_candidates.csv'
START = date(1987, 3, 7)
DAYS = 134
HOURS = 8  # observations at 00, 03, ..., 21 UTC

# 0 = no candidate, 1 = earlier candidate, 2 = newly found by Hint 4
chart = [[0 for _ in range(DAYS)] for _ in range(HOURS)]
counts = {1: 0, 2: 0}

with CSV_FILE.open(newline='', encoding='utf-8-sig') as file:
    for row in csv.DictReader(file):
        day = (date.fromisoformat(row['date']) - START).days
        hour = int(row['hour_UTC'].split(':')[0])
        if not (0 <= day < DAYS and hour in range(0, 24, 3)):
            raise ValueError(f"Invalid date/hour in row: {row}")
        category = 2 if row['classification'] == 'NEW Hint 4' else 1
        chart[hour // 3][day] = category
        counts[category] += 1

fig, ax = plt.subplots(figsize=(27, 5), facecolor='white')
fig.subplots_adjust(top=.79, bottom=.22, left=.047, right=.993)
colors = ListedColormap(['#f6f7f8', '#1f344b', '#df6d27'])
ax.imshow(chart, origin='upper', interpolation='nearest', aspect='auto',
          cmap=colors, vmin=0, vmax=2)

ax.set_xticks(range(0, DAYS, 5), [str(i + 1) for i in range(0, DAYS, 5)], fontsize=10)
ax.set_xticks([i - .5 for i in range(DAYS + 1)], minor=True)
ax.set_yticks(range(HOURS), [f'{i * 3:02d}:00' for i in range(HOURS)], fontsize=11)
ax.set_yticks([i - .5 for i in range(HOURS + 1)], minor=True)
ax.grid(which='minor', axis='x', color='#d5dce2', alpha=.58, lw=.32)
ax.grid(which='minor', axis='y', color='#d5dce2', alpha=.8, lw=.6)
ax.tick_params(which='both', length=0)
ax.set_xlabel('Day of observation  •  1 = 7 March 1987  •  134 days total',
              fontsize=12, labelpad=14)
ax.set_ylabel('Observation hour (UTC)', fontsize=11, labelpad=8)

for day, month in [(1, 'MAR 7'), (26, 'APR 1'), (56, 'MAY 1'),
                   (87, 'JUN 1'), (117, 'JUL 1')]:
    ax.axvline(day - 1.5, lw=1.25, color='#8193a3')
    ax.text(day - 1, -1.2, month, color='#405b73', fontsize=10,
            ha='left', clip_on=False)

fig.text(.047, .945, 'ANTARCTICA   /   WEATHER REPORT CORRUPTION',
         fontsize=20, weight='bold', color='#1c3248')
fig.text(.047, .868,
         f'Hints 1–4    •    {counts[1]} earlier candidates  +  '
         f'{counts[2]} new Hint 4 violations  =  {sum(counts.values())} candidates',
         fontsize=12, color='#596d80')
fig.text(.05, .075, '■  Earlier candidates (Hints 1–3)',
         color='#1f344b', fontsize=12)
fig.text(.31, .075, '■  Newly found with Hint 4',
         color='#df6d27', fontsize=12)
fig.text(.98, .075, 'Each square = one 3-hour weather report',
         color='#566879', fontsize=11, ha='right')

for extension in ('png', 'svg'):
    path = HERE / f'antarctica_hint4_one_chart.{extension}'
    fig.savefig(path, dpi=215, facecolor='white')
    print('Created:', path)
plt.close(fig)
print(f'Total detected candidates: {sum(counts.values())}')
