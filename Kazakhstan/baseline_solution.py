"""
Shows the input and output formats with the simplest forensic idea there is, error level
analysis: save the page again as JPEG and look where it changes most. It finds the stamps,
gives one box per page, writes boxes.csv and draws the box centres of every album.

It runs in about a minute and it does NOT solve the task: its dots do not form letters.

    python baseline_solution.py --data dataset
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

PAGE_W, PAGE_H = 1600, 1200


def stamps_on(page):
    """Boxes (x, y, w, h) of the stamps: everything that is not the dark album background."""
    d = np.abs(page.astype(np.int16) - np.array([38, 36, 34])).max(2)
    m = cv2.morphologyEx((d > 30).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(m)
    return [(x, y, w, h) for x, y, w, h, area in stats[1:] if w > 120 and h > 120]


def error_level(page):
    """Per-pixel change when the page is saved again as JPEG, smoothed."""
    again = cv2.imdecode(cv2.imencode(".jpg", page, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], 1)
    diff = np.abs(page.astype(np.float32) - again.astype(np.float32)).mean(2)
    return cv2.boxFilter(diff, -1, (41, 41))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    args = ap.parse_args()
    data = Path(args.data)

    # the exhibits show what an altered region looks like
    masks = sorted(data.glob("exhibits/*_mask.png"))
    sizes = [cv2.boundingRect(cv2.imread(str(m), 0))[2:] for m in masks]
    print(f"{len(masks)} exhibits, median altered box {np.median(sizes, 0).astype(int).tolist()} px")

    rows = []
    albums = sorted(p for p in (data / "albums").iterdir() if p.is_dir())
    sheet = np.zeros((PAGE_H, PAGE_W * len(albums)), np.uint8)
    for ai, album in enumerate(albums):
        for f in sorted(album.glob("*.jpg")):
            page = cv2.imread(str(f))
            ela = error_level(page)
            inside = np.zeros(ela.shape, bool)
            for x, y, w, h in stamps_on(page):
                inside[y + 20:y + h - 20, x + 20:x + w - 20] = True  # skip the perforation
            ela[~inside] = 0
            cy, cx = np.unravel_index(int(ela.argmax()), ela.shape)
            rows.append([album.name, f.stem, cx - 25, cy - 25, 50, 50])
            cv2.circle(sheet, (cx + PAGE_W * ai, cy), 18, 255, -1)
        print(album.name, "done")

    with open("boxes.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["album", "page", "x", "y", "w", "h"])
        writer.writerows(rows)
    cv2.imwrite("baseline_overlay.png", cv2.resize(sheet, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA))
    print("wrote boxes.csv and baseline_overlay.png (one panel per album)")


if __name__ == "__main__":
    main()
