import cv2
import numpy as np
import os
import glob
import json
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
ALBUMS_DIR = "albums"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_stamps(img):
    """Finds the stamp boxes on the page."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stamp_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 10000:
            x, y, w, h = cv2.boundingRect(cnt)
            stamp_boxes.append((x, y, w, h))

    stamp_boxes.sort(key=lambda b: (b[1], b[0]))
    return stamp_boxes


def detect_tampering_in_page(image_path):
    """Returns the normalized (x, y) of the best anomaly on the page."""
    img = cv2.imread(image_path)
    if img is None: return None

    stamp_boxes = find_stamps(img)
    best_anomaly = None
    max_score = 0

    for i, (x, y, w, h) in enumerate(stamp_boxes):
        stamp = img[y:y + h, x:x + w]
        gray = cv2.cvtColor(stamp, cv2.COLOR_BGR2GRAY)

        # 1. Texture detection (find smooth/erased areas)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, blurred)
        _, smooth_mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY_INV)

        # 2. Ignore pure white
        smooth_mask[gray > 240] = 0

        # 3. Ignore outer edges of the stamp (15% margin)
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)
        border_mask = np.zeros_like(smooth_mask)
        border_mask[margin_y:h - margin_y, margin_x:w - margin_x] = 255
        smooth_mask = cv2.bitwise_and(smooth_mask, border_mask)

        # 4. Clean up small noise dots
        kernel = np.ones((3, 3), np.uint8)
        smooth_mask = cv2.morphologyEx(smooth_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(smooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < 5000:
                bx, by, bw, bh = cv2.boundingRect(cnt)

                # We only care about the largest anomaly on the page
                if area > max_score:
                    max_score = area

                    # Normalize coordinates relative to the stamp (0.0 to 1.0)
                    rel_x = (bx + bw / 2) / w
                    rel_y = (by + bh / 2) / h

                    best_anomaly = (rel_x, rel_y)

    return best_anomaly


def process_albums():
    album_data = {}

    # Step 1: Process all images and collect points
    for album_name in sorted(os.listdir(ALBUMS_DIR)):
        album_path = os.path.join(ALBUMS_DIR, album_name)
        if not os.path.isdir(album_path): continue

        print(f"\nProcessing {album_name}...")
        points = []
        images = sorted(glob.glob(os.path.join(album_path, "*.jpg")))

        for img_path in images:
            anomaly = detect_tampering_in_page(img_path)
            if anomaly:
                points.append(anomaly)
                print(f"  {os.path.basename(img_path)}: ({anomaly[0]:.2f}, {anomaly[1]:.2f})")
            else:
                print(f"  {os.path.basename(img_path)}: No anomaly found.")

        album_data[album_name] = points

    # Save coordinates to file for reference
    with open("album_coordinates.json", "w") as f:
        json.dump(album_data, f, indent=4)

    # Step 2: Generate Heatmaps
    print("\n--- Generating Heatmaps ---")
    for album_name, points in album_data.items():
        if len(points) < 5:
            print(f"Skipping {album_name}: Not enough points.")
            continue

        print(f"Plotting {album_name}...")
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        plt.figure(figsize=(8, 8))

        # Create a 2D histogram (heatmap) of the points
        # 20x20 bins smooths out individual outliers
        heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=20, range=[[0, 1], [0, 1]])

        # Apply a Gaussian blur to connect nearby clusters
        heatmap = ndimage.gaussian_filter(heatmap, sigma=1.5)

        # Plot the heatmap
        plt.imshow(heatmap.T, extent=[0, 1, 1, 0], origin='upper', cmap='hot', aspect='auto')

        # Overlay the raw points in cyan so we can see where they landed
        plt.scatter(xs, ys, c='cyan', s=20, alpha=0.6, edgecolors='black')

        plt.title(f"Heatmap of Altered Regions - {album_name}")
        plt.colorbar(label='Density of Tampering')

        out_file = os.path.join(OUTPUT_DIR, f"{album_name}_heatmap.png")
        plt.savefig(out_file)
        plt.close()
        print(f"Saved: {out_file}")


if __name__ == "__main__":
    process_albums()