import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 0. LOAD DATA  (this is what was missing)
# ============================================================
CSV_PATH = r"C:\Users\ASUS VivoBook\PycharmProjects\PythonProject\france\paris_listings.csv"
df = pd.read_csv(CSV_PATH)
print("Loaded:", df.shape)
assert {"latitude", "longitude"}.issubset(df.columns), "wrong file?"

# ============================================================
# 1. Helper: draw grey background + red subset
# ============================================================
os.makedirs("scan", exist_ok=True)

def plot_subset(sub, title, fname):
    if len(sub) == 0:
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(df["longitude"], df["latitude"], s=1, alpha=0.15, c="grey")
    ax.scatter(sub["longitude"], sub["latitude"], s=3, c="crimson")
    ax.set_title(f"{title}  (n={len(sub)})")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(f"scan/{fname}.png", dpi=120)
    plt.close(fig)

# ============================================================
# 2. Categorical scan
# ============================================================
for col in ["room_type", "host_is_superhost", "property_type",
            "neighbourhood", "bedrooms", "beds", "accommodates"]:
    for val in df[col].dropna().unique():
        sub = df[df[col] == val]
        if 20 < len(sub) < 5000:
            safe = str(val).replace("/", "_").replace(" ", "_")
            plot_subset(sub, f"{col} == {val}", f"{col}_{safe}")

# ============================================================
# 3. Numeric scan: extremes + quantiles + mode
# ============================================================
for col in ["price", "minimum_nights", "availability_365",
            "number_of_reviews", "number_of_reviews_ltm",
            "reviews_per_month", "review_scores_rating",
            "calculated_host_listings_count"]:
    q = df[col].dropna().quantile([0.01, 0.05, 0.95, 0.99])
    plot_subset(df[df[col] <= q[0.01]], f"{col} <= 1% ({q[0.01]:.2f})", f"{col}_low1")
    plot_subset(df[df[col] <= q[0.05]], f"{col} <= 5% ({q[0.05]:.2f})", f"{col}_low5")
    plot_subset(df[df[col] >= q[0.95]], f"{col} >= 95% ({q[0.95]:.2f})", f"{col}_high5")
    plot_subset(df[df[col] >= q[0.99]], f"{col} >= 99% ({q[0.99]:.2f})", f"{col}_high1")
    mode_val = df[col].mode().iloc[0]
    plot_subset(df[df[col] == mode_val], f"{col} == mode ({mode_val})", f"{col}_mode")

print("Done. Open the 'scan' folder and look for letter-shaped red blobs.")
print(f"PNG files written to: {os.path.abspath('scan')}")