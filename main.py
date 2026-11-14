import pandas as pd

# Load the two files
guide = pd.read_csv("office_price_guide.csv")
buys  = pd.read_csv("purchases.csv")

for df in (guide, buys):
    df["Product"] = df["Product"].str.strip().str.title()
# Merge the guide onto the purchases (one guide row per receipt)
df = buys.merge(guide, on="Product", how="left")

# Sanity checks
print("guide shape:    ", guide.shape)    # expect (6, 6)
print("purchases shape:", buys.shape)     # expect (577, 14)
print("merged shape:   ", df.shape)       # expect 577 rows, 14 + 5 = 19 columns
print("unmatched rows: ", df["List_no"].isna().sum())  # expect 0
print(df.head())

# Step 2: price per unit
df["Unit_price"] = df["Total_paid_eur"] / df["Quantity"]

# Sanity checks
print(df[["Receipt_id", "Product", "Quantity", "Total_paid_eur", "Unit", "Unit_price"]].head(10))
print("Missing unit prices:", df["Unit_price"].isna().sum())   # expect 0
print("Zero or negative quantities:", (df["Quantity"] <= 0).sum())  # expect 0

# Step 3: keep only receipts whose unit price is inside the fair range (inclusive)
fair = df[(df["Unit_price"] >= df["Fair_min_eur"]) &
          (df["Unit_price"] <= df["Fair_max_eur"])]

print("Total receipts:", len(df))
print("Fair receipts: ", len(fair))
print(fair["Product"].value_counts())   # expect all 6 products to appear
import matplotlib.pyplot as plt

# Step 4: optional home-city filter
fair_home = fair[fair["City"] == fair["Home_city"]]
print("Fair receipts:", len(fair), "| Fair + home city:", len(fair_home))

# Step 5: pivot table (rows = List_no 1..6, columns = Pilot_no 1..21)
def make_board(data):
    board = (data.groupby(["List_no", "Pilot_no"]).size()
                 .unstack(fill_value=0)
                 .reindex(index=range(1, 7), columns=range(1, 22), fill_value=0))
    return (board > 0).astype(int)

board_all  = make_board(fair)
board_home = make_board(fair_home)
print(board_home)

# Step 6: plot both versions
labels = guide.sort_values("List_no")["Product"].tolist()

fig, axes = plt.subplots(2, 1, figsize=(12, 6))
for ax, board, title in zip(axes,
                            [board_all, board_home],
                            ["Fair price only", "Fair price + home city"]):
    ax.imshow(board.values, cmap="gray_r", aspect="equal")
    ax.set_xticks(range(21)); ax.set_xticklabels(range(1, 22))
    ax.set_yticks(range(6));  ax.set_yticklabels(labels)
    ax.set_title(title)
plt.tight_layout()
plt.savefig("board.png", dpi=200)
plt.show()