import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("potato_fields.csv")   # <- your filename
df["short_m"] = df.area_ha * 10000 / df.long_side_m
df["perim"] = 2 * (df.long_side_m + df.short_m)

X = df[["perim", "ditch_m"]]
m = LinearRegression().fit(X, df.edge_m)
df["resid"] = df.edge_m - m.predict(X)
df["idx"] = np.arange(len(df))
print(m.coef_, m.intercept_)

fig, ax = plt.subplots(3, 3, figsize=(18, 15))
ax = ax.ravel()
ax[0].hist(df.hedge_m, bins=80);                         ax[0].set_title("hist hedge_m")
ax[1].hist(df.resid, bins=80);                           ax[1].set_title("hist resid")
ax[2].scatter(df.hedge_m, df.resid, s=4);                ax[2].set_title("resid vs hedge")
ax[3].scatter(df.edge_m, df.hedge_m, s=4);               ax[3].set_title("hedge vs edge")
ax[4].scatter(df.edge_m, df.resid, s=4);                 ax[4].set_title("resid vs edge")
ax[5].scatter(df.idx, df.resid, s=4);                    ax[5].set_title("resid vs row order")
ax[6].scatter(df.idx, df.hedge_m, s=4);                  ax[6].set_title("hedge vs row order")
ax[7].scatter(df.ditch_m, df.hedge_m, s=4);              ax[7].set_title("hedge vs ditch")
ax[8].scatter(df.perim, df.hedge_m, s=4);                ax[8].set_title("hedge vs perimeter")
plt.tight_layout(); plt.savefig("step3_plots.png", dpi=150); plt.show()

# Fit using hedge_m too, and look at what's left
X3 = df[["perim", "ditch_m", "hedge_m"]]
m3 = LinearRegression().fit(X3, df.edge_m)
print("R2 with hedge:", m3.score(X3, df.edge_m), dict(zip(X3.columns, m3.coef_)))
df["resid3"] = df.edge_m - m3.predict(X3)
print(df.resid3.describe())

# Does the sample look "organised"? e.g. rounding or gaps
print(df.hedge_m.round(1).value_counts().head(15))
print(df[["area_ha","long_side_m","ditch_m","hedge_m","edge_m"]].nunique())