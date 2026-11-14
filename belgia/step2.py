import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("potato_fields.csv")   # <- your filename

# Geometry: assume a rectangle
df["area_m2"] = df.area_ha * 10_000
df["short_m"] = df.area_m2 / df.long_side_m
df["perim"]   = 2 * (df.long_side_m + df.short_m)

print("Correlation with edge_m:")
print(df[["perim","long_side_m","area_ha","ditch_m","hedge_m",
          "fence_quote_eur","lat","lon"]].corrwith(df.edge_m).round(4), "\n")

# Model 1: pure geometry
X1 = df[["perim"]]
m1 = LinearRegression().fit(X1, df.edge_m)
print("Perimeter only   R2 =", round(m1.score(X1, df.edge_m), 4))

# Model 2: declared inputs (NO hedge_m, NO fence_quote_eur: leakage)
X2 = df[["perim", "long_side_m", "area_ha", "ditch_m"]]
m2 = LinearRegression().fit(X2, df.edge_m)
print("Declared inputs  R2 =", round(m2.score(X2, df.edge_m), 4))
print(dict(zip(X2.columns, m2.coef_.round(4))), "\n")

df["resid"] = df.edge_m - m2.predict(X2)
print(df.resid.describe(), "\n")

# Plots
fig, ax = plt.subplots(2, 3, figsize=(18, 10))
ax = ax.ravel()
ax[0].scatter(df.perim, df.edge_m, s=4);                 ax[0].set_title("edge_m vs perimeter")
ax[1].scatter(df.lon, df.lat, c=df.resid, s=6, cmap="coolwarm"); ax[1].set_title("map, residual colour")
ax[2].scatter(df.area_ha, df.resid, s=4);                ax[2].set_title("resid vs area")
ax[3].scatter(df.long_side_m, df.resid, s=4);            ax[3].set_title("resid vs long side")
ax[4].scatter(df.hedge_m, df.resid, s=4);                ax[4].set_title("resid vs hedge")
ax[5].scatter(df.area_ha, df.long_side_m, s=4);          ax[5].set_title("long side vs area")
plt.tight_layout()
plt.savefig("step2_plots.png", dpi=150)
plt.show()