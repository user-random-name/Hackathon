import pandas as pd

df = pd.read_csv("potato_fields.csv")   # <- your dataset filename

# 1. Shape and columns
print("Rows, columns:", df.shape)
print(df.dtypes, "\n")

# 2. First rows
print(df.head(), "\n")

# 3. Missing values
print("Missing values per column:")
print(df.isna().sum(), "\n")

# 4. Summary statistics for the numeric columns
print(df.describe().T, "\n")

# 5. Categorical columns
for col in ["crop_code", "crop", "farm_type"]:
    print(df[col].value_counts(), "\n")

# 6. Duplicate IDs
print("Duplicate field_id:", df.field_id.duplicated().sum())

# 7. Convert area to m² for later geometry
df["area_m2"] = df.area_ha * 10_000