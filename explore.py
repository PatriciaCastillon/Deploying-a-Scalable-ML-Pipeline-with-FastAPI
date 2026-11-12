import pandas as pd

df = pd.read_csv("data/census.csv")

# Strip leading spaces from column names
df.columns = df.columns.str.strip()

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

print("\nInfo:")
df.info()

print("\nNumeric summary:")
print(df.describe())

print("\nCategorical summary:")
print(df.describe(include="object"))

print("\nMissing values (NaN):")
print(df.isna().sum())

print("\nMissing values encoded as '?':")
print(df.apply(lambda col: col.astype(str).str.strip() == "?").sum())

print("\nSalary balance:")
print(df["salary"].value_counts(normalize=True))

print("\nDuplicate rows:", df.duplicated().sum())

print("\nAverages by salary:")
print(df.groupby("salary")[["age", "hours-per-week", "education-num"]].mean())
