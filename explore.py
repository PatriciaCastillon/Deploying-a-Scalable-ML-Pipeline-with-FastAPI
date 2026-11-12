import pandas as pd

df = pd.read_csv("data/census.csv")

# Strip leading spaces from column names
df.columns = df.columns.str.strip()

# Shape gives (rows, columns), so you know how big the dataset is.
print("Shape:", df.shape)
# Column list confirms the names are clean after stripping.
print("\nColumns:", df.columns.tolist())

# Info shows each column's data type and non-null count. It prints on its
# own, so it is not wrapped in print().
print("\nInfo:")
df.info()

# Summary statistics (mean, std, min, max, quartiles) for numeric columns.
# Useful for spotting outliers and skew, such as in capital-gain.
print("\nNumeric summary:")
print(df.describe())

# For text columns: number of unique values, the most common value, and
# how often it appears.
print("\nCategorical summary:")
print(df.describe(include="object"))

# Counts true NaN values per column. Expect zeros, because this dataset
# does not use NaN for missing data.
print("\nMissing values (NaN):")
print(df.isna().sum())

# Missing data here is written as "?". This converts each value to text,
# strips spaces, and counts the "?" entries per column. Expect them in
# workclass, occupation, and native-country.
print("\nMissing values encoded as '?':")
print(df.apply(lambda col: col.astype(str).str.strip() == "?").sum())

# Class balance of the target. normalize=True shows proportions instead
# of counts. About 76% are <=50K, so the classes are imbalanced.
print("\nSalary balance:")
print(df["salary"].value_counts(normalize=True))

# Number of fully identical rows. Duplicates can bias training and make
# test results look better than they are.
print("\nDuplicate rows:", df.duplicated().sum())

# Average age, hours worked, and education level for each salary group.
# A quick check of which features differ most between the two classes.
print("\nAverages by salary:")
print(
    df.groupby("salary")[["age", "hours-per-week", "education-num"]].mean())
