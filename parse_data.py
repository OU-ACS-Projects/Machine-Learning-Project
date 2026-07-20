import pandas as pd

pd.set_option("display.max_rows", 130)
pd.set_option("display.width", 120)


df = pd.read_csv('nonvoters_data.csv')
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print()

# ---------------------------------------------------------------------------
# STEP 2: Column inventory and dtypes
# ---------------------------------------------------------------------------
# What we're looking for: which columns pandas is reading as numeric vs.
# text. Survey data is often numeric-coded (e.g., 1-5 scales) so this is
# a sanity check, not a full data dictionary -- that's a separate pass
# against the codebook PDF.
print('Data types')
print(df.dtypes.value_counts())
print()
print("Non-numeric (object) columns:")
print(df.select_dtypes(include="object").columns.tolist())
print()

# balance check
print("TARGET CLASS BALANCE (voter_category)")
counts = df["voter_category"].value_counts()
pct = df["voter_category"].value_counts(normalize=True).round(3) * 100
print(pd.DataFrame({"count": counts, "pct": pct}))
print()

# check missing data columns
print("MISSINGNESS BY COLUMN (top 20)")
missing = df.isna().mean().sort_values(ascending=False) * 100
print(missing.head(20).round(2))
print()
print(f"Columns with >50% missing: {(missing > 50).sum()}")
print(f"Columns with 0% missing:   {(missing == 0).sum()}")
print()

print("DEMOGRAPHICS")
for col in ["educ", "race", "gender", "income_cat"]:
    print(f"--- {col} ---")
    print(df[col].value_counts())
    print()

print("ppage (age) summary stats:")
print(df["ppage"].describe())
print()


print("SURVEY WEIGHT")
print(df["weight"].describe())
print()