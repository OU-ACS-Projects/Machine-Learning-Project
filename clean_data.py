"""
This acts on what parse_data.py found

Output: data/processed/nonvoters_clean.csv, ready for feature
encoding / train-test split
"""

import pandas as pd
import numpy as np

pd.set_option("display.max_rows", 130)
pd.set_option("display.width", 120)

df = pd.read_csv('nonvoters_data.csv')
print(f"Loaded: {df.shape}")

# ---------------------------------------------------------------------------
# 1: Recode -1, but NOT the same way everywhere
# ---------------------------------------------------------------------------
# Finding from nonvoters_codebook.pdf: a manual check of per-column value counts showed -1 does NOT uniformly mean "refused."
# Two distinct column types are classified under the same numeral:
#
#   (a) "Select all that apply" battery items -- every column takes ONLY
#       {-1, 1} with no 2/3/4 present anywhere. Here -1 means "respondent
#       did NOT check this box," a real, informative negative response,
#       not a missing one. Recoding it to NaN (and later median-imputing)
#       would erase the majority response and manufacture fake "selected"
#       answers that were never given.
#   (b) True ordinal/Likert scale items (e.g. 1-4, 1-5) where -1 sits
#       outside the valid range in small counts -- consistent with an
#       actual refused/skipped code. NaN is the correct recode here.
#
# We detect type (a) programmatically: any Q-column whose full set of
# non-null values is exactly {-1, 1} is a select/not-select flag. For
# those we recode -1 -> 0 (not selected) so the column becomes a clean
# binary indicator (0/1) instead of losing information as NaN. Every
# other Q-column gets the original -1 -> NaN treatment.
Q_COLS = [c for c in df.columns if c.startswith("Q")]

binary_select_cols = [
    c for c in Q_COLS
    if set(df[c].dropna().unique()) == {-1, 1}
]
scale_cols = [c for c in Q_COLS if c not in binary_select_cols]

df[binary_select_cols] = df[binary_select_cols].replace(-1, 0)

before_neg1 = (df[scale_cols] == -1).sum().sum()
df[scale_cols] = df[scale_cols].replace(-1, np.nan)

print(f"\nSTEP 1: {len(binary_select_cols)} select/not-select columns recoded "
      f"-1 -> 0 (not selected), 1 stays 1 (selected):")
print(binary_select_cols)
print(f"STEP 1: Recoded {before_neg1} instances of -1 to NaN across the "
      f"remaining {len(scale_cols)} true scale columns.")

# ---------------------------------------------------------------------------
# STEP 2: Drop columns with high missingness
# ---------------------------------------------------------------------------
# Finding: parse_data.py showed 14 columns >50% missing
MISSING_THRESHOLD = 0.50
missing_pct = df.isna().mean()
cols_to_drop = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
print(f"\nSTEP 2: Dropping {len(cols_to_drop)} columns >{MISSING_THRESHOLD:.0%} missing (post-recode):")
print(cols_to_drop)
df = df.drop(columns=cols_to_drop)
print(f"Shape after drop: {df.shape}")

# ---------------------------------------------------------------------------
# STEP 3: Impute remaining missingness in survey items
# ---------------------------------------------------------------------------
# Finding: after dropping the worst columns, remaining Q-columns still have
# some missingness. Fixing fractional responses that don't make sense
remaining_q_cols = [c for c in Q_COLS if c in df.columns]
missing_before_impute = df[remaining_q_cols].isna().sum().sum()
for col in remaining_q_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())
print(f"\nSTEP 3: Median-imputed {missing_before_impute} remaining missing values across {len(remaining_q_cols)} survey columns.")

# ---------------------------------------------------------------------------
# STEP 4: Encode demographic categorical fields
# ---------------------------------------------------------------------------
# Finding: parse_data.py showed educ, race, gender, income_cat all
# have healthy category sizes (no sparse categories needing collapse).
# educ and income_cat have a natural order and ordinal encoding preserves
# that signal. race and gender have no natural order. one-hot encoding
# avoids implying a false ranking between categories.
educ_order = {"High school or less": 0, "Some college": 1, "College": 2}
income_order = {"Less than $40k": 0, "$40-75k": 1, "$75-125k": 2, "$125k or more": 3}

df["educ_ord"] = df["educ"].map(educ_order)
df["income_ord"] = df["income_cat"].map(income_order)

df = pd.get_dummies(df, columns=["race", "gender"], prefix=["race", "gender"])

print(f"\nSTEP 4: Encoded educ/income_cat as ordinal, race/gender as one-hot.")
print(f"Shape after encoding: {df.shape}")

# ---------------------------------------------------------------------------
# STEP 5: Build composite attitudinal scores
# ---------------------------------------------------------------------------
# Finding: charter's feature engineering plan calls for a composite
# political-trust score built from the Q18_x block (trust in institutions)
# instead of feeding ~10 near-duplicate columns into the model separately.
# Averaging keeps the score on the same 1-2-ish scale as the
# source items, which keeps it interpretable later.
q18_cols = [c for c in df.columns if c.startswith("Q18_")]
if q18_cols:
    df["trust_composite"] = df[q18_cols].mean(axis=1)
    print(f"\nSTEP 5: Built trust_composite from {len(q18_cols)} Q18_x columns.")
else:
    print("\nSTEP 5: Q18_x block was fully dropped in Step 2 (high missingness) -- "
          "no trust_composite built. Flag this for the team: the planned "
          "feature depends on a column group that didn't survive cleaning.")

# ---------------------------------------------------------------------------
# STEP 6: Save
# ---------------------------------------------------------------------------
# Confirm the cleaning process then save.

print("\nSTEP 6: Final class balance:")
print(df["voter_category"].value_counts(normalize=True).round(3) * 100)

df.to_csv("nonvoters_clean.csv", index=False)
print(f"\nSaved cleaned dataset: nonvoters_clean.csv, shape {df.shape}")
