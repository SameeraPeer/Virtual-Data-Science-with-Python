"""
Week 1 - Data Acquisition, Cleaning, and Exploratory Data Analysis (EDA)
Dataset: Titanic passenger data (public, hosted on GitHub)

Run:  python week1_titanic_eda.py
Outputs:
    data/raw/titanic_raw.csv            untouched copy of the downloaded data
    data/processed/titanic_clean.csv    cleaned dataset
    figures/*.png                       all visualizations
    reports/eda_summary.txt             numbers to quote in your write-up
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
RAW_PATH = "data/raw/titanic_raw.csv"
CLEAN_PATH = "data/processed/titanic_clean.csv"
FIG_DIR = "figures"
REPORT_PATH = "reports/eda_summary.txt"

for folder in ("data/raw", "data/processed", FIG_DIR, "reports"):
    os.makedirs(folder, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
report = []  # lines collected for reports/eda_summary.txt


def log(text=""):
    print(text)
    report.append(str(text))


# ------------------------------------------------------------------
# 1. DATA ACQUISITION
# ------------------------------------------------------------------
try:
    df_raw = pd.read_csv(URL)
    df_raw.to_csv(RAW_PATH, index=False)
    log(f"[acquire] Downloaded data from {URL}")
except Exception as err:
    if os.path.exists(RAW_PATH):
        log(f"[acquire] Download failed ({err}); using local copy {RAW_PATH}")
        df_raw = pd.read_csv(RAW_PATH)
    else:
        sys.exit(
            f"Could not download the dataset ({err}).\n"
            f"Download titanic.csv manually and save it as {RAW_PATH}, then re-run."
        )

df = df_raw.copy()

# ------------------------------------------------------------------
# 2. INITIAL INSPECTION
# ------------------------------------------------------------------
log("\n=== INITIAL INSPECTION ===")
log(f"Shape (rows, columns): {df.shape}")
log("\nColumn data types:")
log(df.dtypes.to_string())
log("\nMissing values per column (count and %):")
missing = pd.DataFrame({
    "missing": df.isna().sum(),
    "percent": (df.isna().mean() * 100).round(1),
})
log(missing.to_string())
log(f"\nFully duplicated rows: {df.duplicated().sum()}")
log(f"Duplicated PassengerId values: {df['PassengerId'].duplicated().sum()}")

# Visualization 1: missing values heatmap (BEFORE cleaning)
plt.figure(figsize=(10, 5))
sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap="viridis")
plt.title("Figure 1: Missing values in the raw data (yellow = missing)")
plt.xlabel("Column")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig1_missing_values.png", dpi=200)
plt.close()

# ------------------------------------------------------------------
# 3. DATA CLEANING
# ------------------------------------------------------------------
log("\n=== CLEANING ===")

# 3a. Duplicates
before = len(df)
df = df.drop_duplicates()
df = df.drop_duplicates(subset="PassengerId")  # PassengerId should be unique
log(f"[duplicates] Removed {before - len(df)} duplicate rows.")

# 3b. Feature engineering from Name BEFORE dropping it: extract the title
def extract_title(name):
    match = re.search(r",\s*([^\.]+)\.", str(name))
    return match.group(1).strip() if match else "Unknown"

df["Title"] = df["Name"].apply(extract_title)
rare_titles = df["Title"].value_counts()[lambda s: s < 10].index
df["Title"] = df["Title"].replace(rare_titles, "Rare")

# 3c. Cabin: too many missing values to impute, keep only a "has cabin" flag
cabin_missing_pct = df["Cabin"].isna().mean() * 100
df["HasCabin"] = df["Cabin"].notna().astype(int)
df = df.drop(columns=["Cabin"])
log(f"[cabin] {cabin_missing_pct:.1f}% missing -> dropped, kept HasCabin flag.")

# 3d. Age: impute with the median of passengers in the same Pclass and Sex
age_missing = df["Age"].isna().sum()
df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(lambda s: s.fillna(s.median()))
df["Age"] = df["Age"].fillna(df["Age"].median())  # safety net
log(f"[age] Imputed {age_missing} missing ages using the Pclass x Sex median.")

# 3e. Embarked: only a couple missing -> fill with the most common port
emb_missing = df["Embarked"].isna().sum()
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
log(f"[embarked] Filled {emb_missing} missing values with the mode.")

# 3f. Correct data types
for col in ["Sex", "Embarked", "Title"]:
    df[col] = df[col].astype("category")
df["Pclass"] = pd.Categorical(df["Pclass"], categories=[1, 2, 3], ordered=True)
df["Survived"] = df["Survived"].astype(int)
df["Age"] = df["Age"].round(1)
log("[dtypes] Sex, Embarked, Title -> category; Pclass -> ordered category.")

# 3g. Drop identifier/text columns that carry no analytical value now
df = df.drop(columns=["Name", "Ticket"])

# 3h. Outliers in Fare: flag with the IQR rule, keep the rows (they are real fares)
q1, q3 = df["Fare"].quantile([0.25, 0.75])
upper = q3 + 1.5 * (q3 - q1)
df["FareOutlier"] = (df["Fare"] > upper).astype(int)
df["LogFare"] = np.log1p(df["Fare"])
log(f"[fare] {df['FareOutlier'].sum()} fares above the IQR limit ({upper:.2f}); "
    "flagged and log-transformed, not removed.")

# 3i. Feature: family size
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

assert df.isna().sum().sum() == 0, "There are still missing values!"
df.to_csv(CLEAN_PATH, index=False)
log(f"\nCleaned data saved to {CLEAN_PATH}  shape={df.shape}")

# ------------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------------
log("\n=== SUMMARY STATISTICS ===")
log(df.describe().round(2).T.to_string())

overall = df["Survived"].mean() * 100
log(f"\nOverall survival rate: {overall:.1f}%")
log("\nSurvival rate (%) by Sex:")
log((df.groupby("Sex", observed=True)["Survived"].mean() * 100).round(1).to_string())
log("\nSurvival rate (%) by Pclass:")
log((df.groupby("Pclass", observed=True)["Survived"].mean() * 100).round(1).to_string())
log("\nSurvival rate (%) by Sex and Pclass:")
log((df.groupby(["Sex", "Pclass"], observed=True)["Survived"].mean() * 100).round(1).to_string())
log("\nMedian fare by Pclass:")
log(df.groupby("Pclass", observed=True)["Fare"].median().round(2).to_string())

# Visualization 2: age distribution by survival
plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="Age", hue="Survived", bins=30, kde=True, multiple="layer")
plt.title("Figure 2: Age distribution by survival (0 = died, 1 = survived)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig2_age_by_survival.png", dpi=200)
plt.close()

# Visualization 3: survival rate by sex and class
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="Pclass", y="Survived", hue="Sex", errorbar=None)
plt.title("Figure 3: Survival rate by passenger class and sex")
plt.ylabel("Survival rate")
plt.xlabel("Passenger class")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig3_survival_by_class_sex.png", dpi=200)
plt.close()

# Visualization 4: fare by class (box plot, log scale because of skew)
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Pclass", y="Fare")
plt.yscale("log")
plt.title("Figure 4: Fare by passenger class (log scale)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig4_fare_boxplot.png", dpi=200)
plt.close()

# Visualization 5: correlation heatmap of numeric columns
numeric = df[["Survived", "Age", "SibSp", "Parch", "Fare", "LogFare",
              "HasCabin", "FamilySize"]].copy()
numeric["Pclass"] = df["Pclass"].astype(int)
corr = numeric.corr()
plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Figure 5: Correlation matrix of numeric features")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig5_correlation_heatmap.png", dpi=200)
plt.close()

log("\nCorrelation of each feature with Survived:")
log(corr["Survived"].drop("Survived").sort_values().round(2).to_string())

with open(REPORT_PATH, "w") as fh:
    fh.write("\n".join(report))
log(f"\nDone. Figures in '{FIG_DIR}/', summary in '{REPORT_PATH}'.")
