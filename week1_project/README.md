# Week 1: Data Acquisition, Cleaning and Exploratory Data Analysis

Virtual Data Science internship (Python) - Week 1 task.

**Dataset:** Titanic passenger data (public CSV on GitHub, 891 rows).
**Goal:** acquire a public dataset, clean it with Pandas, explore it with summary
statistics and visualizations, and document the process.

## Project structure

```
week1_titanic_eda.py     main script (acquire -> clean -> explore -> plot)
requirements.txt         Python dependencies
data/raw/                untouched downloaded data
data/processed/          cleaned dataset
figures/                 5 saved visualizations
reports/eda_summary.txt  numbers printed by the script (quote these in the report)
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python week1_titanic_eda.py
```

If the download fails, save the CSV manually as `data/raw/titanic_raw.csv` and run again.

## Cleaning decisions (short version)

| Issue | Action | Why |
|---|---|---|
| Age missing | Median by Pclass and Sex | Age differs by class and sex; keeps rows |
| Cabin mostly missing | Dropped, kept `HasCabin` flag | Too sparse to impute reliably |
| Embarked missing (2 rows) | Mode | Negligible effect |
| Name | Extracted `Title`, then dropped | Title is informative, raw name is not |
| Fare skew and outliers | Flagged (IQR), added `LogFare`, rows kept | Fares are real, not errors |
| Data types | Categorical for Sex, Embarked, Title, Pclass | Correct types, less memory |

## Visualizations

1. Missing-values heatmap (raw data)
2. Age distribution by survival
3. Survival rate by class and sex
4. Fare by class (box plot, log scale)
5. Correlation heatmap
