# Week 3: Statistical Analysis and Hypothesis Testing in Python

Virtual Data Science with Python internship, Week 3.

**Question:** Can routine lab measurements tell three wine cultivars apart, and which measurements genuinely differ?

**Dataset:** UCI Wine recognition dataset (178 wines, 3 cultivars, 13 chemical measurements). It ships inside scikit-learn, so no download is needed. A copy is saved in `data/wine.csv`.

## Hypotheses (alpha = 0.05, two-sided)

| ID | Hypothesis | Test |
|----|------------|------|
| H1 | Alcohol differs between Cultivar A and B | Welch t-test (+ Mann-Whitney, permutation test, bootstrap CI) |
| H2 | Flavanoids differ across the three cultivars | Welch ANOVA (+ classic ANOVA, Kruskal-Wallis), Games-Howell and Tukey post-hoc |
| H3 | Colour-intensity level (low/high) is associated with cultivar | Chi-square test of independence, Cramer's V |
| H4 | Flavanoids and total phenols are linearly related | Pearson correlation (Fisher-z CI), Spearman check |
| H5 | Ash differs between Cultivar A and C (exploratory control) | Welch t-test |

H1-H4 are corrected together with the Holm method. H5 is exploratory.

## Results in one paragraph

H1-H4 are supported with very large effects (alcohol difference A - B about 1.47 units; flavanoids eta-squared 0.73; colour-level Cramer's V 0.73; flavanoids/phenols r = 0.86, but only 0.24 within Cultivar C). H5 shows no detectable difference in ash, which is not the same as proof of equality. Full details, assumption checks, figures and interpretation are in `Week3_Hypothesis_Testing_Report.docx`.

## How to run

```bash
pip install -r requirements.txt
python week3_hypothesis_testing.py
```

This writes the 7 figures to `figures/` and all numbers to `results/` (`results.json`, `summary.txt`, CSV tables). The random seed is fixed (42), so the results are repeatable.

Optional: rebuild the Word report from the results (needs Node.js and `npm install docx`):

```bash
node build_report.js
```

## Files

```
week3_project/
|-- week3_hypothesis_testing.py            analysis script (all tests and figures)
|-- build_report.js                        builds the Word report from results/results.json
|-- Week3_Hypothesis_Testing_Report.docx   final report
|-- requirements.txt
|-- data/wine.csv
|-- figures/                               7 PNG figures
`-- results/                               results.json, summary.txt, CSV tables
```

## Notes

- The cultivar labels are anonymised in the source data and are called A, B, C here.
- The data are observational, so results describe association, not cause.
- The variables were chosen after looking at summary statistics, so this is not a pre-registered study. The report says so openly.
