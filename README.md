# Virtual Data Science Internship with Python

Weekly projects from my virtual data science internship, built with Python.
Each week lives in its own folder with its own code, data, figures and README.

**Author:** Sameera P

## Projects

| Week | Topic | Dataset | Folder | Status |
|---|---|---|---|---|
| 1 | Data Acquisition, Cleaning, and Exploratory Data Analysis | Titanic passengers (891 rows) | [week1_project](week1_project/) | Complete |
| 2 | Advanced Data Visualization and Storytelling | Breast Cancer Wisconsin, Diagnostic (569 samples) | [week2_project](week2_project/) | Complete |
| 3 | Statistical Analysis and Hypothesis Testing | Wine, three cultivars (178 samples) | [week3_project](week3_project/) | Complete |
| 4 | Machine Learning Model Development and Evaluation | Handwritten Digits, 8x8 images (1,797 samples) | [week4_project](week4_project/) | Complete |
| 5 | Comprehensive Project Reporting and Strategic Recommendations | Breast Cancer Wisconsin, Diagnostic (569 samples) | [week5_project](week5_project/) | Complete |

## Highlights

### Week 1: Data cleaning and EDA
- Downloaded a public dataset, then cleaned it with Pandas: imputed 177 missing ages, handled a column that was 77% empty, and flagged 116 fare outliers.
- Five visualizations explore survival by sex, class, age and fare.
- Finding: 74.2% of women survived compared with 18.9% of men.

### Week 2: Data storytelling
- Seven charts tell one story for a non-technical audience: what separates benign from malignant samples, and how much can a few simple clues tell us?
- Finding: two measurements place 74% of samples in zones where 99.3% share the same diagnosis, leaving a grey zone of 26% that needs more evidence.

### Week 3: Statistical analysis and hypothesis testing
- Question: can routine lab measurements tell three wine cultivars apart? Five hypotheses were tested with Welch t-tests, Welch ANOVA with post-hoc comparisons, a chi-square test and Pearson correlation.
- Seven figures, assumption checks, and p-values and confidence intervals interpreted in plain language.
- Findings: alcohol is about 1.47 units higher in cultivar A than in B (95% CI 1.29 to 1.64); flavanoids differ across all three cultivars; flavanoids and total phenols are strongly correlated overall (r = 0.86) but only weakly within cultivar C (r = 0.24). One exploratory test (ash, A vs C) was not significant, which is inconclusive rather than proof of equality.

### Week 4: Machine learning model development and evaluation
- Built a classifier that reads 8x8 pixel images and predicts which digit (0 to 9) each shows, and compared four models using a held-out test set.
- Seven figures cover the data overview, model comparison, confusion matrices, ROC curves, overfitting and underfitting checks, and misclassified images.
- Results: logistic regression 97.2% accuracy (10 errors in 360 test images), random forest 96.9%, decision tree 82.5%, against a 10.0% majority-class baseline.

### Week 5: Capstone report and strategic recommendations
- Combined the skills from Weeks 1 to 4 in one end-to-end analysis of the Breast Cancer Wisconsin data: data audit, exploration, hypothesis tests, modelling and decision analysis.
- Nine figures, code excerpts, a 21-page Word report with an executive summary, six recommendations, business impact, limitations and future work.
- Findings: 9 of 10 average measurements differ between benign and malignant samples after correcting for multiple tests. Logistic regression reached a test AUC of 0.996. Lowering the decision cut-off from 0.50 to 0.20 (chosen using training data only) cut missed malignant cases from 3 to 1 in the test set at the price of 3 extra false alarms. The best cut-off depends on how costly a miss is compared with a false alarm.

## Repository structure

```
.
├── README.md              this overview
├── week1_project/         Week 1: cleaning and EDA
├── week2_project/         Week 2: visualization and storytelling
├── week3_project/         Week 3: statistical analysis and hypothesis testing
├── week4_project/         Week 4: machine learning model development
└── week5_project/         Week 5: capstone report and recommendations
```

Each week folder contains the main Python script, a `requirements.txt`, the figures, the results and its Word report.

## Tools

Python, Pandas, NumPy, SciPy, Matplotlib, Seaborn, scikit-learn

## How to run a project

```bash
cd week1_project                 # or week2_project, week3_project, and so on
pip install -r requirements.txt
python <main_script>.py          # the script name is listed in each week's folder
```

Main scripts: `week1_titanic_eda.py`, `week2_storytelling.py`, `week3_hypothesis_testing.py`, `week4_ml_digits.py`, `week5_capstone.py`.

Weeks 2 to 5 use datasets bundled with scikit-learn, so they run without an internet connection. Week 1 downloads the Titanic data when it runs.

## Note

These are learning projects. The Week 2 and Week 5 datasets are medical, and nothing in this repository is medical advice or a medical device.
