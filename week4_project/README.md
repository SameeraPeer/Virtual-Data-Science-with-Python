# Week 4: Machine Learning Model Development and Evaluation

Handwritten digit recognition (scikit-learn `load_digits`, 1,797 images of 8x8 pixels) with logistic regression,
decision trees and a random forest.

## Run
```
pip install -r requirements.txt
python week4_ml_digits.py
```
Creates `figures/` (7 PNG charts) and `reports/` (`results.json`, `results_summary.txt`). Uses a fixed seed (42).

## Contents
- `week4_ml_digits.py`: full pipeline (checks, split, tuning, evaluation, figures)
- `Week4_ML_Model_Report.docx`: the written report
- `build_report.js`: script that generates the report from `reports/results.json` (needs Node and the `docx` package; optional)
- `figures/`, `reports/`: outputs
