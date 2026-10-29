# Week 2: Advanced Data Visualization and Storytelling

Virtual Data Science internship (Python) - Week 2 task.

**Dataset:** Breast Cancer Wisconsin (Diagnostic): 569 breast tissue samples, 30 measurements of
cell nuclei each, labelled benign or malignant. Public UCI dataset that ships with scikit-learn,
so no download is needed.

**Story:** *What separates benign from malignant samples, and how much can a few simple clues tell us?*
Seven charts walk a non-technical reader from the question to the answer.

## Project structure

```
week2_storytelling.py        main script (load -> check -> 7 figures -> key numbers)
requirements.txt             Python dependencies
data/raw/                    dataset as loaded
data/processed/              dataset with plain-English labels and risk bands
figures/                     the 7 visualizations (PNG)
reports/key_numbers.txt      every number quoted in the report
Week2_Data_Storytelling_Report.docx   final report
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python week2_storytelling.py
```

## The seven visualizations

| # | Chart | Role in the story |
|---|---|---|
| 1 | Stacked bar | Who is in the data? |
| 2 | Violin plots | How do the groups differ? |
| 3 | Ranked bar chart | Which measurements are the best clues? |
| 4 | Correlation heatmap | Which measurements repeat each other? |
| 5 | Annotated scatter with zones | How well do two clues sort the samples? |
| 6 | Risk ladder (bar chart) | How does risk rise with cell size? |
| 7 | PCA scatter with ellipses | The big picture of all 30 measurements |

## Important note

This is a learning project. The thresholds in figure 5 were chosen by inspecting this same dataset
and are **not** a validated diagnostic rule. Nothing here is medical advice.
