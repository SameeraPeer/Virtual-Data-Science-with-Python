# Virtual Data Science Internship with Python

Weekly projects from my virtual data science internship, built with Python.
Each week lives in its own folder with its own code, data, figures and README.

**Author:** Sameera P

## Projects

| Week | Topic | Dataset | Folder | Status |
|---|---|---|---|---|
| 1 | Data Acquisition, Cleaning, and Exploratory Data Analysis | Titanic passengers (891 rows) | [week1_project](week1_project/) | Complete |
| 2 | Advanced Data Visualization and Storytelling | Breast Cancer Wisconsin, Diagnostic (569 samples) | [week2_project](week2_project/) | Complete |
| 3 | To be added | | week3_project | Coming soon |
| 4 | To be added | | week4_project | Coming soon |
| 5 | To be added | | week5_project | Coming soon |

## Highlights

### Week 1: Data cleaning and EDA
- Downloaded a public dataset, then cleaned it with Pandas: imputed 177 missing ages, handled a column that was 77% empty, and flagged 116 fare outliers.
- Five visualizations explore survival by sex, class, age and fare.
- Finding: 74.2% of women survived compared with 18.9% of men.

### Week 2: Data storytelling
- Seven charts tell one story for a non-technical audience: what separates benign from malignant samples, and how much can a few simple clues tell us?
- Finding: two measurements place 74% of samples in zones where 99.3% share the same diagnosis, leaving a grey zone of 26% that needs more evidence.

## Repository structure

```
.
├── README.md              this overview
├── week1_project/         Week 1: cleaning and EDA
├── week2_project/         Week 2: visualization and storytelling
└── ...                    later weeks are added as they are completed
```

Each week folder contains the main Python script, a `requirements.txt`, the data, the figures and its own README.

## Tools

Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn

## How to run a project

```bash
cd week1_project                 # or week2_project, and so on
pip install -r requirements.txt
python <main_script>.py          # the script name is listed in each week's README
```

## Note

These are learning projects. The Week 2 dataset is medical, and nothing in this repository is medical advice.
