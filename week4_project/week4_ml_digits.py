"""
Week 4 - Machine Learning Model Development and Evaluation
Dataset : Optical Recognition of Handwritten Digits (bundled with scikit-learn)
Task    : classify 8x8 pixel images into the digits 0-9
Run     : python week4_ml_digits.py
Outputs : figures/*.png, reports/results.json, reports/results_summary.txt
"""
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, auc)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, learning_curve,
                                     train_test_split, validation_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
REP = os.path.join(HERE, "reports")
os.makedirs(FIG, exist_ok=True)
os.makedirs(REP, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
R = {}  # everything we report is stored here and written to results.json


# ---------------------------------------------------------------- 1. LOAD
digits = load_digits()
X = pd.DataFrame(digits.data, columns=[f"pix_{i}" for i in range(64)])
y = pd.Series(digits.target, name="digit")
R["n_samples"], R["n_features"] = int(X.shape[0]), int(X.shape[1])
R["n_classes"] = int(y.nunique())


# ------------------------------------------------- 2. DATA PREPARATION CHECKS
R["missing_values"] = int(X.isna().sum().sum())
R["duplicate_rows"] = int(X.duplicated().sum())
R["pixel_min"], R["pixel_max"] = float(X.values.min()), float(X.values.max())
R["dtype"] = str(X.dtypes.iloc[0])
const_cols = [c for c in X.columns if X[c].std() == 0]
R["constant_pixels"] = const_cols
R["class_counts"] = {int(k): int(v) for k, v in y.value_counts().sort_index().items()}
R["class_min"], R["class_max"] = min(R["class_counts"].values()), max(R["class_counts"].values())

# Stratified split: the test set is touched only for final evaluation.
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)
R["n_train"], R["n_test"] = int(len(X_tr)), int(len(X_te))

# Exact duplicates that straddle the split would inflate test accuracy.
tr_rows = set(map(tuple, X_tr.values))
R["test_rows_duplicated_in_train"] = int(sum(tuple(r) in tr_rows for r in X_te.values))

# Figure 1: data overview
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [2, 1, 1.1]})
tile = np.vstack([np.hstack([digits.images[np.where(y == d)[0][j]] for d in range(10)])
                  for j in range(5)])
axes[0].imshow(tile, cmap="gray_r")
axes[0].set_title("Five example images for each digit (columns 0 to 9)")
axes[0].set_xticks(np.arange(10) * 8 + 4)
axes[0].set_xticklabels(range(10))
axes[0].set_yticks([])
axes[0].grid(False)
cc = pd.Series(R["class_counts"])
sns.barplot(x=cc.index, y=cc.values, ax=axes[1], color="#4C78A8")
axes[1].set_ylim(150, 200)
axes[1].set_title("Images per digit (axis starts at 150)")
axes[1].set_xlabel("Digit")
axes[1].set_ylabel("Count")
mean_img = X.mean().values.reshape(8, 8)
im = axes[2].imshow(mean_img, cmap="viridis")
axes[2].set_title("Average pixel intensity (0 to 16)")
axes[2].grid(False)
plt.colorbar(im, ax=axes[2], fraction=0.046)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig1_data_overview.png"), dpi=150)
plt.close()


# ----------------------------------------------------------- 3. MODEL SETUP
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def lr_pipe(C=1.0):
    return Pipeline([("drop_constant", VarianceThreshold(0.0)),
                     ("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=3000))])


gs_lr = GridSearchCV(lr_pipe(), {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100]},
                     cv=cv, scoring="accuracy").fit(X_tr, y_tr)
gs_tree = GridSearchCV(DecisionTreeClassifier(random_state=SEED),
                       {"max_depth": [2, 3, 4, 5, 6, 8, 10, 12, 15, None],
                        "min_samples_leaf": [1, 5]},
                       cv=cv, scoring="accuracy").fit(X_tr, y_tr)
R["best_C"] = float(gs_lr.best_params_["clf__C"])
R["best_tree_params"] = {k: (None if v is None else int(v)) for k, v in gs_tree.best_params_.items()}

models = {
    "Majority-class baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic regression (tuned)": gs_lr.best_estimator_,
    "Decision tree (unpruned)": DecisionTreeClassifier(random_state=SEED),
    "Decision tree (tuned)": gs_tree.best_estimator_,
    "Random forest (300 trees)": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
}

rows, fitted = [], {}
for name, m in models.items():
    cvs = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="accuracy")
    m.fit(X_tr, y_tr)
    fitted[name] = m
    p_tr, p_te = m.predict(X_tr), m.predict(X_te)
    row = {"model": name,
           "train_acc": accuracy_score(y_tr, p_tr),
           "cv_acc": cvs.mean(), "cv_sd": cvs.std(),
           "test_acc": accuracy_score(y_te, p_te),
           "macro_precision": precision_score(y_te, p_te, average="macro", zero_division=0),
           "macro_recall": recall_score(y_te, p_te, average="macro", zero_division=0),
           "macro_f1": f1_score(y_te, p_te, average="macro", zero_division=0)}
    if name != "Majority-class baseline":
        row["macro_auc"] = roc_auc_score(y_te, m.predict_proba(X_te), multi_class="ovr", average="macro")
    else:
        row["macro_auc"] = 0.5
    rows.append(row)
res = pd.DataFrame(rows)
R["model_table"] = res.round(4).to_dict(orient="records")
R["gs_lr_cv_scores"] = {str(c): round(float(s), 4) for c, s in
                        zip(gs_lr.cv_results_["param_clf__C"], gs_lr.cv_results_["mean_test_score"])}

# Figure 2: model comparison
pl = res[res.model != "Majority-class baseline"].melt(
    id_vars="model", value_vars=["train_acc", "cv_acc", "test_acc"],
    var_name="Data", value_name="Accuracy")
pl["Data"] = pl["Data"].map({"train_acc": "Training set", "cv_acc": "5-fold CV (training data)",
                             "test_acc": "Held-out test set"})
fig, ax = plt.subplots(figsize=(11, 5))
sns.barplot(data=pl, y="model", x="Accuracy", hue="Data", ax=ax,
            palette=["#9ecae1", "#4C78A8", "#F58518"])
ax.set_xlim(0.7, 1.005)
ax.set_ylabel("")
ax.set_title("Accuracy by model: training vs cross-validation vs test (axis starts at 0.70)")
for c in ax.containers:
    ax.bar_label(c, fmt="%.3f", fontsize=8, padding=2)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig2_model_comparison.png"), dpi=150)
plt.close()


# -------------------------------------------- 4. PRIMARY MODEL: EVALUATION
lr = fitted["Logistic regression (tuned)"]
tree = fitted["Decision tree (tuned)"]
rf = fitted["Random forest (300 trees)"]
p_lr = lr.predict(X_te)
cm_lr = confusion_matrix(y_te, p_lr)
cm_tree = confusion_matrix(y_te, tree.predict(X_te))
R["lr_errors"], R["tree_errors"] = int((p_lr != y_te).sum()), int((tree.predict(X_te) != y_te).sum())
R["rf_errors"] = int((rf.predict(X_te) != y_te).sum())

rep = classification_report(y_te, p_lr, output_dict=True, zero_division=0)
R["lr_per_class"] = {d: {k: round(rep[str(d)][k], 4) for k in ["precision", "recall", "f1-score", "support"]}
                     for d in range(10)}

pairs = [(int(i), int(j), int(cm_lr[i, j])) for i in range(10) for j in range(10)
         if i != j and cm_lr[i, j] > 0]
pairs.sort(key=lambda t: -t[2])
R["lr_top_confusions"] = [{"true": a, "pred": b, "count": c} for a, b, c in pairs[:6]]

# Bootstrap interval for test accuracy (uncertainty from a finite test set)
rng = np.random.default_rng(SEED)
correct = (p_lr == y_te.values).astype(int)
boots = [rng.choice(correct, len(correct)).mean() for _ in range(5000)]
R["lr_acc_ci95"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

# Figure 3: confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
for ax, cm, title in [(axes[0], cm_lr, f"Logistic regression ({R['lr_errors']} errors of {R['n_test']})"),
                      (axes[1], cm_tree, f"Decision tree, tuned ({R['tree_errors']} errors of {R['n_test']})")]:
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig3_confusion_matrices.png"), dpi=150)
plt.close()

# Figure 4: ROC curves (one-vs-rest)
Y_bin = label_binarize(y_te, classes=range(10))
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
P_lr = lr.predict_proba(X_te)
aucs = {}
cmap = plt.get_cmap("tab10")
for d in range(10):
    fpr, tpr, _ = roc_curve(Y_bin[:, d], P_lr[:, d])
    aucs[d] = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color=cmap(d), lw=1.4, label=f"digit {d} (AUC {aucs[d]:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set_title("Logistic regression: one-vs-rest ROC, each digit")
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].legend(fontsize=7, loc="lower right")
axes[0].set_xlim(-0.01, 0.3)
axes[0].set_ylim(0.7, 1.01)
R["lr_auc_per_class"] = {d: round(a, 4) for d, a in aucs.items()}
for name, col in [("Logistic regression (tuned)", "#4C78A8"), ("Decision tree (tuned)", "#E45756"),
                  ("Random forest (300 trees)", "#54A24B")]:
    P = fitted[name].predict_proba(X_te)
    fpr, tpr, _ = roc_curve(Y_bin.ravel(), P.ravel())
    axes[1].plot(fpr, tpr, color=col, lw=2, label=f"{name} (micro AUC {auc(fpr, tpr):.3f})")
axes[1].plot([0, 1], [0, 1], "k--", lw=1)
axes[1].set_xlim(-0.01, 0.3)
axes[1].set_ylim(0.7, 1.01)
axes[1].set_title("Model comparison: micro-averaged ROC")
axes[1].set_xlabel("False positive rate")
axes[1].set_ylabel("True positive rate")
axes[1].legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig4_roc_curves.png"), dpi=150)
plt.close()


# ----------------------------------------- 5. OVERFITTING / UNDERFITTING
depths = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
tr_s, va_s = validation_curve(DecisionTreeClassifier(random_state=SEED), X_tr, y_tr,
                              param_name="max_depth", param_range=depths, cv=cv, scoring="accuracy")
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(depths, tr_s.mean(1), "o-", color="#F58518", label="Training accuracy")
ax.plot(depths, va_s.mean(1), "o-", color="#4C78A8", label="Cross-validation accuracy")
ax.fill_between(depths, va_s.mean(1) - va_s.std(1), va_s.mean(1) + va_s.std(1), alpha=0.2, color="#4C78A8")
ax.set_xlabel("Maximum tree depth (low = too simple, high = too flexible)")
ax.set_ylabel("Accuracy")
ax.set_title("Decision tree: underfitting on the left, overfitting gap on the right")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig5_validation_curve.png"), dpi=150)
plt.close()
R["val_curve"] = {"depth": depths, "train": [round(v, 4) for v in tr_s.mean(1)],
                  "cv": [round(v, 4) for v in va_s.mean(1)]}

sizes = np.linspace(0.1, 1.0, 7)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
lc = {}
for ax, (title, est) in zip(axes, [("Logistic regression", gs_lr.best_estimator_),
                                    ("Decision tree (unpruned)", DecisionTreeClassifier(random_state=SEED))]):
    ts, trn, val = learning_curve(est, X_tr, y_tr, cv=cv, train_sizes=sizes, scoring="accuracy")
    ax.plot(ts, trn.mean(1), "o-", color="#F58518", label="Training accuracy")
    ax.plot(ts, val.mean(1), "o-", color="#4C78A8", label="Cross-validation accuracy")
    ax.fill_between(ts, val.mean(1) - val.std(1), val.mean(1) + val.std(1), alpha=0.2, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel("Number of training images")
    ax.legend(loc="lower right")
    lc[title] = {"final_train": round(float(trn.mean(1)[-1]), 4), "final_cv": round(float(val.mean(1)[-1]), 4),
                 "cv_at_smallest": round(float(val.mean(1)[0]), 4), "smallest_n": int(ts[0])}
axes[0].set_ylabel("Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig6_learning_curves.png"), dpi=150)
plt.close()
R["learning_curves"] = lc


# ------------------------------------------------------- 6. ERROR ANALYSIS
wrong = np.where(p_lr != y_te.values)[0]
n_show = min(12, len(wrong))
fig, axes = plt.subplots(2, 6, figsize=(12, 4.6))
for ax in axes.ravel():
    ax.axis("off")
for k, idx in enumerate(wrong[:n_show]):
    ax = axes.ravel()[k]
    ax.imshow(X_te.iloc[idx].values.reshape(8, 8), cmap="gray_r")
    ax.set_title(f"true {y_te.iloc[idx]}, predicted {p_lr[idx]}", fontsize=9)
fig.suptitle(f"Test images the logistic regression got wrong ({n_show} of {len(wrong)} shown)")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig7_misclassified.png"), dpi=150)
plt.close()

# Stability over different random splits (hyperparameters fixed from the main split)
accs = {"Logistic regression": [], "Random forest": []}
for s in range(10):
    a, b, c, d = train_test_split(X, y, test_size=0.2, stratify=y, random_state=100 + s)
    accs["Logistic regression"].append(lr_pipe(R["best_C"]).fit(a, c).score(b, d))
    accs["Random forest"].append(RandomForestClassifier(300, random_state=SEED, n_jobs=-1).fit(a, c).score(b, d))
R["split_stability"] = {k: {"mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v, ddof=1)), 4),
                            "min": round(float(np.min(v)), 4), "max": round(float(np.max(v)), 4)}
                        for k, v in accs.items()}


# ------------------------------------------------------------- 7. SAVE
with open(os.path.join(REP, "results.json"), "w") as f:
    json.dump(R, f, indent=2, default=str)

with open(os.path.join(REP, "results_summary.txt"), "w") as f:
    f.write("WEEK 4 RESULTS SUMMARY\n" + "=" * 60 + "\n")
    f.write(f"Samples {R['n_samples']}, features {R['n_features']}, classes {R['n_classes']}\n")
    f.write(f"Missing values {R['missing_values']}, duplicate rows {R['duplicate_rows']}, "
            f"constant pixels {R['constant_pixels']}\n")
    f.write(f"Train {R['n_train']}, test {R['n_test']}\n\n")
    f.write(res.round(4).to_string(index=False) + "\n\n")
    f.write(f"Best C = {R['best_C']}; best tree = {R['best_tree_params']}\n")
    f.write(f"LR test accuracy 95% bootstrap CI: {R['lr_acc_ci95'][0]:.4f} to {R['lr_acc_ci95'][1]:.4f}\n")
    f.write(f"Top confusions: {R['lr_top_confusions']}\n")
    f.write(f"Split stability: {R['split_stability']}\n")
print(open(os.path.join(REP, "results_summary.txt")).read())
