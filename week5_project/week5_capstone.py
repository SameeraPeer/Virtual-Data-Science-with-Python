"""
Week 5 Capstone: end-to-end data science project with strategic recommendations.

Integrates the skills from Weeks 1-4 on one dataset (Breast Cancer Wisconsin,
Diagnostic), which ships inside scikit-learn, so the script runs offline.

  Week 1 -> data audit and cleaning checks
  Week 2 -> exploratory visual story
  Week 3 -> hypothesis tests with effect sizes and multiple-testing control
  Week 4 -> model development and evaluation
  Week 5 -> decision analysis (cost-based threshold), model simplification,
            strategic recommendations

Run:  python week5_capstone.py
Outputs: figures/*.png, reports/results.json, reports/eda_summary.txt
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                             recall_score, precision_score, accuracy_score)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEED = 42
ROOT = Path(__file__).parent
FIG = ROOT / "figures"
REP = ROOT / "reports"
FIG.mkdir(exist_ok=True)
REP.mkdir(exist_ok=True)

BENIGN, MALIGNANT = "#2b7bba", "#d9622b"
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 200,
                     "axes.titleweight": "bold", "axes.titlesize": 12})
R = {}  # all numbers used in the report are stored here


def _j(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    return str(o)


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fmt_p(p):
    return "< 1e-10" if p < 1e-10 else f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"


# ----------------------------------------------------------------------------
# STEP 1 - Data acquisition and audit (Week 1 skills)
# ----------------------------------------------------------------------------
bunch = load_breast_cancer(as_frame=True)
df = bunch.frame.copy()
# sklearn codes malignant = 0. Re-code so that 1 = malignant (the event we
# want to catch); this makes recall/sensitivity mean "malignant cases found".
df["malignant"] = 1 - df["target"]
df = df.drop(columns="target")
features = [c for c in df.columns if c != "malignant"]
X, y = df[features], df["malignant"]

R["n_rows"], R["n_features"] = int(df.shape[0]), len(features)
R["n_malignant"] = int(y.sum())
R["n_benign"] = int((1 - y).sum())
R["pct_malignant"] = round(100 * y.mean(), 1)
R["missing_total"] = int(df.isna().sum().sum())
R["duplicate_rows"] = int(df.duplicated().sum())
R["non_numeric_cols"] = int((df.dtypes == "object").sum())
R["negative_values"] = int((X < 0).sum().sum())
R["zero_values"] = int((X == 0).sum().sum())

# Outliers (IQR rule) - flagged, not removed
q1, q3 = X.quantile(0.25), X.quantile(0.75)
iqr = q3 - q1
out_mask = (X < q1 - 1.5 * iqr) | (X > q3 + 1.5 * iqr)
R["outlier_cells"] = int(out_mask.sum().sum())
R["rows_with_any_outlier"] = int(out_mask.any(axis=1).sum())
R["outlier_pct_cells"] = round(100 * out_mask.sum().sum() / X.size, 2)
R["outlier_share_malignant"] = round(
    100 * out_mask.any(axis=1)[y == 1].mean(), 1)
R["outlier_share_benign"] = round(
    100 * out_mask.any(axis=1)[y == 0].mean(), 1)
R["range_ratio"] = round(float(X.median().max() / X.median().min()))
R["skew_gt1"] = int((X.skew() > 1).sum())

# Redundancy
corr = X.corr()
upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1))
R["pairs_corr_gt_090"] = int((upper.abs() > 0.90).sum().sum())
R["corr_radius_area"] = round(float(corr.loc["mean radius", "mean area"]), 3)
R["corr_radius_perimeter"] = round(float(corr.loc["mean radius", "mean perimeter"]), 3)

# Figure 1 - data audit
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2),
                       gridspec_kw={"width_ratios": [1, 1.2, 1.6]})
ax[0].bar(["Benign", "Malignant"], [R["n_benign"], R["n_malignant"]],
          color=[BENIGN, MALIGNANT])
for i, v in enumerate([R["n_benign"], R["n_malignant"]]):
    ax[0].text(i, v + 5, f"{v}\n({100*v/R['n_rows']:.1f}%)", ha="center")
ax[0].set_ylim(0, 440)
ax[0].set_title("Class balance")
ax[0].set_ylabel("Samples")
checks = ["Missing values", "Duplicate rows", "Non-numeric columns",
          "Negative values"]
vals = [R["missing_total"], R["duplicate_rows"], R["non_numeric_cols"],
        R["negative_values"]]
ax[1].barh(checks, [0.04] * len(vals), color="#4c9a6a")
ax[1].set_xlim(0, 1)
ax[1].set_xticks([0, 1])
for i, v in enumerate(vals):
    ax[1].text(0.02, i, f"{v} found" if v else "0 found",
               va="center", fontweight="bold")
ax[1].invert_yaxis()
ax[1].set_title("Integrity checks")
med = X.median().sort_values()
ax[2].scatter(med.values, range(len(med)), color="#555", s=18)
ax[2].set_xscale("log")
ax[2].set_yticks([])
ax[2].set_xlabel("Median of each feature (log scale)")
ax[2].set_title("Features live on very different scales")
ax[2].text(0.02, 0.95, f"Medians span ~{R['range_ratio']:,}x",
           transform=ax[2].transAxes, va="top")
fig.tight_layout()
save(fig, "fig1_data_audit.png")

# ----------------------------------------------------------------------------
# STEP 2 - Exploratory visual story (Week 2 skills)
# ----------------------------------------------------------------------------
mean_cols = [c for c in features if c.startswith("mean ")]
tests = []
for c in mean_cols:
    a, b = X.loc[y == 1, c], X.loc[y == 0, c]
    tests.append((c, a.mean(), b.mean()))
top6 = ["mean concave points", "mean perimeter", "mean radius",
        "mean concavity", "mean area", "mean texture"]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax_, c in zip(axes.ravel(), top6):
    plot_df = pd.DataFrame({"v": X[c],
                            "Diagnosis": np.where(y == 1, "Malignant", "Benign")})
    sns.violinplot(data=plot_df, x="Diagnosis", y="v", hue="Diagnosis",
                   order=["Benign", "Malignant"],
                   palette={"Benign": BENIGN, "Malignant": MALIGNANT},
                   inner="quartile", cut=0, legend=False, ax=ax_)
    ax_.set_title(c.replace("mean ", "").capitalize())
    ax_.set_xlabel("")
    ax_.set_ylabel("")
fig.suptitle("Malignant samples sit higher on most size and shape measurements",
             fontweight="bold", y=1.0)
fig.tight_layout()
save(fig, "fig2_group_differences.png")

fig, ax = plt.subplots(figsize=(8.2, 6.8))
cm_ = X[mean_cols].corr()
order = sns.clustermap(cm_, method="average").dendrogram_row.reordered_ind
plt.close("all")
cm_ord = cm_.iloc[order, order]
sns.heatmap(cm_ord, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
            annot_kws={"size": 7}, linewidths=.4,
            cbar_kws={"label": "Pearson r"}, ax=ax)
ax.set_xticklabels([t.get_text().replace("mean ", "") for t in ax.get_xticklabels()],
                   rotation=45, ha="right")
ax.set_yticklabels([t.get_text().replace("mean ", "") for t in ax.get_yticklabels()],
                   rotation=0)
ax.set_title("Many measurements repeat the same information")
fig.tight_layout()
save(fig, "fig3_correlation.png")

# ----------------------------------------------------------------------------
# STEP 3 - Hypothesis tests (Week 3 skills)
# ----------------------------------------------------------------------------
rows = []
for c in mean_cols:
    a, b = X.loc[y == 1, c], X.loc[y == 0, c]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    u, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / sp
    se = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
    rows.append(dict(feature=c, mean_malignant=a.mean(), mean_benign=b.mean(),
                     t=t, p_welch=p, p_mwu=pu, d=d, d_lo=d - 1.96 * se,
                     d_hi=d + 1.96 * se))
ht = pd.DataFrame(rows).sort_values("d", ascending=False).reset_index(drop=True)
ALPHA = 0.05 / len(ht)          # Bonferroni across 10 tests
ht["significant"] = ht["p_welch"] < ALPHA
ht.to_csv(REP / "hypothesis_tests.csv", index=False)
R["bonferroni_alpha"] = ALPHA
R["n_sig"] = int(ht["significant"].sum())
R["n_tests"] = len(ht)
R["tests"] = ht.round(6).to_dict("records")
R["nonsig_features"] = ht.loc[~ht["significant"], "feature"].tolist()
R["bonf_alpha_str"] = f"{ALPHA:.3f}"

fig, ax = plt.subplots(figsize=(8.5, 5.2))
yy = np.arange(len(ht))[::-1]
colors = [MALIGNANT if s else "#999" for s in ht["significant"]]
ax.hlines(yy, ht["d_lo"], ht["d_hi"], color=colors, lw=2)
ax.scatter(ht["d"], yy, color=colors, zorder=3, s=45)
ax.axvline(0, color="k", lw=1)
for v, lab in [(0.8, "large (0.8)")]:
    ax.axvline(v, color="#bbb", ls="--", lw=1)
    ax.text(v + 0.03, len(ht) - 0.4, lab, color="#777", fontsize=8)
ax.set_yticks(yy)
ax.set_yticklabels([f.replace("mean ", "") for f in ht["feature"]])
ax.set_xlabel("Cohen's d (malignant minus benign) with 95% CI")
ax.set_title("Effect sizes: orange = significant after Bonferroni correction")
fig.tight_layout()
save(fig, "fig4_effect_sizes.png")

# ----------------------------------------------------------------------------
# STEP 4 - Model development and evaluation (Week 4 skills)
# ----------------------------------------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y,
                                          random_state=SEED)
R["n_train"], R["n_test"] = len(X_tr), len(X_te)
R["test_malignant"] = int(y_te.sum())
R["test_benign"] = int((1 - y_te).sum())

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
models = {
    "Majority-class baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic regression": Pipeline([("sc", StandardScaler()),
                                     ("m", LogisticRegression(max_iter=5000))]),
    "Random forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
    "Decision tree": DecisionTreeClassifier(max_depth=4, random_state=SEED),
}
# Tune logistic regression C by CV on the training set only
gs = GridSearchCV(models["Logistic regression"],
                  {"m__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]},
                  scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
gs.fit(X_tr, y_tr)
R["best_C"] = float(gs.best_params_["m__C"])
models["Logistic regression"] = gs.best_estimator_

cvres = {}
for name, m in models.items():
    auc = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc")
    rec = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="recall")
    acc = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="accuracy")
    cvres[name] = dict(auc=auc.mean(), auc_sd=auc.std(), recall=rec.mean(),
                       acc=acc.mean())
R["cv"] = {k: {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in cvres.items()}

test = {}
probs = {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    test[name] = dict(acc=accuracy_score(y_te, pred), auc=roc_auc_score(y_te, p),
                      recall=recall_score(y_te, pred),
                      precision=precision_score(y_te, pred, zero_division=0),
                      tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    probs[name] = p
R["test"] = {k: {kk: (round(float(vv), 4) if isinstance(vv, float) else vv)
                 for kk, vv in v.items()} for k, v in test.items()}

# Bootstrap CIs for the main model on the test set
rng = np.random.default_rng(SEED)
pm = probs["Logistic regression"]
idx_all = np.arange(len(y_te))
bs = {"auc": [], "recall": [], "acc": []}
yt = y_te.values
for _ in range(2000):
    i = rng.choice(idx_all, len(idx_all))
    if yt[i].min() == yt[i].max():
        continue
    bs["auc"].append(roc_auc_score(yt[i], pm[i]))
    bs["recall"].append(recall_score(yt[i], (pm[i] >= .5).astype(int)))
    bs["acc"].append(accuracy_score(yt[i], (pm[i] >= .5).astype(int)))
R["boot"] = {k: [round(float(np.percentile(v, 2.5)), 3),
                 round(float(np.percentile(v, 97.5)), 3)] for k, v in bs.items()}

# Figure 5 - model comparison (CV AUC with sd, test recall)
names = list(models)
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.5))
aucs = [cvres[n]["auc"] for n in names]
sds = [cvres[n]["auc_sd"] for n in names]
cols = ["#bbbbbb", "#1f5f8b", "#7aa6c2", "#7aa6c2"]
ax[0].barh(names, aucs, xerr=sds, color=cols, capsize=3)
ax[0].set_xlim(0.45, 1.02)
ax[0].invert_yaxis()
for i, v in enumerate(aucs):
    ax[0].text(0.47 if i == 0 else v - 0.012, i, f"{v:.3f}", va="center",
               ha="left" if i == 0 else "right",
               color="#333" if i == 0 else "white", fontweight="bold")
ax[0].set_xlabel("ROC AUC (5-fold x 3 repeats, training data)")
ax[0].set_title("Cross-validated ranking quality")
x_ = np.arange(len(names))
w = .38
ax[1].bar(x_ - w / 2, [test[n]["acc"] for n in names], w, label="Accuracy",
          color="#7aa6c2")
ax[1].bar(x_ + w / 2, [test[n]["recall"] for n in names], w,
          label="Recall (malignant)", color=MALIGNANT)
ax[1].set_xticks(x_)
ax[1].set_xticklabels([n.replace(" ", "\n", 1) for n in names], fontsize=8)
ax[1].set_ylim(0, 1.12)
ax[1].legend(loc="lower right", fontsize=8)
for i, n in enumerate(names):
    ax[1].text(i - w / 2, test[n]["acc"] + .02, f"{test[n]['acc']:.2f}",
               ha="center", fontsize=8)
    ax[1].text(i + w / 2, test[n]["recall"] + .02, f"{test[n]['recall']:.2f}",
               ha="center", fontsize=8)
ax[1].set_title("Held-out test results at the default 0.5 cut-off")
fig.tight_layout()
save(fig, "fig5_model_comparison.png")

# ----------------------------------------------------------------------------
# STEP 5 - Decision analysis (Week 5 contribution)
# ----------------------------------------------------------------------------
main = models["Logistic regression"]
# Out-of-fold probabilities on the TRAINING set: thresholds are chosen here,
# never on the test set.
oof = cross_val_predict(main, X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                        method="predict_proba")[:, 1]
ths = np.round(np.arange(0.02, 0.99, 0.01), 2)


def sens_spec(yv, pv, t):
    pred = (pv >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(yv, pred, labels=[0, 1]).ravel()
    return tp / (tp + fn), tn / (tn + fp), fp, fn


curve = np.array([sens_spec(y_tr.values, oof, t) for t in ths])
sens_c, spec_c = curve[:, 0], curve[:, 1]
# Rule: highest threshold that keeps training out-of-fold sensitivity >= 0.98
TARGET_SENS = 0.98
ok = ths[sens_c >= TARGET_SENS]
t_safe = float(ok.max())
R["target_sens"] = TARGET_SENS
R["t_safe"] = t_safe
s_tr, sp_tr, fp_tr, fn_tr = sens_spec(y_tr.values, oof, t_safe)
R["oof_at_tsafe"] = dict(sens=round(float(s_tr), 3), spec=round(float(sp_tr), 3))

pred_def = (pm >= .5).astype(int)
pred_safe = (pm >= t_safe).astype(int)
op = {}
for lab, pr in [("default", pred_def), ("safety", pred_safe)]:
    tn, fp, fn, tp = confusion_matrix(y_te, pr).ravel()
    op[lab] = dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                   sens=round(tp / (tp + fn), 4), spec=round(tn / (tn + fp), 4),
                   ppv=round(tp / (tp + fp), 4) if tp + fp else None,
                   npv=round(tn / (tn + fn), 4) if tn + fn else None,
                   acc=round((tp + tn) / len(yt), 4))
R["op"] = op

# Expected cost per 100 patients for different cost ratios (FN cost : FP cost)
ratios = [1, 5, 10, 20]
cost_tab = {}
for r in ratios:
    best_t, best_c = None, 1e9
    for t in ths:
        s, sp, fp_, fn_ = sens_spec(y_tr.values, oof, t)
        c = (fn_ * r + fp_) / len(y_tr)
        if c < best_c:
            best_c, best_t = c, float(t)
    s, sp, fp_, fn_ = sens_spec(yt, pm, best_t)
    cost_tab[r] = dict(threshold=best_t,
                       test_cost_per_100=round(100 * (fn_ * r + fp_) / len(yt), 2),
                       default_cost_per_100=round(100 * (op["default"]["fn"] * r + op["default"]["fp"]) / len(yt), 2),
                       fn=int(fn_), fp=int(fp_))
R["cost"] = cost_tab


# Human-review band: cases whose probability falls between the safety cut-off
# and 0.80 would be sent to a second reader (illustrative band, fixed in advance)
BAND_LO, BAND_HI = t_safe, 0.80
in_band = (pm >= BAND_LO) & (pm <= BAND_HI)
wrong_default = pred_def != yt
R["band"] = dict(lo=BAND_LO, hi=BAND_HI, n_in_band=int(in_band.sum()),
                 pct_in_band=round(100 * in_band.mean(), 1),
                 errors_total=int(wrong_default.sum()),
                 errors_in_band=int((wrong_default & in_band).sum()))

# Figure 6 - ROC + confusion matrices at two operating points
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6),
                       gridspec_kw={"width_ratios": [1.15, 1, 1]})
for name, c in zip(names[1:], ["#1f5f8b", "#2e9e6b", "#b0872b"]):
    fpr, tpr, _ = roc_curve(y_te, probs[name])
    ax[0].plot(fpr, tpr, color=c, lw=2,
               label=f"{name} (AUC {test[name]['auc']:.3f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
fpr, tpr, thr = roc_curve(y_te, pm)
for t_, lab_, off in [(0.5, "0.5", (.25, .55)), (t_safe, f"{t_safe:.2f}", (.25, .42))]:
    j = np.argmin(np.abs(thr - t_))
    ax[0].scatter(fpr[j], tpr[j], s=70, zorder=5, color="k")
    ax[0].annotate(f"main model at cut-off {lab_}", (fpr[j], tpr[j]), xytext=off,
                   fontsize=8, arrowprops=dict(arrowstyle="-", color="#666"))
ax[0].set_xlabel("False positive rate")
ax[0].set_ylabel("True positive rate (recall)")
ax[0].set_title("ROC curves (test set)")
ax[0].legend(loc="lower right", fontsize=8)
for a_, lab in zip(ax[1:], ["default", "safety"]):
    o = op[lab]
    mat = np.array([[o["tn"], o["fp"]], [o["fn"], o["tp"]]])
    sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", cbar=False, ax=a_,
                annot_kws={"size": 16},
                xticklabels=["Pred. benign", "Pred. malignant"],
                yticklabels=["True benign", "True malignant"])
    thr_txt = "0.50" if lab == "default" else f"{t_safe:.2f}"
    a_.set_title(f"Cut-off {thr_txt}: missed = {o['fn']}, false alarms = {o['fp']}")
fig.tight_layout()
save(fig, "fig6_roc_confusion.png")

# Figure 7 - trade-off and cost
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ax[0].plot(ths, sens_c, color=MALIGNANT, lw=2, label="Sensitivity (malignant caught)")
ax[0].plot(ths, spec_c, color=BENIGN, lw=2, label="Specificity (benign cleared)")
ax[0].axhline(TARGET_SENS, color="#888", ls="--", lw=1)
ax[0].axvline(t_safe, color="k", ls=":", lw=1.2)
ax[0].set_ylim(0.7, 1.03)
ax[0].text(t_safe + .02, 1.012, f"chosen cut-off {t_safe:.2f}", fontsize=9)
ax[0].set_xlabel("Probability cut-off for flagging 'malignant'")
ax[0].set_ylabel("Rate (out-of-fold, training)")
ax[0].set_title("Lowering the cut-off catches more cancers, costs more false alarms")
ax[0].title.set_fontsize(10.5)
ax[0].legend(loc="lower center", fontsize=8)
for r, c in zip(ratios, ["#9ecae1", "#6baed6", "#3182bd", "#08519c"]):
    cc = []
    for t in ths:
        s, sp, fp_, fn_ = sens_spec(y_tr.values, oof, t)
        cc.append(100 * (fn_ * r + fp_) / len(y_tr))
    ax[1].plot(ths, cc, color=c, lw=2, label=f"missed case costs {r}x a false alarm")
ax[1].set_xlabel("Probability cut-off")
ax[1].set_ylabel("Cost per 100 patients\n(1 unit = one false alarm)")
ax[1].set_yscale("log")
ax[1].set_title("The best cut-off depends on how costly a miss is")
ax[1].title.set_fontsize(10.5)
ax[1].legend(fontsize=8, loc="upper left")
ax[1].set_ylim(1, 600)
fig.tight_layout()
save(fig, "fig7_threshold_cost.png")

# ----------------------------------------------------------------------------
# STEP 6 - Can a simpler model do the job? (selection inside CV: no leakage)
# ----------------------------------------------------------------------------
ks = [1, 2, 3, 5, 7, 10, 15, 20, 30]
kres = []
for k in ks:
    pipe = Pipeline([("sel", SelectKBest(f_classif, k=k)), ("sc", StandardScaler()),
                     ("m", LogisticRegression(max_iter=5000, C=R["best_C"]))])
    s = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
    kres.append((k, s.mean(), s.std()))
R["kfeat"] = [dict(k=k, auc=round(float(m), 4), sd=round(float(s_), 4)) for k, m, s_ in kres]
full_auc = [m for k, m, s_ in kres if k == 30][0]
R["k_within_001"] = int(min(k for k, m, s_ in kres if m >= full_auc - 0.005))

fig, ax = plt.subplots(figsize=(7.5, 4.4))
kk = [r[0] for r in kres]
mm = np.array([r[1] for r in kres])
ss = np.array([r[2] for r in kres])
ax.plot(kk, mm, marker="o", color="#1f5f8b", lw=2)
ax.fill_between(kk, mm - ss, mm + ss, color="#1f5f8b", alpha=.15)
ax.set_xlabel("Number of measurements kept (chosen inside each CV fold)")
ax.set_ylabel("Cross-validated ROC AUC")
ax.set_title("Five measurements carry most of the signal; the last gains need many more")
ax.title.set_fontsize(10.5)
ax.set_ylim(min(mm) - .02, 1.0)
fig.tight_layout()
save(fig, "fig8_feature_reduction.png")

# Error analysis on the test set
err_def = np.where(pred_def != yt)[0]
err_safe = np.where(pred_safe != yt)[0]
Xte = X_te.reset_index(drop=True)
R["errors_default"] = [dict(row=int(i), truth="malignant" if yt[i] else "benign",
                            prob=round(float(pm[i]), 3)) for i in err_def]
R["errors_safety"] = [dict(row=int(i), truth="malignant" if yt[i] else "benign",
                           prob=round(float(pm[i]), 3)) for i in err_safe]
coef = pd.Series(main.named_steps["m"].coef_[0], index=features)
R["top_coef"] = coef.reindex(coef.abs().sort_values(ascending=False).index)[:6].round(3).to_dict()

# ----------------------------------------------------------------------------
# STEP 7 - Cross-check of a Week 3 headline number (Wine dataset, deterministic)
# ----------------------------------------------------------------------------
wine = load_wine(as_frame=True).frame
a = wine.loc[wine.target == 0, "alcohol"]
b = wine.loc[wine.target == 1, "alcohol"]
t_, p_ = stats.ttest_ind(a, b, equal_var=False)
diff = a.mean() - b.mean()
se_ = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
dfw = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2 / (
    (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
tc = stats.t.ppf(.975, dfw)
R["wine_check"] = dict(diff=round(float(diff), 2), lo=round(float(diff - tc * se_), 2),
                       hi=round(float(diff + tc * se_), 2))

# ----------------------------------------------------------------------------
# Figure 0 - the pipeline (used on the report cover)
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 2.7))
ax.axis("off")
steps = [("Week 1", "Acquire, audit\nand clean"), ("Week 2", "Explore and\ntell the story"),
         ("Week 3", "Test\nhypotheses"), ("Week 4", "Build and\nevaluate a model"),
         ("Week 5", "Decide and\nrecommend")]
cols = ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08306b"]
for i, ((wk, txt), c) in enumerate(zip(steps, cols)):
    x0 = i * 2.6
    ax.add_patch(mpatches.FancyBboxPatch((x0, 0.1), 2.2, 1.5, boxstyle="round,pad=0.05",
                                         fc=c, ec="none"))
    tcol = "white" if i >= 2 else "#08306b"
    ax.text(x0 + 1.1, 1.3, wk, ha="center", va="center", fontweight="bold",
            color=tcol, fontsize=12)
    ax.text(x0 + 1.1, 0.7, txt, ha="center", va="center", color=tcol, fontsize=10)
    if i < 4:
        ax.annotate("", xy=(x0 + 2.55, 0.85), xytext=(x0 + 2.25, 0.85),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#444"))
ax.set_xlim(-.1, 13.1)
ax.set_ylim(0, 1.8)
save(fig, "fig0_pipeline.png")

# ----------------------------------------------------------------------------
# Save outputs
# ----------------------------------------------------------------------------
with open(REP / "results.json", "w") as f:
    json.dump(R, f, indent=2, default=_j)

lines = [
    "WEEK 5 CAPSTONE - SUMMARY OF RESULTS", "=" * 40,
    f"Rows: {R['n_rows']}  Features: {R['n_features']}  Malignant: {R['n_malignant']} ({R['pct_malignant']}%)",
    f"Missing: {R['missing_total']}  Duplicates: {R['duplicate_rows']}",
    f"Outlier cells (IQR): {R['outlier_cells']} in {R['rows_with_any_outlier']} rows",
    f"Pairs with |r|>0.90: {R['pairs_corr_gt_090']}",
    f"Hypothesis tests significant after Bonferroni: {R['n_sig']} of {R['n_tests']}",
    f"Best C: {R['best_C']}", "", "CV results:",
]
for k, v in R["cv"].items():
    lines.append(f"  {k}: AUC {v['auc']:.3f} +/- {v['auc_sd']:.3f}, recall {v['recall']:.3f}")
lines.append("\nTest results (cut-off 0.5):")
for k, v in R["test"].items():
    lines.append(f"  {k}: acc {v['acc']:.3f} AUC {v['auc']:.3f} recall {v['recall']:.3f} FN {v['fn']} FP {v['fp']}")
lines.append(f"\nSafety cut-off: {t_safe}  ->  {json.dumps(op['safety'], default=_j)}")
lines.append(f"Default cut-off -> {json.dumps(op['default'], default=_j)}")
lines.append(f"Bootstrap 95% CI (test, main model): {R['boot']}")
lines.append(f"Cost table: {json.dumps(R['cost'], default=_j)}")
lines.append(f"Feature-count curve: {R['kfeat']}")
lines.append(f"Wine cross-check (Week 3, A vs B alcohol): {R['wine_check']}")
(REP / "eda_summary.txt").write_text("\n".join(lines))
print("\n".join(lines))
