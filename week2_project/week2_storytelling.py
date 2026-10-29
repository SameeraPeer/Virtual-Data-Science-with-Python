"""
Week 2 - Advanced Data Visualization and Storytelling with Python
Dataset: Breast Cancer Wisconsin (Diagnostic), 569 samples, 30 measurements.
         Public UCI dataset, distributed with scikit-learn (no download needed).

Run:  python week2_storytelling.py

Outputs:
    data/raw/breast_cancer_wisconsin.csv       the dataset as loaded
    data/processed/breast_cancer_prepared.csv  with plain-English labels and risk bands
    figures/fig1 ... fig7 (.png)               the seven visualizations
    reports/key_numbers.txt                    every number quoted in the report
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse, Rectangle
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

for folder in ("data/raw", "data/processed", "figures", "reports"):
    os.makedirs(folder, exist_ok=True)

# ---------------------------------------------------------------- style
BENIGN, MALIGNANT = "#0072B2", "#D55E00"      # colour-blind-safe blue / orange
INK, MUTED, GRID = "#1F2937", "#6B7280", "#E5E7EB"
PALETTE = {"Benign": BENIGN, "Malignant": MALIGNANT}
SOURCE = "Source: Breast Cancer Wisconsin (Diagnostic) dataset, UCI / scikit-learn. n = 569 samples."

sns.set_theme(style="white")
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "text.color": INK, "figure.dpi": 100,
})

lines = []
def log(text=""):
    print(text)
    lines.append(str(text))

def headline(fig, title, subtitle, top=0.97):
    """Story-style title: the takeaway first, plain-English subtitle below."""
    fig.text(0.01, top, title, fontsize=17, fontweight="bold", ha="left", va="top")
    fig.text(0.01, top - 0.075, subtitle, fontsize=11.5, color=MUTED, ha="left", va="top")

def footer(fig, extra=""):
    fig.text(0.01, 0.012, SOURCE + (" " + extra if extra else ""), fontsize=8.5, color=MUTED, ha="left")

def save(fig, name):
    fig.savefig(f"figures/{name}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ---------------------------------------------------------------- 1. data
raw = load_breast_cancer(as_frame=True)
df = raw.frame.copy()
# in scikit-learn, target 0 = malignant and 1 = benign
df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})
df = df.drop(columns="target")
df.to_csv("data/raw/breast_cancer_wisconsin.csv", index=False)

features = [c for c in df.columns if c != "diagnosis"]
df["is_malignant"] = (df["diagnosis"] == "Malignant").astype(int)

log("=== DATA CHECK ===")
log(f"Samples and columns: {df.shape[0]} rows, {len(features)} measurements + diagnosis")
log(f"Missing values: {int(df.isna().sum().sum())}")
log(f"Duplicate rows: {int(df.duplicated().sum())}")
counts = df["diagnosis"].value_counts()
n_b, n_m, n = int(counts["Benign"]), int(counts["Malignant"]), len(df)
log(f"Benign: {n_b} ({n_b / n * 100:.1f}%)   Malignant: {n_m} ({n_m / n * 100:.1f}%)")

# ---------------------------------------------------------------- FIG 1
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.barh([0], [n_b], color=BENIGN, height=0.55)
ax.barh([0], [n_m], left=[n_b], color=MALIGNANT, height=0.55)
ax.text(n_b / 2, 0, f"Benign\n{n_b} samples ({n_b / n * 100:.1f}%)", ha="center", va="center",
        color="white", fontsize=15, fontweight="bold")
ax.text(n_b + n_m / 2, 0, f"Malignant\n{n_m} samples\n({n_m / n * 100:.1f}%)", ha="center", va="center",
        color="white", fontsize=15, fontweight="bold")
ax.set_xlim(0, n); ax.set_ylim(-0.6, 0.6); ax.axis("off")
fig.subplots_adjust(left=0.01, right=0.99, top=0.72, bottom=0.12)
headline(fig, f"Just over a third of the {n} samples are malignant",
         "Each bar segment is one diagnosis group. The data over-represents cancer compared with routine screening.", top=1.02)
footer(fig)
save(fig, "fig1_class_balance.png")

# ---------------------------------------------------------------- FIG 2
panels = [("mean radius", "Size\n(mean radius)", ".1f"),
          ("mean texture", "Roughness\n(mean texture)", ".1f"),
          ("mean area", "Area\n(mean area)", ".0f"),
          ("mean concave points", "Dents in outline\n(mean concave points)", ".3f")]
med = df.groupby("diagnosis")[[p[0] for p in panels]].median()
log("\n=== MEDIANS BY DIAGNOSIS ===")
log(med.round(3).T.to_string())

fig, axes = plt.subplots(1, 4, figsize=(14, 5.6))
for ax, (col, label, fmt) in zip(axes, panels):
    sns.violinplot(data=df, x="diagnosis", y=col, order=["Benign", "Malignant"], hue="diagnosis",
                   palette=PALETTE, inner="quartile", cut=0, linewidth=1.1, ax=ax, legend=False)
    ax.set_title(label, fontsize=12.5, pad=8)
    ax.set_ylabel(""); ax.set_xlabel("")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Benign\nmedian {format(med.loc['Benign', col], fmt)}",
                        f"Malignant\nmedian {format(med.loc['Malignant', col], fmt)}"], fontsize=10)
    sns.despine(ax=ax)
    ax.grid(axis="y", color=GRID)
headline(fig, "Malignant samples are larger and more dented, while texture overlaps much more",
         "Each shape shows how the measurements of a group are spread out. Lines inside mark the lower quartile, median and upper quartile.", top=1.03)
fig.subplots_adjust(top=0.80, bottom=0.14, wspace=0.35)
footer(fig)
save(fig, "fig2_distributions.png")

# ---------------------------------------------------------------- FIG 3
auc = {f: roc_auc_score(df["is_malignant"], df[f]) for f in features}
sep = pd.Series({f: max(a, 1 - a) for f, a in auc.items()}).sort_values(ascending=False)
direction = {f: ("higher in malignant" if auc[f] > 0.5 else "lower in malignant") for f in features}
log("\n=== SEPARATION POWER (AUC) TOP 12 ===")
for f, v in sep.head(12).items():
    log(f"{f:<24s} {v:.3f}  ({direction[f]})")
log(f"Weakest feature: {sep.index[-1]} {sep.iloc[-1]:.3f}")
log(f"mean texture: {sep['mean texture']:.3f}")

top = sep.head(12)[::-1]
def kind(f):
    return "Worst" if f.startswith("worst") else ("Mean" if f.startswith("mean") else "Error")
colours = {"Worst": "#1F3864", "Mean": "#7F8FA9", "Error": "#C9CED8"}
fig, ax = plt.subplots(figsize=(10.5, 6.2))
ax.barh([f.capitalize() for f in top.index], top.values, color=[colours[kind(f)] for f in top.index], height=0.7)
for i, v in enumerate(top.values):
    ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=10.5)
ax.set_xlim(0.5, 1.0)
ax.axvline(0.5, color=MUTED, lw=1.2)
ax.set_xlabel("Separation score (1.0 = perfect, 0.5 = no better than a coin toss)")
ax.grid(axis="x", color=GRID); ax.set_axisbelow(True); sns.despine(ax=ax, left=True)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=colours[k]) for k in ("Worst", "Mean", "Error")],
          labels=["Worst: the most extreme cells in a sample", "Mean: the average cell", "Error: how much cells vary"],
          loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=3, frameon=False, fontsize=10.5)
headline(fig, "The most extreme cells are the best clues",
         "Ranking of the 12 strongest single measurements (separation score is the area under the ROC curve).", top=1.01)
fig.subplots_adjust(top=0.86, bottom=0.2)
footer(fig)
save(fig, "fig3_feature_ranking.png")

# ---------------------------------------------------------------- FIG 4
order = ["radius", "perimeter", "area", "texture", "smoothness", "compactness",
         "concavity", "concave points", "symmetry", "fractal dimension"]
mean_cols = [f"mean {o}" for o in order]
corr = df[mean_cols].corr()
all_corr = df[features].corr().abs()
n_pairs = len(features) * (len(features) - 1) // 2
high = sum(all_corr.loc[a, b] > 0.9 for i, a in enumerate(features) for b in features[i + 1:])
log("\n=== CORRELATIONS ===")
log(f"Pairs of the 30 measurements with |r| > 0.9: {high} of {n_pairs}")
log(f"mean radius vs mean perimeter: {corr.loc['mean radius', 'mean perimeter']:.3f}")
log(f"mean radius vs mean area: {corr.loc['mean radius', 'mean area']:.3f}")
log(f"mean concavity vs mean concave points: {corr.loc['mean concavity', 'mean concave points']:.3f}")
log(f"mean texture vs mean radius: {corr.loc['mean texture', 'mean radius']:.3f}")

mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 9},
            linewidths=0.8, linecolor="white", square=True, cbar_kws={"shrink": 0.7, "label": "Correlation"}, ax=ax)
ax.set_xticklabels([o.capitalize() for o in order], rotation=40, ha="right")
ax.set_yticklabels([o.capitalize() for o in order], rotation=0)
ax.add_patch(Rectangle((0, 0), 3, 3, fill=False, edgecolor=INK, lw=2.5, linestyle="--"))
ax.text(3.35, 1.0, "Size, perimeter and area\nare near-duplicates (0.99+):\nthey carry one clue, not three",
        fontsize=11, va="center", ha="left", color=INK, fontweight="bold")
headline(fig, "Many of the measurements say the same thing twice",
         f"Correlation between the 10 average measurements. {high} of the {n_pairs} pairs among all 30 measurements correlate above 0.9.", top=1.03)
fig.subplots_adjust(top=0.88, bottom=0.17)
footer(fig)
save(fig, "fig4_correlation.png")

# ---------------------------------------------------------------- FIG 5
x_lo, y_lo, x_hi, y_hi = 16.0, 0.11, 17.5, 0.15       # illustrative thresholds
benign_zone = df[(df["worst radius"] < x_lo) & (df["worst concave points"] < y_lo)]
malig_zone = df[(df["worst radius"] > x_hi) & (df["worst concave points"] > y_hi)]
grey_n = n - len(benign_zone) - len(malig_zone)
log("\n=== TWO-CLUE ZONES (illustrative thresholds, not validated) ===")
log(f"Thresholds: benign zone worst radius < {x_lo} and worst concave points < {y_lo}; "
    f"malignant zone worst radius > {x_hi} and worst concave points > {y_hi}")
log(f"Clear benign zone: {len(benign_zone)} samples, {benign_zone['is_malignant'].sum()} malignant "
    f"({benign_zone['is_malignant'].mean() * 100:.1f}%)")
log(f"Clear malignant zone: {len(malig_zone)} samples, {malig_zone['is_malignant'].sum()} malignant "
    f"({malig_zone['is_malignant'].mean() * 100:.1f}%)")
log(f"Grey zone: {grey_n} samples ({grey_n / n * 100:.1f}%)")
grey = df.drop(benign_zone.index).drop(malig_zone.index)
log(f"Grey zone malignant share: {grey['is_malignant'].mean() * 100:.1f}%")
log(f"Samples in the two clear zones: {len(benign_zone) + len(malig_zone)} "
    f"({(len(benign_zone) + len(malig_zone)) / n * 100:.1f}%)")

xmin, xmax = df["worst radius"].min() - 1, df["worst radius"].max() + 1
ymin, ymax = 0, df["worst concave points"].max() + 0.015
fig, ax = plt.subplots(figsize=(11, 6.6))
ax.add_patch(Rectangle((xmin, ymin), x_lo - xmin, y_lo - ymin, color=BENIGN, alpha=0.10, lw=0))
ax.add_patch(Rectangle((x_hi, y_hi), xmax - x_hi, ymax - y_hi, color=MALIGNANT, alpha=0.10, lw=0))
for d in ("Benign", "Malignant"):
    s = df[df["diagnosis"] == d]
    ax.scatter(s["worst radius"], s["worst concave points"], s=34, alpha=0.75, color=PALETTE[d],
               edgecolor="white", linewidth=0.5, label=d)
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
ax.set_xlabel("Worst radius (size of the largest cells in the sample)")
ax.set_ylabel("Worst concave points (dents in the outline)")
box = dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="none", alpha=0.9)
ax.annotate(f"Clear benign zone\n{len(benign_zone)} samples, {benign_zone['is_malignant'].mean() * 100:.1f}% malignant",
            xy=(11.5, 0.095), xytext=(xmin + 0.6, 0.255), color=BENIGN, fontsize=11.5, fontweight="bold",
            va="top", bbox=box, arrowprops=dict(arrowstyle="->", color=BENIGN, lw=1.4))
ax.text(xmax - 0.5, ymax - 0.006, f"Clear malignant zone\n{len(malig_zone)} samples, "
        f"{malig_zone['is_malignant'].mean() * 100:.0f}% malignant", color=MALIGNANT, fontsize=11.5,
        fontweight="bold", va="top", ha="right", bbox=box)
ax.annotate(f"Grey zone: {grey_n} samples ({grey_n / n * 100:.0f}%)\nneed more evidence",
            xy=(17.0, 0.125), xytext=(23.5, 0.052), fontsize=11.5, color=INK, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
ax.legend(frameon=False, loc="lower right", fontsize=11, markerscale=1.4)
sns.despine(ax=ax); ax.grid(color=GRID); ax.set_axisbelow(True)
headline(fig, "Two clues sort most samples with confidence, and flag the rest for a closer look",
         "Each dot is one sample. Shaded areas mark where almost every sample has the same diagnosis (illustrative thresholds).", top=1.02)
fig.subplots_adjust(top=0.88)
footer(fig, "Thresholds chosen by inspecting this dataset; not a validated diagnostic rule.")
save(fig, "fig5_two_clues.png")

# ---------------------------------------------------------------- FIG 6
bins = [0, 12, 14, 16, 18, 20, 100]
labels = ["Under 12", "12 to 14", "14 to 16", "16 to 18", "18 to 20", "Over 20"]
df["radius_band"] = pd.cut(df["worst radius"], bins=bins, labels=labels)
ladder = df.groupby("radius_band", observed=True)["is_malignant"].agg(["mean", "size", "sum"])
log("\n=== RISK LADDER (worst radius bands) ===")
for lab, r in ladder.iterrows():
    log(f"{lab:<9s} n={int(r['size']):>3d}  malignant={int(r['sum']):>3d}  share={r['mean'] * 100:5.1f}%")

fig, ax = plt.subplots(figsize=(10.5, 5.8))
shares = ladder["mean"].values * 100
cmap = plt.cm.YlOrRd(np.linspace(0.25, 0.95, len(shares)))
bars = ax.bar(range(len(shares)), shares, color=cmap, width=0.72)
ax.set_xticks(range(len(shares)))
ax.set_xticklabels([f"{lab}\n{k} samples" for lab, k in zip(ladder.index.astype(str), ladder["size"])], fontsize=11)
for b, s_ in zip(bars, shares):
    ax.text(b.get_x() + b.get_width() / 2, s_ + 2, f"{s_:.0f}%", ha="center", fontsize=14, fontweight="bold")
ax.set_ylim(0, 112); ax.set_ylabel("Share of samples that are malignant (%)")
ax.set_xlabel("Worst radius (size of the largest cells in the sample)", labelpad=10)
ax.grid(axis="y", color=GRID); ax.set_axisbelow(True); sns.despine(ax=ax)
headline(fig, "The bigger the largest cells, the higher the share of malignant samples",
         "Samples grouped by the size of their largest cells. Risk climbs from almost none to every sample.", top=1.01)
fig.subplots_adjust(top=0.86, bottom=0.2)
footer(fig)
save(fig, "fig6_risk_ladder.png")

# ---------------------------------------------------------------- FIG 7
X = StandardScaler().fit_transform(df[features])
pca = PCA(n_components=2, random_state=42)
Z = pca.fit_transform(X)
ev = pca.explained_variance_ratio_ * 100
df["pc1"], df["pc2"] = Z[:, 0], Z[:, 1]
log("\n=== PCA ===")
log(f"Variance explained: PC1 {ev[0]:.1f}%, PC2 {ev[1]:.1f}%, together {ev.sum():.1f}%")
log(f"PC1 mean: benign {df[df.diagnosis == 'Benign'].pc1.mean():.2f}, malignant {df[df.diagnosis == 'Malignant'].pc1.mean():.2f}")

def ellipse(ax, x, y, colour, n_std=2.0):
    vals, vecs = np.linalg.eigh(np.cov(x, y))
    o = vals.argsort()[::-1]
    vals, vecs = vals[o], vecs[:, o]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * n_std * np.sqrt(vals)
    ax.add_patch(Ellipse((np.mean(x), np.mean(y)), w, h, angle=angle, fill=False,
                         edgecolor=colour, lw=2.2, linestyle="--"))

fig, ax = plt.subplots(figsize=(10.5, 6.6))
for d in ("Benign", "Malignant"):
    s = df[df["diagnosis"] == d]
    ax.scatter(s["pc1"], s["pc2"], s=32, alpha=0.7, color=PALETTE[d], edgecolor="white", linewidth=0.5, label=d)
    ellipse(ax, s["pc1"], s["pc2"], PALETTE[d])
ax.set_xlabel(f"Overall pattern 1 ({ev[0]:.0f}% of all variation)")
ax.set_ylabel(f"Overall pattern 2 ({ev[1]:.0f}% of all variation)")
ax.axhline(0, color=GRID, lw=1); ax.axvline(0, color=GRID, lw=1)
ax.legend(frameon=False, fontsize=11, markerscale=1.4, loc="upper right")
sns.despine(ax=ax)
headline(fig, "Seen all at once, the two groups form two clouds with a narrow overlap",
         "All 30 measurements squeezed into 2 summary patterns. Dashed ovals show where roughly 86% of each group falls.", top=1.01)
fig.subplots_adjust(top=0.87)
footer(fig)
save(fig, "fig7_big_picture.png")

# ---------------------------------------------------------------- how many clues are enough?
log("\n=== HOW MANY CLUES ARE ENOUGH? (5-fold cross-validated accuracy, logistic regression) ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
sets = {
    "2 clues (worst radius, worst concave points)": ["worst radius", "worst concave points"],
    "5 strongest single clues": list(sep.head(5).index),
    "All 30 measurements": features,
}
for name, cols in sets.items():
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    sc = cross_val_score(model, df[cols], df["is_malignant"], cv=cv, scoring="accuracy")
    log(f"{name:<48s} accuracy {sc.mean() * 100:.1f}% (+/- {sc.std() * 100:.1f})")

df.drop(columns=["pc1", "pc2"]).to_csv("data/processed/breast_cancer_prepared.csv", index=False)
with open("reports/key_numbers.txt", "w") as fh:
    fh.write("\n".join(lines))
print("\nDone. Figures in 'figures/', numbers in 'reports/key_numbers.txt'.")
