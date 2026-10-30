"""
Week 3 - Statistical Analysis and Hypothesis Testing in Python
===============================================================
Virtual Data Science with Python internship

Domain   : Wine chemistry / quality control (winery authentication problem)
Dataset  : UCI Wine recognition dataset (178 wines, 3 cultivars, 13 chemical
           measurements). It ships inside scikit-learn, so no download is needed.

Business question
-----------------
Can a winery tell its three cultivars apart from routine lab measurements?
Which measurements genuinely differ between cultivars, and which do not?

Hypotheses (alpha = 0.05 for every test)
----------------------------------------
H1  Alcohol content differs between Cultivar A and Cultivar B     (Welch t-test)
H2  Mean flavanoids differ across the three cultivars             (ANOVA + post-hoc)
H3  Colour intensity (low/high) is associated with cultivar       (chi-square)
H4  Flavanoids and total phenols are linearly related             (Pearson correlation)
H5  Exploratory control: ash differs between Cultivar A and C     (Welch t-test)

H1-H4 are the confirmatory family (Holm-corrected). H5 is exploratory.

Run:  python week3_hypothesis_testing.py
Outputs: figures/*.png, results/results.json, results/summary.txt, results/*.csv
"""

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.datasets import load_wine
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.power import FTestAnovaPower, GofChisquarePower, TTestIndPower
import statsmodels.api as sm

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
SEED = 42
ALPHA = 0.05
N_PERM = 10_000
N_BOOT = 10_000
BASE = Path(__file__).resolve().parent
FIG_DIR = BASE / "figures"
RES_DIR = BASE / "results"
DATA_DIR = BASE / "data"
for d in (FIG_DIR, RES_DIR, DATA_DIR):
    d.mkdir(exist_ok=True)

rng = np.random.default_rng(SEED)

# Chart styling (validated categorical slots 1-3: blue, orange, aqua)
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e1"
COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "font.size": 10,
    }
)

R = {}  # every number that ends up in the report is stored here


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def fmt_p(p):
    """Format a p-value for humans (floating-point underflow shows up as exactly 0)."""
    if p < 1e-15:
        return "<1e-15"
    if p < 1e-4:
        return f"{p:.1e}"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def mean_ci(x, level=0.95):
    x = np.asarray(x, float)
    m, se = x.mean(), stats.sem(x)
    lo, hi = stats.t.interval(level, len(x) - 1, loc=m, scale=se)
    return m, lo, hi


def hedges_g(a, b):
    """Cohen's d, Hedges' g and an approximate 95% CI for g."""
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / sp
    g = d * (1 - 3 / (4 * (n1 + n2) - 9))
    se = np.sqrt((n1 + n2) / (n1 * n2) + g**2 / (2 * (n1 + n2)))
    return d, g, (g - 1.96 * se, g + 1.96 * se)


def shapiro(x):
    w, p = stats.shapiro(x)
    return {"W": float(w), "p": float(p), "n": int(len(x))}


def eta_omega(groups):
    allv = np.concatenate(groups)
    gm, k, n = allv.mean(), len(groups), len(allv)
    ssb = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    sst = ssb + ssw
    msw = ssw / (n - k)
    return ssb / sst, (ssb - (k - 1) * msw) / (sst + msw)


def games_howell(groups, names, alpha=ALPHA):
    """Games-Howell post-hoc test (does not assume equal variances).
    Difference convention: later group minus earlier group (same as Tukey)."""
    k, rows = len(groups), []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = groups[i], groups[j]
            n1, n2 = len(a), len(b)
            v1, v2 = a.var(ddof=1), b.var(ddof=1)
            se = np.sqrt(v1 / n1 + v2 / n2)
            df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
            diff = b.mean() - a.mean()
            q = abs(diff) / (se / np.sqrt(2))
            p = float(stats.studentized_range.sf(q, k, df))
            half = float(stats.studentized_range.ppf(1 - alpha, k, df)) * se / np.sqrt(2)
            rows.append(
                {"comparison": f"{names[j]} - {names[i]}", "diff": diff, "lower": diff - half,
                 "upper": diff + half, "p_adj": p, "df": df}
            )
    return pd.DataFrame(rows)


def cramers_v(table):
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def fisher_ci(r, n, level=0.95):
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    crit = stats.norm.ppf(0.5 + level / 2)
    return float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))


def permutation_test(a, b, n_perm=N_PERM):
    obs = a.mean() - b.mean()
    pooled, n1 = np.concatenate([a, b]), len(a)
    perm = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(pooled)
        perm[i] = pooled[:n1].mean() - pooled[n1:].mean()
    p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, perm, float(p)


def bootstrap_diff(a, b, n_boot=N_BOOT):
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()
    return diffs, (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def welch_block(a, b, label_a, label_b):
    """Full two-sample analysis: assumptions, Welch t, Mann-Whitney, effect size."""
    res = stats.ttest_ind(a, b, equal_var=False)
    ci = res.confidence_interval(0.95)
    d, g, g_ci = hedges_g(a, b)
    mw = stats.mannwhitneyu(a, b, alternative="two-sided")
    lev = stats.levene(a, b, center="median")
    n1, n2 = len(a), len(b)
    d_min = TTestIndPower().solve_power(nobs1=n1, ratio=n2 / n1, alpha=ALPHA, power=0.8)
    return {
        "groups": [label_a, label_b],
        "n": [n1, n2],
        "mean": [float(a.mean()), float(b.mean())],
        "sd": [float(a.std(ddof=1)), float(b.std(ddof=1))],
        "shapiro": [shapiro(a), shapiro(b)],
        "levene_p": float(lev.pvalue),
        "t": float(res.statistic),
        "df": float(res.df),
        "p": float(res.pvalue),
        "diff": float(a.mean() - b.mean()),
        "diff_ci": [float(ci.low), float(ci.high)],
        "cohen_d": float(d),
        "hedges_g": float(g),
        "hedges_g_ci": [float(g_ci[0]), float(g_ci[1])],
        "mannwhitney_U": float(mw.statistic),
        "mannwhitney_p": float(mw.pvalue),
        "min_detectable_d_80pct_power": float(d_min),
    }


def strip_box(ax, groups, labels, colors, ylabel):
    """Box + jittered points + mean with 95% CI, direct labels."""
    for i, (g, c) in enumerate(zip(groups, colors), start=1):
        ax.boxplot(g, positions=[i], widths=0.45, showfliers=False, manage_ticks=False,
                   boxprops=dict(color=INK2, lw=1), whiskerprops=dict(color=INK2, lw=1),
                   capprops=dict(color=INK2, lw=1), medianprops=dict(color=INK2, lw=1.6))
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(g)), g, s=16, color=c, alpha=0.65,
                   edgecolor=SURFACE, linewidth=0.4, zorder=3)
        m, lo, hi = mean_ci(g)
        ax.vlines(i + 0.34, lo, hi, color=INK, lw=2, zorder=4)
        ax.scatter([i + 0.34], [m], marker="D", s=34, color=INK, edgecolor=SURFACE, zorder=5)
        ax.text(i + 0.42, m, f"mean {m:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([f"{l}\n(n={len(g)})" for l, g in zip(labels, groups)])
    ax.set_xlim(0.4, len(groups) + 1.0)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)


def forest(ax, rows, xlabel, zero=True):
    """rows: list of (label, estimate, lo, hi, colour)."""
    for i, (lab, est, lo, hi, c) in enumerate(rows[::-1]):
        ax.hlines(i, lo, hi, color=c, lw=2.4)
        ax.scatter([est], [i], color=c, s=46, edgecolor=SURFACE, zorder=4)
        ax.text(est, i + 0.22, f"{est:.2f}  [{lo:.2f}, {hi:.2f}]", fontsize=8.5, color=INK2, ha="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]])
    ax.set_ylim(-0.6, len(rows) - 0.3)
    if zero:
        ax.axvline(0, color=INK2, lw=1, ls="--")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=200)
    plt.close(fig)
    print("  saved", name)


# ----------------------------------------------------------------------------
# 0. Data loading and quality checks
# ----------------------------------------------------------------------------
print("0. Loading data")
raw = load_wine(as_frame=True).frame
df = raw.copy()
df["cultivar"] = df["target"].map({0: "A", 1: "B", 2: "C"})
df = df.drop(columns="target")
df.to_csv(DATA_DIR / "wine.csv", index=False)

R["data"] = {
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1] - 1),
    "missing_values": int(df.isna().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
    "cultivar_counts": {k: int(v) for k, v in df["cultivar"].value_counts().sort_index().items()},
}
KEY_VARS = ["alcohol", "flavanoids", "total_phenols", "color_intensity", "ash"]
desc = df.groupby("cultivar")[KEY_VARS].agg(["count", "mean", "std", "median", "min", "max"]).round(3)
desc.to_csv(RES_DIR / "descriptives_by_cultivar.csv")

A = {c: df.loc[df.cultivar == c] for c in "ABC"}

# ----------------------------------------------------------------------------
# H1  Alcohol: Cultivar A vs B  (Welch t-test)
# ----------------------------------------------------------------------------
print("H1. Alcohol, A vs B")
a1, b1 = A["A"]["alcohol"].to_numpy(), A["B"]["alcohol"].to_numpy()
R["descriptives"] = {
    v: {c: {"n": int(len(A[c])), "mean": float(A[c][v].mean()), "sd": float(A[c][v].std(ddof=1)),
            "median": float(A[c][v].median()), "min": float(A[c][v].min()), "max": float(A[c][v].max())}
        for c in "ABC"} for v in KEY_VARS}
R["H1"] = welch_block(a1, b1, "A", "B")
obs, perm, p_perm = permutation_test(a1, b1)
boot, boot_ci = bootstrap_diff(a1, b1)
R["H1"].update({"perm_p": p_perm, "perm_n": N_PERM, "boot_ci": list(boot_ci), "boot_n": N_BOOT})

# ----------------------------------------------------------------------------
# H2  Flavanoids across A, B, C (ANOVA)
# ----------------------------------------------------------------------------
print("H2. Flavanoids across cultivars")
groups = [A[c]["flavanoids"].to_numpy() for c in "ABC"]
f_cl = stats.f_oneway(*groups)
welch_anova = anova_oneway(groups, use_var="unequal")
kw = stats.kruskal(*groups)
lev2 = stats.levene(*groups, center="median")
eta2, omega2 = eta_omega(groups)
eta_boot = [eta_omega([rng.choice(g, len(g)) for g in groups])[0] for _ in range(5000)]
f_eff = float(np.sqrt(eta2 / (1 - eta2)))
f_min = FTestAnovaPower().solve_power(nobs=len(df), alpha=ALPHA, power=0.8, k_groups=3)
gh = games_howell(groups, list("ABC"))
tk = pairwise_tukeyhsd(df["flavanoids"], df["cultivar"], alpha=ALPHA)
tk_tab = pd.DataFrame(tk.summary().data[1:], columns=tk.summary().data[0])
tk_tab["group1"], tk_tab["group2"] = tk_tab["group1"].astype(str), tk_tab["group2"].astype(str)
for c in ("meandiff", "p-adj", "lower", "upper"):
    tk_tab[c] = tk_tab[c].astype(float)
tk_tab["comparison"] = tk_tab["group2"] + " - " + tk_tab["group1"]
posthoc = gh.merge(
    tk_tab[["comparison", "meandiff", "p-adj", "lower", "upper"]].rename(
        columns={"meandiff": "tukey_diff", "p-adj": "tukey_p", "lower": "tukey_lower", "upper": "tukey_upper"}),
    on="comparison")
posthoc.to_csv(RES_DIR / "h2_posthoc.csv", index=False)

R["H2"] = {
    "n": [len(g) for g in groups],
    "mean": [float(g.mean()) for g in groups],
    "sd": [float(g.std(ddof=1)) for g in groups],
    "shapiro": [shapiro(g) for g in groups],
    "levene_p": float(lev2.pvalue),
    "levene_stat": float(lev2.statistic),
    "anova_F": float(f_cl.statistic),
    "anova_df": [2, int(len(df) - 3)],
    "anova_p": float(f_cl.pvalue),
    "welch_F": float(welch_anova.statistic),
    "welch_df": [float(welch_anova.df_num), float(welch_anova.df_denom)],
    "welch_p": float(welch_anova.pvalue),
    "kruskal_H": float(kw.statistic),
    "kruskal_p": float(kw.pvalue),
    "eta_sq": float(eta2),
    "eta_sq_ci": [float(np.percentile(eta_boot, 2.5)), float(np.percentile(eta_boot, 97.5))],
    "omega_sq": float(omega2),
    "cohen_f": f_eff,
    "min_detectable_f_80pct_power": float(f_min),
    "posthoc": posthoc.to_dict(orient="records"),
}

# ----------------------------------------------------------------------------
# H3  Colour intensity (low/high) vs cultivar (chi-square test of independence)
# ----------------------------------------------------------------------------
print("H3. Colour intensity vs cultivar")
median_ci = float(df["color_intensity"].median())
df["color_level"] = np.where(df["color_intensity"] > median_ci, "High", "Low")
table = pd.crosstab(df["cultivar"], df["color_level"])[["Low", "High"]]
table.to_csv(RES_DIR / "h3_contingency.csv")
chi2, p3, dof, expected = stats.chi2_contingency(table.to_numpy(), correction=False)
n3 = table.to_numpy().sum()
rowp = table.sum(axis=1).to_numpy()[:, None] / n3
colp = table.sum(axis=0).to_numpy()[None, :] / n3
adj_res = (table.to_numpy() - expected) / np.sqrt(expected * (1 - rowp) * (1 - colp))
codes = df["cultivar"].map({"A": 0, "B": 1, "C": 2}).to_numpy()
is_high = (df["color_level"] == "High").astype(int).to_numpy()
v_boot = []
for _ in range(5000):  # bootstrap the wines, rebuild the 3x2 table, recompute Cramer's V
    idx = rng.integers(0, len(df), len(df))
    t = np.bincount(codes[idx] * 2 + is_high[idx], minlength=6).reshape(3, 2)  # cols: Low, High
    v_boot.append(cramers_v(t))
w_min = GofChisquarePower().solve_power(nobs=n3, alpha=ALPHA, power=0.8, n_bins=dof + 1)
R["H3"] = {
    "median_split": median_ci,
    "observed": table.to_dict(),
    "expected": pd.DataFrame(expected, index=table.index, columns=table.columns).round(2).to_dict(),
    "min_expected": float(expected.min()),
    "chi2": float(chi2), "dof": int(dof), "p": float(p3),
    "cramers_v": cramers_v(table.to_numpy()),
    "cramers_v_ci": [float(np.percentile(v_boot, 2.5)), float(np.percentile(v_boot, 97.5))],
    "adj_residuals": pd.DataFrame(adj_res, index=table.index, columns=table.columns).round(2).to_dict(),
    "min_detectable_w_80pct_power": float(w_min),
    "pct_high": {c: float(100 * table.loc[c, "High"] / table.loc[c].sum()) for c in "ABC"},
}

# ----------------------------------------------------------------------------
# H4  Flavanoids vs total phenols (Pearson correlation)
# ----------------------------------------------------------------------------
print("H4. Flavanoids vs total phenols")
x4, y4 = df["flavanoids"].to_numpy(), df["total_phenols"].to_numpy()
r, p4 = stats.pearsonr(x4, y4)
rho, p4s = stats.spearmanr(x4, y4)
within = {}
for c in "ABC":
    xs, ys = A[c]["flavanoids"].to_numpy(), A[c]["total_phenols"].to_numpy()
    rc, pc = stats.pearsonr(xs, ys)
    within[c] = {"r": float(rc), "p": float(pc), "n": len(xs), "ci": list(fisher_ci(rc, len(xs)))}
R["H4"] = {
    "n": int(len(df)),
    "r": float(r), "p": float(p4), "ci": list(fisher_ci(r, len(df))),
    "r_squared": float(r**2),
    "spearman_rho": float(rho), "spearman_p": float(p4s),
    "within_cultivar": within,
}
ols = sm.OLS(y4, sm.add_constant(x4)).fit()
R["H4"]["slope"], R["H4"]["intercept"] = float(ols.params[1]), float(ols.params[0])

# ----------------------------------------------------------------------------
# H5  Exploratory control: ash, Cultivar A vs C
# ----------------------------------------------------------------------------
print("H5. Ash, A vs C (exploratory)")
a5, c5 = A["A"]["ash"].to_numpy(), A["C"]["ash"].to_numpy()
R["H5"] = welch_block(a5, c5, "A", "C")

# ----------------------------------------------------------------------------
# Multiple-testing control (Holm) on the confirmatory family H1-H4
# ----------------------------------------------------------------------------
fam = {"H1": R["H1"]["p"], "H2": R["H2"]["welch_p"], "H3": R["H3"]["p"], "H4": R["H4"]["p"]}
rej, p_holm, _, _ = multipletests(list(fam.values()), alpha=ALPHA, method="holm")
R["holm"] = {k: {"p_raw": float(v), "p_holm": float(ph), "reject": bool(rj)}
             for (k, v), ph, rj in zip(fam.items(), p_holm, rej)}

# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
print("Figures")

# Fig 1 - H1 alcohol A vs B
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
strip_box(ax1, [a1, b1], ["Cultivar A", "Cultivar B"], [COLORS["A"], COLORS["B"]], "Alcohol")
ax1.set_title("H1: alcohol content, Cultivar A vs B")
h1 = R["H1"]
forest(ax2, [("Welch t-test\n95% CI", h1["diff"], *h1["diff_ci"], COLORS["A"]),
             ("Bootstrap\n95% CI", h1["diff"], *h1["boot_ci"], INK2)],
       "Mean difference in alcohol, A - B")
ax2.set_title("Size of the difference")
ax2.set_xlim(-0.15, max(h1["diff_ci"][1], h1["boot_ci"][1]) * 1.3)
save(fig, "fig1_h1_alcohol_boxplot_ci.png")

# Fig 2 - assumption checks (Q-Q plots)
fig, axes = plt.subplots(2, 3, figsize=(10, 6.2))
panels = [("Alcohol, Cultivar A", a1, COLORS["A"]), ("Alcohol, Cultivar B", b1, COLORS["B"]), None,
          ("Flavanoids, Cultivar A", groups[0], COLORS["A"]), ("Flavanoids, Cultivar B", groups[1], COLORS["B"]),
          ("Flavanoids, Cultivar C", groups[2], COLORS["C"])]
for ax, pn in zip(axes.ravel(), panels):
    if pn is None:
        ax.axis("off")
        continue
    title, x, c = pn
    (osm, osr), (slope, intercept, _) = stats.probplot(x, dist="norm")
    ax.scatter(osm, osr, s=14, color=c, alpha=0.75, edgecolor=SURFACE, linewidth=0.4, zorder=3)
    ax.plot(osm, slope * osm + intercept, color=INK2, lw=1.2)
    sw = shapiro(x)
    ax.set_title(f"{title}\nShapiro-Wilk p = {fmt_p(sw['p'])}", fontsize=9.5)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
save(fig, "fig2_assumption_qq_plots.png")

# Fig 3 - H2 flavanoids by cultivar + post-hoc CIs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
strip_box(ax1, groups, ["Cultivar A", "Cultivar B", "Cultivar C"], [COLORS[c] for c in "ABC"], "Flavanoids")
ax1.set_title("H2: flavanoids by cultivar")
rows = [(r_["comparison"], r_["diff"], r_["lower"], r_["upper"], INK) for r_ in R["H2"]["posthoc"]]
forest(ax2, rows, "Mean difference in flavanoids (Games-Howell 95% CI)")
ax2.set_title("Which pairs differ?")
ax2.set_xlim(min(r_[2] for r_ in rows) * 1.15, max(0.35, max(r_[3] for r_ in rows) * 1.3))
save(fig, "fig3_h2_flavanoids_anova_posthoc.png")

# Fig 4 - H3 chi-square
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.4))
pct = table.div(table.sum(axis=1), axis=0) * 100
ax1.barh(range(3), pct["Low"], color="#9aa7b8", edgecolor=SURFACE, linewidth=2, height=0.55, label="Low colour")
ax1.barh(range(3), pct["High"], left=pct["Low"], color=[COLORS[c] for c in "ABC"], edgecolor=SURFACE,
         linewidth=2, height=0.55, label="High colour")
for i, c in enumerate("ABC"):
    lo_, hi_ = pct.loc[c, "Low"], pct.loc[c, "High"]
    ax1.text(lo_ / 2, i, f"Low {lo_:.0f}%" if lo_ >= 25 else f"{lo_:.0f}%", ha="center", va="center", fontsize=9)
    ax1.text(lo_ + hi_ / 2, i, f"High {hi_:.0f}%" if hi_ >= 25 else f"{hi_:.0f}%", ha="center",
             va="center", fontsize=9, color="white" if c == "A" else INK)
ax1.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#9aa7b8", label="Low colour (grey)"),
                    plt.Rectangle((0, 0), 1, 1, color=INK2, label="High colour (cultivar colour)")],
           frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5)
ax1.set_yticks(range(3))
ax1.set_yticklabels([f"Cultivar {c}\n(n={int(table.loc[c].sum())})" for c in "ABC"])
ax1.invert_yaxis()
ax1.set_xlabel("Share of wines in the cultivar (%)", labelpad=2)
ax1.set_title("H3: colour intensity level by cultivar")
ax1.grid(axis="y", visible=False)
cmap = LinearSegmentedColormap.from_list("div", ["#eda100", "#eeeeec", "#6250d6"])
lim = np.abs(adj_res).max()
im = ax2.imshow(adj_res, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
for i in range(3):
    for j in range(2):
        v = adj_res[i, j]
        ax2.text(j, i, f"{v:+.1f}\n(obs {table.iloc[i, j]}, exp {expected[i, j]:.0f})", ha="center", va="center",
                 fontsize=8.5, color="white" if v > 0.55 * lim else INK)
ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Low", "High"])
ax2.set_yticks(range(3)); ax2.set_yticklabels(list("ABC"))
ax2.set_ylabel("Cultivar")
ax2.set_title("Adjusted residuals (violet = above expected)")
ax2.grid(False)
for s in ax2.spines.values():
    s.set_visible(False)
fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03)
save(fig, "fig4_h3_chi_square.png")

# Fig 5 - H4 correlation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
for c in "ABC":
    ax1.scatter(A[c]["flavanoids"], A[c]["total_phenols"], s=22, color=COLORS[c], alpha=0.75,
                edgecolor=SURFACE, linewidth=0.5, label=f"Cultivar {c}", zorder=3)
xs = np.linspace(x4.min(), x4.max(), 100)
pred = ols.get_prediction(sm.add_constant(xs)).summary_frame(alpha=0.05)
ax1.fill_between(xs, pred["mean_ci_lower"], pred["mean_ci_upper"], color=INK2, alpha=0.18, lw=0)
ax1.plot(xs, pred["mean"], color=INK, lw=1.6)
ax1.set_xlabel("Flavanoids")
ax1.set_ylabel("Total phenols")
ax1.set_title("H4: flavanoids vs total phenols")
ax1.legend(frameon=False, loc="upper left", fontsize=9)
ax1.text(0.98, 0.05, f"r = {r:.2f}, R$^2$ = {r**2:.2f}", transform=ax1.transAxes, ha="right", fontsize=9.5)
rows = [("All wines", r, *R["H4"]["ci"], INK)] + [
    (f"Cultivar {c}", within[c]["r"], *within[c]["ci"], COLORS[c]) for c in "ABC"]
forest(ax2, rows, "Pearson r (95% CI, Fisher z)", zero=False)
ax2.set_xlim(-0.1, 1.15)
ax2.set_title("Overall vs within each cultivar")
save(fig, "fig5_h4_correlation.png")

# Fig 6 - permutation null + bootstrap distribution (H1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.3))
ax1.hist(perm, bins=50, color="#9aa7b8", edgecolor=SURFACE, linewidth=0.5)
ax1.axvline(obs, color=COLORS["B"], lw=2)
ax1.axvline(-obs, color=COLORS["B"], lw=2, ls="--")
ax1.text(obs, ax1.get_ylim()[1] * 0.92, " observed\n difference", color=INK, fontsize=9, va="top", ha="right")
ax1.set_xlabel("Mean difference under H0 (labels shuffled)")
ax1.set_ylabel("Count of shuffles")
ax1.set_title(f"Permutation test ({N_PERM:,} shuffles)")
ax1.grid(axis="x", visible=False)
ax2.hist(boot, bins=50, color=COLORS["A"], alpha=0.7, edgecolor=SURFACE, linewidth=0.5)
ax2.axvline(boot_ci[0], color=INK, lw=1.6, ls="--")
ax2.axvline(boot_ci[1], color=INK, lw=1.6, ls="--")
ax2.set_ylim(0, ax2.get_ylim()[1] * 1.15)
ax2.text(np.mean(boot_ci), ax2.get_ylim()[1] * 0.97, "95% bootstrap CI", ha="center", fontsize=9, va="top")
ax2.set_xlabel("Bootstrapped mean difference in alcohol, A - B")
ax2.set_ylabel("Count of resamples")
ax2.set_title(f"Bootstrap distribution ({N_BOOT:,} resamples)")
ax2.grid(axis="x", visible=False)
save(fig, "fig6_h1_permutation_bootstrap.png")

# Fig 7 - H5 exploratory control
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
strip_box(ax1, [a5, c5], ["Cultivar A", "Cultivar C"], [COLORS["A"], COLORS["C"]], "Ash")
ax1.set_title("H5 (exploratory): ash, Cultivar A vs C")
h5 = R["H5"]
forest(ax2, [("Welch t-test\n95% CI", h5["diff"], *h5["diff_ci"], COLORS["A"])], "Mean difference in ash, A - C")
ax2.set_xlim(-0.2, 0.2)
ax2.set_title("The CI includes zero")
save(fig, "fig7_h5_ash_control.png")

# ----------------------------------------------------------------------------
# Save results
# ----------------------------------------------------------------------------
class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


(RES_DIR / "results.json").write_text(json.dumps(R, indent=2, cls=NpEncoder))

L = []
L.append("WEEK 3 - HYPOTHESIS TESTING SUMMARY (alpha = 0.05)")
L.append("=" * 60)
d = R["data"]
L.append(f"Data: {d['rows']} wines, {d['columns']} measurements, missing = {d['missing_values']}, "
         f"duplicates = {d['duplicate_rows']}, cultivars = {d['cultivar_counts']}")
L.append("")
h = R["H1"]
L.append(f"H1 Alcohol A vs B: A={h['mean'][0]:.2f}, B={h['mean'][1]:.2f}; Welch t({h['df']:.1f})={h['t']:.2f}, "
         f"p={fmt_p(h['p'])}; diff={h['diff']:.2f} CI[{h['diff_ci'][0]:.2f},{h['diff_ci'][1]:.2f}]; "
         f"g={h['hedges_g']:.2f}; MWU p={fmt_p(h['mannwhitney_p'])}; perm p={h['perm_p']:.4f}")
h = R["H2"]
L.append(f"H2 Flavanoids: Levene p={fmt_p(h['levene_p'])}; ANOVA F(2,{h['anova_df'][1]})={h['anova_F']:.1f}, "
         f"p={fmt_p(h['anova_p'])}; Welch F={h['welch_F']:.1f}, p={fmt_p(h['welch_p'])}; "
         f"KW H={h['kruskal_H']:.1f}, p={fmt_p(h['kruskal_p'])}; eta2={h['eta_sq']:.2f}, omega2={h['omega_sq']:.2f}")
for ph in h["posthoc"]:
    L.append(f"    {ph['comparison']}: diff={ph['diff']:.2f}, GH p={fmt_p(ph['p_adj'])}, Tukey p={fmt_p(ph['tukey_p'])}")
h = R["H3"]
L.append(f"H3 Colour vs cultivar: chi2({h['dof']})={h['chi2']:.1f}, p={fmt_p(h['p'])}, V={h['cramers_v']:.2f}, "
         f"min expected={h['min_expected']:.1f}, % high={ {k: round(v) for k, v in h['pct_high'].items()} }")
h = R["H4"]
L.append(f"H4 Flavanoids~phenols: r={h['r']:.2f} CI[{h['ci'][0]:.2f},{h['ci'][1]:.2f}], p={fmt_p(h['p'])}; "
         f"rho={h['spearman_rho']:.2f}; within: " + ", ".join(f"{c}={v['r']:.2f}" for c, v in h['within_cultivar'].items()))
h = R["H5"]
L.append(f"H5 Ash A vs C (exploratory): Welch t({h['df']:.1f})={h['t']:.2f}, p={fmt_p(h['p'])}; "
         f"diff={h['diff']:.3f} CI[{h['diff_ci'][0]:.3f},{h['diff_ci'][1]:.3f}]; g={h['hedges_g']:.2f}; "
         f"min detectable d={h['min_detectable_d_80pct_power']:.2f}")
L.append("")
L.append("Holm-corrected family H1-H4: " + "; ".join(
    f"{k}: p_holm={fmt_p(v['p_holm'])} reject={v['reject']}" for k, v in R["holm"].items()))
(RES_DIR / "summary.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
print("\nDone. Figures in", FIG_DIR, "- results in", RES_DIR)
