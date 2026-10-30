// Builds Week3_Hypothesis_Testing_Report.docx from results/results.json and figures/*.png
// Usage: node build_report.js   (needs the "docx" npm package)
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, Footer, Header,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType, PageNumber,
  TabStopType, PageBreak,
} = require("docx");

const BASE = __dirname;
const R = JSON.parse(fs.readFileSync(path.join(BASE, "results", "results.json"), "utf8"));
const REPO = "https://github.com/SameeraPeer/Virtual-Data-Science-with-Python";
const LINK = `${REPO}/tree/main/week3_project`;
const AUTHOR = "Sameera Peer";
const DATE = "31 October 2026";

// ------------------------------------------------------------------ number helpers
const f = (x, d = 2) => Number(x).toFixed(d).replace("-", "\u2212");
const sci = (p) => (p <= 0 ? "< 1e-15" : p < 0.001 ? p.toExponential(1) : p.toFixed(3));
const pTxt = (p) => (p < 0.001 ? "p < 0.001" : `p = ${p.toFixed(3)}`);
const ci = (c, d = 2) => `[${f(c[0], d)}, ${f(c[1], d)}]`;

// ------------------------------------------------------------------ layout constants
const W = 9638; // content width in DXA (A4, 0.79" margins)
const FONT = "Calibri";
const NAVY = "1F3A5F";
const GREY = "52514E";
const LIGHT = "F3F5F8";
const BORDER = { style: BorderStyle.SINGLE, size: 4, color: "C9CED6" };
const BORDERS = { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER };

// ------------------------------------------------------------------ building blocks
function runs(text, base = {}) {
  // **bold** markup, everything else plain
  return String(text)
    .split("**")
    .map((part, i) => new TextRun({ text: part, bold: i % 2 === 1 || base.bold, font: FONT, ...base, ...(i % 2 === 1 ? { bold: true } : {}) }))
    .filter((r) => r);
}
const P = (text, opts = {}) =>
  new Paragraph({ spacing: { after: 120, line: 276 }, alignment: opts.align, keepNext: opts.keepNext, children: runs(text, opts.run || {}) });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, keepNext: true, children: [new TextRun({ text: t, font: FONT })] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, keepNext: true, children: [new TextRun({ text: t, font: FONT })] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, keepNext: true, children: [new TextRun({ text: t, font: FONT })] });
const bullets = (items) =>
  items.map((t) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 60, line: 276 }, children: runs(t) }));
const callout = (title, text) =>
  new Paragraph({
    spacing: { before: 120, after: 160, line: 276 },
    shading: { type: ShadingType.CLEAR, fill: "EAF1FB", color: "auto" },
    border: { left: { style: BorderStyle.SINGLE, size: 24, color: "2A78D6", space: 8 } },
    indent: { left: 200, right: 100 },
    children: [new TextRun({ text: title + " ", bold: true, font: FONT, color: NAVY }), ...runs(text)],
  });

function code(lines) {
  return lines.map(
    (ln, i) =>
      new Paragraph({
        spacing: { before: 0, after: 0, line: 240 },
        shading: { type: ShadingType.CLEAR, fill: "F4F4F2", color: "auto" },
        border: { left: { style: BorderStyle.SINGLE, size: 12, color: "9AA7B8", space: 6 } },
        indent: { left: 160, right: 80 },
        keepNext: i < lines.length - 1,
        children: [new TextRun({ text: ln === "" ? " " : ln, font: "Consolas", size: 17 })],
      })
  ).concat([new Paragraph({ spacing: { after: 120 }, children: [] })]);
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
let figNo = 0;
function figure(file, caption, alt) {
  figNo += 1;
  const full = path.join(BASE, "figures", file);
  const { w, h } = pngSize(full);
  const width = 600;
  const height = Math.round((width * h) / w);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(full), transformation: { width, height },
        altText: { title: `Figure ${figNo}`, description: alt, name: file } })],
    }),
    new Paragraph({
      spacing: { after: 200, line: 252 }, alignment: AlignmentType.LEFT,
      children: [new TextRun({ text: `Figure ${figNo}. `, bold: true, italics: true, size: 19, color: GREY, font: FONT }),
        new TextRun({ text: caption, italics: true, size: 19, color: GREY, font: FONT })],
    }),
  ];
}

function table(headers, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, w, o = {}) =>
    new TableCell({
      width: { size: w, type: WidthType.DXA }, borders: BORDERS,
      shading: o.fill ? { type: ShadingType.CLEAR, fill: o.fill, color: "auto" } : undefined,
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      children: [new Paragraph({ spacing: { after: 0, line: 250 }, alignment: o.align,
        children: runs(text, { size: opts.size || 19, color: o.color, bold: o.bold }) })],
    });
  const head = new TableRow({ tableHeader: true, cantSplit: true,
    children: headers.map((h, i) => cell(h, widths[i], { fill: NAVY, color: "FFFFFF", bold: true })) });
  const body = rows.map((r, ri) =>
    new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, widths[i], { fill: ri % 2 ? LIGHT : undefined })) }));
  return [new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: [head, ...body] }),
    new Paragraph({ spacing: { after: 160 }, children: [] })];
}

// ------------------------------------------------------------------ pull values
const H1r = R.H1, H2r = R.H2, H3r = R.H3, H4r = R.H4, H5r = R.H5, D = R.descriptives, holm = R.holm;
const ph = (cmp) => H2r.posthoc.find((x) => x.comparison === cmp);
const dsc = (v, c, k = "mean") => D[v][c][k];
const dTable = (v) => `${f(dsc(v, "A"))} (${f(dsc(v, "A", "sd"))})`;

const content = [];
const add = (...x) => x.flat().forEach((e) => content.push(e));

// ================================================================== TITLE BLOCK
add(
  new Paragraph({ spacing: { before: 200, after: 80 }, children: [new TextRun({ text: "Virtual Data Science with Python Internship", font: FONT, size: 24, color: GREY })] }),
  new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "Week 3: Statistical Analysis and Hypothesis Testing in Python", font: FONT, size: 48, bold: true, color: NAVY })] }),
  new Paragraph({ spacing: { after: 240 }, border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: "2A78D6", space: 6 } },
    children: [new TextRun({ text: "Can routine lab measurements tell three wine cultivars apart? Five hypotheses, tested with Welch t-tests, ANOVA, chi-square and correlation.", font: FONT, size: 24, italics: true, color: GREY })] }),
);
add(table(["Item", "Details"], [
  ["Prepared by", AUTHOR],
  ["Date", DATE],
  ["Dataset", "UCI Wine recognition dataset (178 wines, 3 cultivars, 13 chemical measurements), loaded from scikit-learn"],
  ["Tools", "Python, pandas, SciPy, statsmodels, scikit-learn, Matplotlib"],
  ["Project code", `${LINK}`],
], [2200, W - 2200]));

// ================================================================== EXECUTIVE SUMMARY
add(H1("Executive summary"));
add(P("A winery that can confirm which cultivar a batch came from, using a handful of routine lab measurements, saves time and money compared with a full chemical profile. This report tests whether five specific measurements really do differ between cultivars, using formal hypothesis tests with effect sizes and confidence intervals rather than eyeballing charts."));
add(...bullets([
  `**Alcohol (H1).** Cultivar A wines average ${f(H1r.diff)} units more alcohol than Cultivar B (95% CI ${ci(H1r.diff_ci)}). Welch t(${f(H1r.df, 1)}) = ${f(H1r.t)}, ${pTxt(H1r.p)}, Hedges' g = ${f(H1r.hedges_g)}.`,
  `**Flavanoids (H2).** They differ across the three cultivars (Welch F(2, ${f(H2r.welch_df[1], 1)}) = ${f(H2r.welch_F, 1)}, ${pTxt(H2r.welch_p)}, η² = ${f(H2r.eta_sq)}). Every pair differs, with the order A > B > C.`,
  `**Colour intensity (H3).** The share of "high colour" wines depends on cultivar: ${f(H3r.pct_high.A, 0)}% of A, ${f(H3r.pct_high.B, 0)}% of B and ${f(H3r.pct_high.C, 0)}% of C (χ²(${H3r.dof}) = ${f(H3r.chi2, 1)}, ${pTxt(H3r.p)}, Cramér's V = ${f(H3r.cramers_v)}).`,
  `**Flavanoids vs total phenols (H4).** Strong positive correlation overall (r = ${f(H4r.r)}), but within Cultivar C it is only ${f(H4r.within_cultivar.C.r)} and not significant. The overall figure is partly a between-cultivar effect.`,
  `**Ash (H5, exploratory control).** No evidence that A and C differ (${pTxt(H5r.p)}; difference ${f(H5r.diff, 3)}, 95% CI ${ci(H5r.diff_ci, 3)}). This is a failure to reject, not proof of equality.`,
]));
add(P("**Results at a glance** (α = 0.05, two-sided; Holm-adjusted p-values for the confirmatory family H1–H4):", { keepNext: true }));
add(table(["Hyp.", "Test", "Statistic", "Adjusted p", "Effect size (95% CI)", "Decision"], [
  ["H1", "Welch t-test", `t(${f(H1r.df, 1)}) = ${f(H1r.t)}`, sci(holm.H1.p_holm), `g = ${f(H1r.hedges_g)} ${ci(H1r.hedges_g_ci)}`, "Reject H0"],
  ["H2", "Welch ANOVA", `F(2, ${f(H2r.welch_df[1], 1)}) = ${f(H2r.welch_F, 1)}`, sci(holm.H2.p_holm), `η² = ${f(H2r.eta_sq)} ${ci(H2r.eta_sq_ci)}`, "Reject H0"],
  ["H3", "Chi-square", `χ²(${H3r.dof}) = ${f(H3r.chi2, 1)}`, sci(holm.H3.p_holm), `V = ${f(H3r.cramers_v)} ${ci(H3r.cramers_v_ci)}`, "Reject H0"],
  ["H4", "Pearson r", `r = ${f(H4r.r)}`, sci(holm.H4.p_holm), `r = ${f(H4r.r)} ${ci(H4r.ci)}`, "Reject H0"],
  ["H5", "Welch t-test", `t(${f(H5r.df, 1)}) = ${f(H5r.t)}`, `${f(H5r.p, 3)} (unadjusted)`, `g = ${f(H5r.hedges_g)} ${ci(H5r.hedges_g_ci)}`, "Fail to reject"],
], [600, 1300, 1900, 1450, 2800, 1588], { size: 18 }));
add(callout("Important caution.", "This is a well-known benchmark dataset in which the three cultivars are known to be chemically distinct, so strong results were expected. The value of this exercise is the method: checking assumptions, choosing robust tests, reporting effect sizes and intervals, and interpreting a non-significant result honestly. The data are observational, so the results describe association, not cause."));

// ================================================================== 1. INTRODUCTION
add(H1("1. Introduction and hypotheses"));
add(H2("1.1 Business problem"));
add(P("Wineries and wine-quality laboratories routinely need to authenticate a batch: is it really the cultivar on the label? A full chemical profile is expensive. If a few cheap measurements already separate the cultivars, the lab can screen faster and send only doubtful batches for full analysis. Before building any screening rule, the first question is statistical: are the differences we see in the data real, how large are they, and how sure are we?"));
add(H2("1.2 Hypotheses"));
add(P("Each hypothesis is stated as a null (H0, no effect) and an alternative (H1). All tests are two-sided at α = 0.05. H1–H4 form the confirmatory family; H5 is an exploratory control chosen to show what a non-significant result looks like (see Section 3.2 for how the variables were chosen)."));
add(table(["ID", "Question", "Null hypothesis (H0)", "Alternative (H1)", "Test"], [
  ["H1", "Do Cultivar A and B differ in alcohol content?", "Mean alcohol of A = mean alcohol of B", "The means differ", "Welch two-sample t-test"],
  ["H2", "Does flavanoid content differ across the 3 cultivars?", "All three mean flavanoid levels are equal", "At least one mean differs", "One-way ANOVA (Welch) + post-hoc"],
  ["H3", "Is colour-intensity level linked to cultivar?", "Colour level (low/high) is independent of cultivar", "Colour level depends on cultivar", "Chi-square test of independence"],
  ["H4", "Are flavanoids and total phenols linearly related?", "Population correlation ρ = 0", "ρ ≠ 0", "Pearson correlation"],
  ["H5", "Does ash differ between A and C? (exploratory)", "Mean ash of A = mean ash of C", "The means differ", "Welch two-sample t-test"],
], [550, 2400, 2350, 2000, 2338], { size: 18 }));

// ================================================================== 2. DATA
add(H1("2. Data description"));
add(P(`The data are the UCI Wine recognition dataset (Forina et al.), a chemical analysis of wines grown in the same region of Italy but derived from three different cultivars. It ships with scikit-learn (load_wine), so the analysis runs offline. There are **${R.data.rows} wines** and **${R.data.columns} continuous measurements**. The cultivar labels are anonymised in the source; this report calls them A, B and C (classes 0, 1, 2), with ${R.data.cultivar_counts.A}, ${R.data.cultivar_counts.B} and ${R.data.cultivar_counts.C} wines respectively.`));
add(P(`**Data quality.** There are ${R.data.missing_values} missing values and ${R.data.duplicate_rows} duplicate rows, so no cleaning was needed (cleaning was the Week 1 focus). The source does not document units in a machine-readable form; alcohol values of about 11–15 are consistent with percent by volume, and the other variables are reported in the dataset's own units. Because only differences and standardised effect sizes matter here, the analysis does not depend on the units.`));
add(P("**Variables used.** alcohol, flavanoids, total phenols, colour intensity and ash, split by cultivar:", { keepNext: true }));
const dRows = [["alcohol", "alcohol"], ["flavanoids", "flavanoids"], ["total phenols", "total_phenols"], ["colour intensity", "color_intensity"], ["ash", "ash"]]
  .map(([lab, v]) => [lab, `${f(dsc(v, "A"))} (${f(dsc(v, "A", "sd"))})`, `${f(dsc(v, "B"))} (${f(dsc(v, "B", "sd"))})`, `${f(dsc(v, "C"))} (${f(dsc(v, "C", "sd"))})`,
    `${f(dsc(v, "A", "min"))} – ${f(dsc(v, "A", "max"))}`, `${f(dsc(v, "B", "min"))} – ${f(dsc(v, "B", "max"))}`, `${f(dsc(v, "C", "min"))} – ${f(dsc(v, "C", "max"))}`]);
add(table(["Variable", `A mean (SD)\nn=${D.alcohol.A.n}`.replace("\n", " "), `B mean (SD) n=${D.alcohol.B.n}`, `C mean (SD) n=${D.alcohol.C.n}`, "A range", "B range", "C range"],
  dRows, [1500, 1300, 1300, 1300, 1400, 1400, 1438], { size: 17 }));
add(P("The table above is the starting point for every hypothesis: the group means already hint at the differences, and the ranges hint at overlap. The tests below quantify how much of that is real signal."));

// ================================================================== 3. METHODOLOGY
add(H1("3. Methodology"));
add(H2("3.1 Study design"));
add(P("This is an observational comparison of independent groups: each wine appears once and belongs to exactly one cultivar. That supports the independence assumption of all tests used, as long as the 178 wines are separate samples (we have no information suggesting repeated measurements). Nothing was randomised, so only association can be claimed."));
add(H2("3.2 Testing framework and its limits"));
add(...bullets([
  "**Significance level:** α = 0.05, two-sided for every test.",
  "**Always report three things:** the test statistic and p-value, an effect size, and a 95% confidence interval. A p-value alone says whether an effect is detectable, not whether it matters.",
  "**Honest note on variable choice.** I looked at the group means before settling on the variables. Alcohol, flavanoids and colour intensity were chosen because they are the classic discriminating measurements for this dataset, and ash was added as a control after the summary table showed A and C with nearly equal means. H5 is therefore labelled exploratory and is not part of the corrected family. A real study would pre-register the hypotheses before looking at any data.",
]));
add(H2("3.3 Choosing the test"));
add(table(["Situation", "Test used", "Assumptions checked", "If an assumption fails"], [
  ["Two independent means (H1, H5)", "Welch t-test", "Normality in each group (Shapiro–Wilk, Q-Q plot); independence", "Welch already allows unequal variances; confirm with Mann–Whitney and a permutation test"],
  ["Three or more means (H2)", "Welch ANOVA, with classic ANOVA shown for comparison", "Normality per group; equal variances (Levene, median-centred)", "Use Welch ANOVA, confirm with Kruskal–Wallis; post-hoc by Games–Howell (Tukey shown for comparison)"],
  ["Two categorical variables (H3)", "Chi-square test of independence", "Independent observations; expected counts of at least 5", "Would switch to Fisher's exact test"],
  ["Two continuous variables (H4)", "Pearson r with Fisher-z CI", "Roughly linear, no extreme outliers", "Confirm with Spearman rank correlation"],
], [2150, 2000, 2700, 2788], { size: 18 }));
add(H2("3.4 Effect sizes, intervals and power"));
add(...bullets([
  "**Hedges' g** (small-sample corrected standardised mean difference) for two-group comparisons. Rough guide: 0.2 small, 0.5 medium, 0.8 large.",
  "**η² and ω²** (share of variance explained by cultivar) for ANOVA, with a bootstrap CI for η².",
  "**Cramér's V** for the chi-square test, with a bootstrap CI.",
  `**Resampling checks for H1:** a permutation test (${H1r.perm_n.toLocaleString()} label shuffles) and a bootstrap CI (${H1r.boot_n.toLocaleString()} resamples), which do not rely on the t-distribution.`,
  "**Sensitivity (power) analysis:** for each test, the smallest effect that would be detected with 80% power at α = 0.05 given the sample sizes. This is more informative than \"post-hoc power\" computed from the observed effect.",
  "**Multiple testing:** the four confirmatory p-values are adjusted together with the Holm method. Post-hoc pairs inside H2 are adjusted by the post-hoc procedure itself.",
]));
add(H2("3.5 Reproducibility"));
add(P("A single script (week3_hypothesis_testing.py) loads the data, runs every test, saves the figures and writes all numbers to results/results.json. The random seed is fixed at 42, so the resampling results are repeatable. This report is generated from that JSON file, so every number in the text comes directly from the analysis output."));

// ================================================================== 4. RESULTS
add(H1("4. Analysis and results"));
add(H2("4.1 Checking the assumptions"));
add(P("Before trusting a t-test or ANOVA I checked normality within each group and equality of variances. Figure 1 shows the Q-Q plots: points close to the line mean the data look normal."));
add(figure("fig2_assumption_qq_plots.png", "Normal Q-Q plots with Shapiro–Wilk p-values. Alcohol (H1) is close to normal in both cultivars. Flavanoids are close to normal in Cultivar A but not in B (one extreme value) or C (right-skewed).", "Q-Q plots of alcohol and flavanoids by cultivar"));
add(table(["Variable / group", "n", "Shapiro–Wilk W", "p-value", "Looks normal?"], [
  ["Alcohol, A", H1r.n[0], f(H1r.shapiro[0].W, 3), f(H1r.shapiro[0].p, 3), "Yes"],
  ["Alcohol, B", H1r.n[1], f(H1r.shapiro[1].W, 3), f(H1r.shapiro[1].p, 3), "Yes"],
  ["Flavanoids, A", H2r.n[0], f(H2r.shapiro[0].W, 3), f(H2r.shapiro[0].p, 3), "Yes"],
  ["Flavanoids, B", H2r.n[1], f(H2r.shapiro[1].W, 3), f(H2r.shapiro[1].p, 4), "**No**"],
  ["Flavanoids, C", H2r.n[2], f(H2r.shapiro[2].W, 3), f(H2r.shapiro[2].p, 4), "**No**"],
  ["Ash, A", H5r.n[0], f(H5r.shapiro[0].W, 3), f(H5r.shapiro[0].p, 3), "Yes"],
  ["Ash, C", H5r.n[1], f(H5r.shapiro[1].W, 3), f(H5r.shapiro[1].p, 3), "Yes"],
], [2900, 900, 1900, 1700, 2238]));
add(P(`Levene's test for equal variances: alcohol (A vs B) p = ${f(H1r.levene_p, 3)}, so equal variances are plausible; flavanoids across the three cultivars p = ${sci(H2r.levene_p)}, so variances are **not** equal (SDs ${f(H2r.sd[0])}, ${f(H2r.sd[1])}, ${f(H2r.sd[2])}); ash (A vs C) p = ${f(H5r.levene_p, 3)}.`));
add(callout("Decision.", "Because flavanoids fail both normality (B, C) and equal variance, classic ANOVA is not the primary analysis for H2. I use Welch's ANOVA, confirm with the rank-based Kruskal–Wallis test, and use the Games–Howell post-hoc test. All of them are reported so the conclusion can be seen to be robust. For two-group tests I use Welch's t-test throughout, which is safe whether or not variances are equal."));

// ---------- H1
add(H2("4.2 H1: Alcohol content, Cultivar A vs B"));
add(P(`**Hypotheses.** H0: μ_A = μ_B. H1: μ_A ≠ μ_B. **Test:** Welch's two-sample t-test.`));
add(...code([
  'a = df.loc[df.cultivar == "A", "alcohol"]',
  'b = df.loc[df.cultivar == "B", "alcohol"]',
  '',
  'print(stats.shapiro(a), stats.shapiro(b))      # normality in each group',
  'print(stats.levene(a, b, center="median"))     # equal variances?',
  '',
  'res = stats.ttest_ind(a, b, equal_var=False)   # Welch t-test',
  'print(res.statistic, res.df, res.pvalue)',
  'print(res.confidence_interval(0.95))           # CI for the mean difference',
  'print(stats.mannwhitneyu(a, b))                # rank-based confirmation',
]));
add(figure("fig1_h1_alcohol_boxplot_ci.png", "Left: alcohol by cultivar (each dot is a wine; box = quartiles; diamond and bar = mean with 95% CI). Right: the estimated difference in means with its Welch and bootstrap 95% intervals; the dashed line marks zero (no difference).", "Box plot of alcohol by cultivar and CI of the difference"));
add(table(["Quantity", "Result"], [
  ["Group means (SD)", `A: ${f(H1r.mean[0])} (${f(H1r.sd[0])}), n = ${H1r.n[0]}   |   B: ${f(H1r.mean[1])} (${f(H1r.sd[1])}), n = ${H1r.n[1]}`],
  ["Welch t-test", `t(${f(H1r.df, 1)}) = ${f(H1r.t)}, p = ${sci(H1r.p)}`],
  ["Mean difference (A − B)", `${f(H1r.diff)}, 95% CI ${ci(H1r.diff_ci)}`],
  ["Effect size", `Hedges' g = ${f(H1r.hedges_g)}, 95% CI ${ci(H1r.hedges_g_ci)} (Cohen's d = ${f(H1r.cohen_d)})`],
  ["Mann–Whitney U (robustness)", `U = ${f(H1r.mannwhitney_U, 0)}, p = ${sci(H1r.mannwhitney_p)}`],
  ["Permutation test", `p ≈ ${f(H1r.perm_p, 4)} (${H1r.perm_n.toLocaleString()} shuffles; none was as extreme, so this is the smallest p the test can report)`],
  ["Bootstrap 95% CI of the difference", ci(H1r.boot_ci)],
  ["Smallest detectable effect (80% power)", `d = ${f(H1r.min_detectable_d_80pct_power)}`],
], [3000, W - 3000]));
add(P(`**Interpretation.** The data strongly contradict the null hypothesis. If A and B really had the same average alcohol, a difference as large as ${f(H1r.diff)} would almost never appear by chance (${pTxt(H1r.p)}). More useful than the p-value is the interval: the true mean difference is plausibly between ${f(H1r.diff_ci[0])} and ${f(H1r.diff_ci[1])} alcohol units, and the interval is nowhere near zero. The effect is very large (g = ${f(H1r.hedges_g)}, almost three pooled standard deviations). The rank-based test, permutation test and bootstrap all agree with the t-test. **H0 is rejected; H1 is supported.**`));
add(figure("fig6_h1_permutation_bootstrap.png", "Left: the distribution of mean differences when cultivar labels are shuffled at random (what \"no effect\" looks like); the observed difference lies far outside it. Right: the bootstrap distribution of the observed difference with its 95% interval.", "Permutation null distribution and bootstrap distribution for H1"));
add(P(`Figure 3 shows the logic of a p-value directly. The grey histogram is the world where cultivar does not matter. The real difference (${f(H1r.diff)}) sits far in the tail, which is why the permutation p-value hits its floor. The bootstrap histogram shows how much the estimate would wobble from sample to sample, which is exactly what the confidence interval summarises.`));

// ---------- H2
add(H2("4.3 H2: Flavanoids across the three cultivars"));
add(P("**Hypotheses.** H0: μ_A = μ_B = μ_C. H1: at least one mean differs. **Test:** Welch's one-way ANOVA with Games–Howell post-hoc comparisons."));
add(...code([
  'groups = [df.loc[df.cultivar == c, "flavanoids"].to_numpy() for c in "ABC"]',
  '',
  'stats.levene(*groups, center="median")                 # equal variances?',
  'stats.f_oneway(*groups)                                # classic ANOVA',
  'anova_oneway(groups, use_var="unequal")               # Welch ANOVA (primary)',
  'stats.kruskal(*groups)                                 # rank-based check',
  '',
  'eta2, omega2 = eta_omega(groups)                       # effect sizes',
  'posthoc = games_howell(groups, list("ABC"))           # pairwise, unequal variances',
  'pairwise_tukeyhsd(df.flavanoids, df.cultivar)         # Tukey for comparison',
]));
add(figure("fig3_h2_flavanoids_anova_posthoc.png", "Left: flavanoids by cultivar with means and 95% CIs. Right: pairwise differences in mean flavanoids with Games–Howell 95% intervals. None of the intervals touches zero.", "Flavanoids by cultivar and post-hoc confidence intervals"));
add(table(["Test", "Result"], [
  ["Classic ANOVA", `F(2, ${H2r.anova_df[1]}) = ${f(H2r.anova_F, 1)}, p = ${sci(H2r.anova_p)}`],
  ["Welch ANOVA (primary)", `F(2, ${f(H2r.welch_df[1], 1)}) = ${f(H2r.welch_F, 1)}, p = ${sci(H2r.welch_p)}`],
  ["Kruskal–Wallis", `H(2) = ${f(H2r.kruskal_H, 1)}, p = ${sci(H2r.kruskal_p)}`],
  ["Effect size", `η² = ${f(H2r.eta_sq)} (95% bootstrap CI ${ci(H2r.eta_sq_ci)}); ω² = ${f(H2r.omega_sq)}; Cohen's f = ${f(H2r.cohen_f)}`],
  ["Smallest detectable effect (80% power)", `Cohen's f = ${f(H2r.min_detectable_f_80pct_power)}`],
], [3000, W - 3000]));
add(P("**Post-hoc comparisons** (difference = later group minus earlier group):", { keepNext: true }));
add(table(["Comparison", "Mean diff", "Games–Howell 95% CI", "GH adj. p", "Tukey adj. p"],
  H2r.posthoc.map((x) => [x.comparison, f(x.diff), ci([x.lower, x.upper]), sci(x.p_adj), sci(x.tukey_p)]),
  [2000, 1500, 2700, 1700, 1738]));
add(P(`**Interpretation.** The three cultivars do not share the same average flavanoid level. All three omnibus tests (classic ANOVA, Welch ANOVA and Kruskal–Wallis) reject H0, so the conclusion does not hinge on any one assumption. Cultivar explains about ${f(H2r.eta_sq * 100, 0)}% of the variation in flavanoids (η² = ${f(H2r.eta_sq)}, CI ${ci(H2r.eta_sq_ci)}), which is a very large effect. The post-hoc table shows that every pair differs: B is ${f(Math.abs(ph("B - A").diff))} lower than A, C is ${f(Math.abs(ph("C - A").diff))} lower than A, and C is ${f(Math.abs(ph("C - B").diff))} lower than B. Games–Howell and Tukey agree on every pair; Tukey's p-values and intervals are slightly different because Tukey assumes equal variances, which this data does not satisfy. Flavanoid ranges also barely overlap between the extremes: A ranges ${f(dsc("flavanoids", "A", "min"))}–${f(dsc("flavanoids", "A", "max"))} and C ranges ${f(dsc("flavanoids", "C", "min"))}–${f(dsc("flavanoids", "C", "max"))}, so no A wine looks like a C wine on this measure. **H0 is rejected; H1 is supported.**`));
add(P("The p-values here are so small that the software reports them as 0 for some pairs; they are shown as \"< 1e-15\", meaning smaller than the software can represent. What matters is that the intervals sit far from zero."));

// ---------- H3
add(H2("4.4 H3: Colour intensity level and cultivar"));
add(P(`**Hypotheses.** H0: colour level and cultivar are independent. H1: they are associated. Colour intensity is continuous, so I split it at the overall median (${f(H3r.median_split, 2)}) into "Low" and "High" to create a 3 × 2 table. **Test:** chi-square test of independence.`));
add(...code([
  'median_ci = df.color_intensity.median()',
  'df["color_level"] = np.where(df.color_intensity > median_ci, "High", "Low")',
  'table = pd.crosstab(df.cultivar, df.color_level)[["Low", "High"]]',
  '',
  'chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)',
  'assert expected.min() >= 5                       # chi-square validity check',
  'v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))   # Cramer\'s V',
  '',
  '# adjusted standardised residuals show WHERE the association comes from',
  'adj = (table - expected) / np.sqrt(expected * (1 - row_p) * (1 - col_p))',
]));
add(figure("fig4_h3_chi_square.png", "Left: share of Low and High colour wines within each cultivar. Right: adjusted residuals (observed minus expected, standardised). Violet cells have more wines than expected under independence; gold cells have fewer. Values beyond about ±2 are notable.", "Stacked bars and residual heatmap for colour level by cultivar"));
const ob = H3r.observed, ex = H3r.expected, rs = H3r.adj_residuals;
add(table(["Cultivar", "Low (obs / exp)", "High (obs / exp)", "% High", "Adj. residual (Low / High)"],
  ["A", "B", "C"].map((c) => [c, `${ob.Low[c]} / ${f(ex.Low[c], 1)}`, `${ob.High[c]} / ${f(ex.High[c], 1)}`, `${f(H3r.pct_high[c], 0)}%`, `${f(rs.Low[c], 1)} / ${f(rs.High[c], 1)}`]),
  [1300, 2100, 2100, 1500, 2638]));
add(P(`**Result.** χ²(${H3r.dof}) = ${f(H3r.chi2, 1)}, p = ${sci(H3r.p)}; Cramér's V = ${f(H3r.cramers_v)} (95% bootstrap CI ${ci(H3r.cramers_v_ci)}). The smallest expected count is ${f(H3r.min_expected, 1)}, well above 5, so the chi-square approximation is valid. The smallest effect detectable with 80% power is w = ${f(H3r.min_detectable_w_80pct_power)}.`));
add(P(`**Interpretation.** Colour level is strongly associated with cultivar. Cultivar B is overwhelmingly low-colour (${f(100 - H3r.pct_high.B, 0)}% low), while A (${f(H3r.pct_high.A, 0)}% high) and C (${f(H3r.pct_high.C, 0)}% high) are mostly high-colour. The residuals show this clearly: B has far more low-colour wines than independence would predict (residual ${f(rs.Low.B, 1)}), and A and C have more high-colour wines than expected. The association is large (V = ${f(H3r.cramers_v)}). Colour intensity therefore helps separate B from the other two, but the chi-square test alone does not say whether A and C differ from each other: their high-colour shares (${f(H3r.pct_high.A, 0)}% vs ${f(H3r.pct_high.C, 0)}%) are closer, and I did not test that pair. **H0 is rejected; H1 is supported.** One cost of the median split is lost information, because a wine just above or just below the median is treated as completely different. A comparison of the continuous values would be more powerful and is a natural follow-up.`));

// ---------- H4
add(H2("4.5 H4: Flavanoids and total phenols"));
add(P("**Hypotheses.** H0: ρ = 0. H1: ρ ≠ 0. **Test:** Pearson correlation, with a confidence interval from Fisher's z transformation and a Spearman rank correlation as a check."));
add(...code([
  'r, p = stats.pearsonr(df.flavanoids, df.total_phenols)',
  'z, se = np.arctanh(r), 1 / np.sqrt(len(df) - 3)',
  'ci = np.tanh([z - 1.96 * se, z + 1.96 * se])            # Fisher-z 95% CI',
  'rho, p_s = stats.spearmanr(df.flavanoids, df.total_phenols)',
  '',
  'for c, g in df.groupby("cultivar"):                      # within-cultivar check',
  '    print(c, stats.pearsonr(g.flavanoids, g.total_phenols))',
]));
add(figure("fig5_h4_correlation.png", "Left: flavanoids against total phenols with the fitted line and its 95% confidence band; colour shows cultivar. Right: the correlation coefficient with 95% CI, for all wines and within each cultivar.", "Scatter plot with regression line and forest plot of correlations"));
add(table(["Quantity", "Result"], [
  ["Pearson correlation, all 178 wines", `r = ${f(H4r.r)}, 95% CI ${ci(H4r.ci)}, p = ${sci(H4r.p)}`],
  ["Variance shared", `R² = ${f(H4r.r_squared)}`],
  ["Spearman rank correlation", `ρ = ${f(H4r.spearman_rho)}, p = ${sci(H4r.spearman_p)}`],
  ["Fitted line", `total phenols = ${f(H4r.intercept)} + ${f(H4r.slope)} × flavanoids`],
  ["Within Cultivar A", `r = ${f(H4r.within_cultivar.A.r)}, 95% CI ${ci(H4r.within_cultivar.A.ci)}, p = ${sci(H4r.within_cultivar.A.p)}`],
  ["Within Cultivar B", `r = ${f(H4r.within_cultivar.B.r)}, 95% CI ${ci(H4r.within_cultivar.B.ci)}, p = ${sci(H4r.within_cultivar.B.p)}`],
  ["Within Cultivar C", `r = ${f(H4r.within_cultivar.C.r)}, 95% CI ${ci(H4r.within_cultivar.C.ci)}, p = ${f(H4r.within_cultivar.C.p, 3)}`],
], [3400, W - 3400]));
add(P(`**Interpretation.** Across all wines the relationship is strong and positive: a wine higher in flavanoids tends to be higher in total phenols (r = ${f(H4r.r)}, about ${f(H4r.r_squared * 100, 0)}% shared variance), and the Spearman correlation (${f(H4r.spearman_rho)}) agrees, so a few outliers are not driving it. **H0 is rejected; H1 is supported.** The within-cultivar results add an important nuance. In A and B the correlation stays strong (${f(H4r.within_cultivar.A.r)} and ${f(H4r.within_cultivar.B.r)}), but in C it drops to ${f(H4r.within_cultivar.C.r)} and its interval ${ci(H4r.within_cultivar.C.ci)} includes zero. Part of the overall correlation therefore comes from the cultivars occupying different regions of the plot (C low on both, A high on both), not only from a wine-by-wine relationship. Also, flavanoids are one family of phenolic compounds, so some positive relationship with total phenols is built in by the chemistry. The correlation shows association only.`));

// ---------- H5
add(H2("4.6 H5 (exploratory control): Ash, Cultivar A vs C"));
add(P("**Hypotheses.** H0: μ_A = μ_C. H1: μ_A ≠ μ_C. **Test:** Welch's t-test. This is deliberately a case where I expected no clear difference, to show how to read a non-significant result."));
add(figure("fig7_h5_ash_control.png", "Left: ash by cultivar. Right: the estimated difference with its 95% interval, which comfortably includes zero.", "Ash by cultivar and CI of the difference"));
add(table(["Quantity", "Result"], [
  ["Group means (SD)", `A: ${f(H5r.mean[0])} (${f(H5r.sd[0])}), n = ${H5r.n[0]}   |   C: ${f(H5r.mean[1])} (${f(H5r.sd[1])}), n = ${H5r.n[1]}`],
  ["Welch t-test", `t(${f(H5r.df, 1)}) = ${f(H5r.t)}, p = ${f(H5r.p, 3)}`],
  ["Mean difference (A − C)", `${f(H5r.diff, 3)}, 95% CI ${ci(H5r.diff_ci, 3)}`],
  ["Effect size", `Hedges' g = ${f(H5r.hedges_g)}, 95% CI ${ci(H5r.hedges_g_ci)}`],
  ["Mann–Whitney U (robustness)", `p = ${f(H5r.mannwhitney_p, 3)}`],
  ["Smallest detectable effect (80% power)", `d = ${f(H5r.min_detectable_d_80pct_power)}`],
], [3400, W - 3400]));
add(P(`**Interpretation.** There is no evidence that average ash differs between A and C (${pTxt(H5r.p)}). **We fail to reject H0.** That is not the same as proving the cultivars have equal ash. The confidence interval ${ci(H5r.diff_ci, 3)} says the true difference could plausibly be anywhere from slightly negative to slightly positive, and the standardised interval for g (${ci(H5r.hedges_g_ci)}) still allows small-to-medium effects. With these sample sizes the test could only reliably detect effects of about d = ${f(H5r.min_detectable_d_80pct_power)} or larger. The safe wording is "no detectable difference", and ash would not be a useful screening measurement for telling A from C.`));

// ---------- Summary table
add(H2("4.7 Multiple testing across the confirmatory family"));
add(P("Running four confirmatory tests raises the chance of at least one false positive. I applied the Holm step-down correction to the four main p-values (H1, H2 Welch, H3, H4). Every result remains significant, because the raw p-values are many orders of magnitude below 0.05."));
add(...code([
  'fam = {"H1": p1, "H2": p2_welch, "H3": p3, "H4": p4}',
  'reject, p_holm, _, _ = multipletests(list(fam.values()), alpha=0.05, method="holm")',
]));
add(table(["Hypothesis", "Raw p", "Holm-adjusted p", "Reject H0 at 0.05?"],
  ["H1", "H2", "H3", "H4"].map((k) => [k, sci(holm[k].p_raw), sci(holm[k].p_holm), holm[k].reject ? "Yes" : "No"]),
  [2000, 2400, 2800, 2438]));

// ================================================================== 5. INTERPRETATION
add(H1("5. Interpretation and validity of the hypotheses"));
add(H2("5.1 Reading p-values and confidence intervals correctly"));
add(...bullets([
  "**A p-value** is the probability of seeing a result at least as extreme as ours if the null hypothesis were true. It is not the probability that the null hypothesis is true, and it says nothing about how large or important the effect is.",
  "**A 95% confidence interval** comes from a procedure that captures the true value in 95% of repeated samples. In practice it is a range of values consistent with the data. If it excludes zero, the effect is statistically significant at α = 0.05; its width shows how precisely the effect is estimated.",
  "**Not significant does not mean no effect.** H5 shows this: the interval is compatible with no difference and with small ones.",
  "**Statistical vs practical significance.** With 178 wines, even modest differences can reach tiny p-values, so the effect sizes (g, η², V) are what tell us the differences here are also large in practical terms.",
]));
add(H2("5.2 Verdict on each hypothesis"));
add(table(["Hypothesis", "Verdict", "Evidence", "What it means for the winery"], [
  ["H1 Alcohol A vs B", "Supported", `Very strong; g = ${f(H1r.hedges_g)}, CI of difference ${ci(H1r.diff_ci)}`, "Alcohol separates A from B well on average, but individual ranges still overlap, so it is a screening clue rather than a rule."],
  ["H2 Flavanoids by cultivar", "Supported", `Very strong; η² = ${f(H2r.eta_sq)}; all pairs differ`, "The most informative single measurement here; cleanly ranks the three cultivars."],
  ["H3 Colour level vs cultivar", "Supported", `Very strong; V = ${f(H3r.cramers_v)}`, "Quickly flags Cultivar B (mostly low colour) from A and C."],
  ["H4 Flavanoids ~ total phenols", "Supported", `Strong overall (r = ${f(H4r.r)}); weak within C (r = ${f(H4r.within_cultivar.C.r)})`, "The two measurements overlap in information overall, so measuring both adds less than expected for some cultivars."],
  ["H5 Ash A vs C", "Not supported (inconclusive)", `p = ${f(H5r.p, 3)}; CI includes zero`, "Ash is not a useful A-vs-C screen with this data."],
], [2000, 1500, 2700, 3438], { size: 18 }));
add(H2("5.3 Business and scientific implications"));
add(...bullets([
  "**Prioritise flavanoids and colour intensity in a routine panel.** Together they split the cultivars: flavanoids rank A > B > C, and colour intensity isolates B.",
  "**Alcohol is a cheap supporting measurement,** especially for A vs B, but should not be used alone because ranges overlap.",
  "**Do not spend lab time on ash for this question,** at least for A vs C.",
  "**Avoid double counting.** Because flavanoids and total phenols move together, a screening rule gains little from using both.",
  "**Next step:** turn these findings into a validated classifier (for example logistic regression or a decision tree) tested on wines it has not seen. A significant group difference does not by itself guarantee good individual-level prediction.",
]));

// ================================================================== 6. LIMITATIONS
add(H1("6. Limitations"));
add(...bullets([
  "**One dataset.** 178 wines from one region, with anonymised cultivars. Results may not hold for other regions or vintages.",
  "**Observational data.** Differences are associations. Growing conditions, vintage or harvest timing may be tangled up with cultivar.",
  "**Exploratory variable choice.** I chose the variables after seeing summary statistics, and H5 especially so. A pre-registered design would be stronger.",
  "**Assumption violations.** Flavanoids are not normal in B and C, and variances are unequal. Robust and rank-based tests agree with the primary analysis, which mitigates this.",
  "**Median split in H3** loses information, and the A-vs-C colour comparison was not tested separately.",
  "**Pooled correlation in H4** mixes between-group and within-group relationships, as the within-cultivar analysis shows.",
  "**Very small p-values** are partly a product of large effects and the chosen tests; they should not be over-read as certainty. The effect sizes and intervals carry the real message.",
]));

// ================================================================== 7. CONCLUSION
add(H1("7. Conclusion"));
add(P(`Four of the five hypotheses were supported and one was inconclusive. Alcohol differs between Cultivars A and B (by ${f(H1r.diff)} units, 95% CI ${ci(H1r.diff_ci)}), flavanoids differ across all three cultivars (η² = ${f(H2r.eta_sq)}), colour intensity level is strongly associated with cultivar (V = ${f(H3r.cramers_v)}), and flavanoids track total phenols (r = ${f(H4r.r)}), although only weakly within Cultivar C. Ash showed no detectable difference between A and C, and the data cannot rule out a small one. The analysis also demonstrated good statistical practice: assumptions were checked rather than assumed, tests were chosen to match the data, results were confirmed by independent methods (rank-based, permutation and bootstrap), effect sizes and intervals were reported with every p-value, and multiple testing was controlled.`));

// ================================================================== 8. REPRODUCIBILITY
add(H1("8. Reproducibility and project files"));
add(P(`**GitHub folder:** ${LINK}`));
add(P("Repository layout (Week 3 sits in its own folder, next to the earlier weeks):", { keepNext: true }));
add(...code([
  'Virtual-Data-Science-with-Python/',
  '|-- week1_titanic_eda.py ... (Week 1)',
  '|-- week2_project/        (Week 2)',
  '`-- week3_project/',
  '    |-- week3_hypothesis_testing.py   # runs every test, makes all figures',
  '    |-- build_report.js               # builds the Word report from the results',
  '    |-- Week3_Hypothesis_Testing_Report.docx',
  '    |-- requirements.txt',
  '    |-- README.md',
  '    |-- data/wine.csv                 # copy of the dataset used',
  '    |-- figures/                      # 7 PNG figures used in this report',
  '    `-- results/                      # results.json, summary.txt, CSV tables',
]));
add(P("To reproduce the analysis:", { keepNext: true }));
add(...code([
  'cd week3_project',
  'pip install -r requirements.txt',
  'python week3_hypothesis_testing.py',
]));

// ================================================================== build
const doc = new Document({
  creator: AUTHOR,
  title: "Week 3: Statistical Analysis and Hypothesis Testing in Python",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: FONT, color: NAVY }, paragraph: { spacing: { before: 360, after: 140 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: FONT, color: "2A78D6" }, paragraph: { spacing: { before: 260, after: 100 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: FONT, color: NAVY }, paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, right: 1134, bottom: 1134, left: 1134 } } },
    footers: {
      default: new Footer({ children: [new Paragraph({ tabStops: [{ type: TabStopType.RIGHT, position: W }],
        border: { top: { style: BorderStyle.SINGLE, size: 4, color: "C9CED6", space: 4 } },
        children: [new TextRun({ text: "Week 3 – Statistical Analysis and Hypothesis Testing", size: 17, color: GREY, font: FONT }),
          new TextRun({ children: ["\tPage ", PageNumber.CURRENT], size: 17, color: GREY, font: FONT })] })] }),
    },
    children: content,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(BASE, "Week3_Hypothesis_Testing_Report.docx");
  fs.writeFileSync(out, buf);
  console.log("Wrote", out, `(${(buf.length / 1024).toFixed(0)} KB)`);
});
