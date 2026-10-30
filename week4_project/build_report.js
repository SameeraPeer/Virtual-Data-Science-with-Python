const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun, Table, TableRow, TableCell,
  WidthType, ShadingType, AlignmentType, BorderStyle, LevelFormat, Footer, PageNumber, PageBreak,
} = require("docx");

const R = JSON.parse(fs.readFileSync("reports/results.json", "utf8"));
const NAME = "Sameera P";
const DATE = "31 October 2026";
const REPO = "https://github.com/SameeraPeer/Virtual-Data-Science-with-Python/tree/main/week4_project";
const f3 = (x) => Number(x).toFixed(3);
const pct = (x) => (100 * x).toFixed(1) + "%";
const M = Object.fromEntries(R.model_table.map((r) => [r.model, r]));
const LR = M["Logistic regression (tuned)"], DT = M["Decision tree (tuned)"], RF = M["Random forest (300 trees)"], BASE = M["Majority-class baseline"];
const SS = R.split_stability;

const W = 9638; // content width in DXA (A4, 2 cm margins)
const FONT = "Calibri";

const p = (text, opts = {}) => new Paragraph({ spacing: { after: 120, line: 276 }, ...opts,
  children: Array.isArray(text) ? text : [new TextRun({ text, ...(opts.run || {}) })] });
const b = (t) => new TextRun({ text: t, bold: true });
const mix = (...parts) => p(parts.map((x) => (typeof x === "string" ? new TextRun(x) : x)));
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const bullet = (parts) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 60, line: 276 },
  children: (Array.isArray(parts) ? parts : [parts]).map((x) => (typeof x === "string" ? new TextRun(x) : x)) });

function code(lines) {
  return lines.map((l, i) => new Paragraph({
    spacing: { after: 0, line: 240 },
    shading: { type: ShadingType.CLEAR, fill: "F3F4F6" },
    border: i === 0 ? { top: { style: BorderStyle.SINGLE, size: 4, color: "D1D5DB", space: 4 } }
      : i === lines.length - 1 ? { bottom: { style: BorderStyle.SINGLE, size: 4, color: "D1D5DB", space: 4 } } : undefined,
    children: [new TextRun({ text: l === "" ? " " : l, font: "Consolas", size: 18 })],
  })).concat([new Paragraph({ spacing: { after: 120 }, children: [] })]);
}

function img(file, caption, maxW = 620) {
  const buf = fs.readFileSync(file);
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const width = maxW, height = Math.round((maxW * h) / w);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: buf, transformation: { width, height },
        altText: { title: caption, description: caption, name: path.basename(file) } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: caption, italics: true, size: 20, color: "444444" })] }),
  ];
}

function table(headers, rows, widths, opts = {}) {
  const border = { style: BorderStyle.SINGLE, size: 4, color: "BFC5CC" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (t, i, head, rowIdx) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: head ? { type: ShadingType.CLEAR, fill: "1F3A5F" } : (rowIdx % 2 ? { type: ShadingType.CLEAR, fill: "F5F7FA" } : undefined),
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ alignment: i === 0 || opts.leftAll ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: [new TextRun({ text: String(t), bold: head, size: 20, color: head ? "FFFFFF" : "000000" })] })],
  });
  return [new Table({
    width: { size: widths.reduce((a, c) => a + c, 0), type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true, 0)) }),
      ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false, ri)) }))],
  }), new Paragraph({ spacing: { after: 160 }, children: [] })];
}

// ---------------------------------------------------------------- content
const kids = [];

// Title block
kids.push(
  new Paragraph({ spacing: { before: 1800, after: 120 }, alignment: AlignmentType.LEFT,
    children: [new TextRun({ text: "Virtual Data Science with Python Internship", size: 26, color: "555555" })] }),
  new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: "Week 4", bold: true, size: 64, color: "1F3A5F" })] }),
  new Paragraph({ spacing: { after: 360 }, border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: "1F3A5F", space: 8 } },
    children: [new TextRun({ text: "Machine Learning Model Development and Evaluation", size: 40, color: "1F3A5F" })] }),
  p("Recognising handwritten digits with logistic regression, decision trees and a random forest", { run: { size: 26, italics: true } }),
  new Paragraph({ spacing: { before: 600, after: 60 }, children: [b("Prepared by: "), new TextRun(NAME)] }),
  new Paragraph({ spacing: { after: 60 }, children: [b("Date: "), new TextRun(DATE)] }),
  new Paragraph({ spacing: { after: 60 }, children: [b("Dataset: "), new TextRun("Optical Recognition of Handwritten Digits (bundled with scikit-learn)")] }),
  new Paragraph({ spacing: { after: 60 }, children: [b("Code repository: "), new TextRun({ text: REPO, color: "1155CC" })] }),
  new Paragraph({ children: [new PageBreak()] }),
);

// 1 Executive summary
kids.push(h1("1. Summary"));
kids.push(p(`This report builds and evaluates a machine learning model that reads handwritten digits. Each image is only 8 by 8 pixels, and the model must say which digit from 0 to 9 it shows. The work covers the full pipeline: data checks, preparation, algorithm choice, training, evaluation, error analysis and suggestions for improvement.`));
kids.push(p("Main findings:"));
kids.push(bullet([b("Primary model: "), `logistic regression reached ${pct(LR.test_acc)} accuracy on ${R.n_test} held-out images (${R.lr_errors} errors). A bootstrap 95% interval for this figure is ${pct(R.lr_acc_ci95[0])} to ${pct(R.lr_acc_ci95[1])}, so the true accuracy on new data could reasonably be a little lower or higher.`]));
kids.push(bullet([b("Baseline: "), `always guessing the most common digit scores only ${pct(BASE.test_acc)}, so the model has learned real structure.`]));
kids.push(bullet([b("Comparison: "), `a random forest scored ${pct(RF.test_acc)} on the test set (${R.rf_errors} errors) and a decision tree only ${pct(DT.test_acc)} (${R.tree_errors} errors). Logistic regression and the random forest are very close to each other; the tree is clearly worse.`]));
kids.push(bullet([b("Overfitting: "), `the decision tree memorises the training data (${pct(DT.train_acc)} training accuracy) but drops to ${pct(DT.cv_acc)} under cross-validation. Logistic regression shows a small gap (${pct(LR.train_acc)} on training data, ${pct(LR.cv_acc)} under cross-validation).`]));
kids.push(bullet([b("Where it fails: "), `most mistakes involve the digits 1 and 8 (recall ${pct(R.lr_per_class["1"].recall)} and ${pct(R.lr_per_class["8"].recall)}). The most common single error is an 8 read as a 1.`]));

// 2 Intro
kids.push(h1("2. Introduction and Objective"));
kids.push(p("Week 4 asks for a basic machine learning model that is built and validated carefully, with the whole process documented. The aim is not just a high score. The aim is to show that each step is sound, that the result is measured honestly, and that the weaknesses of the model are understood."));
kids.push(p("The problem chosen is multi-class image classification. It suits this task because the correct answer is known for every image, the classes are nearly balanced, and the errors are easy to look at and explain. It is also a different dataset from the earlier weeks (Titanic, breast cancer measurements and wine chemistry)."));
kids.push(h2("2.1 Questions this report answers"));
kids.push(bullet("Is the data ready for modelling, and what preparation does it need?"));
kids.push(bullet("Which simple algorithm is a sensible choice, and how does it compare with alternatives?"));
kids.push(bullet("How well does the model perform on data it has never seen, measured with several metrics?"));
kids.push(bullet("Is any model overfitting or underfitting, and what evidence shows it?"));
kids.push(bullet("What causes the remaining errors, and how could the model be improved?"));

// 3 Data
kids.push(h1("3. Data Acquisition and Description"));
kids.push(p(`The dataset is the Optical Recognition of Handwritten Digits data from the UCI Machine Learning Repository. It ships inside scikit-learn (the function load_digits), so it can be loaded with one line of code and the project runs without an internet connection. It contains ${R.n_samples} images. Each image is an 8 by 8 grid of pixels, giving ${R.n_features} numeric features. Each pixel value is an integer from ${R.pixel_min} (blank) to ${R.pixel_max} (fully inked). The target is the digit from 0 to 9.`));
kids.push(...code([
  "from sklearn.datasets import load_digits",
  "digits = load_digits()",
  "X = pd.DataFrame(digits.data, columns=[f\"pix_{i}\" for i in range(64)])",
  "y = pd.Series(digits.target, name=\"digit\")",
]));
kids.push(...img("figures/fig1_data_overview.png", "Figure 1. The data: example images (left), images per digit (middle) and average pixel intensity (right)."));
kids.push(p(`Figure 1 shows three things. The example images are coarse, so some digits are hard to tell apart even for a person. The classes are nearly balanced, with between ${R.class_min} and ${R.class_max} images per digit, so plain accuracy is a fair metric and no resampling is needed. The average intensity map shows that the ink sits in the centre of the grid and the outer columns are almost always blank.`));

// 4 Prep
kids.push(h1("4. Data Preparation"));
kids.push(p("Each check below was run in code, and the result decides what, if anything, needs fixing."));
kids.push(...table(["Check", "Result", "Decision and reason"], [
  ["Missing values", String(R.missing_values), "Nothing to impute. Imputation would only add noise."],
  ["Duplicate rows", String(R.duplicate_rows), "Nothing to remove."],
  ["Data type", `${R.dtype}, range ${R.pixel_min} to ${R.pixel_max}`, "Already numeric. No encoding needed."],
  ["Constant pixels", `${R.constant_pixels.length} (${R.constant_pixels.join(", ")})`, "These pixels are always 0 and carry no information. A variance filter inside the pipeline removes them."],
  ["Class balance", `${R.class_min} to ${R.class_max} per digit`, "Balanced, so use stratified splitting to keep proportions."],
  ["Test rows also in training set", String(R.test_rows_duplicated_in_train), "No exact copies cross the split, so test accuracy is not inflated by memorised images."],
], [2300, 2600, 4738], { leftAll: true }));
kids.push(p("This dataset was already clean, so the preparation work is mostly about doing the remaining steps correctly and without data leakage:"));
kids.push(bullet([b("Stratified 80/20 split. "), `${R.n_train} images for training and ${R.n_test} for the final test. The test set is used once, at the end. All tuning uses cross-validation inside the training set.`]));
kids.push(bullet([b("Scaling and pixel filtering inside a Pipeline. "), "The scaler and the constant-pixel filter are fitted on the training folds only. Fitting them on all the data first would leak information from the test images into training."]));
kids.push(bullet([b("Scaling for the right models only. "), "Logistic regression is sensitive to feature scale, so it gets standardised inputs. Trees and forests split on thresholds and do not need scaling."]));
kids.push(...code([
  "X_tr, X_te, y_tr, y_te = train_test_split(",
  "    X, y, test_size=0.20, stratify=y, random_state=42)",
  "",
  "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
  "",
  "def lr_pipe(C=1.0):",
  "    return Pipeline([",
  "        (\"drop_constant\", VarianceThreshold(0.0)),",
  "        (\"scale\", StandardScaler()),",
  "        (\"clf\", LogisticRegression(C=C, max_iter=3000))])",
]));

// 5 Model selection
kids.push(h1("5. Algorithm Selection and Justification"));
kids.push(p("Four models were trained so that the main choice can be judged against alternatives rather than in isolation."));
kids.push(...table(["Model", "Why it is included"], [
  ["Majority-class baseline", "Always predicts the most frequent digit. Any real model must beat it clearly. It sets the floor."],
  ["Logistic regression (primary)", "Chosen in advance as the main model. It is simple, fast, needs few settings (mainly the regularisation strength C) and its learned weights can be inspected. Pixel intensities relate to digit shape in a roughly linear way, so a linear model is a reasonable first attempt."],
  ["Decision tree", "Very interpretable, and a good example for demonstrating overfitting. Included both unpruned and with settings tuned by cross-validation."],
  ["Random forest (300 trees)", "Averaging many trees reduces the instability of a single tree. It acts as a stronger non-linear comparison to see whether extra complexity pays off."],
], [3000, 6638], { leftAll: true }));
kids.push(p(`Logistic regression was fixed as the primary model before any test results were seen. The choice rests on the task brief (a simple algorithm) and on interpretability, not on which model scored highest.`));

// 6 Training
kids.push(h1("6. Training and Tuning"));
kids.push(p("Hyperparameters were chosen by 5-fold stratified cross-validation on the training set only. The grids were small on purpose, because large grids on a small dataset tend to fit the noise in the validation folds."));
kids.push(...code([
  "gs_lr = GridSearchCV(lr_pipe(), {\"clf__C\": [0.001, 0.01, 0.1, 1, 10, 100]},",
  "                     cv=cv, scoring=\"accuracy\").fit(X_tr, y_tr)",
  "",
  "gs_tree = GridSearchCV(DecisionTreeClassifier(random_state=42),",
  "    {\"max_depth\": [2, 3, 4, 5, 6, 8, 10, 12, 15, None],",
  "     \"min_samples_leaf\": [1, 5]}, cv=cv, scoring=\"accuracy\").fit(X_tr, y_tr)",
]));
const gsRows = Object.entries(R.gs_lr_cv_scores).map(([c, s]) => [c, f3(s), Number(c) === R.best_C ? "Selected" : ""]);
kids.push(p(`For logistic regression the best value was C = ${R.best_C}. Very small C values over-regularise the model (it underfits), and very large values give slightly lower scores again, which is a sign of mild overfitting.`));
kids.push(...table(["C (inverse regularisation strength)", "Mean CV accuracy", ""], gsRows, [4200, 3000, 2438]));
kids.push(p(`For the decision tree the search chose max_depth = ${R.best_tree_params.max_depth} and min_samples_leaf = ${R.best_tree_params.min_samples_leaf}. This is effectively an unpruned tree, so the tuned and unpruned trees give identical results in every table in this report. In other words, within the grid I searched, restricting the tree did not help.`));

// 7 Evaluation
kids.push(h1("7. Evaluation"));
kids.push(h2("7.1 Metrics used and why"));
kids.push(bullet([b("Accuracy: "), "the share of images classified correctly. Appropriate here because the classes are balanced."]));
kids.push(bullet([b("Macro precision, recall and F1: "), "averaged over the ten digits with equal weight, so a weak digit cannot hide behind strong ones."]));
kids.push(bullet([b("Confusion matrix: "), "shows exactly which digits are mistaken for which."]));
kids.push(bullet([b("ROC curve and AUC (one-vs-rest): "), "measures how well the predicted probabilities rank the right digit above the others, independent of a single decision threshold."]));
kids.push(h2("7.2 Results"));
const order = ["Majority-class baseline", "Logistic regression (tuned)", "Decision tree (unpruned)", "Decision tree (tuned)", "Random forest (300 trees)"];
kids.push(...table(["Model", "Train acc.", "CV acc. (±SD)", "Test acc.", "Macro F1", "Macro AUC"],
  order.map((n) => { const r = M[n]; return [n, f3(r.train_acc), `${f3(r.cv_acc)} ± ${f3(r.cv_sd)}`, f3(r.test_acc), f3(r.macro_f1), f3(r.macro_auc)]; }),
  [2900, 1150, 1700, 1150, 1350, 1388]));
kids.push(...img("figures/fig2_model_comparison.png", "Figure 2. Accuracy of each model on training data, in cross-validation and on the held-out test set."));
kids.push(p(`Figure 2 is the most important chart for judging generalisation. For a healthy model the three bars are close together. For the decision tree the training bar is at 1.000 while the other two are much lower, which is the signature of overfitting. Logistic regression and the random forest have small gaps and are the strongest models. The random forest has a slightly higher cross-validation score (${f3(RF.cv_acc)} against ${f3(LR.cv_acc)}), but the difference is smaller than the fold-to-fold spread (about ${f3(LR.cv_sd)}), and on the test set logistic regression is ahead by one image (${R.lr_errors} errors against ${R.rf_errors}). The two cannot be separated reliably with this much data.`));
kids.push(...img("figures/fig3_confusion_matrices.png", "Figure 3. Confusion matrices on the test set: logistic regression (left) and the tuned decision tree (right)."));
kids.push(p(`In Figure 3 the diagonal holds the correct predictions. The logistic regression matrix has almost nothing off the diagonal, with ${R.lr_errors} mistakes in ${R.n_test} images. The tree matrix is visibly messier, with ${R.tree_errors} mistakes spread over many digit pairs.`));
kids.push(...img("figures/fig4_roc_curves.png", "Figure 4. One-vs-rest ROC curves. Left: logistic regression for each digit. Right: micro-averaged ROC for three models. Both axes are zoomed to the top-left corner.", 620));
kids.push(p(`The ROC curves hug the top-left corner for logistic regression and the random forest (macro AUC ${f3(LR.macro_auc)} and ${f3(RF.macro_auc)}). The tree's curve is much lower (macro AUC ${f3(DT.macro_auc)}) because a single tree gives coarse probabilities. Note that the plots are zoomed in, because on the full axes the good models would be indistinguishable. Also note that an AUC close to 1.0 on this dataset does not mean the problem is trivial; it only says the ranking of probabilities is almost always right, and the model still makes ${R.lr_errors} wrong final predictions.`));
kids.push(h2("7.3 Per-digit performance of the logistic regression"));
kids.push(...table(["Digit", "Precision", "Recall", "F1", "Test images", "AUC"],
  Object.keys(R.lr_per_class).map((d) => { const c = R.lr_per_class[d]; return [d, f3(c.precision), f3(c.recall), f3(c["f1-score"]), String(c.support), f3(R.lr_auc_per_class[d])]; }),
  [1300, 1700, 1700, 1700, 1700, 1538]));
kids.push(p(`Seven digits have an F1 of 0.985 or higher. The weak spots are 1 and 8 (F1 about 0.89 each). With only about 36 test images per digit, a single mistake moves a digit's recall by roughly 3 percentage points, so these per-digit numbers are noisy and should not be over-interpreted.`));

// 8 Over/under fitting
kids.push(h1("8. Overfitting and Underfitting"));
kids.push(p("Overfitting means the model learns the quirks of the training images and fails on new ones. Underfitting means it is too simple to capture the pattern at all. The two curves below show both."));
kids.push(...img("figures/fig5_validation_curve.png", "Figure 5. Decision tree accuracy as the allowed depth grows (validation curve).", 520));
kids.push(p(`At depth 1 the tree can only make one split and scores ${pct(R.val_curve.cv[0])} in cross-validation, hardly better than guessing: clear underfitting. As depth grows, cross-validation accuracy rises and then flattens at about ${pct(R.val_curve.cv[R.val_curve.cv.length - 1])} from depth 8 or so onward, while training accuracy keeps climbing to 100%. The widening gap between the two lines is overfitting. More depth no longer helps, which is why the search found no better setting than an unrestricted tree.`));
kids.push(...img("figures/fig6_learning_curves.png", "Figure 6. Learning curves: accuracy as the number of training images grows.", 620));
const lcL = R.learning_curves["Logistic regression"], lcT = R.learning_curves["Decision tree (unpruned)"];
kids.push(p(`A learning curve shows whether more data would help. For logistic regression the cross-validation score rises from ${pct(lcL.cv_at_smallest)} with ${lcL.smallest_n} training images to about ${pct(lcL.final_cv)} with the full training set, and it is still rising slowly, so more data would probably help a little. The training score sits near 100% throughout, so the model has plenty of flexibility for 64 features and a gap of roughly 3 points remains: mild overfitting. For the tree the gap between training (${pct(lcT.final_train)}) and validation (${pct(lcT.final_cv)}) stays large, which means its problem is high variance, not too little data alone.`));
kids.push(p("Conclusion: logistic regression is close to a good fit with a little overfitting. The decision tree is overfit. The majority-class baseline, and shallow trees, are underfit."));

// 9 Error analysis
kids.push(h1("9. Error Analysis"));
kids.push(...img("figures/fig7_misclassified.png", "Figure 7. Test images that the logistic regression classified incorrectly.", 620));
const tc = R.lr_top_confusions;
kids.push(p(`The most frequent error is a true ${tc[0].true} predicted as a ${tc[0].pred} (${tc[0].count} images), followed by a true ${tc[1].true} predicted as ${tc[1].pred} (${tc[1].count} images). Every other error type occurs once. Looking at the images in Figure 7, several of the 8s are drawn as a narrow, mostly vertical stroke with the loops barely visible, so they genuinely resemble a 1 at this resolution. Some of the 1s have a hook or flag at the top that resembles the upper loop of an 8 or a 3. This is my reading of the pictures; I did not run an experiment to prove the cause.`));
kids.push(h2("Stability across different splits"));
kids.push(p(`To check that the result does not depend on one lucky split, I repeated the train/test split with 10 different random seeds, keeping the hyperparameters fixed. Logistic regression averaged ${pct(SS["Logistic regression"].mean)} (SD ${pct(SS["Logistic regression"].sd)}, range ${pct(SS["Logistic regression"].min)} to ${pct(SS["Logistic regression"].max)}). The random forest averaged ${pct(SS["Random forest"].mean)} (SD ${pct(SS["Random forest"].sd)}, range ${pct(SS["Random forest"].min)} to ${pct(SS["Random forest"].max)}). Across these splits the forest is a little ahead on average. One caveat: the hyperparameters were tuned on the training part of the main split, which overlaps with the test parts of other splits, so these averages are slightly optimistic.`));

// 10 Sources of error
kids.push(h1("10. Potential Sources of Error and Limitations"));
kids.push(bullet([b("Small test set. "), `With ${R.n_test} test images, the 95% interval on accuracy is about ${pct(R.lr_acc_ci95[0])} to ${pct(R.lr_acc_ci95[1])}. Differences of one or two images between models are noise.`]));
kids.push(bullet([b("Low image resolution. "), "An 8 by 8 grid throws away detail. Some errors may be unavoidable at this resolution."]));
kids.push(bullet([b("Limited writer variety. "), "The images come from a limited group of people writing in a controlled setting. Real handwriting from other people, pens and scanners may be harder, so the accuracy here should not be expected on new data sources. I did not test this."]));
kids.push(bullet([b("Benchmark effect. "), "This is a well-known teaching dataset, so high scores are expected. The numbers show the pipeline works; they do not show the model would perform this well on a messy real-world task."]));
kids.push(bullet([b("Narrow tuning. "), "Only a few settings were searched, and only accuracy was used for selection. A wider search might shift the ranking of close models."]));
kids.push(bullet([b("Limited cross-validation resolution. "), "Five folds give a noisy estimate of the standard deviation. The SD values should be read as rough guides."]));

// 11 Improvements
kids.push(h1("11. Suggestions for Improvement"));
kids.push(bullet([b("More and better data. "), "The learning curve for logistic regression is still rising, so extra images, especially of 1s and 8s, should help. Data augmentation (small shifts and rotations) is a cheap way to add variety."]));
kids.push(bullet([b("A model that sees shape. "), "A small convolutional neural network, or a support vector machine with an RBF kernel, can capture local stroke patterns that a linear model on raw pixels cannot. These are the natural next step if the accuracy needs to rise above about 97%."]));
kids.push(bullet([b("Feature engineering. "), "Add features such as stroke thickness, number of loops or pixel row and column sums."]));
kids.push(bullet([b("Wider and fairer tuning. "), "Use nested cross-validation or repeated cross-validation, so that choosing hyperparameters and estimating performance do not share the same data."]));
kids.push(bullet([b("Better decision rules for costly errors. "), "If one error type were costly (for example, reading an 8 as a 1 on a bank cheque), the decision threshold could be adjusted for that digit and monitored on its own."]));
kids.push(bullet([b("Probability calibration. "), "Check whether a predicted probability of 90% really means right 90% of the time, and recalibrate if not."]));
kids.push(bullet([b("Ensembling. "), "Averaging logistic regression and the random forest is cheap and often gives a small, stable gain."]));

// 12 Conclusion
kids.push(h1("12. Conclusion"));
kids.push(p(`A properly validated logistic regression pipeline recognises handwritten digits with about ${pct(LR.test_acc)} accuracy on unseen images, far above the ${pct(BASE.test_acc)} baseline. A random forest performs about as well, and a single decision tree is much worse because it overfits. The remaining errors are concentrated in the digits 1 and 8. The measured uncertainty (a bootstrap interval of roughly ${pct(R.lr_acc_ci95[0])} to ${pct(R.lr_acc_ci95[1])}) and the benchmark nature of the dataset mean the result is best read as evidence that the development process is sound, rather than as a guarantee of performance in a real application.`));

// Appendix
kids.push(h1("Appendix: Reproducing This Work"));
kids.push(p("The repository folder week4_project contains:"));
kids.push(...table(["File", "Purpose"], [
  ["week4_ml_digits.py", "Complete pipeline: data checks, split, tuning, evaluation, figures and results file."],
  ["requirements.txt", "Python packages needed."],
  ["figures/", "The 7 figures used in this report (PNG)."],
  ["reports/results.json", "Every number quoted in this report, in machine-readable form."],
  ["reports/results_summary.txt", "Plain-text summary of the main results."],
  ["README.md", "Short project description and run instructions."],
], [3200, 6438], { leftAll: true }));
kids.push(...code([
  "pip install -r requirements.txt",
  "python week4_ml_digits.py",
]));
kids.push(p("The script uses a fixed random seed (42), so repeated runs give the same numbers on the same library versions. Small differences are possible with different scikit-learn versions."));
kids.push(p([b("Repository link: "), new TextRun({ text: REPO, color: "1155CC" })]));

// ---------------------------------------------------------------- document
const doc = new Document({
  creator: NAME,
  title: "Week 4 - Machine Learning Model Development and Evaluation",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: FONT, color: "1F3A5F" }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: FONT, color: "2F5D8C" }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1, keepNext: true } },
    ],
  },
  numbering: { config: [{ reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: `Week 4 - ${NAME} - Page `, size: 18, color: "777777" }),
        new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "777777" })] })] }) },
    children: kids,
  }],
});

Packer.toBuffer(doc).then((buf) => { fs.writeFileSync("Week4_ML_Model_Report.docx", buf); console.log("written", buf.length); });
