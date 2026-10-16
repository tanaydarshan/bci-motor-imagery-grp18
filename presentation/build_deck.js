const pptxgen = require("pptxgenjs");
const path = require("path");
const { applyTheme } = require("/root/.claude/skills/synced/07c2047a-69e4-4483-9136-18c3401beaff_c5e621ae-033f-4b5b-83bd-b04c2510220f/pptx/scripts/apply_theme.js");

const FIG = "/home/user/bci-motor-imagery-grp18/paper/figures_color/";
const OUT = process.argv[2] || "deck.pptx";

const THEME = {
  name: "Group18 Review",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1F2328", lt1: "FFFFFF", dk2: "1F3A5F", lt2: "F2F4F7",
    accent1: "1F3A5F", accent2: "3D7AB8", accent3: "C8642B",
    accent4: "6B7280", accent5: "DCE6F2", accent6: "2E7D5B",
    hlink: "3D7AB8", folHlink: "6B7280",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625
pres.title = "EEG-Based Motor Action and Motor Imagery Classification";
pres.author = "Group 18, Amrita School of Artificial Intelligence";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.35, w: 8.6, h: 1.5, fontSize: 32, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.05, w: 8.6, h: 1.9, fontSize: 14, color: C.accent5, valign: "top", margin: 0 }, text: "" } },
  ],
});

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.4, 0.5, 0.5, 0.5],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.6, fontSize: 24, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Group 18  |  EEG motor action and motor imagery classification", options: { x: 0.5, y: 5.22, w: 7.0, h: 0.25, fontSize: 9, color: C.accent4, margin: 0 } } },
  ],
  slideNumber: { x: 9.0, y: 5.22, w: 0.5, h: 0.25, fontSize: 9, color: C.accent4, align: "right", margin: 0 },
});

// ---------- helpers ----------
const SIZES = {
  fig1_pipeline: [1734, 753], fig2_protocol: [2139, 618], fig3_onset: [2139, 669],
  fig5_timefreq: [2089, 1050], fig6_families: [2139, 638], fig7_confusion: [1969, 770],
  fig8_importance: [2138, 789],
};
function fig(slide, name, x, y, w, h) {
  const [pw, ph] = SIZES[name];
  const ar = pw / ph;
  let iw = w, ih = w / ar;
  if (ih > h) { ih = h; iw = h * ar; }
  slide.addImage({ path: FIG + name + ".png", x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih, objectName: name, altText: name.replace("_", " ") });
}
function content(title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}
function bullets(slide, items, opts) {
  const arr = items.map((t, i) => {
    const o = { bullet: { indent: 14 }, paraSpaceAfter: 6, breakLine: i < items.length - 1 };
    if (Array.isArray(t)) return t.map((r, j) => ({ text: r.text, options: Object.assign({}, r.options || {}, j === 0 ? { bullet: { indent: 14 } } : {}, { paraSpaceAfter: 6 }, j === t.length - 1 ? { breakLine: i < items.length - 1 } : {}) }));
    return [{ text: t, options: o }];
  }).flat();
  slide.addText(arr, Object.assign({ fontSize: 14, color: C.text1, valign: "top", margin: 0, isTextBox: true }, opts));
}
function stat(slide, x, y, w, value, label, sub, color) {
  slide.addText(value, { x, y, w, h: 0.7, fontSize: 34, bold: true, color: color || C.accent1, fontFace: THEME.headFontFace, margin: 0, isTextBox: true, objectName: "stat " + label });
  slide.addText(label, { x, y: y + 0.72, w, h: 0.3, fontSize: 13, bold: true, color: C.text1, margin: 0, isTextBox: true });
  if (sub) slide.addText(sub, { x, y: y + 1.02, w, h: 0.45, fontSize: 11, color: C.accent4, margin: 0, valign: "top", isTextBox: true });
}
function caption(slide, text, x, y, w) {
  slide.addText(text, { x, y, w, h: 0.3, fontSize: 11, italic: true, color: C.accent4, margin: 0, isTextBox: true });
}
const HDR = (t) => ({ text: t, options: { bold: true, color: C.background1, fill: { color: C.text2 }, fontSize: 12 } });
const CELL = (t, extra) => ({ text: String(t), options: Object.assign({ fontSize: 12, color: C.text1 }, extra || {}) });
const BEST = (t) => CELL(t, { bold: true, color: C.accent1, fill: { color: C.accent5 } });

// ---------- 1. title ----------
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
s.addText("EEG-Based Motor Action and Motor Imagery Classification Using Machine Learning", { placeholder: "title" });
s.addText([
  { text: "Harsada K  ·  V Tanay Darshan  ·  K Nikhitha  ·  S Sai Tejesh Reddy", options: { breakLine: true } },
  { text: "Guide: Dr. Abhishek S", options: { breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: 8 } },
  { text: "Group 18  ·  Amrita School of Artificial Intelligence, Coimbatore", options: {} },
], { placeholder: "body" });
s.addNotes("Introduce the team and the topic: classifying real versus imagined limb movements from a 14-channel consumer EEG headset.");

// ---------- 2. problem ----------
s = content("Problem and objectives", "Introduction");
s.addText("Can a low-cost 14-channel headset tell whether a person is moving a limb or only imagining it, and which limb?", { x: 0.5, y: 1.05, w: 9.0, h: 0.6, fontSize: 15, italic: true, color: C.accent2, margin: 0, isTextBox: true });
const objs = [
  ["Action vs imagery", "Separate real movement from imagined movement (2 classes)."],
  ["Which limb", "Identify the body part: right/left arm, leg, wrist (6 classes) and limb × condition (12 classes)."],
  ["Left vs right imagery", "Decide the side of an imagined movement, the core control signal of a motor-imagery BCI."],
];
objs.forEach((o, i) => {
  const x = 0.5 + i * 3.05;
  s.addShape(pres.shapes.RECTANGLE, { x, y: 1.95, w: 2.85, h: 2.4, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "objective card " + (i + 1) });
  s.addText(String(i + 1), { x: x + 0.25, y: 2.15, w: 0.5, h: 0.5, fontSize: 26, bold: true, color: C.accent3, fontFace: THEME.headFontFace, margin: 0, isTextBox: true });
  s.addText(o[0], { x: x + 0.25, y: 2.7, w: 2.4, h: 0.4, fontSize: 15, bold: true, color: C.text2, margin: 0, isTextBox: true });
  s.addText(o[1], { x: x + 0.25, y: 3.1, w: 2.4, h: 1.1, fontSize: 12, color: C.text1, margin: 0, valign: "top", isTextBox: true });
});
s.addText("Constraint throughout: evaluation on unseen people only (person-wise cross-validation), with no information leaking from test subjects.", { x: 0.5, y: 4.55, w: 9.0, h: 0.45, fontSize: 12, color: C.accent4, margin: 0, isTextBox: true });
s.addNotes("Three levels of difficulty. Person-wise validation is the key constraint: every reported number comes from people the model never saw during training.");

// ---------- 3. dataset ----------
pres.addSection({ title: "Data and method" });
s = content("Dataset and recording protocol", "Data and method");
fig(s, "fig2_protocol", 0.5, 1.0, 9.0, 2.6);
[["47", "participants", "51 recordings used"], ["14 ch", "Emotiv EPOC X", "128 Hz, plus motion sensor"], ["12", "tasks", "6 limbs × action / imagery"], ["476", "clean trials", "235 action, 241 imagery"]]
  .forEach((d, i) => stat(s, 0.6 + i * 2.25, 3.65, 2.1, d[0], d[1], d[2]));
s.addNotes("Each session cycles through the six limbs, first performing and then imagining each movement. 59 recordings were collected; 53 passed onset checks; after trial rejection 476 trials from 47 people remain.");

// ---------- 4. pipeline ----------
s = content("Processing pipeline", "Data and method");
fig(s, "fig1_pipeline", 0.6, 0.95, 8.8, 3.85);
caption(s, "All normalisation steps are label-free; every learned step is fitted on training people only.", 0.6, 4.8, 8.8);
s.addNotes("Walk left to right: raw EEG and motion, onset detection, filtering, trial rejection, per-recording normalisation, 2 s windows, 1,771 features, per-recording alignment, ensemble, and finally averaging the 17 window predictions of each trial.");

// ---------- 5. onset ----------
s = content("Finding when the tasks start", "Data and method");
fig(s, "fig3_onset", 0.5, 0.95, 9.0, 2.85);
bullets(s, [
  "Task start varied between recordings, so fixed timing would mislabel windows.",
  "Head-motion energy from the motion sensor is high during real movement and low during imagery; the schedule offset that best separates them is taken as the onset.",
  "Accepted when the contrast is strong (t ≥ 4, all 6 pairs agree): 48 recordings; 5 more checked against recording length; 2 excluded.",
], { x: 0.5, y: 3.85, w: 9.0, h: 1.3, fontSize: 12.5 });
s.addNotes("Without this step the action and imagery labels would be shifted in several sessions. A ±2 s onset shift changes binary accuracy by about 1 to 4 points, reported later.");

// ---------- 6. cleaning ----------
s = content("Filtering, cleaning and normalisation", "Data and method");
bullets(s, [
  [{ text: "Band-pass 4–45 Hz ", options: { bold: true } }, { text: "zero-phase FIR (MNE): removes drift and line noise, keeps theta to gamma." }],
  [{ text: "Trial rejection ", options: { bold: true } }, { text: "flat signal, peak amplitude ≥ 150 µV, or mean |kurtosis| ≥ 10." }],
  [{ text: "Per-recording z-score ", options: { bold: true } }, { text: "of each channel, then of each feature: removes headset fit and session offsets without using labels." }],
  [{ text: "Windows ", options: { bold: true } }, { text: "2 s long, 0.5 s stride: 17 windows per trial, 8,092 windows in total." }],
], { x: 0.5, y: 1.15, w: 5.2, h: 3.9, fontSize: 15, paraSpaceAfter: 10 });
s.addShape(pres.shapes.RECTANGLE, { x: 6.1, y: 1.1, w: 3.4, h: 3.8, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "rejection panel" });
s.addText("Trials", { x: 6.35, y: 1.25, w: 3.0, h: 0.35, fontSize: 13, bold: true, color: C.text2, margin: 0, isTextBox: true });
s.addText([
  { text: "636", options: { fontSize: 30, bold: true, color: C.accent4, fontFace: THEME.headFontFace } },
  { text: "  →  ", options: { fontSize: 22, color: C.accent4 } },
  { text: "476", options: { fontSize: 30, bold: true, color: C.accent1, fontFace: THEME.headFontFace } },
], { x: 6.35, y: 1.65, w: 3.0, h: 0.7, margin: 0, isTextBox: true });
s.addTable([
  [CELL("Amplitude ≥ 150 µV"), CELL("158", { align: "right" })],
  [CELL("Kurtosis ≥ 10"), CELL("29", { align: "right" })],
  [CELL("Window overlap"), CELL("27", { align: "right" })],
], { x: 6.35, y: 2.55, w: 2.95, colW: [2.25, 0.7], rowH: 0.36, border: { type: "solid", pt: 0.5, color: "D0D5DD" }, fill: { color: C.background1 } });
s.addText("Rejection reasons can overlap; 160 trials removed in total.", { x: 6.35, y: 3.75, w: 2.95, h: 0.6, fontSize: 10.5, color: C.accent4, margin: 0, valign: "top", isTextBox: true });
s.addNotes("Most rejections were large amplitude artefacts from movement and blinks, which is expected with a dry consumer headset.");

// ---------- 7. features ----------
s = content("Features: 1,771 per window", "Data and method");
s.addTable([
  [HDR("Family"), HDR("What it captures"), HDR("Count")],
  [CELL("Handcrafted", { bold: true }), CELL("Band powers (Welch), ratios, asymmetry, Hjorth, statistics, fractal, DWT (db4)"), CELL("336", { align: "right" })],
  [CELL("Spectrogram (STFT)", { bold: true }), CELL("Short-time Fourier power over time and frequency"), CELL("560", { align: "right" })],
  [CELL("Scalogram (CWT)", { bold: true }), CELL("Morlet wavelet power: fine time detail at high frequencies"), CELL("714", { align: "right" })],
  [CELL("Connectivity", { bold: true }), CELL("Coherence between channel pairs"), CELL("56", { align: "right" })],
  [CELL("Riemannian", { bold: true }), CELL("Covariance matrix mapped to tangent space, recentred per recording"), CELL("105", { align: "right" })],
  [CELL("Total", { bold: true }), CELL(""), CELL("1,771", { bold: true, align: "right" })],
], { x: 0.5, y: 1.1, w: 9.0, colW: [2.1, 5.9, 1.0], rowH: 0.42, border: { type: "solid", pt: 0.5, color: "D0D5DD" }, valign: "middle" });
caption(s, "Every feature is z-scored within its own recording before classification.", 0.5, 4.55, 9.0);
s.addNotes("The handcrafted set is the classic EEG feature set. Spectrogram and scalogram features add time-frequency detail; the Riemannian features describe how channels co-vary.");

// ---------- 8. time-frequency ----------
s = content("Spectrogram and scalogram analysis", "Data and method");
fig(s, "fig5_timefreq", 0.5, 0.95, 5.6, 4.15);
bullets(s, [
  "Mu (8–13 Hz) power drops during movement over the motor cortex; the drop is weaker for imagery.",
  "The scalogram resolves short bursts that the spectrogram smears.",
  "Used alone: STFT 76.47%, CWT 77.52%, STFT+CWT 78.15% (binary). All features together: 79.62%.",
], { x: 6.3, y: 1.1, w: 3.2, h: 3.9, fontSize: 12.5 });
s.addNotes("Feature-family results are trial-level without mirror augmentation. Each family is useful alone, but combining them is best.");

// ---------- 9. model ----------
s = content("Classifier: a four-model ensemble", "Data and method");
const models = [["Logistic Regression", "C = 0.01"], ["LDA", "shrinkage"], ["Extra Trees", "500 trees"], ["Hist. Gradient Boosting", "200 iterations"]];
models.forEach((m, i) => {
  const y = 1.1 + i * 0.78;
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y, w: 2.6, h: 0.62, fill: { color: C.background2 }, line: { color: "C9D3E0", width: 0.75 }, objectName: "model " + m[0] });
  s.addText([{ text: m[0], options: { bold: true, fontSize: 12.5, color: C.text2, breakLine: true } }, { text: m[1], options: { fontSize: 10.5, color: C.accent4 } }], { x: 0.6, y, w: 2.4, h: 0.62, margin: 0, valign: "middle", isTextBox: true });
  const yc = y + 0.31, yt = 2.575;
  s.addShape(pres.shapes.LINE, { x: 3.1, y: Math.min(yc, yt), w: 0.45, h: Math.max(Math.abs(yt - yc), 0.01), flipV: yc > yt, line: { color: "9AA5B1", width: 1 } });
});
s.addShape(pres.shapes.RECTANGLE, { x: 3.55, y: 2.1, w: 1.6, h: 0.95, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "average box" });
s.addText([{ text: "Average", options: { bold: true, breakLine: true } }, { text: "probabilities", options: { fontSize: 10.5 } }], { x: 3.55, y: 2.1, w: 1.6, h: 0.95, align: "center", valign: "middle", color: C.background1, fontSize: 13, margin: 0, isTextBox: true });
s.addShape(pres.shapes.LINE, { x: 5.15, y: 2.575, w: 0.35, h: 0, line: { color: "9AA5B1", width: 1, endArrowType: "triangle" } });
s.addShape(pres.shapes.RECTANGLE, { x: 5.5, y: 2.1, w: 1.5, h: 0.95, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "trial box" });
s.addText([{ text: "Trial vote", options: { bold: true, breakLine: true } }, { text: "mean of 17 windows", options: { fontSize: 10.5 } }], { x: 5.5, y: 2.1, w: 1.5, h: 0.95, align: "center", valign: "middle", color: C.background1, fontSize: 13, margin: 0, isTextBox: true });
bullets(s, [
  [{ text: "Mirror augmentation: ", options: { bold: true } }, { text: "training windows are copied with left/right channels swapped and labels flipped (RA↔LA)." }],
  [{ text: "Validation: ", options: { bold: true } }, { text: "GroupKFold by person, 5 folds (binary, L/R) and 3 folds (6/12-class)." }],
], { x: 0.5, y: 4.3, w: 9.0, h: 0.85, fontSize: 12.5 });
s.addText("Linear models handle the many weak spectral cues; tree models pick up non-linear interactions. Averaging is more stable than any single model.", { x: 7.25, y: 1.15, w: 2.25, h: 2.9, fontSize: 11.5, color: C.accent4, italic: true, margin: 0, valign: "top", isTextBox: true });
s.addNotes("Each model outputs class probabilities per window; they are averaged, then averaged again over the 17 windows of a trial. Mirror copies are added only to training folds, never to test people.");

// ---------- 10. results ----------
pres.addSection({ title: "Results" });
s = content("Results on unseen participants", "Results");
const res = [
  ["80.25%", "Action vs imagery", "382 / 476 trials\nchance 50%"],
  ["26.68%", "Body part (6 classes)", "127 / 476 trials\nchance 16.7%"],
  ["21.43%", "Limb × condition (12)", "102 / 476 trials\nchance 8.3%"],
  ["58.09%", "Left vs right imagery", "140 / 241 trials\nchance 50%, p = 0.007"],
];
res.forEach((r, i) => {
  const x = 0.5 + i * 2.3;
  s.addShape(pres.shapes.RECTANGLE, { x, y: 1.15, w: 2.1, h: 2.35, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "result " + r[1] });
  s.addText(r[0], { x: x + 0.15, y: 1.3, w: 1.85, h: 0.7, fontSize: 30, bold: true, color: i === 0 ? C.accent3 : C.accent1, fontFace: THEME.headFontFace, margin: 0, isTextBox: true });
  s.addText(r[1], { x: x + 0.15, y: 2.05, w: 1.85, h: 0.5, fontSize: 12.5, bold: true, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  s.addText(r[2], { x: x + 0.15, y: 2.6, w: 1.85, h: 0.75, fontSize: 11, color: C.accent4, margin: 0, valign: "top", isTextBox: true });
});
bullets(s, [
  "Trial-level accuracy, pooled over all test folds. Fold mean ± SD: 80.19 ± 6.59 (binary), 26.68 ± 1.79 (6-class), 21.42 ± 1.21 (12-class).",
  "Body-part and 12-class accuracies are 1.6× and 2.6× chance: above chance, but far from usable control.",
], { x: 0.5, y: 3.75, w: 9.0, h: 1.3, fontSize: 12.5 });
s.addNotes("Window-level accuracies are lower: 68.53% binary, 21.01% six-class, 15.75% twelve-class. Averaging the 17 windows of a trial adds roughly 12 points for binary.");

// ---------- 11. comparison ----------
s = content("Comparison with other models", "Results");
s.addTable([
  [HDR("Model (window level, best feature set)"), HDR("Action vs imagery"), HDR("6-class"), HDR("12-class")],
  [BEST("Our ensemble"), BEST("68.53"), BEST("21.01"), BEST("15.75")],
  [CELL("Hist. Gradient Boosting"), CELL("64.45"), CELL("20.27"), CELL("13.91")],
  [CELL("Random Forest"), CELL("63.81"), CELL("20.37"), CELL("13.35")],
  [CELL("LDA"), CELL("63.35"), CELL("20.30"), CELL("13.44")],
  [CELL("SVM (RBF)"), CELL("62.89"), CELL("20.70"), CELL("14.32")],
  [CELL("Riemannian MDM"), CELL("62.31"), CELL("19.92"), CELL("12.05")],
  [CELL("EEGNet"), CELL("62.15"), CELL("17.49"), CELL("10.61")],
  [CELL("FBCSP + classifier"), CELL("61.60"), CELL("19.47"), CELL("11.50")],
  [CELL("CNN–LSTM"), CELL("58.38"), CELL("–"), CELL("–")],
], { x: 0.5, y: 1.05, w: 6.2, colW: [2.9, 1.3, 1.0, 1.0], rowH: 0.36, border: { type: "solid", pt: 0.5, color: "D0D5DD" }, align: "left" });
bullets(s, [
  "Same folds and same people for every model.",
  "Deep models (EEGNet, CNN–LSTM) trail the feature-based models: 476 trials is too little data to learn filters from raw EEG.",
  "FBCSP alone is weaker than the full feature set.",
  "AdaBoost (62.85) and Naive Bayes (59.80) were tested on the binary task only.",
], { x: 6.95, y: 1.1, w: 2.55, h: 3.95, fontSize: 11.5 });
s.addNotes("These are window-level accuracies so all models are on equal footing. The ensemble's trial-level result is the 80.25% from the previous slide.");

// ---------- 12. errors ----------
s = content("Where the model makes mistakes", "Results");
fig(s, "fig7_confusion", 0.5, 0.95, 9.0, 3.5);
caption(s, "(a) action vs imagery   (b) six body parts   (c) twelve classes", 0.5, 4.4, 9.0);
s.addText([
  { text: "Imagery recall 83.4%, action 77.0%.  ", options: { bold: true } },
  { text: "Arms are recognised best (RA 37.8%); left wrist worst (13.5%). In the 12-class task the condition is right 78.2% of the time but the body part only 25.8%." },
], { x: 0.5, y: 4.7, w: 9.0, h: 0.5, fontSize: 11.5, color: C.text1, margin: 0, isTextBox: true });
s.addNotes("The model separates action from imagery well but confuses limbs, especially wrists and legs, which have small or deep cortical representations that 14 scalp channels capture poorly.");

// ---------- 13. left vs right ----------
s = content("Left vs right imagery", "Results");
s.addChart(pres.charts.BAR, [{ name: "Accuracy (%)", labels: ["All imagery", "Arm", "Leg", "Wrist"], values: [58.09, 74.07, 58.97, 58.54] }], {
  x: 0.5, y: 1.0, w: 5.2, h: 3.9, barDir: "col",
  chartColors: [HEX.accent2],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: HEX.dk1, dataLabelFormatCode: "0.0",
  valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 25, valAxisLabelFontSize: 10, catAxisLabelFontSize: 11,
  catAxisLabelColor: HEX.accent4, valAxisLabelColor: HEX.accent4,
  valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" },
  showLegend: false, showTitle: true, title: "Trial accuracy by limb (chance = 50%)", titleFontSize: 12, titleColor: HEX.dk2,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
});
bullets(s, [
  [{ text: "Requested by the guide. ", options: { bold: true } }, { text: "Imagery trials only, 241 trials, 5-fold person-wise CV." }],
  [{ text: "What helped: ", options: { bold: true } }, { text: "calibrating features on each person's imagery trials only, plus mirror augmentation." }],
  [{ text: "Baselines: ", options: { bold: true } }, { text: "FBCSP + LDA 52.70%; standard calibration + mirror 53.53%." }],
  "Overall 58.09% is significant (p = 0.007); arm imagery reaches 74.07%.",
], { x: 5.95, y: 1.1, w: 3.55, h: 3.9, fontSize: 12.5 });
s.addNotes("Left versus right imagery is the hardest binary task because imagery produces weak, lateralised mu changes. Arms show the clearest lateralisation, matching their large cortical area.");

// ---------- 14. validity ----------
pres.addSection({ title: "Validity" });
s = content("Is the accuracy real? Leakage checks", "Validity");
const checks = [
  ["Person-wise folds", "No participant appears in both training and test sets in any fold; no duplicate EDF files."],
  ["Label permutation", "With shuffled labels the pipeline falls to chance: 49.4 / 54.6 / 48.3% (binary), 18.7 / 17.6 / 18.5% (6-class)."],
  ["Reproducibility", "Same seed, two runs: 382 / 476 both times. Different seed: 384 / 476. The result does not drift upward."],
  ["Onset sensitivity", "Shifting the onset by ±2 s: binary 75.9–79.2%, so the reported figure may be about 3 points optimistic."],
];
checks.forEach((c, i) => {
  const col = i % 2, row = Math.floor(i / 2);
  const x = 0.5 + col * 4.6, y = 1.1 + row * 1.95;
  s.addShape(pres.shapes.RECTANGLE, { x, y, w: 4.4, h: 1.75, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "check " + c[0] });
  s.addText("✓", { x: x + 0.2, y: y + 0.18, w: 0.4, h: 0.4, fontSize: 20, bold: true, color: C.accent6, margin: 0, isTextBox: true });
  s.addText(c[0], { x: x + 0.65, y: y + 0.2, w: 3.6, h: 0.38, fontSize: 14, bold: true, color: C.text2, margin: 0, isTextBox: true });
  s.addText(c[1], { x: x + 0.65, y: y + 0.62, w: 3.6, h: 1.05, fontSize: 13, color: C.text1, margin: 0, valign: "top", isTextBox: true });
});
s.addNotes("Every scaler, alignment and model is refitted inside each fold. Normalisation per recording uses no labels. Permutation tests at chance confirm the pipeline is not memorising subject identity.");

// ---------- 15. guide feedback ----------
s = content("Response to review feedback", "Validity");
s.addTable([
  [HDR("Question"), HDR("Finding / action")],
  [CELL("v10 results differ from earlier report", { bold: true }), CELL("The executed notebook prints both v9 (79.62%) and v10 (80.25%) summaries; the deep models (EEGNet, CNN–LSTM) are not deterministic and shift by about 0.5–2 points between runs. The final numbers are v10.")],
  [CELL("Accuracy rising each run: leakage?", { bold: true }), CELL("No. Fixed seed gives identical results (382/476 twice); permutation tests sit at chance; folds are split by person. Nested CV on all tasks is being completed.")],
  [CELL("Only LR for left vs right", { bold: true }), CELL("All eight models (LR, LDA, SVM, RF, ET, HGB, AdaBoost, NB) and the ensemble are being run on the L/R task with the same folds.")],
  [CELL("Add CSP features to every model?", { bold: true }), CELL("FBCSP alone is weaker (61.60% vs 64.45%; L/R 52.70% vs 58.09%). Each model is being re-run with and without FBCSP features; CSP is kept only where it helps.")],
], { x: 0.5, y: 1.05, w: 9.0, colW: [2.6, 6.4], rowH: [0.36, 0.8, 0.8, 0.6, 0.8], border: { type: "solid", pt: 0.5, color: "D0D5DD" }, valign: "middle", fontSize: 11.5 });
s.addNotes("CSP learns spatial filters for two-class variance differences. Our feature set already includes band power per channel and Riemannian covariance features, which carry the same information, so CSP is expected to add little.");

// ---------- 16. limitations ----------
pres.addSection({ title: "Conclusion" });
s = content("Limitations and next steps", "Conclusion");
s.addText("Limitations", { x: 0.5, y: 1.05, w: 4.3, h: 0.4, fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true });
bullets(s, [
  "14 saline channels with sparse motor-cortex coverage (no C3/C4).",
  "Onset estimated from motion; small timing errors remain.",
  "One session per person: no cross-session tests.",
  "Six- and twelve-class accuracy is above chance but too low for control.",
], { x: 0.5, y: 1.5, w: 4.3, h: 3.5, fontSize: 13 });
s.addText("Next steps", { x: 5.2, y: 1.05, w: 4.3, h: 0.4, fontSize: 16, bold: true, color: C.text2, margin: 0, isTextBox: true });
bullets(s, [
  "Finish nested CV and per-model ± CSP comparison; update paper.",
  "Event markers during recording to remove onset estimation.",
  "Short per-user calibration for left vs right imagery.",
  "More recordings per person to allow deep models to learn.",
], { x: 5.2, y: 1.5, w: 4.3, h: 3.5, fontSize: 13 });
s.addShape(pres.shapes.LINE, { x: 4.95, y: 1.1, w: 0, h: 3.7, line: { color: "D0D5DD", width: 0.75 } });
s.addNotes("Be honest about the multi-class results; the main contribution is a leakage-free pipeline and a strong action versus imagery result on a consumer headset.");

// ---------- 17. close ----------
s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Conclusion" });
s.addText("Thank you", { placeholder: "title" });
s.addText([
  { text: "Action vs imagery 80.25%  ·  Left vs right imagery 58.09%  ·  person-wise validation", options: { breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: 8 } },
  { text: "Questions and suggestions welcome", options: {} },
], { placeholder: "body" });

(async () => {
  await pres.writeFile({ fileName: OUT });
  // pptxgenjs writes a second <a:pPr> after the first run of a mixed-format paragraph; drop it
  const JSZip = require("jszip"), fs = require("fs");
  const zip = await JSZip.loadAsync(fs.readFileSync(OUT));
  for (const f of Object.keys(zip.files).filter((n) => /^ppt\/slides\/slide\d+\.xml$/.test(n))) {
    const x = await zip.file(f).async("string");
    zip.file(f, x.replace(/(<\/a:r>)<a:pPr\b[^>]*?(?:\/>|>.*?<\/a:pPr>)/g, "$1"));
  }
  fs.writeFileSync(OUT, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
