// AgentFuse pitch deck — Team Brocode. Premium OpenAI-inspired dark theme.
const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
p.author = "Team Brocode";
p.title = "AgentFuse";

const DIR = "assets/ppt/";
const F = "Calibri";
const C = {
  bg: "0B0F17", card: "151D2C", card2: "1B2536", line: "2A3A52",
  text: "E6EDF6", dim: "9AA8BC", dim2: "5F6F85",
  teal: "1FD0A0", teal2: "10A37F", cyan: "38BDF8", amber: "F5C542", red: "FB7185", white: "FFFFFF",
};

function shadow() { return { type: "outer", color: "000000", blur: 10, offset: 3, angle: 90, opacity: 0.38 }; }
function card(s, x, y, w, h, o = {}) {
  s.addShape("roundRect", { x, y, w, h, fill: { color: o.fill || C.card },
    line: { color: o.line || C.line, width: 1 }, rectRadius: o.r || 0.09, shadow: shadow() });
}
function disc(s, x, y, d, num, fill, tcolor) {
  s.addShape("ellipse", { x, y, w: d, h: d, fill: { color: fill } });
  s.addText(String(num), { x, y, w: d, h: d, align: "center", valign: "middle", margin: 0,
    fontFace: F, fontSize: d > 0.55 ? 16 : 13, bold: true, color: tcolor || "0B0F17" });
}
function header(s, page) {
  s.addShape("lightningBolt", { x: 0.62, y: 0.42, w: 0.17, h: 0.32, fill: { color: C.amber } });
  s.addText("AgentFuse", { x: 0.86, y: 0.38, w: 2.4, h: 0.4, valign: "middle", margin: 0,
    fontFace: F, fontSize: 12.5, bold: true, color: C.dim });
  s.addText(page + "  /  09", { x: 11.0, y: 0.38, w: 1.73, h: 0.4, align: "right", valign: "middle",
    margin: 0, fontFace: F, fontSize: 11, color: C.dim2 });
}
function title(s, t) {
  s.addText(t, { x: 0.75, y: 0.92, w: 11.8, h: 0.75, margin: 0, fontFace: F, fontSize: 34, bold: true, color: C.white });
}
function subtitle(s, t) {
  s.addText(t, { x: 0.77, y: 1.66, w: 11.6, h: 0.4, margin: 0, fontFace: F, fontSize: 14.5, color: C.cyan });
}
function chip(s, x, y, w, t, color) {
  s.addShape("roundRect", { x, y, w, h: 0.5, fill: { color: C.card2 }, line: { color, width: 1 }, rectRadius: 0.25 });
  s.addText(t, { x, y, w, h: 0.5, align: "center", valign: "middle", margin: 0, fontFace: F, fontSize: 12, bold: true, color });
}
function featureRow(s, x, y, w, h, accent, letter, ttl, desc) {
  card(s, x, y, w, h);
  disc(s, x + 0.26, y + h / 2 - 0.28, 0.56, letter, accent);
  s.addText(ttl, { x: x + 1.02, y: y + 0.16, w: w - 1.25, h: 0.4, margin: 0, fontFace: F, fontSize: 15, bold: true, color: C.text });
  s.addText(desc, { x: x + 1.02, y: y + 0.52, w: w - 1.25, h: h - 0.62, margin: 0, fontFace: F, fontSize: 12, color: C.dim });
}

/* ---------------- Slide 1 : Title ---------------- */
let s = p.addSlide();
s.background = { path: DIR + "bg_title.png" };
s.addText("OPENAI  ·  LONG-RANGE AUTONOMY CHALLENGE", { x: 0.78, y: 0.62, w: 10, h: 0.4,
  margin: 0, fontFace: F, fontSize: 12, bold: true, color: C.teal, charSpacing: 3 });
s.addShape("roundRect", { x: 0.78, y: 2.12, w: 0.98, h: 0.98, fill: { color: C.card2 }, line: { color: C.teal, width: 1.5 }, rectRadius: 0.14 });
s.addShape("lightningBolt", { x: 1.06, y: 2.32, w: 0.42, h: 0.58, fill: { color: C.amber } });
s.addText("AgentFuse", { x: 1.95, y: 2.05, w: 8, h: 1.15, valign: "middle", margin: 0, fontFace: F, fontSize: 54, bold: true, color: C.white });
s.addText("A Logical Circuit Breaker for Long-Range AI Agents", { x: 0.8, y: 3.42, w: 11, h: 0.6, margin: 0, fontFace: F, fontSize: 23, bold: true, color: C.cyan });
s.addText("Autonomous agents fail quietly — infinite loops, goal drift, logic traps, runaway spend. AgentFuse is an independent supervisor that catches these in real time and steers the agent back on track.",
  { x: 0.8, y: 4.18, w: 9.3, h: 0.95, margin: 0, fontFace: F, fontSize: 14.5, color: C.dim, lineSpacingMultiple: 1.1 });
chip(s, 0.8, 5.45, 1.9, "Observability", C.cyan);
chip(s, 2.85, 5.45, 2.45, "Safety / Guardrails", C.teal);
chip(s, 5.45, 5.45, 2.35, "OpenAI AgentKit", C.amber);
s.addText("TEAM", { x: 0.8, y: 6.45, w: 3, h: 0.3, margin: 0, fontFace: F, fontSize: 12, bold: true, color: C.dim2, charSpacing: 3 });
s.addText("Brocode", { x: 0.8, y: 6.72, w: 4, h: 0.5, margin: 0, fontFace: F, fontSize: 24, bold: true, color: C.teal });
chip(s, 9.55, 6.62, 3.15, "ns-0437.github.io/agentfuse", C.cyan);

/* ---------------- Slide 2 : Contents ---------------- */
s = p.addSlide(); s.background = { path: DIR + "bg_content.png" }; header(s, "02");
title(s, "Contents");
const idx = [
  [1, "Introduction", "03"], [2, "Problem Statement", "04"], [3, "Solution", "05"],
  [4, "Tech Stack", "06"], [5, "Live Demo & Dashboard", "07"], [6, "Scope of Improvement", "08"],
  [7, "Thank You", "09"],
];
function idxItem(x, y, num, name, page) {
  card(s, x, y, 5.5, 0.92);
  disc(s, x + 0.24, y + 0.21, 0.5, num, C.teal2, C.white);
  s.addText(name, { x: x + 0.95, y, w: 3.5, h: 0.92, valign: "middle", margin: 0, fontFace: F, fontSize: 17, bold: true, color: C.text });
  s.addText("p." + page, { x: x + 4.45, y, w: 0.95, h: 0.92, valign: "middle", align: "right", margin: 0, fontFace: F, fontSize: 12, color: C.dim2 });
}
idx.forEach((it, i) => {
  const col = i < 4 ? 0 : 1;
  const row = i < 4 ? i : i - 4;
  idxItem(col === 0 ? 0.75 : 7.05, 2.0 + row * 1.13, it[0], it[1], it[2]);
});

/* ---------------- Slide 3 : Introduction ---------------- */
s = p.addSlide(); s.background = { path: DIR + "bg_content.png" }; header(s, "03");
title(s, "Introduction");
subtitle(s, "What exactly is AgentFuse?");
s.addText([
  { text: "Think of the circuit breaker in your home — when current spikes, it trips and cuts the power before damage is done, then a human resets it.\n\n", options: { color: C.dim, fontSize: 15 } },
  { text: "AgentFuse is that breaker, for AI agents.\n\n", options: { color: C.teal, fontSize: 16, bold: true } },
  { text: "Modern agents run for hours across hundreds of steps with no human watching. AgentFuse is an independent supervisor: it watches every action, trips when the agent goes wrong, and uses a separate model to steer it back — or safely escalates to a human.", options: { color: C.dim, fontSize: 15 } },
], { x: 0.75, y: 2.3, w: 6.2, h: 4.4, margin: 0, fontFace: F, lineSpacingMultiple: 1.12, valign: "top" });
featureRow(s, 7.35, 2.25, 5.25, 1.02, C.cyan, "W", "Watches everything", "Tool calls, graph routes, state and token spend");
featureRow(s, 7.35, 3.4, 5.25, 1.02, C.amber, "D", "Detects failure", "Loops, goal drift, logic stalls, runaway spend");
featureRow(s, 7.35, 4.55, 5.25, 1.02, C.teal, "S", "Steers back", "A separate model writes a fix, injected live");
featureRow(s, 7.35, 5.7, 5.25, 1.02, C.red, "H", "Knows its limits", "Escalates to a human when it can't be fixed safely");

/* ---------------- Slide 4 : Problem Statement ---------------- */
s = p.addSlide(); s.background = { path: DIR + "bg_content.png" }; header(s, "04");
title(s, "Problem Statement");
subtitle(s, "Long-range autonomy fails quietly — and the agent can't see it happening.");
function modeCard(x, y, accent, letter, ttl, desc) {
  card(s, x, y, 5.72, 1.82);
  disc(s, x + 0.28, y + 0.3, 0.6, letter, accent);
  s.addText(ttl, { x: x + 1.1, y: y + 0.26, w: 4.4, h: 0.5, margin: 0, fontFace: F, fontSize: 18, bold: true, color: C.text });
  s.addText(desc, { x: x + 1.1, y: y + 0.82, w: 4.45, h: 0.85, margin: 0, fontFace: F, fontSize: 13, color: C.dim, lineSpacingMultiple: 1.05 });
}
modeCard(0.75, 2.2, C.amber, "L", "Infinite Tool Loops", "Repeats the same action over and over, expecting a different result.");
modeCard(6.85, 2.2, C.cyan, "D", "Goal Drift", "Slowly forgets the objective and wanders off onto an unrelated task.");
modeCard(0.75, 4.12, C.teal, "T", "Logic Traps", "Reasons flawlessly — but from a false assumption it never questions.");
modeCard(6.85, 4.12, C.red, "$", "Runaway Spend", "Burns tokens (real money) step after step with nothing to show for it.");
card(s, 0.75, 6.22, 11.82, 0.78, { fill: C.card2, line: C.teal });
s.addShape("lightningBolt", { x: 1.0, y: 6.44, w: 0.2, h: 0.36, fill: { color: C.amber } });
s.addText([
  { text: "The catch:  ", options: { bold: true, color: C.teal } },
  { text: "the agent that failed is the worst judge of its own failure — you need an independent supervisor.", options: { color: C.text } },
], { x: 1.35, y: 6.22, w: 11.0, h: 0.78, valign: "middle", margin: 0, fontFace: F, fontSize: 14.5, italic: true });

/* ---------------- Slide 5 : Solution ---------------- */
s = p.addSlide(); s.background = { path: DIR + "bg_content.png" }; header(s, "05");
title(s, "The Solution — How It Works");
const steps = [
  ["Watch", "Observe every tool call, route and token in real time"],
  ["Detect", "Four sensors: loops · drift · stalls · runaway spend"],
  ["Trip", "Freeze the run the moment a danger threshold is crossed"],
  ["Steer", "A separate reasoning model writes the fix, injected back"],
  ["Resume / Escalate", "Continue on the corrected path — or hand to a human"],
];
steps.forEach((st, i) => {
  const y = 1.95 + i * 0.98;
  card(s, 0.75, y, 6.55, 0.86);
  disc(s, 0.98, y + 0.19, 0.48, i + 1, C.teal2, C.white);
  s.addText(st[0], { x: 1.65, y: y + 0.08, w: 5.4, h: 0.38, margin: 0, fontFace: F, fontSize: 15.5, bold: true, color: C.text });
  s.addText(st[1], { x: 1.65, y: y + 0.44, w: 5.5, h: 0.36, margin: 0, fontFace: F, fontSize: 12, color: C.dim });
});
s.addText("LIVE DEMO", { x: 7.65, y: 1.55, w: 5, h: 0.3, margin: 0, fontFace: F, fontSize: 11, bold: true, color: C.teal, charSpacing: 3 });
card(s, 7.6, 1.88, 5.13, 3.9, { fill: C.card, r: 0.1 });
s.addImage({ path: DIR + "terminal.png", x: 7.72, y: 2.0, w: 4.89, h: 3.67 });
s.addText("A real agent stuck in a loop — caught by the breaker and steered back to finish, in one run.",
  { x: 7.62, y: 5.92, w: 5.1, h: 0.8, margin: 0, fontFace: F, fontSize: 12, italic: true, color: C.dim, lineSpacingMultiple: 1.05 });

/* ---------------- Slide 6 : Tech Stack ---------------- */
s = p.addSlide(); s.background = { path: DIR + "bg_content.png" }; header(s, "06");
title(s, "Tech Stack");
subtitle(s, "Framework-agnostic core, first-class OpenAI AgentKit integration.");
const tech = [
  [C.cyan, "Python", "Stdlib-only core — runs anywhere"],
  [C.teal, "OpenAI AgentKit", "First-class, real RunHooks"],
  [C.teal, "OpenAI SDK", "Guarded tool-use loop"],
  [C.teal, "LangGraph", "Callback-handler adapter"],
  [C.amber, "Reasoning Model", "Separate-model recovery (o4-mini)"],
  [C.cyan, "JSONL Traces", "Full run observability"],
  [C.cyan, "HTML / JS Dashboard", "Self-contained, no server"],
  [C.teal, "GitHub Pages", "Free live deployment"],
];
tech.forEach((t, i) => {
  const col = i % 4, row = Math.floor(i / 4);
  const x = 0.75 + col * 3.0, y = 2.15 + row * 2.05;
  card(s, x, y, 2.85, 1.85);
  s.addShape("ellipse", { x: x + 0.28, y: y + 0.28, w: 0.34, h: 0.34, fill: { color: t[0] } });
  s.addText(t[1], { x: x + 0.26, y: y + 0.72, w: 2.4, h: 0.5, margin: 0, fontFace: F, fontSize: 15, bold: true, color: C.text });
  s.addText(t[2], { x: x + 0.26, y: y + 1.18, w: 2.42, h: 0.55, margin: 0, fontFace: F, fontSize: 11.5, color: C.dim });
});

/* ---------------- Slide 7 : Live Demo & Dashboard ---------------- */
s = p.addSlide(); s.background = { path: DIR + "bg_content.png" }; header(s, "07");
title(s, "Live Demo & Dashboard");
card(s, 0.68, 1.72, 7.86, 5.4, { fill: C.card, r: 0.1 });
s.addImage({ path: DIR + "dashboard_full.png", x: 0.8, y: 1.84, w: 7.62, h: 5.16 });
s.addText("LIVE", { x: 8.85, y: 1.8, w: 4, h: 0.3, margin: 0, fontFace: F, fontSize: 11, bold: true, color: C.teal, charSpacing: 3 });
s.addText("See every intervention", { x: 8.85, y: 2.12, w: 3.9, h: 0.6, margin: 0, fontFace: F, fontSize: 21, bold: true, color: C.white });
s.addText("Every supervised run, replayable — trips, steering recoveries, token spend and graph routes at a glance.",
  { x: 8.85, y: 2.78, w: 3.9, h: 1.0, margin: 0, fontFace: F, fontSize: 13.5, color: C.dim, lineSpacingMultiple: 1.1 });
const feats = ["Timeline of trips & recoveries", "Cumulative token-spend chart", "Per-run detectors & graph route"];
feats.forEach((ft, i) => {
  const y = 4.0 + i * 0.62;
  s.addShape("ellipse", { x: 8.9, y: y + 0.07, w: 0.16, h: 0.16, fill: { color: C.cyan } });
  s.addText(ft, { x: 9.2, y, w: 3.55, h: 0.4, valign: "middle", margin: 0, fontFace: F, fontSize: 13, color: C.text });
});
chip(s, 8.85, 6.2, 3.6, "ns-0437.github.io/agentfuse", C.cyan);

/* ---------------- Slide 8 : Scope of Improvement ---------------- */
s = p.addSlide(); s.background = { path: DIR + "bg_content.png" }; header(s, "08");
title(s, "Scope of Improvement");
subtitle(s, "Where AgentFuse goes next.");
const road = [
  ["Real-time streaming", "A live dashboard, not just run replay"],
  ["More detectors", "Semantic loops, cost forecasting"],
  ["Fleet view", "Supervise many parallel agents at once"],
  ["Adaptive thresholds", "Learned drift baselines per task"],
  ["Alerting & policy", "Slack / PagerDuty + guardrail rules"],
  ["PyPI package", "One-line install:  pip install agentfuse"],
];
road.forEach((r, i) => {
  const col = i % 3, row = Math.floor(i / 3);
  const x = 0.75 + col * 4.03, y = 2.2 + row * 2.05;
  card(s, x, y, 3.78, 1.85);
  disc(s, x + 0.28, y + 0.28, 0.52, i + 1, C.teal2, C.white);
  s.addText(r[0], { x: x + 0.28, y: y + 0.92, w: 3.3, h: 0.42, margin: 0, fontFace: F, fontSize: 15.5, bold: true, color: C.text });
  s.addText(r[1], { x: x + 0.28, y: y + 1.34, w: 3.35, h: 0.44, margin: 0, fontFace: F, fontSize: 11.5, color: C.dim });
});

/* ---------------- Slide 9 : Thank You ---------------- */
s = p.addSlide(); s.background = { path: DIR + "bg_thanks.png" };
s.addShape("lightningBolt", { x: 6.42, y: 1.85, w: 0.5, h: 0.78, fill: { color: C.amber } });
s.addText("Thank You", { x: 0, y: 2.75, w: 13.333, h: 1.0, align: "center", margin: 0, fontFace: F, fontSize: 54, bold: true, color: C.white });
s.addText("AgentFuse — keeping autonomous agents on track.", { x: 0, y: 3.85, w: 13.333, h: 0.5, align: "center", margin: 0, fontFace: F, fontSize: 18, color: C.cyan });
s.addText([
  { text: "TEAM   ", options: { color: C.dim2, fontSize: 14, bold: true, charSpacing: 3 } },
  { text: "Brocode", options: { color: C.teal, fontSize: 22, bold: true } },
], { x: 0, y: 4.55, w: 13.333, h: 0.5, align: "center", margin: 0, fontFace: F });
chip(s, 3.55, 5.55, 3.1, "github.com/ns-0437/agentfuse", C.dim);
chip(s, 6.75, 5.55, 3.1, "ns-0437.github.io/agentfuse", C.cyan);

p.writeFile({ fileName: "AgentFuse_Brocode.pptx" }).then(f => console.log("wrote", f));
