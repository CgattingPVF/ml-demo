const workerState = {
  datasetPath: "",
  referenceData: {},
  history: [],
  latestResult: null,
};

const MAX_SELECT_LABEL_LENGTH = 64;
const REQUIRED_INPUTS = [
  ["scaffold-partner", "Scaffolder"],
  ["homeowner-postcode", "Postcode"],
  ["scaffold-purpose", "Job type"],
  ["start-date", "Erect date"],
  ["due-date", "Install / due date"],
  ["budget-revenue", "Job price"],
];

function formatSelectLabel(value) {
  const text = String(value ?? "").trim();
  if (text.length <= MAX_SELECT_LABEL_LENGTH) {
    return text;
  }
  return `${text.slice(0, MAX_SELECT_LABEL_LENGTH - 1)}\u2026`;
}

function formatNumber(value, options = {}) {
  if (value === null || value === undefined || value === "") return "N/A";
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value);
  return new Intl.NumberFormat("en-GB", options).format(number);
}

function formatPercent(value) {
  if (value === null || value === undefined || value === "") return "N/A";
  const number = Number(value);
  if (!Number.isFinite(number)) return "N/A";
  return `${number.toFixed(number % 1 === 0 ? 0 : 1)}%`;
}

function formatDate(value) {
  if (!value) return "N/A";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
}

function compactCount(values) {
  return formatNumber((values || []).length, { notation: "compact", maximumFractionDigits: 1 });
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

async function api(path, options = {}) {
  const config = { method: "GET", ...options };
  if (config.body && typeof config.body !== "string") {
    config.headers = { ...(config.headers || {}), "Content-Type": "application/json" };
    config.body = JSON.stringify(config.body);
  }

  const response = await fetch(path, config);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || "Request failed");
  }
  return payload;
}

function setText(id, value) {
  const element = document.getElementById(id);
  if (!element) return;

  const text = String(value ?? "");
  element.textContent = text;
  element.title = text;
}

function populateSelect(id, values, placeholder) {
  const select = document.getElementById(id);
  select.innerHTML = "";

  const empty = document.createElement("option");
  empty.value = "";
  empty.textContent = formatSelectLabel(placeholder);
  select.appendChild(empty);

  for (const value of values || []) {
    const textValue = String(value ?? "").trim();
    if (!textValue) continue;

    const option = document.createElement("option");
    option.value = textValue;
    option.textContent = formatSelectLabel(textValue);
    option.title = textValue;
    select.appendChild(option);
  }
}

function populateDatalist(id, values) {
  const datalist = document.getElementById(id);
  datalist.innerHTML = "";
  for (const value of values || []) {
    const option = document.createElement("option");
    option.value = value;
    datalist.appendChild(option);
  }
}

function collectJobRecord() {
  const record = {
    related_contacts: document.getElementById("scaffold-partner").value,
    c_homeownerpostcode: document.getElementById("homeowner-postcode").value.trim(),
    c_scaffoldpurpose: document.getElementById("scaffold-purpose").value,
    date: document.getElementById("start-date").value,
    due_date: document.getElementById("due-date").value,
    contact_name: document.getElementById("business-name").value,
    c_locationofscaffold: document.getElementById("scaffold-location").value,
    description: document.getElementById("job-notes").value.trim(),
  };

  const price = document.getElementById("budget-revenue").value;
  const cost = document.getElementById("budget-cost").value;
  if (price !== "") record.budget_revenue = Number(price);
  if (cost !== "") record.budget_cost = Number(cost);

  return Object.fromEntries(
    Object.entries(record).filter(([, value]) => value !== "" && value !== null)
  );
}

function updateFormReadiness() {
  const completed = REQUIRED_INPUTS.filter(([id]) => {
    const element = document.getElementById(id);
    return String(element?.value || "").trim() !== "";
  });
  const missing = REQUIRED_INPUTS
    .filter(([id]) => String(document.getElementById(id)?.value || "").trim() === "")
    .map(([, label]) => label);
  const ratio = completed.length / REQUIRED_INPUTS.length;
  const percent = Math.round(ratio * 100);

  setText("form-progress-label", `${completed.length} of ${REQUIRED_INPUTS.length} required`);
  const bar = document.getElementById("form-progress-bar");
  if (bar) {
    bar.style.width = `${percent}%`;
  }

  const readiness = document.getElementById("input-readiness");
  if (readiness) {
    readiness.textContent = missing.length
      ? `Missing: ${missing.join(", ")}.`
      : "Ready to calculate risk with operational context.";
    readiness.classList.toggle("ready", missing.length === 0);
  }
}

function metricFor(predictions, name) {
  return predictions.find((prediction) => prediction.name === name);
}

function metricScore(prediction) {
  const score = Number(prediction?.score_pct);
  return Number.isFinite(score) ? score : 0;
}

function metricSeverity(prediction) {
  const level = String(prediction?.level || "").toLowerCase();
  if (level.includes("high") || level.includes("weak")) return 3;
  if (level.includes("medium") || level.includes("watch")) return 2;
  return 1;
}

function summarizeDecision(predictions) {
  const riskSignals = predictions.filter((prediction) => prediction.name !== "Completion likelihood");
  const topSignal = riskSignals
    .slice()
    .sort((a, b) => metricSeverity(b) - metricSeverity(a) || metricScore(b) - metricScore(a))[0];
  const highCount = riskSignals.filter((prediction) => metricSeverity(prediction) >= 3).length;
  const watchCount = riskSignals.filter((prediction) => metricSeverity(prediction) === 2).length;

  if (highCount > 0) {
    return {
      label: "Escalate before booking",
      priority: topSignal?.name || "Operational review",
      className: "decision-high",
      rationale: `${highCount} high-priority signal${highCount === 1 ? "" : "s"} detected.`,
    };
  }

  if (watchCount > 0) {
    return {
      label: "Proceed with controls",
      priority: topSignal?.name || "Standard controls",
      className: "decision-medium",
      rationale: `${watchCount} watch signal${watchCount === 1 ? "" : "s"} detected.`,
    };
  }

  return {
    label: "Proceed standard",
    priority: "Normal handover controls",
    className: "decision-low",
    rationale: "No elevated operational signal detected.",
  };
}

function riskClass(level) {
  const value = String(level || "").toLowerCase();
  if (value.includes("high") || value.includes("weak")) return "risk-high";
  if (value.includes("medium") || value.includes("watch")) return "risk-medium";
  return "risk-low";
}

function formatCurrency(value) {
  if (value === null || value === undefined || value === "") return "N/A";
  const number = Number(value);
  if (!Number.isFinite(number)) return "N/A";
  return new Intl.NumberFormat("en-GB", {
    style: "currency",
    currency: "GBP",
    maximumFractionDigits: 0,
  }).format(number);
}

function formatValue(value, suffix = "") {
  if (value === null || value === undefined || value === "") return "N/A";
  return `${value}${suffix}`;
}

function appendInfoItem(container, label, value, className = "risk-summary-item") {
  const item = document.createElement("div");
  item.className = className;

  const labelNode = document.createElement("span");
  labelNode.textContent = label;

  const valueNode = document.createElement("strong");
  valueNode.textContent = String(value ?? "N/A");
  valueNode.title = valueNode.textContent;

  item.append(labelNode, valueNode);
  container.appendChild(item);
}

function renderMetric(title, prediction, invert = false) {
  const card = document.createElement("article");
  card.className = `risk-metric ${riskClass(prediction?.level)}`;
  if (invert && Number(prediction?.score_pct || 0) >= 82) {
    card.className = "risk-metric risk-low";
  }

  const score = prediction ? `${prediction.score_pct}%` : "N/A";
  const level = prediction?.level || "";

  const titleNode = document.createElement("span");
  titleNode.textContent = title;

  const scoreNode = document.createElement("strong");
  scoreNode.textContent = score;

  const levelNode = document.createElement("em");
  levelNode.textContent = level;

  const detailNode = document.createElement("p");
  detailNode.textContent = prediction?.detail || "No matching historic signal.";

  card.append(titleNode, scoreNode, levelNode, detailNode);
  return card;
}

function renderExecutiveBrief(result, predictions) {
  const panel = document.createElement("section");
  panel.className = "risk-brief";
  const decision = summarizeDecision(predictions);
  panel.classList.add(decision.className);

  const heading = document.createElement("h3");
  heading.textContent = "Executive Brief";

  const grid = document.createElement("div");
  grid.className = "risk-brief-grid";
  const evidence = result.evidence || {};
  const inputs = result.job_inputs || {};
  const price = Number(inputs.job_price);
  const cost = Number(inputs.budget_cost);
  const margin =
    Number.isFinite(price) && price !== 0 && Number.isFinite(cost)
      ? `${formatCurrency(price - cost)} (${formatPercent(((price - cost) / price) * 100)})`
      : "Cost not provided";
  const evidenceBasis = `${formatNumber(evidence.similar_jobs_used)} similar / ${formatNumber(evidence.scaffolder_jobs)} scaffolder jobs`;

  const items = [
    ["Decision", decision.label],
    ["Priority", decision.priority],
    ["Reason", decision.rationale],
    ["Evidence Base", evidenceBasis],
    ["Commercial View", margin],
    ["Work Window", `${formatDate(document.getElementById("start-date")?.value)} to ${formatDate(document.getElementById("due-date")?.value)}`],
  ];

  for (const [label, value] of items) {
    appendInfoItem(grid, label, value, "risk-brief-item");
  }

  panel.append(heading, grid);
  return panel;
}

function renderPredictionBars(predictions) {
  const panel = document.createElement("section");
  panel.className = "risk-bars-panel";

  const heading = document.createElement("h3");
  heading.textContent = "Signal Analysis";

  const rows = document.createElement("div");
  rows.className = "risk-bars";

  for (const prediction of predictions) {
    const score = clamp(metricScore(prediction), 0, 100);
    const row = document.createElement("article");
    row.className = `risk-bar-row ${riskClass(prediction.level)}`;

    const head = document.createElement("div");
    head.className = "risk-bar-head";

    const title = document.createElement("strong");
    title.textContent = prediction.name;

    const value = document.createElement("span");
    value.textContent = `${formatPercent(score)} | ${prediction.level || "N/A"}`;

    const track = document.createElement("div");
    track.className = "risk-bar-track";
    const fill = document.createElement("span");
    fill.style.width = `${score}%`;
    track.appendChild(fill);

    const detail = document.createElement("p");
    detail.textContent = prediction.detail || "No supporting detail returned.";

    head.append(title, value);
    row.append(head, track, detail);
    rows.appendChild(row);
  }

  panel.append(heading, rows);
  return panel;
}

function renderCommercialPanel(result) {
  const panel = document.createElement("section");
  panel.className = "risk-evidence";

  const heading = document.createElement("h3");
  heading.textContent = "Commercial and Benchmarking";

  const grid = document.createElement("div");
  grid.className = "risk-evidence-grid";
  const inputs = result.job_inputs || {};
  const evidence = result.evidence || {};
  const price = Number(inputs.job_price);
  const cost = Number(inputs.budget_cost);
  const median = Number(evidence.median_scaffolder_price);
  const avg = Number(evidence.avg_scaffolder_price);
  const grossMargin = Number.isFinite(price) && Number.isFinite(cost) ? price - cost : null;
  const grossMarginPct =
    Number.isFinite(price) && price !== 0 && Number.isFinite(cost) ? ((price - cost) / price) * 100 : null;
  const medianDelta = Number.isFinite(price) && Number.isFinite(median) ? price - median : null;

  appendInfoItem(grid, "Quoted Price", formatCurrency(price), "risk-evidence-item");
  appendInfoItem(grid, "Scaffold Cost", Number.isFinite(cost) ? formatCurrency(cost) : "Not supplied", "risk-evidence-item");
  appendInfoItem(grid, "Gross Margin", grossMargin === null ? "N/A" : formatCurrency(grossMargin), "risk-evidence-item");
  appendInfoItem(grid, "Margin %", grossMarginPct === null ? "N/A" : formatPercent(grossMarginPct), "risk-evidence-item");
  appendInfoItem(grid, "Median Scaffolder Price", formatCurrency(median), "risk-evidence-item");
  appendInfoItem(grid, "Price vs Median", medianDelta === null ? "N/A" : formatCurrency(medianDelta), "risk-evidence-item");
  appendInfoItem(grid, "Average Scaffolder Price", formatCurrency(avg), "risk-evidence-item");
  appendInfoItem(grid, "Postcode Area", inputs.postcode_area || "N/A", "risk-evidence-item");

  panel.append(heading, grid);
  return panel;
}

function renderSummary(result) {
  const panel = document.createElement("section");
  panel.className = "risk-summary";

  const heading = document.createElement("h3");
  heading.textContent = "Job Snapshot";

  const grid = document.createElement("div");
  grid.className = "risk-summary-grid";
  const inputs = result.job_inputs || {};
  appendInfoItem(grid, "Scaffolder", result.scaffolder || "N/A");
  appendInfoItem(grid, "Postcode", inputs.postcode || "N/A");
  appendInfoItem(grid, "Lead time", formatValue(inputs.lead_time_days, " days"));
  appendInfoItem(grid, "Job price", formatCurrency(inputs.job_price));

  panel.append(heading, grid);
  return panel;
}

function renderRecommendations(recommendations) {
  const panel = document.createElement("section");
  panel.className = "risk-recommendations";

  const heading = document.createElement("h3");
  heading.textContent = "Recommendations";

  const list = document.createElement("ul");
  for (const item of recommendations || []) {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  }

  panel.append(heading, list);
  return panel;
}

function renderEvidence(evidence = {}) {
  const panel = document.createElement("section");
  panel.className = "risk-evidence";

  const heading = document.createElement("h3");
  heading.textContent = "Evidence";

  const grid = document.createElement("div");
  grid.className = "risk-evidence-grid";
  appendInfoItem(grid, "Scaffolder jobs", formatValue(evidence.scaffolder_jobs), "risk-evidence-item");
  appendInfoItem(grid, "Similar jobs", formatValue(evidence.similar_jobs_used), "risk-evidence-item");
  appendInfoItem(grid, "HSE events", formatValue(evidence.hse_events), "risk-evidence-item");
  appendInfoItem(grid, "Damage events", formatValue(evidence.damage_events), "risk-evidence-item");
  appendInfoItem(grid, "Avg price", formatCurrency(evidence.avg_scaffolder_price), "risk-evidence-item");
  appendInfoItem(grid, "Median price", formatCurrency(evidence.median_scaffolder_price), "risk-evidence-item");
  appendInfoItem(grid, "Completion", formatValue(evidence.completion_rate_pct, "%"), "risk-evidence-item");
  appendInfoItem(grid, "Cancellation", formatValue(evidence.cancellation_rate_pct, "%"), "risk-evidence-item");

  panel.append(heading, grid);
  return panel;
}

function renderTechnicalPayload(result) {
  const details = document.createElement("details");
  details.className = "technical-payload";

  const summary = document.createElement("summary");
  summary.textContent = "Technical payload";

  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(result, null, 2);

  details.append(summary, pre);
  return details;
}

function renderResults(result) {
  const output = document.getElementById("worker-output");
  output.innerHTML = "";

  const predictions = result.predictions || [];
  const failure = metricFor(predictions, "Failure / non-completion risk");
  const completion = metricFor(predictions, "Completion likelihood");
  const hse = metricFor(predictions, "HSE / incident risk");
  const programme = metricFor(predictions, "Programme risk");
  const damage = metricFor(predictions, "Damage / claim risk");
  const margin = metricFor(predictions, "Price / margin pressure");
  const cancellation = metricFor(predictions, "Cancellation / postponement risk");

  const grid = document.createElement("div");
  grid.className = "risk-metric-grid";
  grid.append(
    renderMetric("Failure Risk", failure),
    renderMetric("Completion", completion, true),
    renderMetric("HSE Risk", hse),
    renderMetric("Programme", programme),
    renderMetric("Damage Risk", damage),
    renderMetric("Margin", margin),
    renderMetric("Cancellation", cancellation)
  );

  const detailPanel = document.createElement("section");
  detailPanel.className = "risk-breakdown";
  const detailHeading = document.createElement("h3");
  detailHeading.textContent = "Risk Breakdown";
  const detailGrid = document.createElement("div");
  detailGrid.className = "risk-breakdown-grid";
  const breakdownRows = [
    ["Failure Risk", failure?.score_pct, failure?.level],
    ["Completion Likelihood", completion?.score_pct, completion?.level],
    ["HSE Risk", hse?.score_pct, hse?.level],
    ["Programme Risk", programme?.score_pct, programme?.level],
    ["Damage Risk", damage?.score_pct, damage?.level],
    ["Cancellation Risk", cancellation?.score_pct, cancellation?.level],
    ["Margin Pressure", margin?.score_pct, margin?.level],
  ];
  for (const [label, value, level] of breakdownRows) {
    const row = document.createElement("div");
    row.className = "risk-breakdown-row";
    const left = document.createElement("span");
    left.textContent = label;
    const right = document.createElement("strong");
    right.textContent = `${formatValue(value, "%")} | ${level || "N/A"}`;
    row.append(left, right);
    detailGrid.appendChild(row);
  }
  detailPanel.append(detailHeading, detailGrid);

  output.append(
    renderExecutiveBrief(result, predictions),
    renderSummary(result),
    grid,
    renderPredictionBars(predictions),
    detailPanel,
    renderCommercialPanel(result),
    renderRecommendations(result.recommendations),
    renderEvidence(result.evidence),
    renderTechnicalPayload(result)
  );
  setText("risk-state", `${result.scaffolder} | ${result.job_inputs?.postcode || "No postcode"}`);
}

function summarizeResult(result) {
  const predictions = result.predictions || [];
  const failure = metricFor(predictions, "Failure / non-completion risk");
  const completion = metricFor(predictions, "Completion likelihood");
  const hse = metricFor(predictions, "HSE / incident risk");
  const programme = metricFor(predictions, "Programme risk");
  const margin = metricFor(predictions, "Price / margin pressure");
  const decision = summarizeDecision(predictions);
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    at: new Date().toISOString(),
    scaffolder: result.scaffolder || "Unknown",
    postcode: result.job_inputs?.postcode || "N/A",
    jobType: result.job_inputs?.job_type || "N/A",
    price: result.job_inputs?.job_price ?? null,
    failure: failure?.score_pct ?? null,
    completion: completion?.score_pct ?? null,
    hse: hse?.score_pct ?? null,
    programme: programme?.score_pct ?? null,
    margin: margin?.score_pct ?? null,
    decision: decision.label,
    recommendation: (result.recommendations || [])[0] || "No recommendation provided.",
  };
}

function averageOf(items, key) {
  const values = items.map((item) => Number(item[key])).filter((value) => Number.isFinite(value));
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function renderIntelligence() {
  const historyList = document.getElementById("history-list");
  const pinned = document.getElementById("pinned-highlights");
  const count = document.getElementById("history-count-label");
  if (!historyList || !pinned || !count) return;

  count.textContent = `${workerState.history.length} runs`;
  historyList.innerHTML = "";
  pinned.innerHTML = "";

  if (!workerState.history.length) {
    historyList.innerHTML = `<div class="history-empty">No runs yet.</div>`;
    pinned.innerHTML = `<div class="history-empty">Highlights appear after prediction.</div>`;
    return;
  }

  for (const run of workerState.history.slice().reverse().slice(0, 8)) {
    const row = document.createElement("article");
    row.className = "history-row";

    const header = document.createElement("header");
    const scaffolder = document.createElement("strong");
    scaffolder.textContent = run.scaffolder;
    const time = document.createElement("span");
    time.textContent = new Date(run.at).toLocaleTimeString();
    header.append(scaffolder, time);

    const detail = document.createElement("p");
    detail.textContent = `${run.postcode} | ${run.jobType} | ${formatCurrency(run.price)}`;

    const metrics = document.createElement("div");
    metrics.className = "history-metrics";
    for (const label of [
      ["Decision", run.decision],
      ["Failure", formatPercent(run.failure)],
      ["Completion", formatPercent(run.completion)],
      ["Programme", formatPercent(run.programme)],
      ["HSE", formatPercent(run.hse)],
    ]) {
      const metric = document.createElement("span");
      metric.textContent = `${label[0]} ${label[1]}`;
      metrics.appendChild(metric);
    }

    row.append(header, detail, metrics);
    historyList.appendChild(row);
  }

  const topFailure = workerState.history
    .filter((item) => typeof item.failure === "number")
    .sort((a, b) => b.failure - a.failure)[0];
  const lowCompletion = workerState.history
    .filter((item) => typeof item.completion === "number")
    .sort((a, b) => a.completion - b.completion)[0];
  const highHse = workerState.history
    .filter((item) => typeof item.hse === "number")
    .sort((a, b) => b.hse - a.hse)[0];
  const avgFailure = averageOf(workerState.history, "failure");
  const avgCompletion = averageOf(workerState.history, "completion");

  const highlights = [
    ["Highest Failure Risk", topFailure ? `${topFailure.scaffolder} (${topFailure.failure}%)` : "N/A"],
    ["Lowest Completion", lowCompletion ? `${lowCompletion.scaffolder} (${lowCompletion.completion}%)` : "N/A"],
    ["Highest HSE Risk", highHse ? `${highHse.scaffolder} (${highHse.hse}%)` : "N/A"],
    ["Average Failure", avgFailure === null ? "N/A" : formatPercent(avgFailure)],
    ["Average Completion", avgCompletion === null ? "N/A" : formatPercent(avgCompletion)],
    ["Latest Recommendation", workerState.history[workerState.history.length - 1]?.recommendation || "N/A"],
  ];

  for (const [label, value] of highlights) {
    const item = document.createElement("div");
    item.className = "highlight-row";
    const labelNode = document.createElement("span");
    labelNode.textContent = label;
    const valueNode = document.createElement("strong");
    valueNode.textContent = value;
    item.append(labelNode, valueNode);
    pinned.appendChild(item);
  }
}

function exportLastResult() {
  if (!workerState.latestResult) return;
  const blob = new Blob([JSON.stringify(workerState.latestResult, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `risk-assessment-${Date.now()}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

function showError(message) {
  const output = document.getElementById("worker-output");
  const error = document.createElement("div");
  error.className = "risk-error";
  error.textContent = message;
  output.replaceChildren(error);
  setText("risk-state", "Check required fields");
}

async function calculateRisk() {
  try {
    updateFormReadiness();
    const jobRecord = collectJobRecord();
    if (
      !jobRecord.related_contacts ||
      !jobRecord.c_homeownerpostcode ||
      !jobRecord.c_scaffoldpurpose ||
      !jobRecord.date ||
      !jobRecord.due_date ||
      !jobRecord.budget_revenue
    ) {
      throw new Error("Scaffolder, postcode, job type, dates and job price are required.");
    }

    setText("risk-state", "Calculating...");
    const result = await api("/api/scaffolder/predict", {
      method: "POST",
      body: {
        dataset_path: workerState.datasetPath,
        job_record: jobRecord,
      },
    });
    workerState.latestResult = result;
    workerState.history.push(summarizeResult(result));
    if (workerState.history.length > 40) {
      workerState.history = workerState.history.slice(workerState.history.length - 40);
    }
    renderResults(result);
    renderIntelligence();
  } catch (error) {
    showError(error.message);
  }
}

async function loadContext() {
  workerState.datasetPath = document.body.dataset.defaultDataset || "";
  const active = await api("/api/active-model");
  const model = active.experiment || {};
  setText("active-model-label", model.target_column || "Unavailable");

  workerState.referenceData = await api("/api/portal/reference-data", {
    method: "POST",
    body: { dataset_path: workerState.datasetPath },
  });

  populateSelect("scaffold-partner", workerState.referenceData.scaffolders, "Select scaffolder");
  populateSelect("scaffold-purpose", workerState.referenceData.scaffold_purposes, "Select job type");
  populateSelect("business-name", workerState.referenceData.businesses, "Optional business");
  populateSelect("scaffold-location", workerState.referenceData.scaffold_locations, "Optional location");
  populateDatalist("postcode-list", workerState.referenceData.postcodes);

  setText("context-scaffolders", compactCount(workerState.referenceData.scaffolders));
  setText("context-businesses", compactCount(workerState.referenceData.businesses));
  setText("context-job-types", compactCount(workerState.referenceData.scaffold_purposes));
  setText("context-postcodes", compactCount(workerState.referenceData.postcodes));
}

async function init() {
  document.getElementById("predict-job-btn").addEventListener("click", calculateRisk);
  document.getElementById("worker-job-form").addEventListener("input", updateFormReadiness);
  document.getElementById("worker-job-form").addEventListener("change", updateFormReadiness);
  document.getElementById("clear-history-btn")?.addEventListener("click", () => {
    workerState.history = [];
    renderIntelligence();
  });
  document.getElementById("export-last-btn")?.addEventListener("click", exportLastResult);
  try {
    updateFormReadiness();
    await loadContext();
    updateFormReadiness();
    renderIntelligence();
  } catch (error) {
    showError(error.message);
  }
}

window.addEventListener("DOMContentLoaded", init);
