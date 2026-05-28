const workerState = {
  datasetPath: "",
  referenceData: {},
  history: [],
  latestResult: null,
  latestMlPrediction: null,
  activeModel: null,
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

function formatSignedPercent(value) {
  if (value === null || value === undefined || value === "") return "N/A";
  const number = Number(value);
  if (!Number.isFinite(number)) return "N/A";
  return `${number > 0 ? "+" : ""}${formatPercent(number)}`;
}

function taskLabel(task) {
  if (task === "classification") return "Classification";
  if (task === "regression") return "Regression";
  return "Model";
}

function setActiveTaskMode(task) {
  document.body.classList.remove("active-task-classification", "active-task-regression");
  if (task === "classification" || task === "regression") {
    document.body.classList.add(`active-task-${task}`);
  }
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

function selectedOptionText(id, fallback = "") {
  const select = document.getElementById(id);
  if (!select || select.selectedIndex < 0) return fallback;
  const option = select.options[select.selectedIndex];
  return option?.value ? option.textContent : fallback;
}

function setDecisionStrip(items, className = "") {
  const strip = document.getElementById("decision-strip");
  if (!strip) return;

  strip.className = `pvf-decision-strip ${className}`.trim();
  strip.innerHTML = "";

  for (const [label, value] of items) {
    const item = document.createElement("article");
    const labelNode = document.createElement("span");
    const valueNode = document.createElement("strong");
    labelNode.textContent = label;
    valueNode.textContent = value;
    valueNode.title = value;
    item.append(labelNode, valueNode);
    strip.appendChild(item);
  }
}

function formatModelPredictionValue(target, value) {
  const targetText = String(target || "").toLowerCase();
  if (value === null || value === undefined || value === "") return "N/A";
  if (
    targetText.includes("revenue") ||
    targetText.includes("cost") ||
    targetText.includes("price") ||
    targetText.includes("budget")
  ) {
    return formatCurrency(value);
  }
  const number = Number(value);
  if (Number.isFinite(number)) {
    return formatNumber(number, { maximumFractionDigits: 2 });
  }
  return String(value);
}

function updateMlPredictionPanel(payload = null, status = "") {
  const output = document.getElementById("ml-prediction-output");
  const activeModel = workerState.activeModel || {};
  const target = activeModel.target_column || "Active target";
  const task = activeModel.task_type || "";
  setText("ml-panel-target", `${target} | ${taskLabel(task)}`);
  setText("ml-panel-state", status || "Uses reviewed active model");
  if (!output) return;

  output.innerHTML = "";
  if (status && !payload) {
    const state = document.createElement("div");
    state.className = "pvf-model-empty";
    state.textContent = status;
    output.appendChild(state);
    return;
  }
  if (payload?.error) {
    const error = document.createElement("div");
    error.className = "pvf-model-error";
    error.textContent = payload.error;
    output.appendChild(error);
    setText("ml-panel-state", "Prediction unavailable");
    return;
  }
  const prediction = payload?.predictions?.[0];
  if (prediction === undefined) {
    const empty = document.createElement("div");
    empty.className = "pvf-model-empty";
    empty.textContent = "Run an assessment to see the active model prediction for this job.";
    output.appendChild(empty);
    return;
  }

  const grid = document.createElement("div");
  grid.className = "pvf-model-grid";
  const confidence = payload?.confidence?.[0];
  const items = [
    ["Prediction", formatModelPredictionValue(target, prediction)],
    ["Target", target],
    ["Task", taskLabel(task)],
    ["Confidence", confidence === undefined ? "N/A" : formatPercent(Number(confidence) * 100)],
  ];
  for (const [label, value] of items) {
    const item = document.createElement("div");
    const labelNode = document.createElement("span");
    const valueNode = document.createElement("strong");
    labelNode.textContent = label;
    valueNode.textContent = value;
    valueNode.title = value;
    item.append(labelNode, valueNode);
    grid.appendChild(item);
  }
  output.appendChild(grid);
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

function projectDateWindow() {
  const start = document.getElementById("start-date")?.value || "";
  const due = document.getElementById("due-date")?.value || "";
  if (start && due) {
    return start === due ? formatDate(start) : `${formatDate(start)} to ${formatDate(due)}`;
  }
  if (start) return `From ${formatDate(start)}`;
  if (due) return `Due ${formatDate(due)}`;
  return "Set dates";
}

function projectLeadTimeSummary() {
  const start = document.getElementById("start-date")?.value || "";
  const due = document.getElementById("due-date")?.value || "";
  if (!start || !due) return "Set dates";

  const startDate = new Date(start);
  const dueDate = new Date(due);
  if (Number.isNaN(startDate.getTime()) || Number.isNaN(dueDate.getTime())) return "Check dates";

  const days = Math.round((dueDate - startDate) / 86400000);
  if (days < 0) return "Date conflict";
  if (days === 0) return "Same day";
  return `${days} day${days === 1 ? "" : "s"}`;
}

function projectCommercialSummary() {
  const price = Number(document.getElementById("budget-revenue")?.value);
  const cost = Number(document.getElementById("budget-cost")?.value);
  if (!Number.isFinite(price) || price <= 0) return "Add price";
  if (Number.isFinite(cost) && cost > 0) {
    const margin = price - cost;
    const marginPct = price ? (margin / price) * 100 : null;
    return `${formatCurrency(price)} | margin ${formatCurrency(margin)} (${formatPercent(marginPct)})`;
  }
  return formatCurrency(price);
}

function updateProjectPreview() {
  const team = selectedOptionText("scaffold-partner", "Select team");
  const postcode = document.getElementById("homeowner-postcode")?.value.trim().toUpperCase() || "Add postcode";
  const jobType = selectedOptionText("scaffold-purpose", "Select type");

  setText("preview-team", team);
  setText("preview-postcode", postcode);
  setText("preview-window", projectDateWindow());
  setText("preview-lead-time", projectLeadTimeSummary());
  setText("preview-job-type", jobType);
  setText("preview-commercial", projectCommercialSummary());
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
  setText("overview-readiness", `${percent}%`);
  setText(
    "form-status-chip",
    missing.length ? `${missing.length} missing` : "Ready"
  );
  document.body.classList.toggle("is-ready", missing.length === 0);
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
  updateProjectPreview();
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

const TEAM_STAT_HELP = {
  Fit: "Overall match score for this team on the current job, combining delivery history, similar work, safety signals and price fit.",
  Comp: "Completion rate from this team's historical jobs. Higher means more of their jobs reached completion rather than staying open, cancelled or postponed.",
  Cancel: "Cancellation or postponement rate for this team. Lower is better and suggests fewer disrupted jobs.",
  Similar: "Number of historic jobs used as a close comparison for this team, based on job type, postcode area and commercial profile.",
  Median: "Median historic price for comparable work by this team. Use this as the central benchmark rather than an outlier-sensitive average.",
  Delta: "Difference between the quoted job price and this team's median comparable price. Negative means the quote is below the team's usual benchmark; positive means above.",
  HSE: "Historic health, safety or incident signal rate for this team's jobs. Lower is better.",
  Jobs: "Total historic jobs found for this team in the dataset. Higher counts usually make the other signals more reliable.",
};

const INFO_ITEM_HELP = {
  Decision: "Recommended operating decision for this assessment based on the strongest risk and delivery signals.",
  "Priority Signal": "The signal that most needs review before booking or handover.",
  "Best Match": "Top-ranked team based on the current job details and historical team performance.",
  "Compared Teams": "Number of teams considered when creating the shortlist.",
  Completion: "Estimated or historical completion performance. Higher is usually better.",
  Cancellation: "Estimated or historical cancellation/postponement pressure. Lower is usually better.",
  Priority: "The most important issue to address before the job proceeds.",
  Reason: "Plain-language summary of why the recommendation was chosen.",
  "Evidence Base": "How many comparable jobs and team records support the assessment.",
  "Commercial View": "Margin or commercial context calculated from the supplied job price and cost.",
  "Work Window": "The start-to-due date window entered for this assessment.",
  "Quoted Price": "The job price entered in the project form.",
  "Scaffold Cost": "Optional cost input used to estimate gross margin.",
  "Gross Margin": "Quoted price minus supplied scaffold cost.",
  "Margin %": "Gross margin as a percentage of quoted price.",
  "Median Scaffolder Price": "Median comparable price for the selected scaffolder or matched team context.",
  "Price vs Median": "Difference between the quoted price and the relevant median benchmark.",
  "Average Scaffolder Price": "Average comparable price for the selected scaffolder or matched team context.",
  "Postcode Area": "Outward postcode area used when comparing similar jobs.",
  "Scaffolder jobs": "Total historical jobs available for the selected scaffolder.",
  "Similar jobs": "Comparable historical jobs used to support this assessment.",
  "HSE events": "Historical health and safety events detected in the matched evidence.",
  "Damage events": "Historical damage or claim events detected in the matched evidence.",
  "Avg price": "Average price across matched historical evidence.",
  "Median price": "Median price across matched historical evidence.",
};

function addMetricTooltip(item, label, value, detail) {
  if (!detail) return;

  item.classList.add("has-metric-tooltip");
  item.tabIndex = 0;
  item.setAttribute("aria-label", `${label}: ${value}. ${detail}`);

  const tooltip = document.createElement("span");
  tooltip.className = "metric-tooltip";
  tooltip.setAttribute("role", "tooltip");
  tooltip.textContent = detail;
  item.appendChild(tooltip);
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
  addMetricTooltip(item, label, valueNode.textContent, INFO_ITEM_HELP[label]);
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

function selectedTeamFromResult(result) {
  const leaderboard = result.team_leaderboard || [];
  const selected = leaderboard.find((team) => team.is_selected);
  if (selected) return selected;
  return leaderboard.find((team) => String(team.team || "").toLowerCase() === String(result.scaffolder || "").toLowerCase());
}

function teamFitClass(score) {
  const value = Number(score);
  if (value >= 78) return "team-fit-strong";
  if (value >= 62) return "team-fit-watch";
  return "team-fit-review";
}

function appendTeamStat(container, label, value) {
  const stat = document.createElement("div");
  stat.className = "team-stat";

  const labelNode = document.createElement("span");
  labelNode.textContent = label;

  const valueNode = document.createElement("strong");
  valueNode.textContent = String(value ?? "N/A");
  valueNode.title = valueNode.textContent;

  stat.append(labelNode, valueNode);
  addMetricTooltip(stat, label, valueNode.textContent, TEAM_STAT_HELP[label]);
  container.appendChild(stat);
}

function renderFitMeter(score) {
  const meter = document.createElement("div");
  meter.className = "team-fit-meter";
  const fill = document.createElement("span");
  fill.style.width = `${clamp(Number(score) || 0, 0, 100)}%`;
  meter.appendChild(fill);
  return meter;
}

function renderSelectedTeamAssessment(result, predictions) {
  const panel = document.createElement("section");
  panel.className = "team-assessment";

  const decision = summarizeDecision(predictions);
  panel.classList.add(decision.className);

  const selectedTeam = selectedTeamFromResult(result);
  const summary = result.team_match_summary || {};
  const topTeam = (result.team_leaderboard || [])[0];
  const fitScore = selectedTeam?.fit_score ?? summary.selected_fit_score;

  const main = document.createElement("div");
  main.className = "team-assessment-main";

  const copy = document.createElement("div");
  const kicker = document.createElement("span");
  kicker.className = "team-kicker";
  kicker.textContent = "Selected Team Assessment";

  const heading = document.createElement("h3");
  heading.textContent = result.scaffolder || "Selected team";
  heading.title = heading.textContent;

  const reason = document.createElement("p");
  reason.textContent = selectedTeam?.reason || decision.rationale;

  copy.append(kicker, heading, reason);

  const scoreBox = document.createElement("div");
  scoreBox.className = `team-score-box ${teamFitClass(fitScore)}`;
  const scoreLabel = document.createElement("span");
  scoreLabel.textContent = "Fit Score";
  const scoreValue = document.createElement("strong");
  scoreValue.textContent = fitScore === undefined || fitScore === null ? "N/A" : formatPercent(fitScore);
  const scoreRank = document.createElement("em");
  scoreRank.textContent = selectedTeam?.rank ? `Rank #${selectedTeam.rank}` : "Unranked";
  scoreBox.append(scoreLabel, scoreValue, scoreRank, renderFitMeter(fitScore));

  main.append(copy, scoreBox);

  const grid = document.createElement("div");
  grid.className = "team-assessment-grid";
  appendInfoItem(grid, "Decision", decision.label, "team-assessment-item");
  appendInfoItem(grid, "Priority Signal", decision.priority, "team-assessment-item");
  appendInfoItem(grid, "Best Match", topTeam?.team || summary.best_team || "N/A", "team-assessment-item");
  appendInfoItem(grid, "Compared Teams", formatNumber(summary.teams_compared), "team-assessment-item");
  appendInfoItem(grid, "Completion", formatPercent(selectedTeam?.completion_rate_pct), "team-assessment-item");
  appendInfoItem(grid, "Cancellation", formatPercent(selectedTeam?.cancellation_rate_pct), "team-assessment-item");

  panel.append(main, grid);
  return panel;
}

function renderTeamLeaderboard(result) {
  const leaderboard = result.team_leaderboard || [];
  const panel = document.createElement("section");
  panel.className = "team-leaderboard-panel";

  const header = document.createElement("header");
  const heading = document.createElement("h3");
  heading.textContent = "Ranked Team Shortlist";
  const meta = document.createElement("span");
  const summary = result.team_match_summary || {};
  meta.textContent = `${formatNumber(summary.teams_compared)} compared`;
  header.append(heading, meta);

  const list = document.createElement("div");
  list.className = "team-rank-list";

  if (!leaderboard.length) {
    const empty = document.createElement("div");
    empty.className = "team-rank-empty";
    empty.textContent = "No ranked teams returned for this job.";
    list.appendChild(empty);
    panel.append(header, list);
    return panel;
  }

  for (const team of leaderboard) {
    const card = document.createElement("article");
    card.className = `team-rank-card ${teamFitClass(team.fit_score)}`;
    if (team.is_selected) card.classList.add("is-selected");

    const rank = document.createElement("div");
    rank.className = "team-rank-badge";
    const rankLabel = document.createElement("span");
    rankLabel.textContent = "Rank";
    const rankValue = document.createElement("strong");
    rankValue.textContent = `#${team.rank || "-"}`;
    rank.append(rankLabel, rankValue);

    const identity = document.createElement("div");
    identity.className = "team-rank-identity";
    const nameRow = document.createElement("div");
    nameRow.className = "team-rank-name-row";
    const name = document.createElement("strong");
    name.textContent = team.team || "Unknown team";
    name.title = name.textContent;
    nameRow.appendChild(name);
    if (team.is_selected) {
      const selected = document.createElement("span");
      selected.textContent = "Selected";
      nameRow.appendChild(selected);
    }
    const reason = document.createElement("p");
    reason.textContent = team.reason || "Ranked from historical delivery, safety and commercial signals.";
    identity.append(nameRow, renderFitMeter(team.fit_score), reason);

    const stats = document.createElement("div");
    stats.className = "team-rank-stats";
    appendTeamStat(stats, "Fit", formatPercent(team.fit_score));
    appendTeamStat(stats, "Comp", formatPercent(team.completion_rate_pct));
    appendTeamStat(stats, "Cancel", formatPercent(team.cancellation_rate_pct));
    appendTeamStat(stats, "Similar", formatNumber(team.similar_jobs));
    appendTeamStat(stats, "Median", formatCurrency(team.median_price));
    appendTeamStat(stats, "Delta", formatSignedPercent(team.price_delta_pct));
    appendTeamStat(stats, "HSE", formatPercent(team.hse_rate_pct));
    appendTeamStat(stats, "Jobs", formatNumber(team.total_jobs));

    card.append(rank, identity, stats);
    list.appendChild(card);
  }

  panel.append(header, list);
  return panel;
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
  const decision = summarizeDecision(predictions);
  const failure = metricFor(predictions, "Failure / non-completion risk");
  const completion = metricFor(predictions, "Completion likelihood");
  const hse = metricFor(predictions, "HSE / incident risk");
  const programme = metricFor(predictions, "Programme risk");
  const damage = metricFor(predictions, "Damage / claim risk");
  const margin = metricFor(predictions, "Price / margin pressure");
  const cancellation = metricFor(predictions, "Cancellation / postponement risk");
  const evidence = result.evidence || {};
  const matchSummary = result.team_match_summary || {};
  const selectedTeam = selectedTeamFromResult(result);
  const bestTeam = (result.team_leaderboard || [])[0];
  const selectedRank = selectedTeam?.rank ? `#${selectedTeam.rank}` : "Unranked";

  setText("overview-best-team", bestTeam?.team || matchSummary.best_team || "N/A");

  setDecisionStrip(
    [
      ["Decision", decision.label],
      ["Selected Rank", selectedRank],
      ["Best Team", bestTeam?.team || matchSummary.best_team || "N/A"],
      [
        "Evidence",
        `${formatNumber(evidence.similar_jobs_used)} similar / ${formatNumber(evidence.scaffolder_jobs)} team jobs`,
      ],
    ],
    decision.className
  );

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
    renderSelectedTeamAssessment(result, predictions),
    renderTeamLeaderboard(result),
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
  const matchSummary = result.team_match_summary || {};
  const selectedTeam = selectedTeamFromResult(result);
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    at: new Date().toISOString(),
    scaffolder: result.scaffolder || "Unknown",
    bestTeam: matchSummary.best_team || result.team_leaderboard?.[0]?.team || "N/A",
    selectedRank: selectedTeam?.rank ?? matchSummary.selected_rank ?? null,
    fitScore: selectedTeam?.fit_score ?? matchSummary.selected_fit_score ?? null,
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
  setText("overview-runs", workerState.history.length);
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
      ["Rank", run.selectedRank ? `#${run.selectedRank}` : "N/A"],
      ["Fit", formatPercent(run.fitScore)],
      ["Failure", formatPercent(run.failure)],
      ["Completion", formatPercent(run.completion)],
      ["Programme", formatPercent(run.programme)],
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
  const avgFit = averageOf(workerState.history, "fitScore");
  const latestRun = workerState.history[workerState.history.length - 1];

  const highlights = [
    ["Latest Best Match", latestRun?.bestTeam || "N/A"],
    ["Latest Selected Rank", latestRun?.selectedRank ? `#${latestRun.selectedRank}` : "N/A"],
    ["Highest Failure Risk", topFailure ? `${topFailure.scaffolder} (${topFailure.failure}%)` : "N/A"],
    ["Lowest Completion", lowCompletion ? `${lowCompletion.scaffolder} (${lowCompletion.completion}%)` : "N/A"],
    ["Highest HSE Risk", highHse ? `${highHse.scaffolder} (${highHse.hse}%)` : "N/A"],
    ["Average Fit", avgFit === null ? "N/A" : formatPercent(avgFit)],
    ["Average Failure", avgFailure === null ? "N/A" : formatPercent(avgFailure)],
    ["Average Completion", avgCompletion === null ? "N/A" : formatPercent(avgCompletion)],
    ["Latest Recommendation", latestRun?.recommendation || "N/A"],
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
  const payload = {
    operational_assessment: workerState.latestResult,
    active_model_prediction: workerState.latestMlPrediction,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
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
  setText("overview-best-team", "N/A");
  setDecisionStrip(
    [
      ["Decision", "Input check required"],
      ["Selected Rank", "Not calculated"],
      ["Best Team", "Not calculated"],
      ["Evidence", "Run paused"],
    ],
    "decision-high"
  );
}

function setDefaultProjectDates() {
  const start = document.getElementById("start-date");
  const due = document.getElementById("due-date");
  const today = new Date().toISOString().slice(0, 10);
  if (start && !start.value) start.value = today;
  if (due && !due.value) due.value = today;
}

function initThemeToggle() {
  const toggle = document.getElementById("theme-toggle");
  if (!toggle) return;

  const setTheme = (isDark) => {
    document.body.classList.toggle("is-dark", isDark);
    toggle.textContent = `Dark Mode: ${isDark ? "On" : "Off"}`;
    try {
      localStorage.setItem("pvf-theme", isDark ? "dark" : "light");
    } catch {
      // Theme persistence is a convenience; the toggle still works without storage.
    }
  };

  let savedTheme = "light";
  try {
    savedTheme = localStorage.getItem("pvf-theme") || "light";
  } catch {
    savedTheme = "light";
  }

  setTheme(savedTheme === "dark");
  toggle.addEventListener("click", () => setTheme(!document.body.classList.contains("is-dark")));
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
    setDecisionStrip(
      [
        ["Decision", "Calculating"],
        ["Selected Rank", "Ranking teams"],
        ["Best Team", "Matching shortlist"],
        ["Evidence", "Matching model and historical jobs"],
      ],
      "is-calculating"
    );
    updateMlPredictionPanel(null, "Calculating active model prediction...");
    const [result, mlPrediction] = await Promise.all([
      api("/api/scaffolder/predict", {
        method: "POST",
        body: {
          dataset_path: workerState.datasetPath,
          job_record: jobRecord,
        },
      }),
      api("/api/active-model/predict", {
        method: "POST",
        body: {
          job_record: jobRecord,
        },
      }).catch((error) => ({ error: error.message })),
    ]);
    workerState.latestResult = result;
    workerState.latestMlPrediction = mlPrediction;
    workerState.history.push(summarizeResult(result));
    if (workerState.history.length > 40) {
      workerState.history = workerState.history.slice(workerState.history.length - 40);
    }
    renderResults(result);
    updateMlPredictionPanel(mlPrediction);
    renderIntelligence();
  } catch (error) {
    showError(error.message);
    updateMlPredictionPanel(workerState.latestMlPrediction);
  }
}

async function loadContext() {
  workerState.datasetPath = document.body.dataset.defaultDataset || "";
  setText("overview-context", "Loading reference data");
  const active = await api("/api/active-model");
  const model = active.experiment || {};
  workerState.activeModel = model;
  setText("active-model-label", model.target_column || "Unavailable");
  setText("active-task-label", taskLabel(model.task_type));
  setText("overview-target", model.target_column || "Unavailable");
  setActiveTaskMode(model.task_type);
  updateMlPredictionPanel();

  try {
    workerState.referenceData = await api("/api/portal/reference-data", {
      method: "POST",
      body: { dataset_path: workerState.datasetPath },
    });
  } catch (error) {
    workerState.referenceData = {};
    setText("overview-context", "Reference data unavailable");
    throw error;
  }

  populateSelect("scaffold-partner", workerState.referenceData.scaffolders, "Select scaffolder");
  populateSelect("scaffold-purpose", workerState.referenceData.scaffold_purposes, "Select job type");
  populateSelect("business-name", workerState.referenceData.businesses, "Optional business");
  populateSelect("scaffold-location", workerState.referenceData.scaffold_locations, "Optional location");

  setText("context-scaffolders", compactCount(workerState.referenceData.scaffolders));
  setText("context-businesses", compactCount(workerState.referenceData.businesses));
  setText("context-job-types", compactCount(workerState.referenceData.scaffold_purposes));
  setText("context-postcodes", compactCount(workerState.referenceData.postcodes));
  setText(
    "overview-context",
    `${compactCount(workerState.referenceData.scaffolders)} teams / ${compactCount(workerState.referenceData.scaffold_purposes)} job types`
  );
}

async function init() {
  initThemeToggle();
  setDefaultProjectDates();
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
