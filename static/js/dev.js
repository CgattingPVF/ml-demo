const devState = {
  defaultDatasetPath: "",
  activeModelId: "",
  activeExperiment: null,
  experiments: [],
};

async function devApi(path, options = {}) {
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

function setOutput(containerId, html, isError = false) {
  const el = document.getElementById(containerId);
  el.innerHTML = `<div class="output-box ${isError ? "error" : ""}">${html}</div>`;
}

function safeNum(value) {
  if (value === null || value === undefined || value === "") return "N/A";
  if (typeof value === "number") return value.toLocaleString();
  return String(value);
}

function setDevText(id, value) {
  const element = document.getElementById(id);
  if (!element) return;
  const text = String(value ?? "N/A");
  element.textContent = text;
  element.title = text;
}

function formatMetric(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "N/A";
  return number.toFixed(4);
}

function formatDateTime(value) {
  if (!value) return "N/A";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).replace("T", " ").slice(0, 19);
  return date.toLocaleString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function shortId(value) {
  const text = String(value || "");
  return text.length > 12 ? `${text.slice(0, 8)}...${text.slice(-4)}` : text;
}

function replaceWithNode(containerId, node) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  container.appendChild(node);
}

function createStatGrid(items) {
  const grid = document.createElement("div");
  grid.className = "ops-stat-grid";
  for (const [label, value] of items) {
    const item = document.createElement("article");
    item.className = "ops-stat";
    const labelNode = document.createElement("span");
    labelNode.textContent = label;
    const valueNode = document.createElement("strong");
    valueNode.textContent = String(value ?? "N/A");
    valueNode.title = valueNode.textContent;
    item.append(labelNode, valueNode);
    grid.appendChild(item);
  }
  return grid;
}

function createObjectTable(title, data, limit = 12) {
  const panel = document.createElement("section");
  panel.className = "ops-subpanel";
  const heading = document.createElement("h3");
  heading.textContent = title;
  const tableWrap = document.createElement("div");
  tableWrap.className = "table-wrap compact-table";
  const table = document.createElement("table");
  const tbody = document.createElement("tbody");

  for (const [key, value] of Object.entries(data || {}).slice(0, limit)) {
    const row = document.createElement("tr");
    const keyCell = document.createElement("td");
    const valueCell = document.createElement("td");
    keyCell.textContent = key;
    valueCell.textContent = safeNum(value);
    row.append(keyCell, valueCell);
    tbody.appendChild(row);
  }

  table.appendChild(tbody);
  tableWrap.appendChild(table);
  panel.append(heading, tableWrap);
  return panel;
}

function updateOpsSummary() {
  const active = devState.activeExperiment || {};
  const experiments = devState.experiments || [];
  const latest = experiments
    .slice()
    .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))[0];

  setDevText("ops-active-target", active.target_column || "Unavailable");
  setDevText("ops-experiment-count", experiments.length);
  setDevText(
    "ops-best-score",
    active.metric_name ? `${active.metric_name}: ${formatMetric(active.metric_value)}` : "No active model"
  );
  setDevText("ops-last-trained", latest ? formatDateTime(latest.created_at) : "No runs");
  setDevText("ops-dataset-name", devState.defaultDatasetPath.split(/[\\/]/).pop() || "Dataset");
}

function renderDatasetProfile(profile) {
  const root = document.createElement("div");
  root.className = "ops-output-grid";

  root.appendChild(
    createStatGrid([
      ["Rows", safeNum(profile.rows)],
      ["Columns", safeNum((profile.columns || []).length || profile.columns)],
      ["Target Candidates", safeNum((profile.target_candidates || []).length)],
      ["Missing Fields Tracked", safeNum(Object.keys(profile.top_missing_pct || {}).length)],
    ])
  );

  const targetPanel = document.createElement("section");
  targetPanel.className = "ops-subpanel";
  const targetHeading = document.createElement("h3");
  targetHeading.textContent = "Recommended Target Columns";
  const targetList = document.createElement("div");
  targetList.className = "target-chip-list";
  for (const target of (profile.target_candidates || []).slice(0, 24)) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "target-chip";
    button.textContent = target;
    button.title = `Use ${target} as the training target`;
    button.addEventListener("click", () => {
      document.getElementById("dev-target-column").value = target;
      document.getElementById("dev-train-dataset-path").value =
        document.getElementById("dev-dataset-path").value.trim() || devState.defaultDatasetPath;
      document.getElementById("dev-target-column").focus();
    });
    targetList.appendChild(button);
  }
  if (!targetList.children.length) {
    const empty = document.createElement("p");
    empty.className = "ops-muted";
    empty.textContent = "No target suggestions were returned for this dataset.";
    targetList.appendChild(empty);
  }
  targetPanel.append(targetHeading, targetList);

  const insightGrid = document.createElement("div");
  insightGrid.className = "ops-three-grid";
  insightGrid.append(
    createObjectTable("Data Types", profile.dtype_breakdown, 8),
    createObjectTable("Highest Missing %", profile.top_missing_pct, 10),
    createObjectTable("Highest Cardinality", profile.top_cardinality, 10)
  );

  root.append(targetPanel, insightGrid);
  return root;
}

function renderTrainingResult(result) {
  const root = document.createElement("div");
  root.className = "ops-output-grid";
  const exp = result.experiment || {};

  root.appendChild(
    createStatGrid([
      ["Experiment", shortId(exp.id)],
      ["Best Model", result.best_model_name],
      ["Primary Metric", `${exp.metric_name}: ${formatMetric(exp.metric_value)}`],
      ["Rows", safeNum(exp.total_rows)],
      ["Columns", safeNum(exp.total_columns)],
      ["Task", exp.task_type],
    ])
  );

  const candidatePanel = document.createElement("section");
  candidatePanel.className = "ops-subpanel";
  const candidateHeading = document.createElement("h3");
  candidateHeading.textContent = "Candidate Model Scores";
  const candidateWrap = document.createElement("div");
  candidateWrap.className = "table-wrap compact-table";
  const candidateTable = document.createElement("table");
  const candidateHead = document.createElement("thead");
  candidateHead.innerHTML = "<tr><th>Model</th><th>Primary</th><th>Metrics</th></tr>";
  const candidateBody = document.createElement("tbody");
  for (const item of result.all_results || []) {
    const row = document.createElement("tr");
    const modelCell = document.createElement("td");
    const primaryCell = document.createElement("td");
    const metricsCell = document.createElement("td");
    modelCell.textContent = item.model_name;
    primaryCell.textContent = `${item.primary_metric_name}: ${formatMetric(item.primary_metric_value)}`;
    metricsCell.textContent = Object.entries(item.metrics || {})
      .map(([key, value]) => `${key} ${formatMetric(value)}`)
      .join(" | ");
    row.append(modelCell, primaryCell, metricsCell);
    candidateBody.appendChild(row);
  }
  candidateTable.append(candidateHead, candidateBody);
  candidateWrap.appendChild(candidateTable);
  candidatePanel.append(candidateHeading, candidateWrap);

  const featurePanel = document.createElement("section");
  featurePanel.className = "ops-subpanel";
  const featureHeading = document.createElement("h3");
  featureHeading.textContent = "Top Feature Signals";
  const featureList = document.createElement("div");
  featureList.className = "feature-list";
  for (const feature of (result.feature_importance || []).slice(0, 12)) {
    const score = Number(feature.importance);
    const row = document.createElement("div");
    row.className = "feature-row";
    const label = document.createElement("span");
    label.textContent = feature.feature;
    const value = document.createElement("strong");
    value.textContent = Number.isFinite(score) ? score.toFixed(5) : "N/A";
    row.append(label, value);
    featureList.appendChild(row);
  }
  if (!featureList.children.length) {
    const empty = document.createElement("p");
    empty.className = "ops-muted";
    empty.textContent = "This model type did not expose feature importance.";
    featureList.appendChild(empty);
  }
  featurePanel.append(featureHeading, featureList);

  root.append(candidatePanel, featurePanel);
  return root;
}

function renderExperimentDetails(experiment) {
  const root = document.createElement("div");
  root.className = "ops-output-grid";
  const metadata = experiment.metadata || {};
  const training = metadata.training_summary || {};

  root.appendChild(
    createStatGrid([
      ["Experiment", shortId(experiment.id)],
      ["Target", experiment.target_column],
      ["Task", experiment.task_type],
      ["Model", experiment.best_model_name],
      ["Metric", `${experiment.metric_name}: ${formatMetric(experiment.metric_value)}`],
      ["Training Rows", safeNum(training.train_rows)],
      ["Test Rows", safeNum(training.test_rows)],
      ["Features", safeNum((metadata.feature_columns || []).length)],
      ["Created", formatDateTime(experiment.created_at)],
    ])
  );

  const featurePanel = document.createElement("section");
  featurePanel.className = "ops-subpanel";
  const featureHeading = document.createElement("h3");
  featureHeading.textContent = "Feature Columns";
  const featureColumns = document.createElement("div");
  featureColumns.className = "target-chip-list";
  for (const feature of (metadata.feature_columns || []).slice(0, 80)) {
    const chip = document.createElement("span");
    chip.className = "target-chip static";
    chip.textContent = feature;
    featureColumns.appendChild(chip);
  }
  featurePanel.append(featureHeading, featureColumns);

  const resultPanel = document.createElement("section");
  resultPanel.className = "ops-subpanel";
  const resultHeading = document.createElement("h3");
  resultHeading.textContent = "Training Run Results";
  const resultWrap = document.createElement("div");
  resultWrap.className = "table-wrap compact-table";
  const table = document.createElement("table");
  const head = document.createElement("thead");
  head.innerHTML = "<tr><th>Model</th><th>Primary</th><th>Metrics</th></tr>";
  const body = document.createElement("tbody");
  for (const item of training.results || []) {
    const row = document.createElement("tr");
    const modelCell = document.createElement("td");
    const primaryCell = document.createElement("td");
    const metricCell = document.createElement("td");
    modelCell.textContent = item.model_name;
    primaryCell.textContent = `${item.primary_metric_name}: ${formatMetric(item.primary_metric_value)}`;
    metricCell.textContent = Object.entries(item.metrics || {})
      .map(([key, value]) => `${key} ${formatMetric(value)}`)
      .join(" | ");
    row.append(modelCell, primaryCell, metricCell);
    body.appendChild(row);
  }
  table.append(head, body);
  resultWrap.appendChild(table);
  resultPanel.append(resultHeading, resultWrap);

  root.append(featurePanel, resultPanel);
  return root;
}

async function loadExperimentDetails(experimentId) {
  try {
    const experiment = await devApi(`/api/experiments/${experimentId}`);
    replaceWithNode("dev-detail-output", renderExperimentDetails(experiment));
  } catch (error) {
    setOutput("dev-detail-output", error.message, true);
  }
}

async function refreshActiveModel() {
  const active = await devApi("/api/active-model");
  devState.activeModelId = active.active_experiment_id;
  devState.activeExperiment = active.experiment || null;
  const exp = active.experiment || {};
  setDevText(
    "dev-active-model",
    `${shortId(exp.id)} | ${exp.target_column} | ${exp.best_model_name} | ${exp.metric_name}: ${formatMetric(exp.metric_value)}`
  );
  updateOpsSummary();
}

async function refreshExperimentsTable() {
  devState.experiments = await devApi("/api/experiments?limit=200");
  const tbody = document.querySelector("#dev-experiments-table tbody");
  tbody.innerHTML = "";

  for (const exp of devState.experiments) {
    const row = document.createElement("tr");
    const isActive = exp.id === devState.activeModelId;

    const cells = [
      shortId(exp.id),
      formatDateTime(exp.created_at),
      exp.target_column,
      exp.task_type,
      exp.best_model_name,
      `${exp.metric_name}: ${formatMetric(exp.metric_value)}`,
      safeNum(exp.total_rows),
    ];

    for (const value of cells) {
      const cell = document.createElement("td");
      cell.textContent = value;
      cell.title = String(value);
      row.appendChild(cell);
    }

    const actionTd = document.createElement("td");
    actionTd.className = "dev-actions-cell";

    const detailsBtn = document.createElement("button");
    detailsBtn.className = "dev-action-btn";
    detailsBtn.textContent = "Details";
    detailsBtn.addEventListener("click", () => loadExperimentDetails(exp.id));

    const btn = document.createElement("button");
    btn.className = `dev-action-btn ${isActive ? "active" : ""}`;
    btn.textContent = isActive ? "Active" : "Set Active";
    btn.disabled = isActive;
    btn.addEventListener("click", async () => {
      try {
        await devApi("/api/active-model", {
          method: "POST",
          body: { experiment_id: exp.id },
        });
        await refreshAllDevData();
      } catch (error) {
        setOutput("dev-train-output", error.message, true);
      }
    });
    actionTd.append(detailsBtn, btn);
    row.appendChild(actionTd);
    tbody.appendChild(row);
  }

  updateOpsSummary();
}

async function runDevProfile() {
  try {
    const datasetPath = document.getElementById("dev-dataset-path").value.trim();
    setDevText("ops-dataset-name", datasetPath.split(/[\\/]/).pop() || "Dataset");
    const profile = await devApi("/api/profile", {
      method: "POST",
      body: { dataset_path: datasetPath },
    });
    replaceWithNode("dev-profile-output", renderDatasetProfile(profile));
  } catch (error) {
    setOutput("dev-profile-output", error.message, true);
  }
}

async function runDevTraining() {
  try {
    const payload = {
      dataset_path: document.getElementById("dev-train-dataset-path").value.trim(),
      target_column: document.getElementById("dev-target-column").value.trim(),
      task_type: document.getElementById("dev-task-type").value || null,
      test_size: Number(document.getElementById("dev-test-size").value || 0.2),
      random_state: Number(document.getElementById("dev-random-state").value || 42),
    };
    if (!payload.target_column) {
      throw new Error("Target column is required before training.");
    }

    const result = await devApi("/api/train", {
      method: "POST",
      body: payload,
    });
    replaceWithNode("dev-train-output", renderTrainingResult(result));
    await refreshAllDevData();
  } catch (error) {
    setOutput("dev-train-output", error.message, true);
  }
}

async function refreshAllDevData() {
  await refreshActiveModel();
  await refreshExperimentsTable();
  if (devState.activeModelId) {
    await loadExperimentDetails(devState.activeModelId);
  }
}

async function initDevPage() {
  devState.defaultDatasetPath = document.body.dataset.defaultDataset || "";
  document.getElementById("dev-dataset-path").value = devState.defaultDatasetPath;
  document.getElementById("dev-train-dataset-path").value = devState.defaultDatasetPath;

  document.getElementById("dev-profile-btn").addEventListener("click", runDevProfile);
  document.getElementById("dev-train-btn").addEventListener("click", runDevTraining);

  try {
    await refreshAllDevData();
  } catch (error) {
    setOutput("dev-train-output", error.message, true);
  }
}

window.addEventListener("DOMContentLoaded", initDevPage);
