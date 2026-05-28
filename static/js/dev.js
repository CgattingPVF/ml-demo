const devState = {
  defaultDatasetPath: "",
  activeModelId: "",
  activeExperiment: null,
  experiments: [],
  profile: null,
  targetProfiles: [],
  trainingProgressTimer: null,
  trainingProgressValue: 0,
};

const TASK_LABELS = {
  classification: "Classification",
  regression: "Regression",
  auto: "Auto",
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

function taskLabel(task) {
  return TASK_LABELS[task] || "Auto";
}

function selectedTarget() {
  return document.getElementById("dev-target-column").value.trim();
}

function taskOverride() {
  return document.getElementById("dev-task-type").value || "";
}

function effectiveTask(profile) {
  return taskOverride() || profile?.recommended_task_type || "auto";
}

function targetProfileFor(column) {
  return devState.targetProfiles.find((profile) => profile.column === column) || null;
}

function setDevText(id, value) {
  const element = document.getElementById(id);
  if (!element) return;
  const text = String(value ?? "N/A");
  element.textContent = text;
  element.title = text;
}

function setTrainingProgress(value, stepText = "") {
  const progress = Math.max(0, Math.min(100, Math.round(value)));
  devState.trainingProgressValue = progress;
  setDevText("dev-progress-percent", `${progress}%`);
  const bar = document.getElementById("dev-progress-bar");
  if (bar) {
    bar.style.width = `${progress}%`;
  }
  if (stepText) {
    setDevText("dev-progress-step", stepText);
  }
}

function startTrainingProgress(task, target) {
  const panel = document.getElementById("dev-training-progress");
  const profileButton = document.getElementById("dev-profile-btn");
  if (!panel) return;

  const taskName = taskLabel(task);
  const steps =
    task === "classification"
      ? [
          "Validating class balance and train/test split.",
          "Encoding categorical fields for classifier candidates.",
          "Fitting logistic, forest, and boosting classifiers.",
          "Scoring weighted precision, recall, and F1.",
          "Preparing release review and feature signals.",
        ]
      : [
          "Validating numeric target coverage.",
          "Encoding features for regression candidates.",
          "Fitting elastic net, forest, and boosting regressors.",
          "Scoring R2, MAE, and RMSE.",
          "Preparing release review and feature signals.",
        ];
  let tick = 0;

  clearInterval(devState.trainingProgressTimer);
  document.body.classList.add("ops-is-training");
  panel.hidden = false;
  panel.classList.remove("complete", "error");
  setDevText("dev-progress-kicker", `${taskName} Training`);
  setDevText("dev-progress-title", target ? `Training candidate for ${target}` : "Training candidate model");
  setTrainingProgress(8, steps[0]);
  if (profileButton) {
    profileButton.disabled = true;
  }

  devState.trainingProgressTimer = setInterval(() => {
    tick += 1;
    const nextStep = steps[Math.min(steps.length - 1, Math.floor(tick / 4))];
    const increment = tick < 8 ? 5 : tick < 18 ? 3 : 1;
    setTrainingProgress(Math.min(92, devState.trainingProgressValue + increment), nextStep);
  }, 700);
}

function stopTrainingProgress(status = "complete", message = "") {
  const panel = document.getElementById("dev-training-progress");
  const profileButton = document.getElementById("dev-profile-btn");
  clearInterval(devState.trainingProgressTimer);
  devState.trainingProgressTimer = null;
  document.body.classList.remove("ops-is-training");
  if (profileButton) {
    profileButton.disabled = false;
  }
  if (!panel) return;

  panel.classList.toggle("complete", status === "complete");
  panel.classList.toggle("error", status === "error");
  if (status === "complete") {
    setTrainingProgress(100, message || "Candidate training complete. Review the registered run below.");
  } else if (status === "error") {
    setTrainingProgress(devState.trainingProgressValue || 100, message || "Training stopped before a candidate was registered.");
  } else {
    panel.hidden = true;
  }
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
  setDevText("ops-dataset-name", currentDatasetName());
}

function currentDatasetPath() {
  const datasetInput = document.getElementById("dev-dataset-path");
  return datasetInput?.value.trim() || devState.defaultDatasetPath;
}

function currentDatasetName() {
  return currentDatasetPath().split(/[\\/]/).pop() || "Dataset";
}

function setBodyTaskMode(task, quality = "ready") {
  document.body.classList.remove(
    "ops-mode-auto",
    "ops-mode-classification",
    "ops-mode-regression",
    "ops-mode-blocked"
  );
  const mode = quality === "blocked" ? "blocked" : task || "auto";
  document.body.classList.add(`ops-mode-${mode}`);
}

function populateTargetSelect() {
  const select = document.getElementById("dev-target-select");
  const previous = select.value || selectedTarget();
  select.innerHTML = "";

  const blank = document.createElement("option");
  blank.value = "";
  blank.textContent = devState.targetProfiles.length
    ? "Choose a dataset header"
    : "Profile dataset to choose target";
  select.appendChild(blank);

  for (const profile of devState.targetProfiles) {
    const option = document.createElement("option");
    option.value = profile.column;
    option.textContent = `${profile.column} | ${taskLabel(profile.recommended_task_type)} | ${profile.quality}`;
    option.title = profile.reasons?.join(" ") || profile.warnings?.join(" ") || profile.column;
    select.appendChild(option);
  }

  if (previous && devState.targetProfiles.some((profile) => profile.column === previous)) {
    select.value = previous;
  }
}

function renderIssueList(profile) {
  const issues = [...(profile?.reasons || []), ...(profile?.warnings || [])];
  const panel = document.createElement("section");
  panel.className = "ops-subpanel ops-issue-panel";
  const heading = document.createElement("h3");
  heading.textContent = profile?.reasons?.length ? "Training Blockers" : "Checks";
  const list = document.createElement("ul");
  if (!issues.length) {
    const item = document.createElement("li");
    item.textContent = "No target quality issues detected.";
    list.appendChild(item);
  } else {
    for (const issue of issues) {
      const item = document.createElement("li");
      item.textContent = issue;
      list.appendChild(item);
    }
  }
  panel.append(heading, list);
  return panel;
}

function renderTargetIntel(profile) {
  const root = document.createElement("div");
  root.className = "ops-output-grid";
  if (!profile) {
    const empty = document.createElement("div");
    empty.className = "ops-empty-state";
    empty.textContent = "Choose a target header to see task type, quality checks, and distribution.";
    root.appendChild(empty);
    return root;
  }

  const task = effectiveTask(profile);
  root.appendChild(
    createStatGrid([
      ["Target", profile.column],
      ["Task", taskLabel(task)],
      ["Quality", profile.quality],
      ["Rows", safeNum(profile.non_null_rows)],
      ["Unique", safeNum(profile.unique_count)],
      ["Missing", `${formatMetric(profile.missing_pct)}%`],
    ])
  );

  if (task === "regression") {
    const stats = profile.stats || {};
    root.appendChild(
      createObjectTable("Numeric Shape", {
        "Valid Numeric Rows": stats.valid_numeric_rows,
        "Numeric Unique": stats.numeric_unique_count,
        Min: stats.min,
        Median: stats.median,
        Mean: stats.mean,
        Max: stats.max,
        Std: stats.std,
      })
    );
  } else {
    const stats = profile.stats || {};
    root.appendChild(
      createObjectTable("Class Shape", {
        Classes: stats.class_count,
        "Smallest Class": stats.min_class_rows,
        "Largest Class": stats.max_class_rows,
        ...stats.top_classes,
      })
    );
  }

  root.appendChild(renderIssueList(profile));
  return root;
}

function updateTargetExperience() {
  const target = selectedTarget();
  const profile = targetProfileFor(target);
  const task = effectiveTask(profile);
  const hasOverride = Boolean(taskOverride());
  const isBlocked = Boolean(profile && !profile.is_trainable && !hasOverride);
  const quality = isBlocked ? "blocked" : profile?.quality || "ready";
  const trainButton = document.getElementById("dev-train-btn");

  setBodyTaskMode(task, quality);
  setDevText("dev-mode-pill", taskLabel(task));
  setDevText("dev-intel-mode", target ? `${taskLabel(task)} target` : "Awaiting target");
  setDevText(
    "dev-page-title",
    target ? `${taskLabel(task)} Training: ${target}` : "Training and Release Control"
  );
  setDevText(
    "dev-page-subtitle",
    target
      ? "Train a candidate, compare metrics, then set active only after review."
      : "Profile a dataset, choose a target header, then review candidate models."
  );
  setDevText("dev-intel-title", target ? "Selected Header Intelligence" : "Header Intelligence");

  trainButton.disabled = !target || isBlocked;
  trainButton.textContent =
    target && task !== "auto" ? `Train ${taskLabel(task)} Candidate` : "Train Candidate";

  const status = document.getElementById("dev-target-status");
  if (!target) {
    status.textContent = "Choose a target header before training.";
    status.className = "ops-target-status";
  } else if (!profile) {
    status.textContent = "Target has not been profiled yet. Refresh the profile before training.";
    status.className = "ops-target-status review";
  } else if (isBlocked) {
    status.textContent = profile.reasons?.[0] || "Target is blocked by quality checks.";
    status.className = "ops-target-status blocked";
  } else if (hasOverride) {
    status.textContent = `Manual ${taskLabel(task)} override selected. Backend validation will confirm this target can train.`;
    status.className = "ops-target-status review";
  } else {
    status.textContent = `${profile.column} is ready for ${taskLabel(task).toLowerCase()} training.`;
    status.className = `ops-target-status ${profile.quality}`;
  }

  replaceWithNode("dev-target-intel", renderTargetIntel(profile));
}

function selectTarget(column) {
  document.getElementById("dev-target-select").value = column;
  document.getElementById("dev-target-column").value = column;
  updateTargetExperience();
}

function syncLegacyDatasetInput() {
  const legacyInput = document.getElementById("dev-train-dataset-path");
  if (legacyInput) {
    legacyInput.value = currentDatasetPath();
  }
}

function renderDatasetProfile(profile) {
  const root = document.createElement("div");
  root.className = "ops-output-grid";
  const targetProfiles = profile.target_profiles || [];
  const trainableCount = targetProfiles.filter((item) => item.is_trainable).length;
  const blockedCount = targetProfiles.filter((item) => !item.is_trainable).length;

  root.appendChild(
    createStatGrid([
      ["Rows", safeNum(profile.rows)],
      ["Columns", safeNum(profile.column_count || profile.columns?.length || profile.columns)],
      ["Trainable Targets", safeNum(trainableCount)],
      ["Blocked Headers", safeNum(blockedCount)],
    ])
  );

  const targetPanel = document.createElement("section");
  targetPanel.className = "ops-subpanel";
  const targetHeading = document.createElement("h3");
  targetHeading.textContent = "Dataset Headers";
  const targetList = document.createElement("div");
  targetList.className = "target-chip-list";
  const sortedProfiles = targetProfiles
    .slice()
    .sort((a, b) => Number(!a.is_trainable) - Number(!b.is_trainable));
  for (const item of sortedProfiles.slice(0, 80)) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `target-chip ${item.quality}`;
    button.textContent = item.column;
    button.title = item.reasons?.join(" ") || `${taskLabel(item.recommended_task_type)} target`;
    button.addEventListener("click", () => selectTarget(item.column));
    targetList.appendChild(button);
  }
  if (!targetList.children.length) {
    const empty = document.createElement("p");
    empty.className = "ops-muted";
    empty.textContent = "No headers were returned for this dataset.";
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

function metricText(item) {
  if (item.error) return item.error;
  return Object.entries(item.metrics || {})
    .map(([key, value]) => `${key} ${formatMetric(value)}`)
    .join(" | ");
}

function renderTrainingResult(result) {
  const root = document.createElement("div");
  root.className = "ops-output-grid";
  const exp = result.experiment || {};
  const targetProfile = result.target_profile || {};

  root.appendChild(
    createStatGrid([
      ["Experiment", shortId(exp.id)],
      ["Target", exp.target_column],
      ["Task", taskLabel(exp.task_type)],
      ["Best Model", result.best_model_name],
      ["Primary Metric", `${exp.metric_name}: ${formatMetric(exp.metric_value)}`],
      ["Activation", "Review required"],
    ])
  );

  const review = document.createElement("section");
  review.className = "ops-subpanel ops-review-callout";
  const reviewTitle = document.createElement("h3");
  reviewTitle.textContent = "Release Review";
  const reviewCopy = document.createElement("p");
  reviewCopy.textContent =
    "This model has been registered as a candidate. Use Set Active in the registry after reviewing metrics and feature signals.";
  review.append(reviewTitle, reviewCopy);

  const candidatePanel = document.createElement("section");
  candidatePanel.className = "ops-subpanel";
  const candidateHeading = document.createElement("h3");
  candidateHeading.textContent = "Candidate Model Scores";
  const candidateWrap = document.createElement("div");
  candidateWrap.className = "table-wrap compact-table";
  const candidateTable = document.createElement("table");
  const candidateHead = document.createElement("thead");
  candidateHead.innerHTML = "<tr><th>Model</th><th>Primary</th><th>Metrics / Status</th></tr>";
  const candidateBody = document.createElement("tbody");
  for (const item of result.all_results || []) {
    const row = document.createElement("tr");
    if (item.error) row.className = "ops-row-error";
    const modelCell = document.createElement("td");
    const primaryCell = document.createElement("td");
    const metricsCell = document.createElement("td");
    modelCell.textContent = item.model_name;
    primaryCell.textContent = item.error
      ? "Failed"
      : `${item.primary_metric_name}: ${formatMetric(item.primary_metric_value)}`;
    metricsCell.textContent = metricText(item);
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

  root.append(review, candidatePanel, featurePanel);
  if (targetProfile.reasons?.length || targetProfile.warnings?.length) {
    root.appendChild(renderIssueList(targetProfile));
  }
  return root;
}

function renderExperimentDetails(experiment) {
  const root = document.createElement("div");
  root.className = "ops-output-grid";
  const metadata = experiment.metadata || {};
  const training = metadata.training_summary || {};
  const targetProfile = training.target_profile || {};

  root.appendChild(
    createStatGrid([
      ["Experiment", shortId(experiment.id)],
      ["Target", experiment.target_column],
      ["Task", taskLabel(experiment.task_type)],
      ["Model", experiment.best_model_name],
      ["Metric", `${experiment.metric_name}: ${formatMetric(experiment.metric_value)}`],
      ["Training Rows", safeNum(training.train_rows)],
      ["Test Rows", safeNum(training.test_rows)],
      ["Features", safeNum((metadata.feature_columns || []).length)],
      ["Created", formatDateTime(experiment.created_at)],
    ])
  );

  if (targetProfile.column) {
    const profilePanel = document.createElement("section");
    profilePanel.className = "ops-subpanel";
    const heading = document.createElement("h3");
    heading.textContent = "Target Profile";
    const summary = document.createElement("p");
    summary.className = "ops-muted";
    summary.textContent = `${targetProfile.column} | ${taskLabel(targetProfile.effective_task_type)} | ${targetProfile.quality}`;
    profilePanel.append(heading, summary);
    root.appendChild(profilePanel);
  }

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
  head.innerHTML = "<tr><th>Model</th><th>Primary</th><th>Metrics / Status</th></tr>";
  const body = document.createElement("tbody");
  for (const item of training.results || []) {
    const row = document.createElement("tr");
    if (item.error) row.className = "ops-row-error";
    const modelCell = document.createElement("td");
    const primaryCell = document.createElement("td");
    const metricCell = document.createElement("td");
    modelCell.textContent = item.model_name;
    primaryCell.textContent = item.error
      ? "Failed"
      : `${item.primary_metric_name}: ${formatMetric(item.primary_metric_value)}`;
    metricCell.textContent = metricText(item);
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
    if (isActive) row.className = "ops-active-row";

    const cells = [
      shortId(exp.id),
      formatDateTime(exp.created_at),
      exp.target_column,
      taskLabel(exp.task_type),
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
        setOutput("dev-detail-output", error.message, true);
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
    const datasetPath = currentDatasetPath();
    syncLegacyDatasetInput();
    setDevText("ops-dataset-name", currentDatasetName());
    setDevText("dev-profile-summary", "Profiling...");
    setOutput("dev-profile-output", "Reading dataset and scoring target headers...");
    const profile = await devApi("/api/profile", {
      method: "POST",
      body: { dataset_path: datasetPath },
    });
    devState.profile = profile;
    devState.targetProfiles = profile.target_profiles || [];
    populateTargetSelect();
    const currentTarget = selectedTarget();
    const currentTargetProfile = targetProfileFor(currentTarget);
    const preferredTarget =
      (currentTargetProfile?.is_trainable ? currentTarget : "") ||
      (profile.target_candidates || [])[0] ||
      devState.targetProfiles.find((item) => item.is_trainable)?.column ||
      "";
    if (preferredTarget) {
      selectTarget(preferredTarget);
    } else {
      document.getElementById("dev-target-column").value = "";
      document.getElementById("dev-target-select").value = "";
      updateTargetExperience();
    }
    setDevText(
      "dev-profile-summary",
      `${safeNum(profile.rows)} rows | ${safeNum(profile.column_count)} columns | ${safeNum((profile.target_candidates || []).length)} trainable targets`
    );
    replaceWithNode("dev-profile-output", renderDatasetProfile(profile));
  } catch (error) {
    setOutput("dev-profile-output", error.message, true);
    setDevText("dev-profile-summary", "Profile failed");
    updateTargetExperience();
  }
}

async function runDevTraining() {
  const trainButton = document.getElementById("dev-train-btn");
  try {
    const profile = targetProfileFor(selectedTarget());
    const task = effectiveTask(profile);
    const payload = {
      dataset_path: currentDatasetPath(),
      target_column: selectedTarget(),
      task_type: taskOverride() || null,
      test_size: Number(document.getElementById("dev-test-size").value || 0.2),
      random_state: Number(document.getElementById("dev-random-state").value || 42),
    };
    if (!payload.target_column) {
      throw new Error("Target column is required before training.");
    }

    trainButton.disabled = true;
    startTrainingProgress(task, payload.target_column);
    setOutput("dev-train-output", `Training ${taskLabel(task).toLowerCase()} candidate...`);
    const result = await devApi("/api/train", {
      method: "POST",
      body: payload,
    });
    stopTrainingProgress("complete");
    replaceWithNode("dev-train-output", renderTrainingResult(result));
    await refreshAllDevData();
  } catch (error) {
    stopTrainingProgress("error", error.message);
    setOutput("dev-train-output", error.message, true);
  } finally {
    updateTargetExperience();
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
  const datasetInput = document.getElementById("dev-dataset-path");
  if (!datasetInput) {
    setOutput("dev-train-output", "The /dev page markup is out of date. Hard refresh the page and try again.", true);
    return;
  }
  datasetInput.value = devState.defaultDatasetPath;
  syncLegacyDatasetInput();

  document.getElementById("dev-profile-btn").addEventListener("click", runDevProfile);
  document.getElementById("dev-train-btn").addEventListener("click", runDevTraining);
  document.getElementById("dev-target-select").addEventListener("change", (event) => {
    document.getElementById("dev-target-column").value = event.target.value;
    updateTargetExperience();
  });
  document.getElementById("dev-target-column").addEventListener("input", (event) => {
    const select = document.getElementById("dev-target-select");
    select.value = targetProfileFor(event.target.value.trim()) ? event.target.value.trim() : "";
    updateTargetExperience();
  });
  document.getElementById("dev-task-type").addEventListener("change", updateTargetExperience);
  datasetInput.addEventListener("change", () => {
    devState.profile = null;
    devState.targetProfiles = [];
    document.getElementById("dev-target-column").value = "";
    syncLegacyDatasetInput();
    populateTargetSelect();
    updateTargetExperience();
  });

  updateTargetExperience();
  try {
    await refreshAllDevData();
    await runDevProfile();
  } catch (error) {
    setOutput("dev-train-output", error.message, true);
  }
}

window.addEventListener("DOMContentLoaded", initDevPage);
