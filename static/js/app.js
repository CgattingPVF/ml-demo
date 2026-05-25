const state = {
  experiments: [],
  intakeTemplate: null,
};

function formatNumber(value) {
  if (value === null || value === undefined || value === "") return "N/A";
  if (typeof value === "number") return value.toLocaleString();
  return String(value);
}

async function apiRequest(path, options = {}) {
  const config = { method: "GET", ...options };
  if (config.body && typeof config.body !== "string") {
    config.headers = { ...(config.headers || {}), "Content-Type": "application/json" };
    config.body = JSON.stringify(config.body);
  }

  const response = await fetch(path, config);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload.detail || "Request failed";
    throw new Error(message);
  }
  return payload;
}

function outputError(container, message) {
  container.innerHTML = "";
  const box = document.createElement("div");
  box.className = "output-box error";
  box.textContent = message;
  container.appendChild(box);
}

function outputJson(container, data) {
  container.innerHTML = "";
  const box = document.createElement("pre");
  box.className = "output-box";
  box.textContent = JSON.stringify(data, null, 2);
  container.appendChild(box);
}

function setExperimentOptions(select) {
  select.innerHTML = "";
  if (!state.experiments.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No experiments available";
    select.appendChild(option);
    return;
  }

  for (const exp of state.experiments) {
    const option = document.createElement("option");
    option.value = exp.id;
    option.textContent = `${exp.id.slice(0, 8)}... | ${exp.target_column} | ${exp.metric_name}: ${Number(exp.metric_value).toFixed(4)}`;
    select.appendChild(option);
  }
}

async function refreshExperiments() {
  state.experiments = await apiRequest("/api/experiments?limit=200");
  setExperimentOptions(document.getElementById("intake-experiment-select"));
}

function renderProfileCard(profile) {
  const template = document.getElementById("profile-card-template");
  const fragment = template.content.cloneNode(true);
  const root = fragment.querySelector(".profile-card");

  root.querySelector(".profile-title").textContent = `${profile.profile_type.toUpperCase()} | ${profile.name}`;
  root.querySelector(".profile-subtitle").textContent = `Records analyzed: ${formatNumber(profile.snapshot?.records_analyzed)}`;

  const metrics = [
    { label: "Total Jobs", value: profile.portfolio?.total_jobs },
    { label: "Active Jobs", value: profile.portfolio?.active_jobs },
    { label: "Completed Jobs", value: profile.portfolio?.completed_jobs },
    { label: "Overdue Open Jobs", value: profile.portfolio?.overdue_open_jobs },
  ];
  const metricsGrid = root.querySelector(".metrics-grid");
  for (const metric of metrics) {
    const node = document.createElement("div");
    node.className = "metric";
    node.innerHTML = `<div class="label">${metric.label}</div><div class="value">${formatNumber(metric.value)}</div>`;
    metricsGrid.appendChild(node);
  }

  const portfolioRows = [
    ["Completion Rate %", profile.portfolio?.completion_rate_pct],
    ["Cancellation Rate %", profile.portfolio?.cancellation_rate_pct],
    ["First Job Date", profile.snapshot?.date_range?.first_job_date],
    ["Last Job Date", profile.snapshot?.date_range?.last_job_date],
    ["Avg Cycle Days", profile.delivery?.avg_cycle_time_days],
    ["Median Cycle Days", profile.delivery?.median_cycle_time_days],
    ["Avg To-Do Hours", profile.delivery?.avg_total_to_do_hours],
    ["Avg Done Hours", profile.delivery?.avg_total_done_hours],
  ];

  const financialRows = [
    ["Budget Revenue", profile.financials?.budget_revenue_total],
    ["Actual Revenue", profile.financials?.actual_revenue_total],
    ["Budget Cost", profile.financials?.budget_cost_total],
    ["Actual Cost", profile.financials?.actual_cost_total],
    ["Budget Margin", profile.financials?.budget_margin_total],
    ["Actual Margin", profile.financials?.actual_margin_total],
    ["Revenue Variance %", profile.financials?.revenue_variance_pct],
    ["Margin Variance %", profile.financials?.margin_variance_pct],
  ];

  const portfolioTable = root.querySelector(".portfolio-table");
  const financialTable = root.querySelector(".financial-table");
  for (const [label, value] of portfolioRows) {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<span>${label}</span><strong>${formatNumber(value)}</strong>`;
    portfolioTable.appendChild(row);
  }
  for (const [label, value] of financialRows) {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<span>${label}</span><strong>${formatNumber(value)}</strong>`;
    financialTable.appendChild(row);
  }

  const jobs = profile.recent_jobs || [];
  const table = root.querySelector(".recent-table");
  if (jobs.length) {
    const headers = Object.keys(jobs[0]);
    const thead = document.createElement("thead");
    const headerRow = document.createElement("tr");
    for (const header of headers) {
      const th = document.createElement("th");
      th.textContent = header;
      headerRow.appendChild(th);
    }
    thead.appendChild(headerRow);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");
    for (const job of jobs.slice(0, 10)) {
      const row = document.createElement("tr");
      for (const header of headers) {
        const td = document.createElement("td");
        td.textContent = String(job[header] ?? "");
        row.appendChild(td);
      }
      tbody.appendChild(row);
    }
    table.appendChild(tbody);
  }

  return root;
}

function renderIntakeForm(template) {
  const form = document.getElementById("intake-form");
  form.innerHTML = "";

  for (const field of template.recommended_fields) {
    const label = document.createElement("label");
    label.textContent = field.required ? `${field.field} *` : field.field;

    let input;
    if (field.type === "category") {
      input = document.createElement("select");
      const blank = document.createElement("option");
      blank.value = "";
      blank.textContent = "-- select --";
      input.appendChild(blank);
      for (const value of field.suggested_values || []) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        input.appendChild(option);
      }
      if (field.default) input.value = String(field.default);
    } else {
      input = document.createElement("input");
      input.type = field.type === "number" ? "number" : field.type === "date" ? "date" : "text";
      if (field.type === "number") input.step = "any";
      if (field.default !== null && field.default !== undefined && field.default !== "") {
        input.placeholder = String(field.default);
      }
    }

    input.dataset.field = field.field;
    input.dataset.type = field.type;
    if (field.required) input.required = true;
    label.appendChild(input);
    form.appendChild(label);
  }
}

function readIntakeRecord() {
  const inputs = document.querySelectorAll("#intake-form [data-field]");
  const record = {};
  for (const input of inputs) {
    const field = input.dataset.field;
    const type = input.dataset.type;
    const raw = (input.value || "").trim();
    if (!raw) continue;
    if (type === "number") {
      const parsed = Number(raw);
      record[field] = Number.isFinite(parsed) ? parsed : raw;
    } else {
      record[field] = raw;
    }
  }
  return record;
}

async function runDatasetProfile() {
  const container = document.getElementById("train-output");
  try {
    const datasetPath = document.getElementById("train-dataset-path").value.trim();
    const result = await apiRequest("/api/profile", { method: "POST", body: { dataset_path: datasetPath } });
    const box = document.createElement("div");
    box.className = "output-box";
    box.innerHTML = `
      <div class="pill-row">
        <span class="pill">Rows: ${formatNumber(result.rows)}</span>
        <span class="pill">Columns: ${formatNumber(result.columns)}</span>
        <span class="pill accent">Target Candidates: ${formatNumber((result.target_candidates || []).length)}</span>
      </div>
      <p><strong>Top Candidate Targets:</strong> ${(result.target_candidates || []).slice(0, 12).join(", ")}</p>
    `;
    container.innerHTML = "";
    container.appendChild(box);
  } catch (error) {
    outputError(container, error.message);
  }
}

async function runTraining() {
  const container = document.getElementById("train-output");
  try {
    const payload = {
      dataset_path: document.getElementById("train-dataset-path").value.trim(),
      target_column: document.getElementById("train-target-column").value.trim(),
      task_type: document.getElementById("train-task-type").value || null,
      test_size: Number(document.getElementById("train-test-size").value || 0.2),
      random_state: Number(document.getElementById("train-random-state").value || 42),
    };
    if (!payload.target_column) {
      throw new Error("Target column is required.");
    }
    const result = await apiRequest("/api/train", { method: "POST", body: payload });
    const experiment = result.experiment || {};

    const box = document.createElement("div");
    box.className = "output-box";
    box.innerHTML = `
      <div class="pill-row">
        <span class="pill">Experiment: ${experiment.id || "N/A"}</span>
        <span class="pill">Model: ${result.best_model_name}</span>
        <span class="pill accent">${experiment.metric_name}: ${Number(experiment.metric_value || 0).toFixed(4)}</span>
      </div>
      <p><strong>Best Metrics</strong></p>
    `;
    const metricsPre = document.createElement("pre");
    metricsPre.className = "output-box";
    metricsPre.textContent = JSON.stringify(result.best_metrics, null, 2);

    container.innerHTML = "";
    container.appendChild(box);
    container.appendChild(metricsPre);
    await refreshExperiments();
  } catch (error) {
    outputError(container, error.message);
  }
}

async function loadIntakeTemplate() {
  const output = document.getElementById("intake-output");
  try {
    const experimentId = document.getElementById("intake-experiment-select").value;
    if (!experimentId) throw new Error("Select an experiment.");

    const maxFields = Number(document.getElementById("intake-max-fields").value || 18);
    const template = await apiRequest("/api/intake/template", {
      method: "POST",
      body: { experiment_id: experimentId, max_fields: maxFields },
    });
    state.intakeTemplate = template;
    document.getElementById("intake-profile-path").value = template.dataset_path || "";
    renderIntakeForm(template);

    const box = document.createElement("div");
    box.className = "output-box";
    box.innerHTML = `
      <div class="pill-row">
        <span class="pill">Target: ${template.target_column}</span>
        <span class="pill">${template.task_type}</span>
        <span class="pill accent">Fields: ${(template.recommended_fields || []).length}</span>
      </div>
    `;
    output.innerHTML = "";
    output.appendChild(box);
  } catch (error) {
    outputError(output, error.message);
  }
}

async function predictNewJob() {
  const output = document.getElementById("intake-output");
  try {
    if (!state.intakeTemplate) throw new Error("Load an intake form first.");
    const experimentId = document.getElementById("intake-experiment-select").value;
    if (!experimentId) throw new Error("Select an experiment.");
    const datasetPath = document.getElementById("intake-profile-path").value.trim();
    const jobRecord = readIntakeRecord();
    if (!Object.keys(jobRecord).length) throw new Error("Enter at least one intake field.");

    const result = await apiRequest("/api/intake/predict", {
      method: "POST",
      body: {
        experiment_id: experimentId,
        job_record: jobRecord,
        dataset_path: datasetPath || null,
      },
    });

    output.innerHTML = "";
    const prediction = result.prediction || {};
    const outcome = (prediction.predictions || [])[0];
    const confidence = (prediction.confidence || [])[0];

    const topBox = document.createElement("div");
    topBox.className = "output-box";
    const confidenceText = typeof confidence === "number" ? `${(confidence * 100).toFixed(2)}%` : "N/A";
    topBox.innerHTML = `
      <div class="pill-row">
        <span class="pill accent">Predicted: ${outcome ?? "N/A"}</span>
        <span class="pill">Confidence: ${confidenceText}</span>
      </div>
    `;
    output.appendChild(topBox);

    const submitted = document.createElement("pre");
    submitted.className = "output-box";
    submitted.textContent = JSON.stringify(jobRecord, null, 2);
    output.appendChild(submitted);

    if (result.profiles?.client) output.appendChild(renderProfileCard(result.profiles.client));
    if (result.profiles?.partner) output.appendChild(renderProfileCard(result.profiles.partner));
    if (result.profiles?.client_error) {
      const warn = document.createElement("div");
      warn.className = "output-box error";
      warn.textContent = `Client profile warning: ${result.profiles.client_error}`;
      output.appendChild(warn);
    }
    if (result.profiles?.partner_error) {
      const warn = document.createElement("div");
      warn.className = "output-box error";
      warn.textContent = `Partner profile warning: ${result.profiles.partner_error}`;
      output.appendChild(warn);
    }
  } catch (error) {
    outputError(output, error.message);
  }
}

async function loadProfileEntities() {
  const output = document.getElementById("profile-output");
  try {
    const datasetPath = document.getElementById("profile-dataset-path").value.trim();
    const profileType = document.getElementById("profile-type").value;
    const response = await apiRequest("/api/profiles/entities", {
      method: "POST",
      body: {
        dataset_path: datasetPath,
        profile_type: profileType,
        limit: 600,
      },
    });

    const select = document.getElementById("profile-entity-select");
    select.innerHTML = "";
    for (const entity of response.entities || []) {
      const option = document.createElement("option");
      option.value = entity;
      option.textContent = entity;
      select.appendChild(option);
    }

    const box = document.createElement("div");
    box.className = "output-box";
    box.innerHTML = `<p>Loaded ${formatNumber((response.entities || []).length)} ${profileType} entities.</p>`;
    output.innerHTML = "";
    output.appendChild(box);
  } catch (error) {
    outputError(output, error.message);
  }
}

async function generateProfileCardFromInputs() {
  const output = document.getElementById("profile-output");
  try {
    const datasetPath = document.getElementById("profile-dataset-path").value.trim();
    const profileType = document.getElementById("profile-type").value;
    const chosen = document.getElementById("profile-entity-select").value;
    const manual = document.getElementById("profile-manual-name").value.trim();
    const name = manual || chosen;
    if (!name) throw new Error("Choose or enter a profile name.");

    const card = await apiRequest("/api/profiles/card", {
      method: "POST",
      body: {
        dataset_path: datasetPath,
        profile_type: profileType,
        name,
      },
    });

    output.innerHTML = "";
    output.appendChild(renderProfileCard(card));
  } catch (error) {
    outputError(output, error.message);
  }
}

async function init() {
  const defaultDataset = document.body.dataset.defaultDataset || "";
  document.getElementById("train-dataset-path").value = defaultDataset;
  document.getElementById("profile-dataset-path").value = defaultDataset;
  document.getElementById("intake-profile-path").value = defaultDataset;

  document.getElementById("profile-dataset-btn").addEventListener("click", runDatasetProfile);
  document.getElementById("train-btn").addEventListener("click", runTraining);
  document.getElementById("load-intake-template-btn").addEventListener("click", loadIntakeTemplate);
  document.getElementById("predict-intake-btn").addEventListener("click", predictNewJob);
  document.getElementById("load-entities-btn").addEventListener("click", loadProfileEntities);
  document.getElementById("generate-profile-btn").addEventListener("click", generateProfileCardFromInputs);

  try {
    await refreshExperiments();
  } catch (error) {
    outputError(document.getElementById("train-output"), error.message);
  }
}

window.addEventListener("DOMContentLoaded", init);

