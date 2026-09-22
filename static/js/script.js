/* ── AttritionIQ Frontend JS ─────────────────────────────────────────── */

// ─── Nav toggle ────────────────────────────────────────────────────────────
function toggleNav() {
  document.querySelector(".nav-links").classList.toggle("open");
}

// ─── Plotly colour palette ─────────────────────────────────────────────────
const COLORS = {
  yes:    "#dc2626",
  no:     "#3b82d4",
  scale:  ["#dbeafe","#93c5fd","#3b82d4","#1d4ed8","#1e3a8a"],
  accent: "#3b82d4",
};

const LAYOUT_BASE = {
  margin: { t: 20, r: 20, b: 60, l: 50 },
  paper_bgcolor: "transparent",
  plot_bgcolor:  "transparent",
  font: { family: "-apple-system,'Segoe UI',system-ui,sans-serif", size: 12 },
  showlegend: false,
};

const CONFIG = { responsive: true, displayModeBar: false };

// ─── Dashboard ─────────────────────────────────────────────────────────────

function initDashboard(initialStats) {
  loadFilterOptions();
  renderDashboard(initialStats);
}

function loadFilterOptions() {
  fetch("/api/filter-options")
    .then(r => r.json())
    .then(opts => {
      populateSelect("f-dept",   opts.departments,    "All Departments");
      populateSelect("f-role",   opts.job_roles,      "All Roles");
      populateSelect("f-gender", opts.genders,        "All");
      populateSelect("f-travel", opts.business_travel,"All");
      populateSelect("f-ot",     opts.overtime,       "All");
      populateSelect("f-level",  opts.job_levels,     "All");
    })
    .catch(() => {});
}

function populateSelect(id, values, allLabel) {
  const sel = document.getElementById(id);
  if (!sel) return;
  sel.innerHTML = `<option value="All">${allLabel}</option>`;
  values.forEach(v => {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    sel.appendChild(opt);
  });
}

function applyFilters() {
  const params = new URLSearchParams({
    department:     document.getElementById("f-dept")?.value   || "All",
    job_role:       document.getElementById("f-role")?.value   || "All",
    gender:         document.getElementById("f-gender")?.value || "All",
    business_travel:document.getElementById("f-travel")?.value || "All",
    overtime:       document.getElementById("f-ot")?.value     || "All",
    job_level:      document.getElementById("f-level")?.value  || "All",
  });
  fetch(`/api/dashboard-data?${params}`)
    .then(r => r.json())
    .then(data => {
      if (data.error) { alert(data.error); return; }
      renderDashboard(data);
    })
    .catch(err => console.error("Filter error:", err));
}

function resetFilters() {
  ["f-dept","f-role","f-gender","f-travel","f-ot","f-level"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = "All";
  });
  applyFilters();
}

function renderDashboard(stats) {
  if (!stats) return;
  updateKPIs(stats.kpi);
  renderDeptChart(stats.by_department);
  renderOvertimeChart(stats.by_overtime);
  renderRoleChart(stats.by_job_role);
  renderJobSatChart(stats.by_job_satisfaction);
  renderAgeChart(stats.by_age_group);
  renderTenureChart(stats.by_tenure);
  renderTravelChart(stats.by_business_travel);
}

function updateKPIs(kpi) {
  if (!kpi) return;
  setText("k-total",    kpi.total_employees?.toLocaleString());
  setText("k-left",     kpi.employees_left?.toLocaleString());
  setText("k-retained", kpi.employees_retained?.toLocaleString());
  setText("k-rate",     kpi.attrition_rate + "%");
  setText("k-income",   "$" + (kpi.avg_monthly_income || 0).toLocaleString());
  setText("k-tenure",   kpi.avg_years_at_company);
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el && val !== undefined) el.textContent = val;
}

function hBar(labels, values, color) {
  return {
    type: "bar",
    orientation: "h",
    x: values,
    y: labels,
    marker: { color },
    hovertemplate: "%{y}: %{x:.1f}%<extra></extra>",
  };
}

function vBar(x, y, color) {
  return {
    type: "bar",
    x, y,
    marker: { color },
    hovertemplate: "%{x}: %{y:.1f}%<extra></extra>",
  };
}

function renderDeptChart(data) {
  if (!data?.length) return;
  const sorted = [...data].sort((a,b) => b.attrition_rate - a.attrition_rate);
  Plotly.newPlot("chart-dept",
    [hBar(sorted.map(d => d.Department), sorted.map(d => d.attrition_rate), COLORS.yes)],
    { ...LAYOUT_BASE, xaxis: { title: "Attrition Rate (%)" }, margin: { t:20,r:20,b:40,l:150 } },
    CONFIG
  );
}

function renderOvertimeChart(data) {
  if (!data?.length) return;
  Plotly.newPlot("chart-ot",
    [vBar(data.map(d => d.OverTime), data.map(d => d.attrition_rate),
      data.map(d => d.OverTime === "Yes" ? COLORS.yes : COLORS.no))],
    { ...LAYOUT_BASE, yaxis: { title: "Attrition Rate (%)" } },
    CONFIG
  );
}

function renderRoleChart(data) {
  if (!data?.length) return;
  const sorted = [...data].sort((a,b) => b.attrition_rate - a.attrition_rate);
  Plotly.newPlot("chart-role",
    [hBar(sorted.map(d => d.JobRole), sorted.map(d => d.attrition_rate), COLORS.accent)],
    { ...LAYOUT_BASE, xaxis: { title: "Attrition Rate (%)" }, margin: { t:20,r:20,b:40,l:180 } },
    CONFIG
  );
}

function renderJobSatChart(data) {
  if (!data?.length) return;
  const labels = { 1:"Low",2:"Medium",3:"High",4:"Very High" };
  Plotly.newPlot("chart-js",
    [vBar(data.map(d => labels[d.JobSatisfaction] || d.JobSatisfaction),
          data.map(d => d.attrition_rate), COLORS.accent)],
    { ...LAYOUT_BASE, xaxis: { title: "Job Satisfaction" }, yaxis: { title: "Attrition Rate (%)" } },
    CONFIG
  );
}

function renderAgeChart(data) {
  if (!data?.length) return;
  Plotly.newPlot("chart-age",
    [vBar(data.map(d => d.AgeGroup), data.map(d => d.attrition_rate), "#7c5cd8")],
    { ...LAYOUT_BASE, xaxis: { title: "Age Group" }, yaxis: { title: "Attrition Rate (%)" } },
    CONFIG
  );
}

function renderTenureChart(data) {
  if (!data?.length) return;
  Plotly.newPlot("chart-tenure",
    [vBar(data.map(d => d.TenureBucket), data.map(d => d.attrition_rate), COLORS.accent)],
    { ...LAYOUT_BASE, xaxis: { title: "Years at Company" }, yaxis: { title: "Attrition Rate (%)" } },
    CONFIG
  );
}

function renderTravelChart(data) {
  if (!data?.length) return;
  Plotly.newPlot("chart-travel",
    [vBar(data.map(d => d.BusinessTravel), data.map(d => d.attrition_rate), "#f59e0b")],
    { ...LAYOUT_BASE, xaxis: { title: "Business Travel" }, yaxis: { title: "Attrition Rate (%)" } },
    CONFIG
  );
}

// ─── Prediction form ────────────────────────────────────────────────────────

function submitPrediction() {
  const form    = document.getElementById("predForm");
  const btn     = document.getElementById("predictBtn");
  const section = document.getElementById("resultSection");

  if (!form.checkValidity()) { form.reportValidity(); return; }

  btn.disabled = true;
  btn.textContent = "Predicting…";

  // Collect form data
  const fd = new FormData(form);
  const payload = {};
  fd.forEach((v, k) => { payload[k] = v; });

  fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
    .then(r => r.json())
    .then(data => {
      if (data.error) { showError(data.error); return; }
      renderResult(data);
      section.style.display = "block";
      section.scrollIntoView({ behavior: "smooth", block: "start" });
    })
    .catch(err => showError("Network error: " + err.message))
    .finally(() => {
      btn.disabled = false;
      btn.textContent = "Predict Attrition";
    });
}

function renderResult(data) {
  const isYes = data.prediction === "Yes";
  const prob  = Math.round(data.probability * 100);
  const risk  = data.risk_level;

  // Prediction label
  const predEl = document.getElementById("resultPrediction");
  predEl.textContent = isYes ? "Likely to Leave" : "Likely to Stay";
  predEl.className   = "result-prediction " + (isYes ? "pred-yes" : "pred-no");

  // Badge
  const badge = document.getElementById("resultBadge");
  badge.textContent = risk + " Risk";
  badge.className   = "result-badge badge-" + risk.toLowerCase();

  // Result card border color
  const card = document.getElementById("resultCard");
  card.style.borderColor = risk === "High" ? "#dc2626" : risk === "Medium" ? "#d97706" : "#16a34a";

  // Probability bar
  document.getElementById("probFill").style.width  = prob + "%";
  document.getElementById("probFill").className    = "prob-fill " + risk.toLowerCase();
  document.getElementById("probValue").textContent = prob + "%";

  // Factors
  const fl = document.getElementById("factorList");
  fl.innerHTML = "";
  (data.top_factors || []).forEach(f => {
    const li = document.createElement("li");
    li.textContent = f;
    fl.appendChild(li);
  });

  // Recommendations
  const rl = document.getElementById("recList");
  rl.innerHTML = "";
  (data.recommendations || []).forEach(r => {
    const li = document.createElement("li");
    li.textContent = r;
    rl.appendChild(li);
  });
}

function resetForm() {
  document.getElementById("predForm").reset();
  document.getElementById("resultSection").style.display = "none";
}

function showError(msg) {
  alert("Error: " + msg);
}

// ─── Insights page ──────────────────────────────────────────────────────────

function loadInsightsData() {
  fetch("/api/metrics")
    .then(r => r.json())
    .then(data => {
      if (data.error) { console.error(data.error); return; }
      renderModelTable(data.model_comparison, data.best_model);
      renderRocCurve(data.roc_curve);
      renderConfusionMatrix(data.confusion_matrix);
    })
    .catch(err => console.error("Metrics load error:", err));

  fetch("/api/feature-importance")
    .then(r => r.json())
    .then(data => {
      if (data.feature_importance) renderFeatureImportance(data.feature_importance);
    })
    .catch(() => {});
}

function renderModelTable(rows, best) {
  const cont = document.getElementById("modelTableContainer");
  if (!cont) return;
  let html = `<table class="data-table">
    <thead><tr>
      <th>Model</th><th>Accuracy</th><th>Precision</th>
      <th>Recall</th><th>F1</th><th>ROC-AUC</th>
    </tr></thead><tbody>`;
  rows.forEach(r => {
    const cls = r.is_best ? ' class="best-model"' : "";
    html += `<tr${cls}>
      <td><strong>${r.model}</strong>${r.is_best ? ' <span class="badge badge-info">Best</span>' : ""}</td>
      <td>${pct(r.accuracy)}</td>
      <td>${pct(r.precision)}</td>
      <td>${pct(r.recall)}</td>
      <td>${pct(r.f1)}</td>
      <td>${pct(r.roc_auc)}</td>
    </tr>`;
  });
  html += "</tbody></table>";
  cont.innerHTML = html;
}

function pct(v) { return (v * 100).toFixed(1) + "%"; }

function renderRocCurve(roc) {
  if (!roc || !document.getElementById("chart-roc")) return;
  const auc = roc.auc;
  Plotly.newPlot("chart-roc", [
    { x: [0,1], y: [0,1], mode: "lines", line: { dash: "dot", color: "#e5e7eb" }, hoverinfo: "none" },
    { x: roc.fpr, y: roc.tpr, mode: "lines",
      line: { color: COLORS.yes, width: 2.5 },
      name: `ROC (AUC = ${auc})`,
      hovertemplate: "FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>" }
  ], {
    ...LAYOUT_BASE,
    showlegend: true,
    legend: { x: .6, y: .1 },
    xaxis: { title: "False Positive Rate", range: [0,1] },
    yaxis: { title: "True Positive Rate", range: [0,1] },
    annotations: [{ x: .7, y: .15, text: `AUC = ${auc}`, showarrow: false, font: { size: 13, color: COLORS.yes } }]
  }, CONFIG);
}

function renderConfusionMatrix(cm) {
  if (!cm || !document.getElementById("chart-cm")) return;
  const matrix = cm.matrix;
  Plotly.newPlot("chart-cm", [{
    type: "heatmap",
    z: matrix,
    x: cm.labels,
    y: cm.labels,
    colorscale: [[0,"#eff6ff"],[1,"#1d4ed8"]],
    showscale: false,
    text: matrix.map(row => row.map(v => v.toString())),
    texttemplate: "<b>%{text}</b>",
    hovertemplate: "Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
  }], {
    ...LAYOUT_BASE,
    xaxis: { title: "Predicted" },
    yaxis: { title: "Actual", autorange: "reversed" },
  }, CONFIG);
}

function renderFeatureImportance(features) {
  if (!features?.length || !document.getElementById("chart-fi")) return;
  const top = features.slice(0, 15);
  Plotly.newPlot("chart-fi", [
    hBar(top.map(f => humanizeFeature(f.feature)).reverse(),
         top.map(f => f.importance).reverse(),
         COLORS.accent)
  ], {
    ...LAYOUT_BASE,
    xaxis: { title: "Importance Score" },
    margin: { t: 20, r: 20, b: 40, l: 200 },
    height: 400,
  }, CONFIG);
}

function humanizeFeature(name) {
  return name
    .replace(/_/g, " ")
    .replace(/([A-Z])/g, " $1")
    .replace(/\s+/g, " ")
    .trim();
}
