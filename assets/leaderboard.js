(() => {
  const errorMetrics = {
    rmse: { label: "RMSE", rolloutKey: "rmse", value: (item) => item.metrics.rmse },
    rolloutRmse: { label: "Rollout RMSE", rolloutKey: "rolloutRmse", value: (item) => item.metrics.rolloutRmse },
    mae: { label: "MAE", rolloutKey: "mae", value: (item) => item.metrics.mae },
    rolloutMae: { label: "Rollout MAE", rolloutKey: "rolloutMae", value: (item) => item.metrics.rolloutMae },
    relativeL2: { label: "Rel. L₂", rolloutKey: "relativeL2", value: (item) => item.metrics.relativeL2 },
    rolloutRelativeL2: { label: "Rollout Rel. L₂", rolloutKey: "rolloutRelativeL2", value: (item) => item.metrics.rolloutRelativeL2 },
  };
  const state = {
    data: null,
    category: "overall",
    errorKey: "rmse",
    representationFilter: "all",
    query: new URLSearchParams(window.location.search).get("q") || "",
    sort: { key: "score", direction: "asc", groupRepresentation: false, representationDirection: "asc" },
  };
  const defaultSortKeys = { overall: "score", rollout: "step100", efficiency: "parameters", perVariable: "velocityRmse" };
  const defaultDirections = { submitted: "desc" };
  const $ = (selector) => document.querySelector(selector);
  const head = $("#leaderboard-head");
  const body = $("#leaderboard-body");
  const resultCount = $("#result-count");
  const datasetFilter = $("#dataset-filter");
  const searchInput = $("#model-search");
  searchInput.value = state.query;

  const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
  const formatError = (value) => Number(value).toExponential(1).replace("e-0", "e-").replace("e+0", "e+");
  const formatCount = (value) => new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);
  const formatDate = (value) => new Date(`${value}T00:00:00`).toLocaleDateString("en-US", { month: "short", day: "2-digit", year: "numeric" });
  const currentError = () => errorMetrics[state.errorKey];

  function selectedDataset() {
    return state.data.datasets.find((dataset) => dataset.id === datasetFilter.value) || state.data.datasets[0];
  }

  function fillDatasetFilter() {
    datasetFilter.innerHTML = state.data.datasets.map((dataset) => `<option value="${escapeHtml(dataset.id)}">${escapeHtml(dataset.name)}</option>`).join("");
    const requestedDataset = new URLSearchParams(window.location.search).get("dataset");
    if (state.data.datasets.some((dataset) => dataset.id === requestedDataset)) datasetFilter.value = requestedDataset;
  }

  function scoreControl() {
    const options = Object.entries(errorMetrics).map(([key, metric]) => `<button type="button" role="menuitemradio" aria-checked="${key === state.errorKey}" class="error-picker-option${key === state.errorKey ? " is-selected" : ""}" data-error-key="${key}"><span>${escapeHtml(metric.label)}</span><span class="error-picker-check" aria-hidden="true">${key === state.errorKey ? "✓" : ""}</span></button>`).join("");
    return `<div class="error-picker"><button type="button" id="score-metric-button" class="error-picker-button" aria-haspopup="menu" aria-expanded="false" aria-controls="score-metric-menu"><span class="error-picker-caption">Error</span><span class="error-picker-value">${escapeHtml(currentError().label)}</span></button><div id="score-metric-menu" class="error-picker-menu" role="menu" aria-label="Choose error metric" hidden>${options}</div></div>`;
  }

  function sortHeader(label, key) {
    const direction = key === "representation"
      ? (state.sort.groupRepresentation ? state.sort.representationDirection : null)
      : (state.sort.key === key ? state.sort.direction : null);
    const arrow = direction ? (direction === "asc" ? "↑" : "↓") : "↕";
    const ariaSort = direction ? ` aria-sort="${direction === "asc" ? "ascending" : "descending"}"` : "";
    const sortIndicator = key === "representation" ? "" : `<span class="sort-arrow" aria-hidden="true">${arrow}</span>`;
    const ariaLabel = key === "representation"
      ? `Representation filter: ${state.representationFilter === "all" ? "all submissions" : `${state.representationFilter} only`}. Click to cycle through all, Graph, and Matrix.`
      : `Sort by ${label}`;
    return `<th scope="col" class="sortable-heading"${ariaSort}><button type="button" class="sort-button${key === "representation" ? " representation-sort-button" : ""}" data-sort-key="${key}" aria-label="${escapeHtml(ariaLabel)}"><span>${escapeHtml(label)}</span>${sortIndicator}</button></th>`;
  }

  function scoreHeader() {
    const ariaSort = state.sort.key === "score" ? ` aria-sort="${state.sort.direction === "asc" ? "ascending" : "descending"}"` : "";
    return `<th scope="col" class="score-heading"${ariaSort}>${scoreControl()}</th>`;
  }

  function columns() {
    const common = ["<th scope=\"col\">#</th>", sortHeader("Model", "model"), sortHeader("Representation", "representation")];
    if (state.category === "overall") {
      return [...common, scoreHeader(), sortHeader("Inference / sample", "inference"), sortHeader("Params", "parameters"), sortHeader("Submitted", "submitted"), "<th scope=\"col\" aria-label=\"Details\"></th>"].join("");
    }
    if (state.category === "rollout") {
      return [...common, sortHeader("Step 1", "step1"), sortHeader("Step 25", "step25"), sortHeader("Step 100", "step100"), sortHeader("Submitted", "submitted"), "<th scope=\"col\" aria-label=\"Details\"></th>"].join("");
    }
    if (state.category === "efficiency") {
      return [...common, sortHeader("Params", "parameters"), sortHeader("Training time", "trainingHours"), sortHeader("Inference / sample", "inference"), sortHeader("GPU memory", "memoryGb"), sortHeader("Submitted", "submitted"), "<th scope=\"col\" aria-label=\"Details\"></th>"].join("");
    }
    return [...common, sortHeader("Velocity RMSE", "velocityRmse"), sortHeader("Pressure RMSE", "pressureRmse"), sortHeader("Submitted", "submitted"), "<th scope=\"col\" aria-label=\"Details\"></th>"].join("");
  }

  function valueFor(item, key) {
    if (key === "score") return errorMetrics[state.errorKey].value(item);
    if (key === "model") return item.model;
    if (key === "representation") return item.representation;
    if (key === "inference") return item.metrics.efficiency.inferenceMs;
    if (key === "parameters") return item.metrics.efficiency.parameters;
    if (key === "submitted") return new Date(`${item.submittedAt}T00:00:00`).getTime();
    if (key === "trainingHours") return item.metrics.efficiency.trainingHours;
    if (key === "memoryGb") return item.metrics.efficiency.memoryGb;
    if (key.startsWith("step")) return item.metrics.rollout[key][errorMetrics[state.errorKey].rolloutKey];
    if (key === "velocityRmse") return item.metrics.perVariable.velocityRmse;
    if (key === "pressureRmse") return item.metrics.perVariable.pressureRmse;
    return 0;
  }

  function errorCell(score, label, max) {
    const width = max > 0 ? Math.max(12, (score / max) * 58) : 12;
    const title = `${label}: ${formatError(score)}`;
    return `<td class="metric-value" title="${escapeHtml(title)}">${formatError(score)}<span class="metric-bar" role="img" aria-label="${escapeHtml(title)}" title="${escapeHtml(title)}"><span style="width:${width}px"></span></span></td>`;
  }

  function row(item, index) {
    const model = escapeHtml(item.model);
    const authors = escapeHtml(item.authors);
    const detailHref = window.FluidBenchPaths.url(`/pages/models.html#${encodeURIComponent(item.id)}`);
    const compareHref = window.FluidBenchPaths.url(`/pages/compare.html?model=${encodeURIComponent(item.id)}`);
    const medal = index < 3 ? `<span class="medal" aria-label="Rank ${index + 1}">${["🥇", "🥈", "🥉"][index]}</span>` : index + 1;
    const modelCell = `<td><a class="model-name" href="${detailHref}">${model}</a><span class="model-author">${authors}</span></td>`;
    const representationCell = `<td><span class="representation-badge ${item.representation.toLowerCase()}">${escapeHtml(item.representation)}</span></td>`;
    let metricCells = "";
    if (state.category === "overall") {
      const value = currentError().value(item);
      const max = Math.max(...state.data.submissions.map(currentError().value));
      metricCells += errorCell(value, currentError().label, max);
      metricCells += `<td>${Number(item.metrics.efficiency.inferenceMs).toFixed(1)} ms</td>`;
      metricCells += `<td>${formatCount(item.metrics.efficiency.parameters)}</td>`;
    } else if (state.category === "rollout") {
      const selected = currentError();
      for (const step of ["step1", "step25", "step100"]) {
        const value = item.metrics.rollout[step][selected.rolloutKey];
        const title = `${selected.label} at ${step.replace("step", "Step ")}: ${formatError(value)}`;
        const max = Math.max(...state.data.submissions.map((submission) => submission.metrics.rollout[step][selected.rolloutKey]));
        const width = max > 0 ? Math.max(12, (value / max) * 58) : 12;
        metricCells += `<td class="metric-value" title="${escapeHtml(title)}">${formatError(value)}<span class="metric-bar" role="img" aria-label="${escapeHtml(title)}" title="${escapeHtml(title)}"><span style="width:${width}px"></span></span></td>`;
      }
    } else if (state.category === "efficiency") {
      metricCells += `<td>${formatCount(item.metrics.efficiency.parameters)}</td><td>${Number(item.metrics.efficiency.trainingHours).toFixed(1)} h</td><td>${Number(item.metrics.efficiency.inferenceMs).toFixed(1)} ms</td><td>${Number(item.metrics.efficiency.memoryGb).toFixed(1)} GB</td>`;
    } else {
      const velocity = item.metrics.perVariable.velocityRmse;
      const pressure = item.metrics.perVariable.pressureRmse;
      const maxVelocity = Math.max(...state.data.submissions.map((submission) => submission.metrics.perVariable.velocityRmse));
      const maxPressure = Math.max(...state.data.submissions.map((submission) => submission.metrics.perVariable.pressureRmse));
      metricCells += errorCell(velocity, "Velocity RMSE", maxVelocity);
      metricCells += errorCell(pressure, "Pressure RMSE", maxPressure);
    }
    return `<tr><td class="rank-cell">${medal}</td>${modelCell}${representationCell}${metricCells}<td>${formatDate(item.submittedAt)}</td><td><a class="table-arrow" href="${compareHref}" aria-label="Compare ${model} with another model">›</a></td></tr>`;
  }

  function filteredSubmissions() {
    const dataset = selectedDataset();
    const query = state.query.trim().toLowerCase();
    return state.data.submissions.filter((item) => {
      if (item.datasetId !== dataset.id) return false;
      if (state.representationFilter !== "all" && item.representation !== state.representationFilter) return false;
      if (query && !`${item.model} ${item.authors} ${item.representation}`.toLowerCase().includes(query)) return false;
      return true;
    });
  }

  function compareValues(a, b, key, direction) {
    const first = valueFor(a, key);
    const second = valueFor(b, key);
    const comparison = typeof first === "number" && typeof second === "number"
      ? first - second
      : String(first).localeCompare(String(second), undefined, { sensitivity: "base", numeric: true });
    return direction === "desc" ? -comparison : comparison;
  }

  function setSort(key) {
    if (key === "representation") {
      state.representationFilter = state.representationFilter === "all"
        ? "Graph"
        : state.representationFilter === "Graph" ? "Matrix" : "all";
      state.sort.groupRepresentation = true;
      state.sort.representationDirection = "asc";
      render();
      return;
    }

    const wasGrouped = state.sort.groupRepresentation;
    if (state.sort.key === key && !wasGrouped) {
      state.sort.direction = state.sort.direction === "asc" ? "desc" : "asc";
    } else if (state.sort.key !== key) {
      state.sort.direction = defaultDirections[key] || "asc";
    }
    state.sort.key = key;
    state.sort.groupRepresentation = false;
    state.representationFilter = "all";
    render();
  }

  function closeErrorMenu() {
    const button = $("#score-metric-button");
    const menu = $("#score-metric-menu");
    if (!button || !menu) return;
    button.setAttribute("aria-expanded", "false");
    menu.hidden = true;
  }

  function selectError(key) {
    if (!errorMetrics[key]) return;
    state.errorKey = key;
    state.sort.key = state.category === "rollout" ? "step100" : "score";
    state.sort.direction = "asc";
    state.sort.groupRepresentation = false;
    state.representationFilter = "all";
    render();
    $("#score-metric-button")?.focus();
  }

  function render() {
    const items = filteredSubmissions().sort((a, b) => {
      if (state.sort.groupRepresentation) {
        const groupResult = compareValues(a, b, "representation", state.sort.representationDirection);
        if (groupResult) return groupResult;
      }
      return compareValues(a, b, state.sort.key, state.sort.direction);
    });
    head.innerHTML = `<tr>${columns()}</tr>`;
    body.innerHTML = items.length ? items.map(row).join("") : `<tr><td class="empty-state" colspan="9">No submissions match these filters.</td></tr>`;
    const sortLabels = {
      model: "Model",
      representation: "Representation",
      inference: "Inference / sample",
      parameters: "Params",
      submitted: "Submitted",
      trainingHours: "Training time",
      memoryGb: "GPU memory",
      velocityRmse: "Velocity RMSE",
      pressureRmse: "Pressure RMSE",
    };
    let sortLabel = sortLabels[state.sort.key] || state.sort.key;
    if (state.sort.key === "score") sortLabel = currentError().label;
    if (state.sort.key.startsWith("step")) sortLabel = `${state.sort.key.replace("step", "Step ")} ${currentError().label}`;
    const errorSort = ["score", "step1", "step25", "step100", "velocityRmse", "pressureRmse"].includes(state.sort.key);
    let orderLabel = errorSort
      ? (state.sort.direction === "asc" ? "best to worst" : "worst to best")
      : (state.sort.direction === "asc" ? "smallest to largest" : "largest to smallest");
    if (state.sort.key === "model") orderLabel = state.sort.direction === "asc" ? "A to Z" : "Z to A";
    if (state.sort.key === "submitted") orderLabel = state.sort.direction === "desc" ? "newest first" : "oldest first";
    const sortDescription = state.sort.groupRepresentation
      ? `Representation: ${state.representationFilter === "all" ? "all submissions" : `${state.representationFilter} only`} · ${sortLabel} ${orderLabel}`
      : `${sortLabel} · ${orderLabel}`;
    resultCount.textContent = `${items.length} submissions · ${sortDescription} ${state.sort.direction === "asc" ? "↑" : "↓"}`;
  }

  head.addEventListener("click", (event) => {
    const errorOption = event.target.closest("[data-error-key]");
    if (errorOption) {
      selectError(errorOption.dataset.errorKey);
      return;
    }
    const errorButton = event.target.closest("#score-metric-button");
    if (errorButton) {
      const menu = $("#score-metric-menu");
      const isOpen = errorButton.getAttribute("aria-expanded") === "true";
      errorButton.setAttribute("aria-expanded", String(!isOpen));
      menu.hidden = isOpen;
      if (!isOpen) menu.querySelector("[role=menuitemradio]")?.focus();
      return;
    }
    const button = event.target.closest("[data-sort-key]");
    if (!button) return;
    setSort(button.dataset.sortKey);
  });
  document.addEventListener("click", (event) => {
    if (!head.contains(event.target)) closeErrorMenu();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || $("#score-metric-menu")?.hidden) return;
    closeErrorMenu();
    $("#score-metric-button")?.focus();
  });
  document.querySelectorAll("[data-category]").forEach((button) => button.addEventListener("click", () => {
    state.category = button.dataset.category;
    state.sort.key = defaultSortKeys[state.category];
    state.sort.direction = "asc";
    state.sort.groupRepresentation = false;
    state.sort.representationDirection = "asc";
    state.representationFilter = "all";
    document.querySelectorAll("[data-category]").forEach((tab) => tab.setAttribute("aria-selected", String(tab === button)));
    render();
  }));
  datasetFilter.addEventListener("change", () => {
    state.representationFilter = "all";
    render();
  });
  searchInput.addEventListener("input", () => { state.query = searchInput.value; render(); });

  window.FluidBenchData.load().then((data) => {
    state.data = data;
    fillDatasetFilter();
    document.dispatchEvent(new CustomEvent("fluidbench:benchmark-selected", { detail: { id: datasetFilter.value } }));
    render();
  }).catch((error) => {
    head.innerHTML = "";
    body.innerHTML = `<tr><td class="empty-state" colspan="9">${escapeHtml(error.message)}</td></tr>`;
    resultCount.textContent = "Submissions could not be loaded.";
  });
})();
