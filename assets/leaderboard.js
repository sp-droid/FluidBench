(() => {
  const state = {
    data: null,
    options: null,
    dataCategory: null,
    errorKey: null,
    representationFilter: "all",
    query: new URLSearchParams(window.location.search).get("q") || "",
    sort: { key: "score", direction: "asc", groupRepresentation: false, representationDirection: "asc" },
  };
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
  const formatError = (value) => {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    return Number(value).toExponential(1).replace("e-0", "e-").replace("e+0", "e+");
  };
  const formatCount = (value) => value === null || value === undefined || !Number.isFinite(Number(value))
    ? "—"
    : new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);
  const formatDate = (value) => new Date(`${value}T00:00:00`).toLocaleDateString("en-US", { month: "short", day: "2-digit", year: "numeric" });

  function selectedDataset() {
    return state.data.datasets.find((dataset) => dataset.id === datasetFilter.value) || state.data.datasets[0];
  }

  function problemOptions() {
    const problem = selectedDataset()?.problem;
    return state.options?.problems?.[problem] || null;
  }

  function activeDataCategory() {
    const categories = problemOptions()?.categories || [];
    return categories.find((category) => category.id === state.dataCategory) || null;
  }

  function activeErrors() {
    return activeDataCategory()?.errors || [];
  }

  function currentError() {
    const errors = activeErrors();
    return errors.find((error) => error.id === state.errorKey) || errors[0] || null;
  }

  function activeMetrics() {
    const definitions = problemOptions()?.metrics || [];
    const ids = problemOptions()?.overallMetrics || [];
    return ids.map((id) => definitions.find((metric) => metric.id === id)).filter(Boolean);
  }

  function getPath(source, path) {
    if (!source || typeof path !== "string") return null;
    return path.split(".").reduce((value, part) => value?.[part], source) ?? null;
  }

  function metricSortKey(metric) {
    return `metric:${metric.id}`;
  }

  function fillDatasetFilter() {
    datasetFilter.innerHTML = state.data.datasets.map((dataset) => `<option value="${escapeHtml(dataset.id)}">${escapeHtml(dataset.name)}</option>`).join("");
    const requestedDataset = new URLSearchParams(window.location.search).get("dataset");
    if (state.data.datasets.some((dataset) => dataset.id === requestedDataset)) datasetFilter.value = requestedDataset;
  }

  function syncDataCategory(reset = false) {
    const config = problemOptions();
    const categories = config?.categories || [];
    if (reset || !categories.some((category) => category.id === state.dataCategory)) {
      state.dataCategory = config?.defaultCategory && categories.some((category) => category.id === config.defaultCategory)
        ? config.defaultCategory
        : categories[0]?.id || null;
    }
    if (reset || !activeErrors().some((error) => error.id === state.errorKey)) {
      state.errorKey = activeDataCategory()?.defaultError || activeErrors()[0]?.id || null;
    }
  }

  function scoreControl() {
    const category = activeDataCategory();
    const selected = currentError();
    if (!category) return "";
    const errorOptions = activeErrors().map((metric) => `<button type="button" role="menuitemradio" aria-checked="${metric.id === selected?.id}" class="error-picker-option${metric.id === selected?.id ? " is-selected" : ""}" data-picker-kind="error" data-picker-value="${escapeHtml(metric.id)}"><span>${escapeHtml(metric.label)}</span><span class="error-picker-check" aria-hidden="true">${metric.id === selected?.id ? "✓" : ""}</span></button>`).join("");
    const errorPicker = selected ? `<div class="error-picker"><button type="button" id="score-metric-button" class="error-picker-button" aria-label="Choose error metric" aria-haspopup="menu" aria-expanded="false" aria-controls="score-metric-menu" data-picker-button data-menu-id="score-metric-menu"><span class="error-picker-value">${escapeHtml(selected.label)}</span></button><div id="score-metric-menu" class="error-picker-menu" role="menu" aria-label="Choose error metric" hidden>${errorOptions}</div></div>` : "";
    return `<div class="score-picker">${errorPicker}</div>`;
  }

  function sortHeader(label, key) {
    const direction = key === "representation"
      ? (state.sort.groupRepresentation ? state.sort.representationDirection : null)
      : (state.sort.key === key ? state.sort.direction : null);
    const arrow = direction ? (direction === "asc" ? "↑" : "↓") : "↕";
    const ariaSort = direction ? ` aria-sort="${direction === "asc" ? "ascending" : "descending"}"` : "";
    const sortIndicator = key === "representation" ? "" : `<span class="sort-arrow" aria-hidden="true">${arrow}</span>`;
    const ariaLabel = key === "representation"
      ? `Representation filter: ${state.representationFilter === "all" ? "all submissions" : `${state.representationFilter} only`}. Click to cycle through all, Graph, and Grid.`
      : `Sort by ${label}`;
    return `<th scope="col" class="sortable-heading"${ariaSort}><button type="button" class="sort-button${key === "representation" ? " representation-sort-button" : ""}" data-sort-key="${escapeHtml(key)}" aria-label="${escapeHtml(ariaLabel)}"><span>${escapeHtml(label)}</span>${sortIndicator}</button></th>`;
  }

  function scoreHeader() {
    const ariaSort = state.sort.key === "score" ? ` aria-sort="${state.sort.direction === "asc" ? "ascending" : "descending"}"` : "";
    const control = scoreControl();
    return `<th scope="col" class="score-heading"${ariaSort}>${control || "Metric"}</th>`;
  }

  function metricDefinitions() {
    return activeMetrics();
  }

  function columns() {
    const common = ["<th scope=\"col\">#</th>", sortHeader("Model", "model"), sortHeader("Representation", "representation")];
    const metrics = metricDefinitions();
    // The bar column has no heading so the score number and its bar never share a cell.
    const content = [...common, scoreHeader(), '<th class="score-bar-heading" aria-hidden="true"></th>'];
    for (const metric of metrics) content.push(sortHeader(metric.label, metricSortKey(metric)));
    content.push(sortHeader("Submitted", "submitted"));
    return { html: content.join(""), count: content.length };
  }

  function errorValue(item, error = currentError()) {
    if (!error) return null;
    return getPath(item.metrics, error.path);
  }

  function valueFor(item, key) {
    if (key === "score") return errorValue(item);
    if (key === "model") return item.model;
    if (key === "representation") return item.representation;
    if (key === "submitted") return new Date(`${item.submittedAt}T00:00:00`).getTime();
    if (key.startsWith("metric:")) {
      const id = key.slice("metric:".length);
      const metric = (problemOptions()?.metrics || []).find((entry) => entry.id === id);
      return metric ? getPath(item.metrics, metric.path) : null;
    }
    return null;
  }

  // Log-scaled bar length between the best and worst visible scores, so close results stay distinguishable.
  function scoreRange(items) {
    const scores = items.map((submission) => Number(errorValue(submission))).filter((value) => Number.isFinite(value) && value > 0);
    return scores.length ? { min: Math.min(...scores), max: Math.max(...scores) } : null;
  }

  function barWidth(score, range) {
    const full = 58;
    const minimum = 8;
    if (!range || range.max <= range.min) return full;
    const value = Math.max(Number(score), range.min);
    const position = (Math.log10(value) - Math.log10(range.min)) / (Math.log10(range.max) - Math.log10(range.min));
    return minimum + position * (full - minimum);
  }

  function errorCell(score, label, range) {
    if (score === null || score === undefined || !Number.isFinite(Number(score))) {
      return `<td class="metric-value" title="${escapeHtml(label)}: unavailable">—</td><td class="score-bar-cell"></td>`;
    }
    const title = `${label}: ${formatError(score)}`;
    return `<td class="metric-value" title="${escapeHtml(title)}">${formatError(score)}</td><td class="score-bar-cell"><span class="metric-bar" role="img" aria-label="${escapeHtml(`${title} (log scale)`)}" title="${escapeHtml(`${title} · log scale`)}"><span style="width:${barWidth(score, range).toFixed(1)}px"></span></span></td>`;
  }

  function formattedValue(value, format) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
    const number = Number(value);
    if (format === "error") return formatError(number);
    if (format === "count") return formatCount(number);
    if (format === "milliseconds") return `${number.toFixed(1)} ms`;
    if (format === "hours") return `${number.toFixed(1)} h`;
    if (format === "gigabytes") return `${number.toFixed(1)} GB`;
    if (format === "percent") return `${number.toFixed(1)}%`;
    return new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 }).format(number);
  }

  function metricCell(item, metric, items) {
    const value = getPath(item.metrics, metric.path);
    if (metric.format === "error") {
      const rendered = Number.isFinite(Number(value)) && value !== null ? formatError(value) : "—";
      return `<td class="metric-value" title="${escapeHtml(`${metric.label}: ${rendered}`)}">${rendered}</td>`;
    }
    const rendered = formattedValue(value, metric.format);
    return `<td${value == null ? "" : ` title=\"${escapeHtml(`${metric.label}: ${rendered}`)}\"`}>${rendered}</td>`;
  }

  function row(item, index, items) {
    const model = escapeHtml(item.model);
    const authors = escapeHtml(item.authors);
    const compareHref = window.FluidBenchPaths.url(`/pages/compare.html?dataset=${encodeURIComponent(item.datasetId)}&model=${encodeURIComponent(item.id)}`);
    const medal = index < 3 ? `<span class="medal" aria-label="Rank ${index + 1}">${["🥇", "🥈", "🥉"][index]}</span>` : index + 1;
    const modelCell = `<td><span class="model-name">${model}</span><span class="model-author">${authors}</span></td>`;
    const representationCell = `<td><span class="representation-badge ${escapeHtml(String(item.representation || "").toLowerCase())}">${escapeHtml(item.representation)}</span></td>`;
    let metricCells = "";
    const selected = currentError();
    metricCells += errorCell(errorValue(item), selected?.label || "Error", scoreRange(items));
    for (const metric of metricDefinitions()) metricCells += metricCell(item, metric, items);
    return `<tr class="submission-row" data-href="${escapeHtml(compareHref)}" tabindex="0" aria-label="Open comparison for ${model}"><td class="rank-cell">${medal}</td>${modelCell}${representationCell}${metricCells}<td>${formatDate(item.submittedAt)}</td></tr>`;
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
    const firstMissing = first === null || first === undefined || (typeof first === "number" && !Number.isFinite(first));
    const secondMissing = second === null || second === undefined || (typeof second === "number" && !Number.isFinite(second));
    if (firstMissing || secondMissing) return firstMissing === secondMissing ? 0 : firstMissing ? 1 : -1;
    const comparison = typeof first === "number" && typeof second === "number"
      ? first - second
      : String(first).localeCompare(String(second), undefined, { sensitivity: "base", numeric: true });
    return direction === "desc" ? -comparison : comparison;
  }

  function setSort(key) {
    if (key === "representation") {
      state.representationFilter = state.representationFilter === "all"
        ? "Graph"
        : state.representationFilter === "Graph" ? "Grid" : "all";
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

  function closePickerMenus() {
    document.querySelectorAll(".error-picker-menu").forEach((menu) => {
      const button = document.getElementById(menu.dataset.buttonId || "");
      button?.setAttribute("aria-expanded", "false");
      if (menu.dataset.portaled === "true") {
        button?.closest(".error-picker")?.append(menu);
        delete menu.dataset.portaled;
        delete menu.dataset.buttonId;
        menu.removeAttribute("style");
      }
      menu.hidden = true;
    });
  }

  function openPickerMenu(button, menu) {
    if (!menu) return;
    closePickerMenus();
    menu.hidden = false;
    menu.dataset.portaled = "true";
    menu.dataset.buttonId = button.id;
    document.body.append(menu);

    const anchor = button.getBoundingClientRect();
    const menuWidth = menu.offsetWidth;
    const menuHeight = menu.offsetHeight;
    const edge = 8;
    const left = Math.max(edge, Math.min(anchor.left, window.innerWidth - menuWidth - edge));
    let top = anchor.bottom + 6;
    if (top + menuHeight > window.innerHeight - edge && anchor.top - menuHeight - 6 >= edge) {
      top = anchor.top - menuHeight - 6;
    } else {
      top = Math.min(top, window.innerHeight - menuHeight - edge);
    }
    menu.style.left = `${left}px`;
    menu.style.top = `${Math.max(edge, top)}px`;
    button.setAttribute("aria-expanded", "true");
    menu.querySelector("[role=menuitemradio]")?.focus();
  }

  function selectError(key) {
    if (!activeErrors().some((error) => error.id === key)) return;
    state.errorKey = key;
    state.sort.key = "score";
    state.sort.direction = "asc";
    state.sort.groupRepresentation = false;
    state.representationFilter = "all";
    render();
    $("#score-metric-button")?.focus();
  }

  function sortLabel(key) {
    if (key === "score") return currentError()?.label || "Error";
    if (key.startsWith("metric:")) {
      return (problemOptions()?.metrics || []).find((metric) => metric.id === key.slice(7))?.label || key;
    }
    return ({ model: "Model", representation: "Representation", submitted: "Submitted" })[key] || key;
  }

  function isErrorSort(key) {
    if (key === "score") return true;
    if (!key.startsWith("metric:")) return false;
    return (problemOptions()?.metrics || []).find((metric) => metric.id === key.slice(7))?.format === "error";
  }

  function render() {
    closePickerMenus();
    const visibleItems = filteredSubmissions();
    const items = visibleItems.sort((a, b) => {
      if (state.sort.groupRepresentation) {
        const groupResult = compareValues(a, b, "representation", state.sort.representationDirection);
        if (groupResult) return groupResult;
      }
      return compareValues(a, b, state.sort.key, state.sort.direction);
    });
    const tableColumns = columns();
    head.innerHTML = `<tr>${tableColumns.html}</tr>`;
    body.innerHTML = items.length
      ? items.map((item, index) => row(item, index, items)).join("")
      : `<tr><td class="empty-state" colspan="${tableColumns.count}">No submissions match these filters.</td></tr>`;
    const label = sortLabel(state.sort.key);
    const orderLabel = state.sort.key === "model"
      ? (state.sort.direction === "asc" ? "A to Z" : "Z to A")
      : state.sort.key === "submitted"
        ? (state.sort.direction === "desc" ? "newest first" : "oldest first")
        : isErrorSort(state.sort.key)
          ? (state.sort.direction === "asc" ? "best to worst" : "worst to best")
          : (state.sort.direction === "asc" ? "smallest to largest" : "largest to smallest");
    const sortDescription = state.sort.groupRepresentation
      ? `Representation: ${state.representationFilter === "all" ? "all submissions" : `${state.representationFilter} only`} · ${label} ${orderLabel}`
      : `${label} · ${orderLabel}`;
    resultCount.textContent = `${items.length} submissions · ${sortDescription} ${state.sort.direction === "asc" ? "↑" : "↓"}`;
  }

  head.addEventListener("click", (event) => {
    const button = event.target.closest("[data-sort-key]");
    if (!button) return;
    setSort(button.dataset.sortKey);
  });
  body.addEventListener("click", (event) => {
    const row = event.target.closest("tr.submission-row[data-href]");
    if (!row || event.target.closest("a, button, input, select, textarea")) return;
    window.location.href = row.dataset.href;
  });
  body.addEventListener("keydown", (event) => {
    const row = event.target.closest("tr.submission-row[data-href]");
    if (!row || event.target !== row || (event.key !== "Enter" && event.key !== " ")) return;
    event.preventDefault();
    window.location.href = row.dataset.href;
  });
  document.addEventListener("click", (event) => {
    const pickerOption = event.target.closest("[data-picker-kind][data-picker-value]");
    if (pickerOption) {
      selectError(pickerOption.dataset.pickerValue);
      return;
    }
    const pickerButton = event.target.closest("[data-picker-button]");
    if (pickerButton) {
      const menu = document.getElementById(pickerButton.dataset.menuId);
      if (pickerButton.getAttribute("aria-expanded") === "true") closePickerMenus();
      else openPickerMenu(pickerButton, menu);
      return;
    }
    if (!event.target.closest(".error-picker-menu")) closePickerMenus();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    const openMenu = document.querySelector(".error-picker-menu[data-portaled='true']");
    if (!openMenu) return;
    const button = document.getElementById(openMenu.dataset.buttonId || "");
    closePickerMenus();
    button?.focus();
  });
  window.addEventListener("resize", closePickerMenus);
  window.addEventListener("scroll", () => {
    if (document.querySelector(".error-picker-menu[data-portaled='true']")) closePickerMenus();
  }, true);
  datasetFilter.addEventListener("change", () => {
    state.representationFilter = "all";
    state.sort.groupRepresentation = false;
    state.sort.representationDirection = "asc";
    syncDataCategory(true);
    state.sort.key = "score";
    state.sort.direction = "asc";
    document.dispatchEvent(new CustomEvent("fluidbench:benchmark-selected", { detail: { id: datasetFilter.value } }));
    render();
  });
  searchInput.addEventListener("input", () => { state.query = searchInput.value; render(); });

  Promise.all([window.FluidBenchData.load(), window.FluidBenchData.loadOptions()]).then(([data, options]) => {
    state.data = data;
    state.options = options;
    fillDatasetFilter();
    syncDataCategory(true);
    state.sort.key = "score";
    document.dispatchEvent(new CustomEvent("fluidbench:benchmark-selected", { detail: { id: datasetFilter.value } }));
    render();
  }).catch((error) => {
    head.innerHTML = "";
    body.innerHTML = `<tr><td class="empty-state" colspan="8">${escapeHtml(error.message)}</td></tr>`;
    resultCount.textContent = "Submissions could not be loaded.";
  });
})();
