(() => {
  const originCategories = document.querySelectorAll("[data-origin-category]");
  const benchmarkTags = document.querySelectorAll("[data-benchmark-tags]");
  const detailFields = document.querySelectorAll("[data-dataset-field]");
  const datasetImages = document.querySelectorAll("[data-dataset-image]");
  if (!originCategories.length && !benchmarkTags.length && !detailFields.length && !datasetImages.length) return;

  const datasetFilter = document.querySelector("#dataset-filter");
  let loadedDatasets = [];
  const originLabel = (value) => ({
    none: "None",
    synthetic: "Synthetic",
    real: "Real",
    "synthetic-real": "Synthetic & real",
  })[value] || "Unknown";
  const categories = {
    fullySynthetic: {
      id: "fully-synthetic",
      name: "Fully synthetic",
      description: "Training, validation, and test data are synthetic; pretraining data is absent or synthetic.",
    },
    simulatedTraining: {
      id: "simulated-training",
      name: "Simulated training",
      description: "Training data is synthetic and test data is real; validation data may be synthetic or real.",
    },
    realTraining: {
      id: "real-training",
      name: "Real training",
      description: "Training, validation, and test data are real; any pretraining data is real too.",
    },
    realFinetuning: {
      id: "real-finetuning",
      name: "Real finetuning",
      description: "Pretraining data is synthetic; training, validation, and test data are real.",
    },
    other: {
      id: "other",
      name: "Other setup",
      description: "This data-origin combination does not match one of the four named categories yet.",
    },
  };
  function classifyDataOrigin(dataOrigin) {
    if (!dataOrigin || typeof dataOrigin !== "object" || Array.isArray(dataOrigin)) return categories.other;
    const { pretrain, train, validation, test } = dataOrigin;
    const all = (value) => train === value && validation === value && test === value;
    const noPretraining = pretrain == null || pretrain === "none";
    if (pretrain === "synthetic" && all("real")) return categories.realFinetuning;
    if (train === "synthetic" && test === "real" && ["synthetic", "real"].includes(validation)) return categories.simulatedTraining;
    if (all("synthetic") && (noPretraining || pretrain === "synthetic")) return categories.fullySynthetic;
    if (all("real") && (noPretraining || pretrain === "real")) return categories.realTraining;
    return categories.other;
  }
  window.FluidBenchDataOrigins = { classify: classifyDataOrigin };
  const compressibilityClass = (value) => String(value || "").toLowerCase() === "compressible" ? "compressible" : "incompressible";
  const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
  const formatSplits = (splits) => {
    if (!splits || typeof splits !== "object" || Array.isArray(splits)) return "—";
    const keys = ["train", "validation", "test"];
    const values = keys.map((key) => splits[key] == null ? NaN : Number(splits[key]));
    if (values.some((value) => !Number.isFinite(value))) return "—";
    const percentages = values.map((value) => {
      const percent = value * 100;
      return Number.isInteger(percent) ? String(percent) : percent.toFixed(1).replace(/\.0$/, "");
    });
    return `${percentages.join("-")}%`;
  };
  function fieldValue(dataset, key) {
    if (key === "sampleCount") return dataset.sample_count == null ? "—" : Number(dataset.sample_count).toLocaleString();
    if (key === "splitFractions") return formatSplits(dataset.splitFractions);
    const value = dataset[key];
    if (value == null) return "—";
    if (Array.isArray(value)) return value.join(", ");
    if (typeof value === "object") return "—";
    return String(value);
  }

  function datasetFor(element, datasets) {
    const scope = element.closest("[data-follow-selection], [data-dataset-id]");
    const id = scope?.hasAttribute("data-follow-selection") ? datasetFilter?.value : scope?.dataset.datasetId;
    return datasets.find((item) => item.id === id) || datasets[0];
  }

  function render(datasets) {
    const selectedId = datasetFilter?.value;
    originCategories.forEach((container) => {
      const id = container.hasAttribute("data-follow-selection") ? selectedId : container.dataset.datasetId;
      const dataset = datasets.find((item) => item.id === id) || datasets[0];
      if (!dataset) return;
      const category = classifyDataOrigin(dataset.dataOrigin);
      container.setAttribute("aria-label", `Data origin category: ${category.name}. ${category.description}`);
      container.innerHTML = `<span class="dataset-origin-badge dataset-origin-${category.id}" title="${escapeHtml(category.description)}">${escapeHtml(category.name)}</span>`;
    });
    benchmarkTags.forEach((container) => {
      const dataset = datasetFor(container, datasets);
      if (!dataset) return;
      const compressibility = dataset.compressibility || "Incompressible";
      const mesh = dataset.mesh || "Unknown";
      const problem = dataset.problem || "Unknown";
      container.innerHTML = `<span class="benchmark-class-tag benchmark-problem-tag" title="Problem category">${escapeHtml(problem)}</span><span class="benchmark-class-tag benchmark-mesh-tag">${escapeHtml(mesh)} mesh</span><span class="benchmark-class-tag benchmark-${compressibilityClass(compressibility)}">${escapeHtml(compressibility)}</span>`;
    });
    datasetImages.forEach((image) => {
      const dataset = datasetFor(image, datasets);
      if (!dataset) return;
      if (dataset.image) image.src = window.FluidBenchPaths.url(dataset.image);
      image.alt = `${dataset.name} benchmark visualization`;
    });
    document.querySelectorAll("[data-dataset-origin]").forEach((cell) => {
      const dataset = datasetFor(cell, datasets);
      const value = dataset?.dataOrigin?.[cell.dataset.datasetOrigin];
      cell.textContent = originLabel(value);
    });
    detailFields.forEach((field) => {
      const dataset = datasetFor(field, datasets);
      if (!dataset) return;
      const key = field.dataset.datasetField;
      field.textContent = fieldValue(dataset, key);
    });
    document.querySelectorAll("[data-dataset-details-link]").forEach((link) => {
      const dataset = datasetFor(link, datasets);
      if (dataset) link.href = window.FluidBenchPaths.url(`/pages/benchmarks.html?dataset=${encodeURIComponent(dataset.id)}#benchmark-details`);
    });
  }

  document.addEventListener("fluidbench:benchmark-selected", () => {
    if (loadedDatasets.length) render(loadedDatasets);
  });

  window.FluidBenchData.loadBenchmarks()
    .then((datasets) => {
      loadedDatasets = datasets;
      render(datasets);
      datasetFilter?.addEventListener("change", () => render(datasets));
    })
    .catch(() => {
      originCategories.forEach((container) => { container.textContent = "Data origin unavailable"; });
    });
})();
