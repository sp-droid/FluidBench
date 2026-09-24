(() => {
  const tags = document.querySelectorAll("[data-dataset-tags]");
  const benchmarkTags = document.querySelectorAll("[data-benchmark-tags]");
  const detailFields = document.querySelectorAll("[data-dataset-field]");
  const datasetImages = document.querySelectorAll("[data-dataset-image]");
  if (!tags.length && !benchmarkTags.length && !detailFields.length && !datasetImages.length) return;

  const datasetFilter = document.querySelector("#dataset-filter");
  let loadedDatasets = [];
  const typeLabel = (value) => ({
    synthetic: "Synthetic",
    real: "Real",
    "synthetic-real": "Synthetic & real",
  })[value] || "Unknown";
  const typeClass = (value) => ({ synthetic: "synthetic", real: "real", "synthetic-real": "synthetic-real" })[value] || "unknown";
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
    if (key === "resolution") return dataset.width == null || dataset.height == null ? "—" : `${dataset.width} × ${dataset.height}`;
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
    tags.forEach((container) => {
      const id = container.hasAttribute("data-follow-selection") ? selectedId : container.dataset.datasetId;
      const dataset = datasets.find((item) => item.id === id) || datasets[0];
      if (!dataset) return;
      const trainingClass = typeClass(dataset.trainingType);
      const validationClass = typeClass(dataset.validationType);
      const trainingLabel = `${typeLabel(dataset.trainingType)} data training`;
      const validationLabel = `${typeLabel(dataset.validationType)} data validation`;
      container.innerHTML = `<span class="dataset-type-tag dataset-type-${trainingClass}">${trainingLabel}</span><span class="dataset-type-tag dataset-type-${validationClass}">${validationLabel}</span>`;
    });
    benchmarkTags.forEach((container) => {
      const dataset = datasetFor(container, datasets);
      if (!dataset) return;
      const compressibility = dataset.compressibility || "Incompressible";
      const mesh = dataset.mesh || "Unknown";
      container.innerHTML = `<span class="benchmark-class-tag benchmark-mesh-tag">${escapeHtml(mesh)} mesh</span><span class="benchmark-class-tag benchmark-${compressibilityClass(compressibility)}">${escapeHtml(compressibility)}</span>`;
    });
    datasetImages.forEach((image) => {
      const dataset = datasetFor(image, datasets);
      if (!dataset) return;
      if (dataset.image) image.src = dataset.image;
      image.alt = `${dataset.name} benchmark visualization`;
    });
    document.querySelectorAll("[data-dataset-type]").forEach((cell) => {
      const dataset = datasetFor(cell, datasets);
      const value = cell.dataset.datasetType === "training" ? dataset?.trainingType : dataset?.validationType;
      cell.textContent = typeLabel(value);
    });
    detailFields.forEach((field) => {
      const dataset = datasetFor(field, datasets);
      if (!dataset) return;
      const key = field.dataset.datasetField;
      field.textContent = fieldValue(dataset, key);
    });
    document.querySelectorAll("[data-dataset-details-link]").forEach((link) => {
      const dataset = datasetFor(link, datasets);
      if (dataset) link.href = `/pages/benchmarks.html#${encodeURIComponent(dataset.id)}`;
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
      tags.forEach((container) => { container.textContent = "Dataset types unavailable"; });
    });
})();
