document.addEventListener("DOMContentLoaded", async () => {
  const tableBody = document.querySelector("#benchmark-directory-body");
  const details = document.querySelector("[data-selected-benchmark]");
  if (!tableBody || !details) return;

  const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
  const typeLabel = (value) => ({ synthetic: "Synthetic", real: "Real", "synthetic-real": "Synthetic & real" })[value] || "Unknown";
  const displayList = (value) => Array.isArray(value) ? value.join(", ") : "—";
  const displaySplits = (splits) => {
    if (!splits || typeof splits !== "object") return "—";
    const values = ["train", "validation", "test"].map((key) => splits[key] == null ? NaN : Number(splits[key]));
    if (values.some((value) => !Number.isFinite(value))) return "—";
    const percentages = values.map((value) => {
      const percent = value * 100;
      return Number.isInteger(percent) ? String(percent) : percent.toFixed(1).replace(/\.0$/, "");
    });
    return `${percentages.join("-")}%`;
  };

  let datasets = [];
  let selectedId = "";

  function renderRows() {
    tableBody.innerHTML = datasets.map((dataset) => {
      const selected = dataset.id === selectedId;
      const reynolds = displayList(dataset.reynoldsNumbers);
      const mach = dataset.machNumber == null ? "—" : dataset.machNumber;
      const samples = dataset.sample_count == null ? "—" : Number(dataset.sample_count).toLocaleString();
      const resolution = dataset.width == null || dataset.height == null ? "—" : `${dataset.width} × ${dataset.height}`;
      return `<tr id="benchmark-row-${escapeHtml(dataset.id)}" data-benchmark-id="${escapeHtml(dataset.id)}" tabindex="0" aria-label="Show ${escapeHtml(dataset.name)} details" class="${selected ? "is-selected" : ""}"><td><strong>${escapeHtml(dataset.name)}</strong></td><td>${escapeHtml(dataset.compressibility || "—")}</td><td>${escapeHtml(dataset.mesh || "—")}</td><td>${escapeHtml(typeLabel(dataset.trainingType))}</td><td>${escapeHtml(typeLabel(dataset.validationType))}</td><td>${escapeHtml(reynolds)}</td><td>${escapeHtml(mach)}</td><td>${escapeHtml(samples)}</td><td>${escapeHtml(resolution)}</td><td>${escapeHtml(displaySplits(dataset.splitFractions))}</td><td><a class="text-link" href="${window.FluidBenchPaths.url(`/pages/leaderboard.html?dataset=${encodeURIComponent(dataset.id)}`)}">View results →</a></td></tr>`;
    }).join("");
  }

  function selectBenchmark(id) {
    const dataset = datasets.find((item) => item.id === id);
    if (!dataset) return;
    selectedId = dataset.id;
    document.querySelectorAll("[data-dataset-id]").forEach((element) => {
      element.dataset.datasetId = dataset.id;
    });
    document.querySelectorAll("[data-benchmark-results-link]").forEach((link) => {
      link.href = window.FluidBenchPaths.url(`/pages/leaderboard.html?dataset=${encodeURIComponent(dataset.id)}`);
    });
    renderRows();
    document.dispatchEvent(new CustomEvent("fluidbench:benchmark-selected", { detail: { id: dataset.id } }));
  }

  tableBody.addEventListener("click", (event) => {
    if (event.target.closest("a")) return;
    const row = event.target.closest("tr[data-benchmark-id]");
    if (row) selectBenchmark(row.dataset.benchmarkId);
  });
  tableBody.addEventListener("keydown", (event) => {
    if (event.target.closest("a") || !["Enter", " "].includes(event.key)) return;
    const row = event.target.closest("tr[data-benchmark-id]");
    if (!row) return;
    event.preventDefault();
    selectBenchmark(row.dataset.benchmarkId);
  });

  try {
    datasets = await window.FluidBenchData.loadBenchmarks();
    if (!datasets.length) {
      tableBody.innerHTML = `<tr><td colspan="11">No benchmarks are available.</td></tr>`;
      return;
    }
    const requestedId = decodeURIComponent(window.location.hash.slice(1));
    const initialId = datasets.some((dataset) => dataset.id === requestedId)
      ? requestedId
      : datasets.find((dataset) => dataset.id === "cylinder-flow")?.id || datasets[0].id;
    selectBenchmark(initialId);
  } catch (error) {
    tableBody.innerHTML = `<tr><td colspan="11">${escapeHtml(error.message)}</td></tr>`;
  }
});
