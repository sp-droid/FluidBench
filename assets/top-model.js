document.addEventListener("DOMContentLoaded", async () => {
  const title = document.getElementById("top-model-title");
  if (!title) return;
  try {
    const data = await window.FluidBenchData.load();
    const leaders = data.submissions.slice().sort((a, b) => a.metrics.rmse - b.metrics.rmse);
    const leader = leaders[0];
    if (!leader) return;
    const dataset = data.datasets.find((item) => item.id === leader.datasetId);
    const representation = document.getElementById("top-model-representation");
    title.textContent = leader.model;
    document.getElementById("top-model-authors").textContent = leader.authors;
    document.getElementById("top-model-dataset").textContent = dataset?.name || "Benchmark";
    document.getElementById("top-model-rmse").textContent = leader.metrics.rmse.toExponential(1);
    document.getElementById("top-model-params").textContent = (leader.metrics.efficiency.parameters / 1000000).toFixed(1) + "M";
    document.getElementById("top-model-count").textContent = leaders.length + (leaders.length === 1 ? " submission" : " submissions");
    document.getElementById("site-submission-count").textContent = `${leaders.length} submissions, ${new Set(leaders.map((item) => item.authors)).size} teams`;
    const flowRegimeCount = new Set(data.datasets.map((item) => item.compressibility).filter(Boolean)).size;
    document.getElementById("site-benchmark-stats").textContent = `${flowRegimeCount} flow regime${flowRegimeCount === 1 ? "" : "s"}, ${data.datasets.length} datasets, 6 metrics`;
    representation.textContent = leader.representation;
    representation.className = "badge " + (leader.representation === "Graph" ? "badge-graph" : "badge-matrix");
  } catch {
    // The static fallback content remains visible if sample data is unavailable.
  }
});
