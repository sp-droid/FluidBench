document.addEventListener("DOMContentLoaded", async () => {
  const list = document.getElementById("model-list");
  if (!list) return;
  try {
    const data = await window.FluidBenchData.load();
    const rows = data.submissions.slice().sort((a, b) => a.metrics.rmse - b.metrics.rmse);
    list.innerHTML = rows.map((item, index) => {
      const badge = item.representation === "Graph" ? "badge-graph" : "badge-matrix";
      const params = (item.metrics.efficiency.parameters / 1000000).toFixed(1) + "M";
      const rmse = item.metrics.rmse.toExponential(1);
      const l2 = item.metrics.relativeL2.toExponential(1);
      const dataset = data.datasets.find((entry) => entry.id === item.datasetId);
      return '<article class="model-card" id="' + item.id + '"><div class="model-rank">' + (index === 0 ? "♛" : String(index + 1)) + '</div><div class="model-card-main"><p class="eyebrow">' + (dataset?.name || "Benchmark") + ' · ' + item.year + '</p><h2>' + item.model + '</h2><p class="model-card-authors">' + item.authors + '</p><span class="badge ' + badge + '">' + item.representation + '</span></div><div class="model-card-metrics"><div><span>RMSE</span><strong>' + rmse + '</strong></div><div><span>Relative L2</span><strong>' + l2 + '</strong></div><div><span>Parameters</span><strong>' + params + '</strong></div><div><span>Training</span><strong>' + item.metrics.efficiency.trainingHours.toFixed(1) + ' h</strong></div></div><a class="text-link model-detail-link" href="' + window.FluidBenchPaths.url('/pages/leaderboard.html?q=' + encodeURIComponent(item.model)) + '">View result →</a></article>';
    }).join("");
    if (window.location.hash) {
      const target = document.getElementById(decodeURIComponent(window.location.hash.slice(1)));
      if (target) window.setTimeout(() => target.scrollIntoView({ block: "center", behavior: "smooth" }), 80);
    }
  } catch (error) {
    list.innerHTML = '<div class="callout">Sample models could not be loaded. Serve this site over HTTP and reload.</div>';
  }
});
