(() => {
  const jsonCache = new Map();

  async function readJson(path) {
    const url = window.FluidBenchPaths.url(path);
    if (!jsonCache.has(url)) {
      jsonCache.set(url, fetch(url).then((response) => {
        if (!response.ok) throw new Error(`Could not load ${url} (${response.status}).`);
        return response.json();
      }));
    }
    return jsonCache.get(url);
  }

  function safeIds(ids, label) {
    if (!Array.isArray(ids) || ids.some((id) => typeof id !== "string" || !/^[a-z0-9_-]+$/i.test(id))) {
      throw new Error(`The ${label} manifest contains an invalid ID.`);
    }
    return ids;
  }

  async function loadSubmissions() {
    const manifest = await readJson("/submissions/submissions.json");
    const submissionsByBenchmark = manifest.submissionsByBenchmark;
    if (!submissionsByBenchmark || typeof submissionsByBenchmark !== "object" || Array.isArray(submissionsByBenchmark)) {
      throw new Error("The submission manifest must group submission IDs by benchmark.");
    }
    const benchmarkIds = safeIds(Object.keys(submissionsByBenchmark), "benchmark");
    const routes = benchmarkIds.flatMap((benchmarkId) => {
      const submissionIds = safeIds(submissionsByBenchmark[benchmarkId], `${benchmarkId} submission`);
      return submissionIds.map((id) => ({ benchmarkId, id }));
    });
    return Promise.all(routes.map(({ benchmarkId, id }) => readJson(`/submissions/${encodeURIComponent(benchmarkId)}/${encodeURIComponent(id)}.json`)));
  }

  async function loadBenchmarks() {
    const manifest = await readJson("/benchmarks/benchmarks.json");
    const ids = safeIds(manifest.benchmarkIds, "benchmark");
    return Promise.all(ids.map((id) => readJson(`/benchmarks/${encodeURIComponent(id)}.json`)));
  }

  async function load() {
    const [submissions, datasets] = await Promise.all([loadSubmissions(), loadBenchmarks()]);
    return { submissions, datasets };
  }

  async function loadTrainingHistory(id, benchmarkId) {
    if (!/^[a-z0-9_-]+$/i.test(id)) throw new Error("The submission ID is invalid.");
    if (!/^[a-z0-9_-]+$/i.test(benchmarkId || "")) throw new Error("The benchmark ID is invalid.");
    const history = await readJson(`/submissions/${encodeURIComponent(benchmarkId)}/${encodeURIComponent(id)}/training-history.json`);
    if (!Array.isArray(history.epochs) || !Array.isArray(history.trainingLosses) || !Array.isArray(history.validationLosses)) {
      throw new Error(`The training history for ${id} is incomplete.`);
    }
    if (history.epochs.length !== history.trainingLosses.length || history.epochs.length !== history.validationLosses.length) {
      throw new Error(`The training history for ${id} has mismatched list lengths.`);
    }
    return {
      lossName: "RMSE",
      points: history.epochs.map((epoch, index) => ({
        epoch,
        trainingLoss: history.trainingLosses[index],
        validationLoss: history.validationLosses[index],
      })),
    };
  }

  window.FluidBenchData = { load, loadSubmissions, loadBenchmarks, loadTrainingHistory };
})();
