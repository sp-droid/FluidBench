(() => {
  const jsonCache = new Map();

  async function readJson(path) {
    const url = window.FluidBenchPaths.url(path);
    if (!jsonCache.has(url)) {
      jsonCache.set(url, fetch(url).then((response) => {
        if (!response.ok) {
          const error = new Error(`Could not load ${url} (${response.status}).`);
          error.status = response.status;
          throw error;
        }
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

  // submissions/submissions.json lists the benchmark folders; each folder has its own manifest,
  // submissions/{benchmark}/submissions.json, listing the submission folders inside it, and each record lives at
  // submissions/{benchmark}/{id}/submission.json next to its artifacts.
  async function benchmarkSubmissionIds(benchmarkId) {
    const manifest = await readJson(`/submissions/${encodeURIComponent(benchmarkId)}/submissions.json`);
    return safeIds(manifest.submissionIds, `${benchmarkId} submission`);
  }

  // Fast path: update.py also writes one bundle with every lightweight record; the manifest chain is the fallback.
  async function readBundle(path, key) {
    try {
      const bundle = await readJson(path);
      if (!Array.isArray(bundle[key])) throw new Error(`${path} must contain a "${key}" list.`);
      return bundle[key];
    } catch (error) {
      if (error.status === 404) return null;
      throw error;
    }
  }

  async function loadSubmissions() {
    const bundled = await readBundle("/submissions/bundle.json", "submissions");
    if (bundled) return bundled;
    const manifest = await readJson("/submissions/submissions.json");
    const benchmarkIds = safeIds(manifest.benchmarkIds, "benchmark");
    const idsByBenchmark = await Promise.all(benchmarkIds.map(benchmarkSubmissionIds));
    const routes = benchmarkIds.flatMap((benchmarkId, index) => idsByBenchmark[index].map((id) => ({ benchmarkId, id })));
    return Promise.all(routes.map(({ benchmarkId, id }) => readJson(`/submissions/${encodeURIComponent(benchmarkId)}/${encodeURIComponent(id)}/submission.json`)));
  }

  async function loadBenchmarks() {
    const bundled = await readBundle("/benchmarks/bundle.json", "benchmarks");
    if (bundled) return bundled;
    const manifest = await readJson("/benchmarks/benchmarks.json");
    const ids = safeIds(manifest.benchmarkIds, "benchmark");
    return Promise.all(ids.map((id) => readJson(`/benchmarks/${encodeURIComponent(id)}.json`)));
  }

  async function load() {
    const [submissions, datasets] = await Promise.all([loadSubmissions(), loadBenchmarks()]);
    return { submissions, datasets };
  }

  async function loadOptions() {
    return readJson("/benchmarks/options.json");
  }

  function artifactPath(id, benchmarkId, artifact, fallback) {
    if (!/^[a-z0-9_-]+$/i.test(id)) throw new Error("The submission ID is invalid.");
    if (!/^[a-z0-9_-]+$/i.test(benchmarkId || "")) throw new Error("The benchmark ID is invalid.");
    const filename = artifact || fallback;
    if (typeof filename !== "string" || !/^[a-z0-9_-]+(?:\.[a-z0-9_-]+)*$/i.test(filename)) {
      throw new Error(`The artifact filename for ${id} is invalid.`);
    }
    return `/submissions/${encodeURIComponent(benchmarkId)}/${encodeURIComponent(id)}/${encodeURIComponent(filename)}`;
  }

  async function loadTrainingHistory(id, benchmarkId, artifact) {
    // Training curves are optional: submissions only carry them when the author provided them.
    if (!artifact) return null;
    const history = await readJson(artifactPath(id, benchmarkId, artifact, "training-history.json"));
    if (!Array.isArray(history.epochs) || !Array.isArray(history.trainingLosses) || !Array.isArray(history.validationLosses)) {
      throw new Error(`The training history for ${id} is incomplete.`);
    }
    if (history.epochs.length !== history.trainingLosses.length || history.epochs.length !== history.validationLosses.length) {
      throw new Error(`The training history for ${id} has mismatched list lengths.`);
    }
    return {
      lossName: typeof history.lossName === "string" && history.lossName ? history.lossName : "RMSE",
      validationSplit: Number.isFinite(history.validationSplit) ? history.validationSplit : null,
      points: history.epochs.map((epoch, index) => ({
        epoch,
        trainingLoss: history.trainingLosses[index],
        validationLoss: history.validationLosses[index],
      })),
    };
  }

  async function loadRolloutAnalysis(id, benchmarkId, artifact) {
    try {
      const analysis = await readJson(artifactPath(id, benchmarkId, artifact, "rollout-analysis.json"));
      if (!Array.isArray(analysis.trajectories) || !Array.isArray(analysis.timeSteps)) {
        throw new Error(`The rollout analysis for ${id} is incomplete.`);
      }
      return analysis;
    } catch (error) {
      if (error.status === 404) return null;
      throw error;
    }
  }

  window.FluidBenchData = { load, loadSubmissions, loadBenchmarks, loadOptions, loadTrainingHistory, loadRolloutAnalysis };
})();
