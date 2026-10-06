(() => {
  const leftPanel = document.querySelector("#compare-left");
  const rightPanel = document.querySelector("#compare-right");
  const params = new URLSearchParams(window.location.search);
  const requestedLeft = params.get("model");
  const requestedRight = params.get("compare");
  const rolloutErrorMetrics = [
    { label: "RMSE", id: "rmse" },
    { label: "MAE", id: "mae" },
    { label: "Relative L₂", id: "relativeL2" },
  ];

  const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
  const formatLoss = (value) => Number(value).toExponential(2).replace("e-0", "e-").replace("e+0", "e+");
  const formatCount = (value) => new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);
  const formatFixed = (value, digits = 1) => Number(value).toFixed(digits);
  const comparisonTone = (value, otherValue) => {
    if (value == null || otherValue == null) return "";
    if (!Number.isFinite(Number(value)) || !Number.isFinite(Number(otherValue))) return "";
    if (Number(value) < Number(otherValue)) return "is-better";
    if (Number(value) > Number(otherValue)) return "is-worse";
    return "is-tied";
  };

  function trajectoryLabel(trajectory) {
    const conditions = Object.entries(trajectory.conditions || {})
      .map(([name, value]) => `${name} ${Array.isArray(value) ? value.join(", ") : value}`)
      .join(" · ");
    return `${trajectory.label || trajectory.id}${conditions ? ` · ${conditions}` : ""}`;
  }

  function summarizeValues(values) {
    const finite = values.filter((value) => value !== null && value !== undefined).map(Number).filter(Number.isFinite);
    if (!finite.length) return { mean: null, standardDeviation: null, count: 0 };
    const mean = finite.reduce((sum, value) => sum + value, 0) / finite.length;
    const variance = finite.reduce((sum, value) => sum + (value - mean) ** 2, 0) / finite.length;
    return { mean, standardDeviation: Math.sqrt(variance), count: finite.length };
  }

  function selectedTrajectories(analysis, trajectoryId) {
    if (trajectoryId === "all") return analysis.trajectories || [];
    return (analysis.trajectories || []).filter((trajectory) => trajectory.id === trajectoryId);
  }

  function rolloutErrorSeries(analysis, metricId, trajectoryId) {
    const trajectories = selectedTrajectories(analysis, trajectoryId);
    const steps = analysis.timeSteps || [];
    const summaries = steps.map((step, index) => {
      // For "all", this is the arithmetic mean of every trajectory's error at this snapshot.
      const values = trajectories.map((trajectory) => trajectory.errors?.[metricId]?.[index]);
      return { step, ...summarizeValues(values) };
    });
    return {
      steps: summaries.map((summary) => summary.step),
      mean: summaries.map((summary) => summary.count ? summary.mean : null),
      standardDeviation: summaries.map((summary) => summary.count ? summary.standardDeviation : null),
    };
  }

  function rolloutCheckpointTable(analysis, metricId, trajectoryId, comparisonAnalysis = null) {
    const series = rolloutErrorSeries(analysis, metricId, trajectoryId);
    if (!series.steps.length) return `<div class="compare-chart-empty rollout-checkpoint-empty">No checkpoint values are available.</div>`;
    const comparisonSeries = comparisonAnalysis ? rolloutErrorSeries(comparisonAnalysis, metricId, trajectoryId) : null;
    const comparisonByStep = new Map((comparisonSeries?.steps || []).map((step, index) => [String(step), comparisonSeries.mean[index]]));
    const lastIndex = series.steps.length - 1;
    const percentileIndex = (percentile) => Math.max(0, Math.min(lastIndex, Math.round(series.steps.length * percentile) - 1));
    const checkpoints = [
      { label: "First step", index: 0 },
      { label: "P25", index: percentileIndex(.25) },
      { label: "P50", index: percentileIndex(.5) },
      { label: "P75", index: percentileIndex(.75) },
      { label: "Last step", index: lastIndex },
    ];
    const headers = checkpoints.map(({ label, index }) => `<th scope="col"><span>${label}</span><small>Snapshot ${series.steps[index]}</small></th>`).join("");
    const values = checkpoints.map(({ index }) => {
      const value = series.mean[index];
      const comparisonValue = comparisonByStep.get(String(series.steps[index]));
      const tone = comparisonSeries ? comparisonTone(value, comparisonValue) : "";
      const detail = value == null ? "" : trajectoryId === "all"
        ? `Mean across ${selectedTrajectories(analysis, trajectoryId).length} trajectories at snapshot ${series.steps[index]}: ${formatLoss(value)}`
        : `Snapshot ${series.steps[index]}: ${formatLoss(value)}`;
      const comparisonDetail = comparisonValue == null ? "" : ` · Compared value: ${formatLoss(comparisonValue)}`;
      const title = detail ? ` title="${escapeHtml(`${detail}${comparisonDetail}`)}"` : "";
      return `<td${tone ? ` class="${tone}"` : ""}${title}>${value == null ? "—" : formatLoss(value)}</td>`;
    }).join("");
    return `<div class="rollout-checkpoint-scroll"><table class="rollout-checkpoint-table"><thead><tr>${headers}</tr></thead><tbody><tr>${values}</tr></tbody></table></div>`;
  }

  function combinedSpectrum(trajectories, field) {
    const entries = trajectories.map((trajectory) => {
      const spectrum = trajectory.kineticEnergySpectrum?.[field];
      if (!spectrum || !Array.isArray(spectrum.mean)) return null;
      return {
        mean: spectrum.mean,
        standardDeviation: spectrum.standardDeviation || spectrum.mean.map(() => 0),
        count: Number(spectrum.snapshotCount) || (trajectory.errors?.rmse || []).length || 1,
      };
    }).filter(Boolean);
    if (!entries.length) return null;
    const length = Math.min(...entries.map((entry) => entry.mean.length));
    const totalCount = entries.reduce((sum, entry) => sum + entry.count, 0);
    const mean = Array.from({ length }, (_, index) => entries.reduce(
      (sum, entry) => sum + Number(entry.mean[index]) * entry.count,
      0,
    ) / totalCount);
    const standardDeviation = Array.from({ length }, (_, index) => Math.sqrt(entries.reduce((sum, entry) => {
      const entryMean = Number(entry.mean[index]);
      const entryStd = Number(entry.standardDeviation[index] || 0);
      return sum + entry.count * (entryStd ** 2 + (entryMean - mean[index]) ** 2);
    }, 0) / totalCount));
    return { mean, standardDeviation };
  }

  function axisNumber(value) {
    if (!Number.isFinite(value) || value === 0) return "0";
    return Number(value).toExponential(1).replace("e-0", "e-").replace("e+0", "e+");
  }

  function linePath(xs, values, yFor) {
    let drawing = false;
    return values.map((value, index) => {
      if (value == null || !Number.isFinite(Number(value))) {
        drawing = false;
        return "";
      }
      const command = drawing ? "L" : "M";
      drawing = true;
      return `${command}${xs[index].toFixed(1)} ${yFor(Number(value)).toFixed(1)}`;
    }).filter(Boolean).join(" ");
  }

  function bandPath(xs, means, standardDeviations, yFor) {
    if (!means.length || !standardDeviations.some((value) => Number(value) > 0)) return "";
    const paths = [];
    let segment = [];
    const finishSegment = () => {
      if (!segment.length || !segment.some((index) => Number(standardDeviations[index]) > 0)) {
        segment = [];
        return;
      }
      const upper = segment.map((index) => `${xs[index].toFixed(1)} ${yFor(Number(means[index]) + Number(standardDeviations[index] || 0)).toFixed(1)}`);
      const lower = segment.map((index) => `${xs[index].toFixed(1)} ${yFor(Math.max(0, Number(means[index]) - Number(standardDeviations[index] || 0))).toFixed(1)}`).reverse();
      paths.push(`M${upper.join(" L")} L${lower.join(" L")} Z`);
      segment = [];
    };
    means.forEach((mean, index) => {
      if (mean == null || !Number.isFinite(Number(mean))) finishSegment();
      else segment.push(index);
    });
    finishSegment();
    return paths.join(" ");
  }

  function bandBoundaryPaths(xs, means, standardDeviations, yFor, seriesClass) {
    if (!means.length || !standardDeviations.some((value) => Number(value) > 0)) return "";
    const upper = means.map((mean, index) => mean == null ? null : Number(mean) + Number(standardDeviations[index] || 0));
    const lower = means.map((mean, index) => mean == null ? null : Math.max(0, Number(mean) - Number(standardDeviations[index] || 0)));
    return `<path class="rollout-band-edge ${seriesClass}" d="${linePath(xs, upper, yFor)}"></path><path class="rollout-band-edge ${seriesClass}" d="${linePath(xs, lower, yFor)}"></path>`;
  }

  function chartGrid(yMax, yFor, xLabels, xPositions, xLabel, yLabel) {
    const yTicks = [0, 0.5, 1].map((fraction) => {
      const value = yMax * fraction;
      const y = yFor(value);
      return `<g class="rollout-gridline"><line x1="52" y1="${y.toFixed(1)}" x2="428" y2="${y.toFixed(1)}"/><text x="46" y="${(y + 3).toFixed(1)}" text-anchor="end">${axisNumber(value)}</text></g>`;
    }).join("");
    const xTicks = xLabels.map((label, index) => `<g class="rollout-tick"><line x1="${xPositions[index].toFixed(1)}" y1="181" x2="${xPositions[index].toFixed(1)}" y2="185"/><text x="${xPositions[index].toFixed(1)}" y="198" text-anchor="middle">${escapeHtml(label)}</text></g>`).join("");
    return `${yTicks}${xTicks}<line class="rollout-axis" x1="52" y1="20" x2="52" y2="181"/><line class="rollout-axis" x1="52" y1="181" x2="428" y2="181"/><text class="rollout-axis-label rollout-y-axis-label" transform="translate(13 100) rotate(-90)" text-anchor="middle">${escapeHtml(yLabel)}</text><text class="rollout-axis-label" x="240" y="220" text-anchor="middle">${escapeHtml(xLabel)}</text>`;
  }

  function rolloutErrorChart(analysis, metricId, trajectoryId, comparisonAnalysis = null) {
    const series = rolloutErrorSeries(analysis, metricId, trajectoryId);
    const spreads = series.mean.map((mean, index) => mean == null ? null : Number(mean) + Number(series.standardDeviation[index] || 0)).filter(Number.isFinite);
    if (!series.steps.length || !spreads.length) return `<div class="compare-chart-empty">No rollout error data is available.</div>`;
    const yMax = Math.max(...spreads, Number.EPSILON) * 1.08;
    const yFor = (value) => 181 - (Math.max(0, value) / yMax) * 161;
    const xFor = (index) => 52 + (series.steps.length === 1 ? 0 : index / (series.steps.length - 1)) * 376;
    const xs = series.steps.map((_, index) => xFor(index));
    const comparisonSeries = comparisonAnalysis ? rolloutErrorSeries(comparisonAnalysis, metricId, trajectoryId) : null;
    const comparisonIndicesByStep = new Map((comparisonSeries?.steps || []).map((step, index) => [String(step), index]));
    const xLabels = [series.steps[0], series.steps[Math.floor((series.steps.length - 1) / 2)], series.steps[series.steps.length - 1]];
    const xPositions = [xs[0], xs[Math.floor((xs.length - 1) / 2)], xs[xs.length - 1]];
    const label = rolloutErrorMetrics.find((metric) => metric.id === metricId)?.label || "Error";
    const comparisonSegments = comparisonSeries
      ? series.steps.slice(0, -1).map((step, index) => {
        const nextStep = series.steps[index + 1];
        const comparisonIndex = comparisonIndicesByStep.get(String(step));
        const comparisonNextIndex = comparisonIndicesByStep.get(String(nextStep));
        if (comparisonIndex == null || comparisonNextIndex == null) return "";
        const ownStart = series.mean[index];
        const ownEnd = series.mean[index + 1];
        const otherStart = comparisonSeries.mean[comparisonIndex];
        const otherEnd = comparisonSeries.mean[comparisonNextIndex];
        if ([ownStart, ownEnd, otherStart, otherEnd].some((value) => value == null || !Number.isFinite(Number(value)))) return "";
        const ownMean = (Number(ownStart) + Number(ownEnd)) / 2;
        const otherMean = (Number(otherStart) + Number(otherEnd)) / 2;
        const tone = comparisonTone(ownMean, otherMean);
        if (!tone) return "";
        const result = ownMean < otherMean ? "Lower error" : ownMean > otherMean ? "Higher error" : "Equal error";
        const title = `Snapshots ${step}–${nextStep} · ${label}: ${ownMean.toPrecision(5)} · Comparison: ${otherMean.toPrecision(5)} · ${result}`;
        return `<line class="rollout-comparison-segment ${tone}" x1="${xs[index].toFixed(1)}" y1="181" x2="${xs[index + 1].toFixed(1)}" y2="181"><title>${escapeHtml(title)}</title></line>`;
      }).join("")
      : "";
    const points = series.mean.map((value, index) => value == null ? "" : `<circle class="rollout-error-point" cx="${xs[index].toFixed(1)}" cy="${yFor(Number(value)).toFixed(1)}" r="2"><title>Snapshot ${series.steps[index]} · Mean ${label}: ${Number(value).toPrecision(5)} · SD: ${Number(series.standardDeviation[index] || 0).toPrecision(5)}</title></circle>`).join("");
    const grid = chartGrid(yMax, yFor, xLabels, xPositions, "Prediction snapshot", label);
    const legend = trajectoryId === "all"
      ? `<g class="rollout-in-graph-legend"><rect x="316" y="23" width="107" height="22" rx="5"/><rect class="rollout-legend-mean-band" x="324" y="29" width="14" height="10" rx="2"/><line class="rollout-legend-mean" x1="324" y1="34" x2="338" y2="34"/><text x="343" y="37">Mean ± 1 SD</text></g>`
      : `<g class="rollout-in-graph-legend"><rect x="337" y="23" width="86" height="22" rx="5"/><line class="rollout-legend-mean" x1="344" y1="34" x2="358" y2="34"/><text x="363" y="37">Snapshot error</text></g>`;
    return `<div class="rollout-chart-wrap"><svg class="rollout-chart" viewBox="0 0 440 232" role="img" aria-label="${escapeHtml(label)} rollout error by prediction snapshot">
      ${grid}<path class="rollout-band rollout-band-model" d="${bandPath(xs, series.mean, series.standardDeviation, yFor)}"></path>${bandBoundaryPaths(xs, series.mean, series.standardDeviation, yFor, "rollout-band-edge-model")}<path class="rollout-line rollout-error-line" d="${linePath(xs, series.mean, yFor)}"></path>${points}${comparisonSegments}
      ${legend}
    </svg></div>`;
  }

  function rolloutSpectrumSeries(analysis, trajectoryId) {
    const trajectories = selectedTrajectories(analysis, trajectoryId);
    return {
      wavenumbers: analysis.wavenumbers || [],
      target: combinedSpectrum(trajectories, "target"),
      prediction: combinedSpectrum(trajectories, "prediction"),
    };
  }

  function spectrumTrajectorySelect(analysis) {
    const options = analysis.trajectories.map((trajectory) => `<option value="${escapeHtml(trajectory.id)}">${escapeHtml(trajectoryLabel(trajectory))}</option>`).join("");
    return `<div class="rollout-spectrum-controls"><label>Test trajectory<select data-spectrum-trajectory aria-label="Choose a test trajectory for the kinetic energy spectrum"><option value="all">All trajectories · mean ± 1 SD</option>${options}</select></label></div>`;
  }

  function kineticSpectrumChart(analysis, trajectoryId, comparisonAnalysis = null) {
    const series = rolloutSpectrumSeries(analysis, trajectoryId);
    if (!series.wavenumbers.length || !series.target || !series.prediction) {
      return `<div class="compare-chart-empty">Kinetic energy spectra are not available for this benchmark.</div>`;
    }
    const maximum = Math.max(
      ...series.target.mean.map((value, index) => Number(value) + Number(series.target.standardDeviation[index] || 0)),
      ...series.prediction.mean.map((value, index) => Number(value) + Number(series.prediction.standardDeviation[index] || 0)),
      Number.EPSILON,
    );
    const yMax = maximum * 1.08;
    const yFor = (value) => 181 - (Math.max(0, value) / yMax) * 161;
    const xFor = (index) => 52 + (series.wavenumbers.length === 1 ? 0 : index / (series.wavenumbers.length - 1)) * 376;
    const xs = series.wavenumbers.map((_, index) => xFor(index));
    const comparisonSeries = comparisonAnalysis ? rolloutSpectrumSeries(comparisonAnalysis, trajectoryId) : null;
    const comparisonIndicesByWavenumber = new Map((comparisonSeries?.wavenumbers || []).map((wavenumber, index) => [String(wavenumber), index]));
    const xLabels = [series.wavenumbers[0], series.wavenumbers[Math.floor((series.wavenumbers.length - 1) / 2)], series.wavenumbers[series.wavenumbers.length - 1]];
    const xPositions = [xs[0], xs[Math.floor((xs.length - 1) / 2)], xs[xs.length - 1]];
    const grid = chartGrid(yMax, yFor, xLabels, xPositions, "Wavenumber k", "Kinetic energy");
    const markers = (values, name, className) => values.map((value, index) => `<circle class="rollout-spectrum-point ${className}" cx="${xs[index].toFixed(1)}" cy="${yFor(Number(value)).toFixed(1)}" r="2"><title>k = ${series.wavenumbers[index]} · ${name}: ${Number(value).toPrecision(5)}</title></circle>`).join("");
    const comparisonSegments = comparisonSeries?.target && comparisonSeries?.prediction
      ? series.wavenumbers.slice(0, -1).map((wavenumber, index) => {
        const nextWavenumber = series.wavenumbers[index + 1];
        const comparisonIndex = comparisonIndicesByWavenumber.get(String(wavenumber));
        const comparisonNextIndex = comparisonIndicesByWavenumber.get(String(nextWavenumber));
        if (comparisonIndex == null || comparisonNextIndex == null) return "";
        const rawValues = [
          series.prediction.mean[index], series.target.mean[index],
          series.prediction.mean[index + 1], series.target.mean[index + 1],
          comparisonSeries.prediction.mean[comparisonIndex], comparisonSeries.target.mean[comparisonIndex],
          comparisonSeries.prediction.mean[comparisonNextIndex], comparisonSeries.target.mean[comparisonNextIndex],
        ];
        if (rawValues.some((value) => value == null || !Number.isFinite(Number(value)))) return "";
        const [ownPredictionStart, ownReferenceStart, ownPredictionEnd, ownReferenceEnd,
          otherPredictionStart, otherReferenceStart, otherPredictionEnd, otherReferenceEnd] = rawValues.map(Number);
        const ownStart = Math.abs(ownPredictionStart - ownReferenceStart);
        const ownEnd = Math.abs(ownPredictionEnd - ownReferenceEnd);
        const otherStart = Math.abs(otherPredictionStart - otherReferenceStart);
        const otherEnd = Math.abs(otherPredictionEnd - otherReferenceEnd);
        const ownGap = (ownStart + ownEnd) / 2;
        const otherGap = (otherStart + otherEnd) / 2;
        const tone = comparisonTone(ownGap, otherGap);
        if (!tone) return "";
        const result = ownGap < otherGap ? "Closer to reference" : ownGap > otherGap ? "Farther from reference" : "Equal distance from reference";
        const title = `Wavenumbers ${wavenumber}–${nextWavenumber} · Prediction/reference gap: ${ownGap.toPrecision(5)} · Comparison: ${otherGap.toPrecision(5)} · ${result}`;
        return `<line class="rollout-comparison-segment ${tone}" x1="${xs[index].toFixed(1)}" y1="181" x2="${xs[index + 1].toFixed(1)}" y2="181"><title>${escapeHtml(title)}</title></line>`;
      }).join("")
      : "";
    const legend = `<g class="rollout-in-graph-legend"><rect x="258" y="23" width="165" height="22" rx="5"/><rect class="rollout-legend-reference-band" x="264" y="29" width="13" height="10" rx="2"/><line class="rollout-legend-reference" x1="264" y1="34" x2="277" y2="34"/><text x="281" y="37">Reference</text><rect class="rollout-legend-prediction-band" x="335" y="29" width="13" height="10" rx="2"/><line class="rollout-legend-prediction" x1="335" y1="34" x2="348" y2="34"/><text x="352" y="37">Prediction</text></g>`;
    return `<div class="rollout-chart-wrap"><svg class="rollout-chart" viewBox="0 0 440 232" role="img" aria-label="Kinetic energy spectrum of predicted and reference trajectories">
      ${grid}<path class="rollout-band rollout-band-reference" d="${bandPath(xs, series.target.mean, series.target.standardDeviation, yFor)}"></path><path class="rollout-band rollout-band-model" d="${bandPath(xs, series.prediction.mean, series.prediction.standardDeviation, yFor)}"></path>${bandBoundaryPaths(xs, series.target.mean, series.target.standardDeviation, yFor, "rollout-band-edge-reference")}${bandBoundaryPaths(xs, series.prediction.mean, series.prediction.standardDeviation, yFor, "rollout-band-edge-model")}
      <path class="rollout-line rollout-reference-line" d="${linePath(xs, series.target.mean, yFor)}"></path><path class="rollout-line rollout-prediction-line" d="${linePath(xs, series.prediction.mean, yFor)}"></path>
      ${markers(series.target.mean, "Reference", "rollout-target-point")}${markers(series.prediction.mean, "Prediction", "rollout-prediction-point")}
      ${comparisonSegments}
      ${legend}
    </svg></div>`;
  }

  function rolloutAnalysisSection(item, other = null) {
    const analysis = item.rolloutAnalysis;
    if (!analysis || !analysis.trajectories?.length) {
      return `<div class="compare-section"><h3>Rollout analysis</h3><div class="compare-chart-empty">No per-trajectory rollout analysis is available for this submission.</div></div>`;
    }
    const errorOptions = rolloutErrorMetrics.map((metric) => `<option value="${metric.id}">${metric.label}</option>`).join("");
    const trajectoryOptions = analysis.trajectories.map((trajectory) => `<option value="${escapeHtml(trajectory.id)}">${escapeHtml(trajectoryLabel(trajectory))}</option>`).join("");
    return `<div class="compare-section rollout-analysis-section">
      <div class="compare-section-title"><div><h3>Rollout analysis</h3><p>Per-snapshot errors and time-averaged kinetic spectra</p></div></div>
      <div class="rollout-chart-block"><h4>Snapshot error</h4><div class="rollout-analysis-controls"><label>Error type<select data-rollout-metric>${errorOptions}</select></label><label>Test trajectory<select data-rollout-trajectory><option value="all">All trajectories · mean ± 1 SD</option>${trajectoryOptions}</select></label></div><div data-rollout-checkpoints>${rolloutCheckpointTable(analysis, "rmse", "all", other?.rolloutAnalysis)}</div><div data-rollout-error-chart>${rolloutErrorChart(analysis, "rmse", "all", other?.rolloutAnalysis)}</div></div>
      <div class="rollout-chart-block"><h4>Kinetic energy spectrum</h4>${spectrumTrajectorySelect(analysis)}<div data-rollout-spectrum-chart>${kineticSpectrumChart(analysis, "all", other?.rolloutAnalysis)}</div><p class="rollout-chart-note">Shading and thin edges show ±1 SD across snapshots and selected cases.</p></div>
    </div>`;
  }

  function modelPicker(submissions, leftId, benchmarkId, selectedId = "") {
    const candidates = submissions.filter((submission) => submission.id !== leftId && submission.datasetId === benchmarkId);
    const options = candidates.map((submission) => `<option value="${escapeHtml(submission.id)}"${submission.id === selectedId ? " selected" : ""}>${escapeHtml(submission.model)}</option>`).join("");
    const placeholder = candidates.length ? "Choose a model…" : "No other submissions yet";
    return `<label class="compare-model-select"><span>Compare against</span><select id="compare-model-select" aria-label="Choose a different model"${candidates.length ? "" : " disabled"}><option value=""${selectedId ? "" : " selected"}>${placeholder}</option>${options}</select></label>`;
  }

  function modelFacts(item, other, dataset) {
    const metrics = item.metrics;
    const efficiency = metrics.efficiency;
    const facts = [
      { label: "Benchmark", value: escapeHtml(dataset?.name || item.datasetId) },
      { label: "Representation", value: `<span class="representation-badge ${item.representation.toLowerCase()}">${escapeHtml(item.representation)}</span>` },
      { label: "Parameters", value: formatCount(efficiency.parameters), metric: efficiency.parameters, other: other?.metrics.efficiency.parameters },
      { label: "Inference / sample", value: `${formatFixed(efficiency.inferenceMs)} ms`, metric: efficiency.inferenceMs, other: other?.metrics.efficiency.inferenceMs },
      { label: "GPU memory", value: `${formatFixed(efficiency.memoryGb)} GB`, metric: efficiency.memoryGb, other: other?.metrics.efficiency.memoryGb },
    ];
    return `<dl class="compare-facts">${facts.map((fact) => {
      const tone = other && fact.metric != null ? ` ${comparisonTone(fact.metric, fact.other)}` : "";
      return `<div class="compare-fact-item${tone}"><dt>${fact.label}</dt><dd>${fact.value}</dd></div>`;
    }).join("")}</dl>`;
  }

  function averageTestRolloutError(item, metricId) {
    const values = (item.rolloutAnalysis?.trajectories || []).flatMap((trajectory) => {
      const errors = trajectory.errors?.[metricId];
      return Array.isArray(errors) ? errors : [];
    });
    return summarizeValues(values).mean;
  }

  function evaluationLosses(item, other) {
    const columns = [
      { label: "Test rollout", value: averageTestRolloutError },
    ];
    const header = columns.map(({ label }) => `<th scope="col">${label}</th>`).join("");
    const rows = rolloutErrorMetrics.map(({ label, id }) => {
      const values = columns.map(({ value }) => {
        const result = value(item, id);
        const comparison = other ? value(other, id) : null;
        const tone = other ? comparisonTone(result, comparison) : "";
        return `<td${tone ? ` class="${tone}"` : ""}>${result == null ? "—" : formatLoss(result)}</td>`;
      }).join("");
      return `<tr><th scope="row">${escapeHtml(label)}</th>${values}</tr>`;
    }).join("");
    return `<div class="evaluation-loss-scroll"><table class="evaluation-loss-table" aria-label="Average evaluation losses by split"><thead><tr><th scope="col">Error type</th>${header}</tr></thead><tbody>${rows}</tbody></table></div>`;
  }

  function renderPanel(item, other, dataset, side, submissions, leftId) {
    const controls = side === "right" ? modelPicker(submissions, leftId, item.datasetId, item.id) : `<span class="compare-fixed-label">Selected from leaderboard</span>`;
    return `<div class="compare-panel-top"><div class="compare-model-title"><span class="compare-side-label">${side === "left" ? "Model A" : "Model B"}</span><h2>${escapeHtml(item.model)}</h2><p>${escapeHtml(item.authors)} · ${item.year}</p></div>${controls}</div>
      <div class="compare-section"><h3>Submission details</h3>${modelFacts(item, other, dataset)}</div>
      <div class="compare-section"><div class="compare-section-title"><div><h3>Evaluation losses</h3><p>Test rollout averages all snapshots and trajectories.</p></div></div>${evaluationLosses(item, other)}</div>
      ${rolloutAnalysisSection(item, other)}`;
  }

  function emptyComparePanel(submissions, leftId, benchmarkId) {
    const hasCandidates = submissions.some((submission) => submission.id !== leftId && submission.datasetId === benchmarkId);
    const guidance = hasCandidates
      ? "Pick another submission from this benchmark to compare its results."
      : "There are no other submissions for this benchmark yet.";
    return `<div class="compare-panel-top"><div class="compare-model-title"><span class="compare-side-label">Model B</span><h2>No model selected</h2><p>${guidance}</p></div>${modelPicker(submissions, leftId, benchmarkId)}</div><div class="compare-empty-state"><span class="compare-empty-icon" aria-hidden="true">↔</span><h3>Ready to compare</h3><p>${guidance}</p></div>`;
  }

  function updateUrl(left, right) {
    const url = new URL(window.location.href);
    url.searchParams.set("model", left.id);
    if (right) url.searchParams.set("compare", right.id);
    else url.searchParams.delete("compare");
    window.history.replaceState({}, "", url);
  }

  function refreshRolloutPanel(panel, item, other = null) {
    if (!item?.rolloutAnalysis) return;
    const metric = panel.querySelector("[data-rollout-metric]")?.value || "rmse";
    const trajectory = panel.querySelector("[data-rollout-trajectory]")?.value || "all";
    const errorChart = panel.querySelector("[data-rollout-error-chart]");
    const checkpointTable = panel.querySelector("[data-rollout-checkpoints]");
    if (errorChart) errorChart.innerHTML = rolloutErrorChart(item.rolloutAnalysis, metric, trajectory, other?.rolloutAnalysis);
    if (checkpointTable) checkpointTable.innerHTML = rolloutCheckpointTable(item.rolloutAnalysis, metric, trajectory, other?.rolloutAnalysis);

    const spectrumChart = panel.querySelector("[data-rollout-spectrum-chart]");
    const spectrumTrajectory = panel.querySelector("[data-spectrum-trajectory]")?.value || "all";
    if (spectrumChart) spectrumChart.innerHTML = kineticSpectrumChart(item.rolloutAnalysis, spectrumTrajectory, other?.rolloutAnalysis);
  }

  function bindRolloutControls(panel, item, peerPanel = null, peerItem = null) {
    const controlSelectors = ["[data-rollout-metric]", "[data-rollout-trajectory]", "[data-spectrum-trajectory]"];
    const syncPeer = (selector, value) => {
      const peerControl = peerPanel?.querySelector(selector);
      if (peerControl && Array.from(peerControl.options).some((option) => option.value === value)) peerControl.value = value;
    };
    const refreshBoth = () => {
      refreshRolloutPanel(panel, item, peerItem);
      if (peerPanel && peerItem) refreshRolloutPanel(peerPanel, peerItem, item);
    };
    controlSelectors.forEach((selector) => {
      const control = panel.querySelector(selector);
      if (!control) return;
      control.addEventListener("change", () => {
        syncPeer(selector, control.value);
        refreshBoth();
      });
    });
    refreshBoth();
  }

  window.FluidBenchData.load().then((data) => {
    const submissions = data.submissions || [];
    if (submissions.length < 2) throw new Error("Add at least two submissions to compare models.");
    const sortedByRmse = submissions.slice().sort((a, b) => a.metrics.rmse - b.metrics.rmse);
    const left = submissions.find((item) => item.id === requestedLeft) || sortedByRmse[0];
    const rightCandidates = submissions.filter((item) => item.id !== left.id && item.datasetId === left.datasetId);
    let right = rightCandidates.find((item) => item.id === requestedRight) || null;
    const datasets = data.datasets || [];

    async function render() {
      const [leftRollout, rightRollout] = await Promise.all([
        window.FluidBenchData.loadRolloutAnalysis(left.id, left.datasetId, left.artifacts?.rolloutAnalysis),
        right ? window.FluidBenchData.loadRolloutAnalysis(right.id, right.datasetId, right.artifacts?.rolloutAnalysis) : Promise.resolve(null),
      ]);
      left.rolloutAnalysis = leftRollout;
      if (right) {
        right.rolloutAnalysis = rightRollout;
      }
      leftPanel.className = "compare-panel compare-panel-left";
      rightPanel.className = `compare-panel compare-panel-right${right ? "" : " is-unselected"}`;
      leftPanel.innerHTML = renderPanel(left, right, datasets.find((dataset) => dataset.id === left.datasetId), "left", submissions, left.id);
      rightPanel.innerHTML = right
        ? renderPanel(right, left, datasets.find((dataset) => dataset.id === right.datasetId), "right", submissions, left.id)
        : emptyComparePanel(submissions, left.id, left.datasetId);
      bindRolloutControls(leftPanel, left, right ? rightPanel : null, right);
      if (right) bindRolloutControls(rightPanel, right, leftPanel, left);
      document.querySelector("#compare-model-select")?.addEventListener("change", (event) => {
        right = rightCandidates.find((item) => item.id === event.target.value) || null;
        render().catch(showError);
      });
      updateUrl(left, right);
    }

    function showError(error) {
      const message = `<div class="compare-error">${escapeHtml(error.message)}</div>`;
      rightPanel.innerHTML = message;
    }

    render().catch(showError);
  }).catch((error) => {
    const message = `<div class="compare-error">${escapeHtml(error.message)}</div>`;
    leftPanel.innerHTML = message;
    rightPanel.innerHTML = message;
  });
})();
