(() => {
  const leftPanel = document.querySelector("#compare-left");
  const rightPanel = document.querySelector("#compare-right");
  const params = new URLSearchParams(window.location.search);
  const requestedLeft = params.get("model");
  const requestedRight = params.get("compare");
  // Submission IDs are only unique within a benchmark, so the benchmark is part of the address.
  const requestedDataset = params.get("dataset");
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

  // Conditions are stored as float32 (e.g. Re 109.00000762939453); show them rounded.
  const formatCondition = (value) => Number.isFinite(Number(value)) ? String(Number(Number(value).toPrecision(6))) : String(value);

  function trajectoryLabel(trajectory) {
    const conditions = Object.entries(trajectory.conditions || {})
      .map(([name, value]) => `${name} ${Array.isArray(value) ? value.map(formatCondition).join(", ") : formatCondition(value)}`)
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

  function bandBoundaryPaths(xs, means, standardDeviations, yFor, seriesClass, series = "") {
    if (!means.length || !standardDeviations.some((value) => Number(value) > 0)) return "";
    const upper = means.map((mean, index) => mean == null ? null : Number(mean) + Number(standardDeviations[index] || 0));
    const lower = means.map((mean, index) => mean == null ? null : Math.max(0, Number(mean) - Number(standardDeviations[index] || 0)));
    const tag = series ? ` data-series="${series}"` : "";
    return `<path class="rollout-band-edge ${seriesClass}"${tag} d="${linePath(xs, upper, yFor)}"></path><path class="rollout-band-edge ${seriesClass}"${tag} d="${linePath(xs, lower, yFor)}"></path>`;
  }

  // Log range fitted to the data (not rounded out to whole decades), with a small margin.
  function fittedLogRange(minimum, maximum) {
    let low = Math.log10(minimum);
    let high = Math.log10(maximum);
    if (high - low < 0.2) {
      const middle = (low + high) / 2;
      low = middle - 0.1;
      high = middle + 0.1;
    }
    const margin = (high - low) * 0.04;
    return { minExponent: low - margin, maxExponent: high + margin };
  }

  // Tick values at whole decades inside a log range; falls back to its ends when it spans less than two decades.
  function logTicks(range, maxTicks = 6) {
    const first = Math.ceil(range.minExponent);
    const last = Math.floor(range.maxExponent);
    if (last - first < 1) return [10 ** range.minExponent, 10 ** range.maxExponent];
    const step = Math.max(1, Math.ceil((last - first + 1) / maxTicks));
    const ticks = [];
    for (let exponent = first; exponent <= last; exponent += step) ticks.push(10 ** exponent);
    return ticks;
  }

  function chartGrid(yMax, yFor, xLabels, xPositions, xLabel, yLabel, yTickValues = null) {
    const yTicks = (yTickValues || [0, 0.5, 1].map((fraction) => yMax * fraction)).map((value) => {
      const y = yFor(value);
      return `<g class="rollout-gridline"><line x1="52" y1="${y.toFixed(1)}" x2="428" y2="${y.toFixed(1)}"/><text x="46" y="${(y + 3).toFixed(1)}" text-anchor="end">${axisNumber(value)}</text></g>`;
    }).join("");
    const xTicks = xLabels.map((label, index) => `<g class="rollout-tick"><line x1="${xPositions[index].toFixed(1)}" y1="181" x2="${xPositions[index].toFixed(1)}" y2="185"/><text x="${xPositions[index].toFixed(1)}" y="198" text-anchor="middle">${escapeHtml(label)}</text></g>`).join("");
    return `${yTicks}${xTicks}<line class="rollout-axis" x1="52" y1="8" x2="52" y2="181"/><line class="rollout-axis" x1="52" y1="181" x2="428" y2="181"/><text class="rollout-axis-label rollout-y-axis-label" transform="translate(13 100) rotate(-90)" text-anchor="middle">${escapeHtml(yLabel)}</text><text class="rollout-axis-label" x="240" y="220" text-anchor="middle">${escapeHtml(xLabel)}</text>`;
  }

  function rolloutErrorChart(analysis, metricId, trajectoryId, comparisonAnalysis = null) {
    const series = rolloutErrorSeries(analysis, metricId, trajectoryId);
    const upperEdges = (part) => part.mean.map((mean, index) => mean == null ? null : Number(mean) + Number(part.standardDeviation[index] || 0)).filter(Number.isFinite);
    const spreads = upperEdges(series);
    if (!series.steps.length || !spreads.length) return `<div class="compare-chart-empty">No rollout error data is available.</div>`;
    // Same y-range in both panels: cover this submission and the one it is compared with.
    const comparisonSpreads = comparisonAnalysis ? upperEdges(rolloutErrorSeries(comparisonAnalysis, metricId, trajectoryId)) : [];
    const yMax = Math.max(...spreads, ...comparisonSpreads, Number.EPSILON) * 1.03;
    const yFor = (value) => 181 - (Math.max(0, value) / yMax) * 173;
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
        return `<line class="rollout-comparison-segment ${tone}" data-series="comparison" x1="${xs[index].toFixed(1)}" y1="181" x2="${xs[index + 1].toFixed(1)}" y2="181"><title>${escapeHtml(title)}</title></line>`;
      }).join("")
      : "";
    const grid = chartGrid(yMax, yFor, xLabels, xPositions, "Prediction snapshot", label);
    const legend = trajectoryId === "all"
      ? `<g class="rollout-in-graph-legend"><rect x="58" y="11" width="107" height="22" rx="5"/><g class="chart-legend-entry" data-legend-series="mean"><rect class="chart-legend-hit" x="60" y="13" width="103" height="18"/><rect class="rollout-legend-mean-band" x="66" y="17" width="14" height="10" rx="2"/><line class="rollout-legend-mean" x1="66" y1="22" x2="80" y2="22"/><text x="85" y="25">Mean ± 1 SD</text></g></g>`
      : `<g class="rollout-in-graph-legend"><rect x="58" y="11" width="86" height="22" rx="5"/><g class="chart-legend-entry" data-legend-series="mean"><rect class="chart-legend-hit" x="60" y="13" width="82" height="18"/><line class="rollout-legend-mean" x1="65" y1="22" x2="79" y2="22"/><text x="84" y="25">Snapshot error</text></g></g>`;
    return `<div class="rollout-chart-wrap"><svg class="rollout-chart" viewBox="0 0 440 232" role="img" aria-label="${escapeHtml(label)} rollout error by prediction snapshot">
      ${grid}<path class="rollout-band rollout-band-model" data-series="mean" d="${bandPath(xs, series.mean, series.standardDeviation, yFor)}"></path>${bandBoundaryPaths(xs, series.mean, series.standardDeviation, yFor, "rollout-band-edge-model", "mean")}<path class="rollout-line rollout-error-line" data-series="mean" d="${linePath(xs, series.mean, yFor)}"></path>${comparisonSegments}
      ${legend}
    </svg></div>`;
  }

  // Each test case's error averaged over all of its rollout snapshots.
  function caseAverages(analysis, metricId) {
    return (analysis?.trajectories || []).map((trajectory, index) => {
      const re = trajectory.conditions?.Re;
      return {
        id: trajectory.id,
        label: re == null ? String(index + 1) : formatCondition(re),
        name: trajectoryLabel(trajectory),
        value: summarizeValues(trajectory.errors?.[metricId] || []).mean,
      };
    });
  }

  function caseErrorChart(analysis, metricId, comparisonAnalysis = null) {
    const cases = caseAverages(analysis, metricId);
    const comparison = new Map(caseAverages(comparisonAnalysis, metricId).map((entry) => [entry.id, entry.value]));
    // Same y-range in both panels.
    const values = [...cases.map((entry) => entry.value), ...comparison.values()].filter((value) => value != null && Number.isFinite(Number(value)));
    if (!cases.length || !values.length) return `<div class="compare-chart-empty">No per-case errors are available.</div>`;
    const yMax = Math.max(...values.map(Number), Number.EPSILON) * 1.03;
    const yFor = (value) => 181 - (Math.max(0, value) / yMax) * 173;
    const slot = 376 / cases.length;
    const barWidth = Math.min(40, Math.max(1.5, slot * 0.7));
    const metricLabel = rolloutErrorMetrics.find((metric) => metric.id === metricId)?.label || "Error";
    const bars = cases.map((entry, index) => {
      if (entry.value == null || !Number.isFinite(Number(entry.value))) return "";
      const other = comparison.get(entry.id);
      const tone = comparisonTone(entry.value, other);
      const y = yFor(Number(entry.value));
      const x = 52 + slot * index + (slot - barWidth) / 2;
      const versus = other == null ? "" : ` · Comparison: ${formatLoss(other)}`;
      return `<rect class="case-bar${tone ? ` ${tone}` : ""}" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barWidth.toFixed(1)}" height="${(181 - y).toFixed(1)}"><title>${escapeHtml(`${entry.name} · Average ${metricLabel}: ${formatLoss(entry.value)}${versus}`)}</title></rect>`;
    }).join("");
    // Label every case when they fit, otherwise about twelve evenly spread ones.
    const every = Math.max(1, Math.ceil(cases.length / 12));
    const tickIndices = cases.map((_, index) => index).filter((index) => index % every === 0 || index === cases.length - 1);
    const hasRe = cases.some((entry, index) => entry.label !== String(index + 1));
    const grid = chartGrid(yMax, yFor, tickIndices.map((index) => cases[index].label), tickIndices.map((index) => 52 + slot * (index + 0.5)), hasRe ? "Test case (Re)" : "Test case", `Average ${metricLabel}`);
    return `<div class="rollout-chart-wrap"><svg class="rollout-chart" viewBox="0 0 440 232" role="img" aria-label="Average ${escapeHtml(metricLabel)} for each test case">${grid}${bars}</svg></div>`;
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
    // Same y-range in both panels: cover this submission's curves and the comparison's.
    const comparisonSpectrum = comparisonAnalysis ? rolloutSpectrumSeries(comparisonAnalysis, trajectoryId) : null;
    const parts = [series.target, series.prediction, comparisonSpectrum?.target, comparisonSpectrum?.prediction].filter(Boolean);
    const maximum = Math.max(
      ...parts.flatMap((part) => part.mean.map((value, index) => Number(value) + Number(part.standardDeviation[index] || 0))).filter(Number.isFinite),
      Number.EPSILON,
    );
    // Log-log axes: energy and wavenumber both span orders of magnitude.
    const positive = parts.flatMap((part) => part.mean.flatMap((value, index) => {
      const sd = Number(part.standardDeviation[index] || 0);
      return [Number(value), Number(value) + sd, Number(value) - sd];
    })).filter((value) => Number.isFinite(value) && value > 0);
    const energyRange = fittedLogRange(Math.min(...positive, maximum), maximum);
    const yFor = (value) => {
      const clamped = Math.min(Math.max(Number(value), 10 ** energyRange.minExponent), 10 ** energyRange.maxExponent);
      return 181 - ((Math.log10(clamped) - energyRange.minExponent) / (energyRange.maxExponent - energyRange.minExponent)) * 173;
    };
    const yTickValues = logTicks(energyRange);
    const kMin = Math.max(Number(series.wavenumbers[0]), Number.EPSILON);
    const kMax = Number(series.wavenumbers[series.wavenumbers.length - 1]);
    const logK = kMax > kMin;
    const xFor = (index) => {
      if (series.wavenumbers.length === 1) return 52;
      if (!logK) return 52 + (index / (series.wavenumbers.length - 1)) * 376;
      return 52 + ((Math.log10(Math.max(Number(series.wavenumbers[index]), kMin)) - Math.log10(kMin)) / (Math.log10(kMax) - Math.log10(kMin))) * 376;
    };
    const xs = series.wavenumbers.map((_, index) => xFor(index));
    const comparisonSeries = comparisonAnalysis ? rolloutSpectrumSeries(comparisonAnalysis, trajectoryId) : null;
    const comparisonIndicesByWavenumber = new Map((comparisonSeries?.wavenumbers || []).map((wavenumber, index) => [String(wavenumber), index]));
    // Ticks at the powers of two present in the spectrum, plus its last wavenumber.
    const tickIndices = series.wavenumbers
      .map((wavenumber, index) => (Number.isInteger(Math.log2(Number(wavenumber))) || index === series.wavenumbers.length - 1 ? index : null))
      .filter((index) => index != null);
    const xLabels = tickIndices.map((index) => series.wavenumbers[index]);
    const xPositions = tickIndices.map((index) => xs[index]);
    const grid = chartGrid(null, yFor, xLabels, xPositions, "Wavenumber k · log scale", "Kinetic energy · log scale", yTickValues);
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
        return `<line class="rollout-comparison-segment ${tone}" data-series="comparison" x1="${xs[index].toFixed(1)}" y1="181" x2="${xs[index + 1].toFixed(1)}" y2="181"><title>${escapeHtml(title)}</title></line>`;
      }).join("")
      : "";
    const legend = `<g class="rollout-in-graph-legend"><rect x="258" y="11" width="165" height="22" rx="5"/><g class="chart-legend-entry" data-legend-series="reference"><rect class="chart-legend-hit" x="260" y="13" width="70" height="18"/><rect class="rollout-legend-reference-band" x="264" y="17" width="13" height="10" rx="2"/><line class="rollout-legend-reference" x1="264" y1="22" x2="277" y2="22"/><text x="281" y="25">Reference</text></g><g class="chart-legend-entry" data-legend-series="prediction"><rect class="chart-legend-hit" x="331" y="13" width="88" height="18"/><rect class="rollout-legend-prediction-band" x="335" y="17" width="13" height="10" rx="2"/><line class="rollout-legend-prediction" x1="335" y1="22" x2="348" y2="22"/><text x="352" y="25">Prediction</text></g></g>`;
    return `<div class="rollout-chart-wrap"><svg class="rollout-chart" viewBox="0 0 440 232" role="img" aria-label="Kinetic energy spectrum of predicted and reference trajectories">
      ${grid}<path class="rollout-band rollout-band-reference" data-series="reference" d="${bandPath(xs, series.target.mean, series.target.standardDeviation, yFor)}"></path><path class="rollout-band rollout-band-model" data-series="prediction" d="${bandPath(xs, series.prediction.mean, series.prediction.standardDeviation, yFor)}"></path>${bandBoundaryPaths(xs, series.target.mean, series.target.standardDeviation, yFor, "rollout-band-edge-reference", "reference")}${bandBoundaryPaths(xs, series.prediction.mean, series.prediction.standardDeviation, yFor, "rollout-band-edge-model", "prediction")}
      <path class="rollout-line rollout-reference-line" data-series="reference" d="${linePath(xs, series.target.mean, yFor)}"></path><path class="rollout-line rollout-prediction-line" data-series="prediction" d="${linePath(xs, series.prediction.mean, yFor)}"></path>
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
      <div class="compare-section-title"><div><h3>Rollout analysis</h3><p>Test-set errors per case and per snapshot, and time-averaged kinetic spectra</p></div></div>
      <div class="rollout-chart-block"><h4>Average error</h4>${evaluationLosses(item, other)}</div>
      <div class="rollout-chart-block"><h4>Error per test case</h4><div class="rollout-spectrum-controls"><label>Error type<select data-case-metric>${errorOptions}</select></label></div><div data-case-error-chart>${caseErrorChart(analysis, "rmse", other?.rolloutAnalysis)}</div><p class="rollout-chart-note">Each bar is one test case, averaged over all its rollout snapshots. Green or orange marks a lower or higher error than the compared submission on that case.</p></div>
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

  function chartY(value, logScale) {
    const clamped = Math.min(Math.max(Number(value), 10 ** logScale.minExponent), 10 ** logScale.maxExponent);
    const normalized = (Math.log10(clamped) - logScale.minExponent) / (logScale.maxExponent - logScale.minExponent);
    return 12 + (1 - normalized) * 164;
  }

  function epochBounds(points) {
    const epochs = points.map((point) => Number(point.epoch)).filter(Number.isFinite);
    return epochs.length ? { min: Math.min(...epochs), max: Math.max(...epochs) } : { min: 0, max: 0 };
  }

  function formatEpoch(epoch) {
    return Number.isInteger(epoch) ? String(epoch) : String(Number(epoch.toPrecision(3)));
  }

  function epochTicks(maxEpoch) {
    if (maxEpoch <= 0) return [0];
    const divisions = Math.min(4, Math.max(1, Math.ceil(maxEpoch)));
    return Array.from({ length: divisions + 1 }, (_, index) => formatEpoch((maxEpoch * index) / divisions));
  }

  function chartPath(points, field, logScale, maxEpoch) {
    const left = 55;
    const right = 12;
    const width = 440;
    const plotWidth = width - left - right;
    return points.map((point, index) => {
      const x = left + (Number(point.epoch) / maxEpoch) * plotWidth;
      const y = chartY(point[field], logScale);
      return `${index ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`;
    }).join(" ");
  }

  function trainingChart(item, logScale) {
    const points = item.trainingHistory?.points || [];
    if (!points.length) return `<div class="training-chart-wrap"><div class="compare-chart-empty training-chart-empty">No training history provided.</div></div>`;
    const { max: dataMaxEpoch } = epochBounds(points);
    const maxEpoch = dataMaxEpoch > 0 ? dataMaxEpoch : 1;
    const gridLines = logTicks(logScale).map((value) => {
      const y = chartY(value, logScale);
      return `<g class="chart-gridline"><line x1="55" y1="${y}" x2="428" y2="${y}"/><text x="48" y="${y + 3}" text-anchor="end">${formatLoss(value)}</text></g>`;
    }).join("");
    const xTicks = epochTicks(dataMaxEpoch).map((epoch) => {
      const x = 55 + (Number(epoch) / maxEpoch) * 373;
      return `<g class="chart-tick"><line x1="${x}" y1="176" x2="${x}" y2="180"/><text x="${x}" y="195" text-anchor="middle">${epoch}</text></g>`;
    }).join("");
    const label = `${item.model} ${item.trainingHistory.lossName || "RMSE"} training and validation loss by epoch`;
    return `<div class="training-chart-wrap"><svg class="training-chart" viewBox="0 0 440 210" role="img" aria-label="${escapeHtml(label)}" preserveAspectRatio="xMidYMid meet">
      ${gridLines}${xTicks}<line class="chart-axis" x1="55" y1="12" x2="55" y2="176"/><line class="chart-axis" x1="55" y1="176" x2="428" y2="176"/>
      <path class="chart-series chart-validation-line" data-series="validation" d="${chartPath(points, "validationLoss", logScale, maxEpoch)}"></path><path class="chart-series chart-training-line" data-series="training" d="${chartPath(points, "trainingLoss", logScale, maxEpoch)}"></path>
      <g class="chart-legend-in-graph"><rect x="277" y="17" width="147" height="32" rx="5"/><g class="chart-legend-entry" data-legend-series="training"><rect class="chart-legend-hit" x="282" y="20" width="66" height="14"/><line class="chart-legend-training" x1="286" y1="28" x2="302" y2="28"/><text x="307" y="31">Training</text></g><g class="chart-legend-entry" data-legend-series="validation"><rect class="chart-legend-hit" x="350" y="20" width="70" height="14"/><line class="chart-legend-validation" x1="354" y1="28" x2="370" y2="28"/><text x="375" y="31">Validation</text></g><text class="chart-legend-loss-name" x="286" y="43">${escapeHtml(item.trainingHistory.lossName || "RMSE")} loss</text></g>
      <text class="chart-axis-label chart-y-label" transform="translate(13 95) rotate(-90)" text-anchor="middle">${escapeHtml(item.trainingHistory.lossName || "RMSE")} loss · log scale</text>
      <text class="chart-axis-label" x="241" y="208" text-anchor="middle">Training epoch</text>
    </svg></div>`;
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

  function formatValidationSplit(split) {
    if (!Number.isFinite(split)) return "—";
    const validation = Math.round(split * 1000) / 10;
    return `${Math.round((100 - validation) * 10) / 10}-${validation}%`;
  }

  // Error name, split and final epoch losses, as reported with the training curves.
  function trainingSummary(item, other) {
    const history = item.trainingHistory;
    const last = history?.points?.[history.points.length - 1];
    const otherHistory = other?.trainingHistory;
    const otherLast = otherHistory?.points?.[otherHistory.points.length - 1];
    const comparable = history && otherHistory && history.lossName === otherHistory.lossName;
    const facts = [
      { label: "Error type", value: history ? escapeHtml(history.lossName) : "—" },
      { label: "Train-val split", value: formatValidationSplit(history?.validationSplit) },
      { label: "Final train error", value: last ? formatLoss(last.trainingLoss) : "—", metric: last?.trainingLoss, other: comparable ? otherLast?.trainingLoss : null },
      { label: "Final validation error", value: last ? formatLoss(last.validationLoss) : "—", metric: last?.validationLoss, other: comparable ? otherLast?.validationLoss : null },
    ];
    return `<dl class="compare-facts training-summary">${facts.map((fact) => {
      const tone = fact.metric != null && fact.other != null ? ` ${comparisonTone(fact.metric, fact.other)}` : "";
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
      { label: "Average", value: averageTestRolloutError },
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
    return `<div class="evaluation-loss-scroll"><table class="evaluation-loss-table" aria-label="Average test errors"><thead><tr><th scope="col">Error type</th>${header}</tr></thead><tbody>${rows}</tbody></table></div>`;
  }

  function renderPanel(item, other, dataset, side, logScale, submissions, leftId) {
    const controls = side === "right" ? modelPicker(submissions, leftId, item.datasetId, item.id) : `<span class="compare-fixed-label">Selected from leaderboard</span>`;
    const points = item.trainingHistory?.points || [];
    const range = epochBounds(points);
    const rangeLabel = points.length
      ? (range.min === range.max ? `Epoch ${formatEpoch(range.min)}` : `Epoch ${formatEpoch(range.min)}–${formatEpoch(range.max)}`)
      : "No epoch data";
    return `<div class="compare-panel-top"><div class="compare-model-title"><span class="compare-side-label">${side === "left" ? "Model A" : "Model B"}</span><h2>${escapeHtml(item.model)}</h2><p>${escapeHtml(item.authors)} · ${item.year}</p></div>${controls}</div>
      <div class="compare-section"><h3>Submission details</h3>${modelFacts(item, other, dataset)}</div>
      <div class="compare-section"><div class="compare-section-title"><div><h3>Training loss</h3><p>${item.trainingHistory ? `Epoch versus ${escapeHtml(item.trainingHistory.lossName)}` : "Epoch versus training error"}</p></div><span class="chart-range">${rangeLabel} · log scale</span></div>${trainingChart(item, logScale)}${trainingSummary(item, other)}</div>
      ${rolloutAnalysisSection(item, other)}`;
  }

  function emptyComparePanel(submissions, leftId, benchmarkId) {
    const hasCandidates = submissions.some((submission) => submission.id !== leftId && submission.datasetId === benchmarkId);
    const guidance = hasCandidates
      ? "Pick another submission from this benchmark to compare its curves and losses."
      : "There are no other submissions for this benchmark yet.";
    return `<div class="compare-panel-top"><div class="compare-model-title"><span class="compare-side-label">Model B</span><h2>No model selected</h2><p>${guidance}</p></div>${modelPicker(submissions, leftId, benchmarkId)}</div><div class="compare-empty-state"><span class="compare-empty-icon" aria-hidden="true">↔</span><h3>Ready to compare</h3><p>${guidance}</p></div>`;
  }

  function updateUrl(left, right) {
    const url = new URL(window.location.href);
    url.searchParams.set("dataset", left.datasetId);
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

    const caseChart = panel.querySelector("[data-case-error-chart]");
    const caseMetric = panel.querySelector("[data-case-metric]")?.value || "rmse";
    if (caseChart) caseChart.innerHTML = caseErrorChart(item.rolloutAnalysis, caseMetric, other?.rolloutAnalysis);

    const spectrumChart = panel.querySelector("[data-rollout-spectrum-chart]");
    const spectrumTrajectory = panel.querySelector("[data-spectrum-trajectory]")?.value || "all";
    if (spectrumChart) spectrumChart.innerHTML = kineticSpectrumChart(item.rolloutAnalysis, spectrumTrajectory, other?.rolloutAnalysis);
  }

  function bindRolloutControls(panel, item, peerPanel = null, peerItem = null) {
    const controlSelectors = ["[data-rollout-metric]", "[data-rollout-trajectory]", "[data-case-metric]", "[data-spectrum-trajectory]"];
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

  // Hovering a legend entry fades every other series in the same chart to near invisibility.
  function focusSeries(svg, series) {
    svg?.querySelectorAll("[data-series]").forEach((element) => {
      element.style.opacity = series && element.dataset.series !== series ? "0.02" : "";
    });
  }
  document.addEventListener("pointerover", (event) => {
    const entry = event.target.closest?.("[data-legend-series]");
    if (entry) focusSeries(entry.closest("svg"), entry.dataset.legendSeries);
  });
  document.addEventListener("pointerout", (event) => {
    const entry = event.target.closest?.("[data-legend-series]");
    if (entry && !entry.contains(event.relatedTarget)) focusSeries(entry.closest("svg"), null);
  });

  window.FluidBenchData.load().then((data) => {
    const submissions = data.submissions || [];
    if (submissions.length < 2) throw new Error("Add at least two submissions to compare models.");
    const sortedByRmse = submissions.slice().sort((a, b) => a.metrics.rmse - b.metrics.rmse);
    const left = submissions.find((item) => item.id === requestedLeft && (!requestedDataset || item.datasetId === requestedDataset)) || sortedByRmse[0];
    const rightCandidates = submissions.filter((item) => item.id !== left.id && item.datasetId === left.datasetId);
    let right = rightCandidates.find((item) => item.id === requestedRight) || null;
    const datasets = data.datasets || [];

    async function render() {
      const [leftHistory, rightHistory, leftRollout, rightRollout] = await Promise.all([
        window.FluidBenchData.loadTrainingHistory(left.id, left.datasetId, left.artifacts?.trainingHistory),
        right ? window.FluidBenchData.loadTrainingHistory(right.id, right.datasetId, right.artifacts?.trainingHistory) : Promise.resolve(null),
        window.FluidBenchData.loadRolloutAnalysis(left.id, left.datasetId, left.artifacts?.rolloutAnalysis),
        right ? window.FluidBenchData.loadRolloutAnalysis(right.id, right.datasetId, right.artifacts?.rolloutAnalysis) : Promise.resolve(null),
      ]);
      left.trainingHistory = leftHistory;
      left.rolloutAnalysis = leftRollout;
      if (right) {
        right.trainingHistory = rightHistory;
        right.rolloutAnalysis = rightRollout;
      }
      const compared = right ? [left, right] : [left];
      const allTrainingLosses = compared.flatMap((item) => (item.trainingHistory?.points || []).flatMap((point) => [point.trainingLoss, point.validationLoss]));
      const positiveLosses = allTrainingLosses.filter((value) => Number(value) > 0).map(Number);
      const minLoss = positiveLosses.length ? Math.min(...positiveLosses) : 1e-8;
      const maxLoss = positiveLosses.length ? Math.max(...positiveLosses) : minLoss * 10;
      const logScale = fittedLogRange(minLoss, maxLoss);
      leftPanel.className = "compare-panel compare-panel-left";
      rightPanel.className = `compare-panel compare-panel-right${right ? "" : " is-unselected"}`;
      leftPanel.innerHTML = renderPanel(left, right, datasets.find((dataset) => dataset.id === left.datasetId), "left", logScale, submissions, left.id);
      rightPanel.innerHTML = right
        ? renderPanel(right, left, datasets.find((dataset) => dataset.id === right.datasetId), "right", logScale, submissions, left.id)
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
