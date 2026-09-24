(() => {
  const leftPanel = document.querySelector("#compare-left");
  const rightPanel = document.querySelector("#compare-right");
  const params = new URLSearchParams(window.location.search);
  const requestedLeft = params.get("model");
  const requestedRight = params.get("compare");
  const lossMetrics = [
    { label: "RMSE", key: "rmse" },
    { label: "Rollout RMSE", key: "rolloutRmse" },
    { label: "MAE", key: "mae" },
    { label: "Rollout MAE", key: "rolloutMae" },
    { label: "Rel. L₂", key: "relativeL2" },
    { label: "Rollout Rel. L₂", key: "rolloutRelativeL2" },
  ];
  const rolloutMetrics = [
    { label: "Rollout RMSE", key: "rolloutRmse" },
    { label: "Rollout MAE", key: "rolloutMae" },
    { label: "Rollout Rel. L₂", key: "rolloutRelativeL2" },
  ];

  const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]);
  const formatLoss = (value) => Number(value).toExponential(2).replace("e-0", "e-").replace("e+0", "e+");
  const formatCount = (value) => new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);
  const formatFixed = (value, digits = 1) => Number(value).toFixed(digits);
  const comparisonTone = (value, otherValue) => {
    if (!Number.isFinite(Number(value)) || !Number.isFinite(Number(otherValue))) return "";
    if (Number(value) < Number(otherValue)) return "is-better";
    if (Number(value) > Number(otherValue)) return "is-worse";
    return "is-tied";
  };

  function modelPicker(submissions, leftId, selectedId = "") {
    const options = submissions.filter((submission) => submission.id !== leftId).map((submission) => `<option value="${escapeHtml(submission.id)}"${submission.id === selectedId ? " selected" : ""}>${escapeHtml(submission.model)}</option>`).join("");
    return `<label class="compare-model-select"><span>Compare against</span><select id="compare-model-select" aria-label="Choose a different model"><option value=""${selectedId ? "" : " selected"}>Choose a model…</option>${options}</select></label>`;
  }

  function chartY(value, logScale) {
    const clamped = Math.max(Number(value), 10 ** logScale.minExponent);
    const normalized = (Math.log10(clamped) - logScale.minExponent) / (logScale.maxExponent - logScale.minExponent);
    return 12 + (1 - normalized) * 164;
  }

  function chartPath(points, field, logScale) {
    const left = 55;
    const right = 12;
    const width = 440;
    const plotWidth = width - left - right;
    return points.map((point, index) => {
      const x = left + (point.epoch / 100) * plotWidth;
      const y = chartY(point[field], logScale);
      return `${index ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`;
    }).join(" ");
  }

  function chartMarkers(points, field, logScale, label) {
    const left = 55;
    const plotWidth = 373;
    const markerClass = field === "trainingLoss" ? "chart-point-training" : "chart-point-validation";
    return points.map((point) => {
      const x = left + (point.epoch / 100) * plotWidth;
      const y = chartY(point[field], logScale);
      const exactValue = String(Number(point[field]));
      const hoverInfo = `Epoch ${point.epoch} · ${label} loss: ${exactValue}`;
      return `<circle class="chart-point ${markerClass}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3.1" tabindex="0" role="img" aria-label="${escapeHtml(hoverInfo)}" data-hover-info="${escapeHtml(hoverInfo)}"></circle>`;
    }).join("");
  }

  function trainingChart(item, logScale) {
    const points = item.trainingHistory?.points || [];
    if (!points.length) return `<div class="compare-chart-empty">No training history provided.</div>`;
    const gridLines = Array.from({ length: logScale.maxExponent - logScale.minExponent + 1 }, (_, index) => logScale.maxExponent - index).map((exponent) => {
      const y = chartY(10 ** exponent, logScale);
      const value = 10 ** exponent;
      return `<g class="chart-gridline"><line x1="55" y1="${y}" x2="428" y2="${y}"/><text x="48" y="${y + 3}" text-anchor="end">${formatLoss(value)}</text></g>`;
    }).join("");
    const xTicks = [0, 25, 50, 75, 100].map((epoch) => {
      const x = 55 + (epoch / 100) * 373;
      return `<g class="chart-tick"><line x1="${x}" y1="176" x2="${x}" y2="180"/><text x="${x}" y="195" text-anchor="middle">${epoch}</text></g>`;
    }).join("");
    const label = `${item.model} ${item.trainingHistory.lossName || "RMSE"} training and validation loss by epoch`;
    return `<div class="training-chart-wrap"><svg class="training-chart" viewBox="0 0 440 210" role="img" aria-label="${escapeHtml(label)}" preserveAspectRatio="xMidYMid meet">
      ${gridLines}${xTicks}<line class="chart-axis" x1="55" y1="12" x2="55" y2="176"/><line class="chart-axis" x1="55" y1="176" x2="428" y2="176"/>
      <path class="chart-series chart-training-line" d="${chartPath(points, "trainingLoss", logScale)}"></path><path class="chart-series chart-validation-line" d="${chartPath(points, "validationLoss", logScale)}"></path>
      ${chartMarkers(points, "trainingLoss", logScale, "Training")}${chartMarkers(points, "validationLoss", logScale, "Validation")}
      <g class="chart-legend-in-graph"><rect x="277" y="17" width="147" height="32" rx="5"/><line class="chart-legend-training" x1="286" y1="28" x2="302" y2="28"/><text x="307" y="31">Training</text><line class="chart-legend-validation" x1="354" y1="28" x2="370" y2="28"/><text x="375" y="31">Validation</text><text class="chart-legend-loss-name" x="286" y="43">${escapeHtml(item.trainingHistory.lossName || "RMSE")} loss</text></g>
      <text class="chart-axis-label chart-y-label" transform="translate(13 95) rotate(-90)" text-anchor="middle">Loss · log scale</text>
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
      { label: "Training time", value: `${formatFixed(efficiency.trainingHours)} h`, metric: efficiency.trainingHours, other: other?.metrics.efficiency.trainingHours },
      { label: "Inference / sample", value: `${formatFixed(efficiency.inferenceMs)} ms`, metric: efficiency.inferenceMs, other: other?.metrics.efficiency.inferenceMs },
      { label: "GPU memory", value: `${formatFixed(efficiency.memoryGb)} GB`, metric: efficiency.memoryGb, other: other?.metrics.efficiency.memoryGb },
    ];
    return `<dl class="compare-facts">${facts.map((fact) => {
      const tone = other && fact.metric != null ? ` ${comparisonTone(fact.metric, fact.other)}` : "";
      return `<div class="compare-fact-item${tone}"><dt>${fact.label}</dt><dd>${fact.value}</dd></div>`;
    }).join("")}</dl>`;
  }

  function overallLosses(item, other) {
    const values = lossMetrics.map(({ label, key }) => {
      const tone = other ? ` ${comparisonTone(item.metrics[key], other.metrics[key])}` : "";
      return `<div class="compare-loss-item${tone}"><dt>${escapeHtml(label)}</dt><dd>${formatLoss(item.metrics[key])}</dd></div>`;
    }).join("");
    return `<dl class="compare-loss-grid">${values}</dl>`;
  }

  function rolloutTable(item, other) {
    const headings = ["step1", "step25", "step100"];
    const header = headings.map((step) => `<th scope="col">${step.replace("step", "Step ")}</th>`).join("");
    const rows = rolloutMetrics.map(({ label, key }) => `<tr><th scope="row">${escapeHtml(label)}</th>${headings.map((step) => {
      const value = item.metrics.rollout[step][key];
      const otherValue = other?.metrics.rollout[step][key];
      const tone = other ? ` ${comparisonTone(value, otherValue)}` : "";
      return `<td class="${tone.trim()}">${formatLoss(value)}</td>`;
    }).join("")}</tr>`).join("");
    return `<div class="compare-table-scroll"><table class="compare-loss-table"><thead><tr><th scope="col">Rollout loss</th>${header}</tr></thead><tbody>${rows}</tbody></table></div>`;
  }

  function variableLosses(item, other) {
    const { velocityRmse, pressureRmse } = item.metrics.perVariable;
    const otherVelocity = other?.metrics.perVariable.velocityRmse;
    const otherPressure = other?.metrics.perVariable.pressureRmse;
    const velocityTone = other ? ` ${comparisonTone(velocityRmse, otherVelocity)}` : "";
    const pressureTone = other ? ` ${comparisonTone(pressureRmse, otherPressure)}` : "";
    return `<div class="variable-loss-grid"><div class="${velocityTone.trim()}"><span>Velocity RMSE</span><strong>${formatLoss(velocityRmse)}</strong></div><div class="${pressureTone.trim()}"><span>Pressure RMSE</span><strong>${formatLoss(pressureRmse)}</strong></div></div>`;
  }

  function renderPanel(item, other, dataset, side, logScale, submissions, leftId) {
    const controls = side === "right" ? modelPicker(submissions, leftId, item.id) : `<span class="compare-fixed-label">Selected from leaderboard</span>`;
    return `<div class="compare-panel-top"><div class="compare-model-title"><span class="compare-side-label">${side === "left" ? "Model A" : "Model B"}</span><h2>${escapeHtml(item.model)}</h2><p>${escapeHtml(item.authors)} · ${item.year}</p></div>${controls}</div>
      <div class="compare-section"><h3>Submission details</h3>${modelFacts(item, other, dataset)}</div>
      <div class="compare-section"><div class="compare-section-title"><div><h3>Training loss</h3><p>Epoch versus ${escapeHtml(item.trainingHistory?.lossName || "RMSE")} loss</p></div><span class="chart-range">Epoch 1–100 · log scale</span></div>${trainingChart(item, logScale)}</div>
      <div class="compare-section"><div class="compare-section-title"><div><h3>Evaluation losses</h3><p>Lower is better</p></div></div>${overallLosses(item, other)}</div>
      <div class="compare-section"><div class="compare-section-title"><div><h3>Rollout losses</h3><p>Errors across prediction horizons</p></div></div>${rolloutTable(item, other)}</div>
      <div class="compare-section"><div class="compare-section-title"><div><h3>Per-variable losses</h3><p>Velocity and pressure fields</p></div></div>${variableLosses(item, other)}</div>`;
  }

  function emptyComparePanel(submissions, leftId) {
    return `<div class="compare-panel-top"><div class="compare-model-title"><span class="compare-side-label">Model B</span><h2>No model selected</h2><p>Choose a submission to compare against Model A.</p></div>${modelPicker(submissions, leftId)}</div><div class="compare-empty-state"><span class="compare-empty-icon" aria-hidden="true">↔</span><h3>Ready to compare</h3><p>Pick a different model above to see its training curve and losses alongside the selected submission.</p></div>`;
  }

  function updateUrl(left, right) {
    const url = new URL(window.location.href);
    url.searchParams.set("model", left.id);
    if (right) url.searchParams.set("compare", right.id);
    else url.searchParams.delete("compare");
    window.history.replaceState({}, "", url);
  }

  function bindChartTooltips(panel) {
    panel.querySelectorAll(".training-chart-wrap").forEach((wrap) => {
      const tooltip = document.createElement("div");
      tooltip.className = "chart-hover-tooltip";
      tooltip.setAttribute("role", "tooltip");
      tooltip.hidden = true;
      wrap.appendChild(tooltip);

      let timer = 0;
      let activePoint = null;
      const hide = () => {
        window.clearTimeout(timer);
        activePoint = null;
        tooltip.hidden = true;
      };
      const schedule = (point) => {
        window.clearTimeout(timer);
        activePoint = point;
        timer = window.setTimeout(() => {
          if (!activePoint || !wrap.contains(activePoint)) return;
          tooltip.textContent = activePoint.dataset.hoverInfo;
          tooltip.hidden = false;
          const wrapRect = wrap.getBoundingClientRect();
          const pointRect = activePoint.getBoundingClientRect();
          const pointX = pointRect.left + pointRect.width / 2 - wrapRect.left;
          const pointY = pointRect.top - wrapRect.top;
          const halfWidth = tooltip.offsetWidth / 2;
          const left = Math.min(Math.max(halfWidth + 4, pointX), wrap.clientWidth - halfWidth - 4);
          tooltip.style.left = `${left}px`;
          if (pointY > tooltip.offsetHeight + 8) {
            tooltip.style.top = `${pointY - 6}px`;
            tooltip.style.transform = "translate(-50%, -100%)";
          } else {
            tooltip.style.top = `${pointY + pointRect.height + 7}px`;
            tooltip.style.transform = "translateX(-50%)";
          }
        }, 100);
      };

      wrap.addEventListener("pointerover", (event) => {
        const point = event.target.closest?.(".chart-point");
        if (point && wrap.contains(point)) schedule(point);
      });
      wrap.addEventListener("pointerout", (event) => {
        const point = event.target.closest?.(".chart-point");
        const nextPoint = event.relatedTarget?.closest?.(".chart-point");
        if (point && point !== nextPoint) hide();
      });
      wrap.addEventListener("focusin", (event) => {
        const point = event.target.closest?.(".chart-point");
        if (point) schedule(point);
      });
      wrap.addEventListener("focusout", (event) => {
        if (event.target.closest?.(".chart-point")) hide();
      });
    });
  }

  window.FluidBenchData.load().then((data) => {
    const submissions = data.submissions || [];
    if (submissions.length < 2) throw new Error("Add at least two submissions to compare models.");
    const sortedByRmse = submissions.slice().sort((a, b) => a.metrics.rmse - b.metrics.rmse);
    const left = submissions.find((item) => item.id === requestedLeft) || sortedByRmse[0];
    const rightCandidates = submissions.filter((item) => item.id !== left.id);
    let right = rightCandidates.find((item) => item.id === requestedRight) || null;
    const datasets = data.datasets || [];

    async function render() {
      const [leftHistory, rightHistory] = await Promise.all([
        window.FluidBenchData.loadTrainingHistory(left.id, left.datasetId),
        right ? window.FluidBenchData.loadTrainingHistory(right.id, right.datasetId) : Promise.resolve(null),
      ]);
      left.trainingHistory = leftHistory;
      if (right) right.trainingHistory = rightHistory;
      const compared = right ? [left, right] : [left];
      const allTrainingLosses = compared.flatMap((item) => (item.trainingHistory?.points || []).flatMap((point) => [point.trainingLoss, point.validationLoss]));
      const positiveLosses = allTrainingLosses.filter((value) => Number(value) > 0).map(Number);
      const minLoss = positiveLosses.length ? Math.min(...positiveLosses) : 1e-8;
      const maxLoss = positiveLosses.length ? Math.max(...positiveLosses) : minLoss * 10;
      const logScale = { minExponent: Math.floor(Math.log10(minLoss)), maxExponent: Math.ceil(Math.log10(maxLoss)) };
      if (logScale.minExponent === logScale.maxExponent) logScale.maxExponent += 1;
      leftPanel.className = "compare-panel compare-panel-left";
      rightPanel.className = `compare-panel compare-panel-right${right ? "" : " is-unselected"}`;
      leftPanel.innerHTML = renderPanel(left, right, datasets.find((dataset) => dataset.id === left.datasetId), "left", logScale, submissions, left.id);
      rightPanel.innerHTML = right
        ? renderPanel(right, left, datasets.find((dataset) => dataset.id === right.datasetId), "right", logScale, submissions, left.id)
        : emptyComparePanel(submissions, left.id);
      bindChartTooltips(leftPanel);
      bindChartTooltips(rightPanel);
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
