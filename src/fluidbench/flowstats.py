"""Statistical, physical and modal rollout metrics (beyond snapshot-to-snapshot error).

Every function takes one test trajectory as numpy arrays shaped [time, channel, ...spatial] (grid or mesh
cells alike), prediction first. Per-step statistics are computed per channel over space and averaged over
channels, so variables with different units weigh the same. Values that are undefined (e.g. the correlation
of a constant prediction) are None.

Metric definitions follow L. Lacasa, "Beyond Snapshot Matching" (2026): VRMSE (1.3), Pearson and anomaly
correlation (2.1, 2.2), the Murphy MSE decomposition (2.3), the energy ratio (0.3), one-point Wasserstein
distance and moments (6.1, 6.2), POD modal energy and subspaces (10.1) and DMD frequencies (10.2).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

_EPS = 1e-12
HIGH_CORRELATION = 0.8  # threshold of the high-correlation time
STEP_KEYS = (
    "vrmse",
    "correlation",
    "anomalyCorrelation",
    "murphyBias",
    "murphyAmplitude",
    "murphyPhase",
    "logEnergyRatio",
    "wasserstein",
)
_MAX_MODES = 10
_POD_ENERGY = 0.99  # modes kept: enough to hold this fraction of the true fluctuation energy
_DMD_MODES = 6
_STEADY_FLUCTUATION = 1e-4  # settled fluctuation energy / mean-field energy below this = steady flow, no modes
_VELOCITY_ALIASES = {
    "u", "v", "w", "ux", "uy", "uz", "vx", "vy", "vz",
    "velocity", "velocityx", "velocityy", "velocityz",
}


def _velocity_channels(variables: Sequence[str], channel_count: int) -> list[int]:
    selected = [
        index for index, name in enumerate(variables)
        if name.casefold().replace("_", "").replace("-", "").replace(" ", "") in _VELOCITY_ALIASES
    ]
    return selected or list(range(channel_count))


def _nanmean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else None


def _channel_mean(per_channel: np.ndarray) -> list[float | None]:
    """[time, channel] -> per time the mean over channels with a defined value."""
    return [_nanmean(row) for row in per_channel]


def step_statistics(
    prediction: np.ndarray, target: np.ndarray, variables: Sequence[str]
) -> dict[str, list[float | None]]:
    """Per-snapshot statistics of one trajectory (keys in STEP_KEYS)."""
    steps, channels = target.shape[:2]
    pred = prediction.reshape(steps, channels, -1)
    true = target.reshape(steps, channels, -1)

    mu_p, mu_t = pred.mean(axis=2), true.mean(axis=2)
    sd_p, sd_t = pred.std(axis=2), true.std(axis=2)
    var_t = np.where(sd_t > _EPS, sd_t**2, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        covariance = ((pred - mu_p[..., None]) * (true - mu_t[..., None])).mean(axis=2)
        rho = np.where((sd_p > _EPS) & (sd_t > _EPS), covariance / (sd_p * sd_t), np.nan)
        mse = np.square(pred - true).mean(axis=2)
        vrmse = np.sqrt(mse / var_t)

        # Anomalies from the true trajectory's time-mean field (climatology of this case).
        climatology = true.mean(axis=0, keepdims=True)
        anomaly_p, anomaly_t = pred - climatology, true - climatology
        norm = np.sqrt(np.square(anomaly_p).mean(axis=2) * np.square(anomaly_t).mean(axis=2))
        acc = np.where(norm > _EPS**2, (anomaly_p * anomaly_t).mean(axis=2) / norm, np.nan)

        # Murphy (1988): MSE = bias^2 + (sd_p - sd_t)^2 + 2 sd_p sd_t (1 - rho), in units of the true variance.
        bias = np.square(mu_p - mu_t) / var_t
        amplitude = np.square(sd_p - sd_t) / var_t
        phase = np.where(np.isfinite(rho), 2 * sd_p * sd_t * (1 - np.nan_to_num(rho)), 2 * sd_p * sd_t) / var_t

        # One-point law: W1 between the pooled values of each channel, in units of the true std.
        wasserstein = np.abs(np.sort(pred, axis=2) - np.sort(true, axis=2)).mean(axis=2) / np.sqrt(var_t)

    velocity = _velocity_channels(variables, channels)
    energy_p = 0.5 * np.square(pred[:, velocity]).sum(axis=1).mean(axis=1)
    energy_t = 0.5 * np.square(true[:, velocity]).sum(axis=1).mean(axis=1)
    log_energy = [
        float(math.log(p / t)) if p > _EPS and t > _EPS else None
        for p, t in zip(energy_p, energy_t)
    ]
    return {
        "vrmse": _channel_mean(vrmse),
        "correlation": _channel_mean(rho),
        "anomalyCorrelation": _channel_mean(acc),
        "murphyBias": _channel_mean(bias),
        "murphyAmplitude": _channel_mean(amplitude),
        "murphyPhase": _channel_mean(phase),
        "logEnergyRatio": log_energy,
        "wasserstein": _channel_mean(wasserstein),
    }


def _moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Skewness and kurtosis over the last axis (NaN where the field is constant)."""
    centred = values - values.mean(axis=-1, keepdims=True)
    sd = centred.std(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = centred / np.where(sd > _EPS, sd, np.nan)[..., None]
        return (scaled**3).mean(axis=-1), (scaled**4).mean(axis=-1)


def moment_errors(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | None]:
    """Mean |skewness| and |kurtosis| errors of the one-point distributions, over steps and channels."""
    steps, channels = target.shape[:2]
    skew_p, kurt_p = _moments(prediction.reshape(steps, channels, -1))
    skew_t, kurt_t = _moments(target.reshape(steps, channels, -1))
    return {
        "skewnessError": _nanmean(np.abs(skew_p - skew_t)),
        "kurtosisError": _nanmean(np.abs(kurt_p - kurt_t)),
    }


def _mean(values: Sequence[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def summarize(series: dict[str, list[float | None]], mask: np.ndarray | None = None) -> dict[str, float | None]:
    """Time means of the per-step statistics (optionally over a step mask), in lower-is-better form too."""
    def pick(key: str) -> list[float | None]:
        values = series[key]
        return values if mask is None else [value for value, keep in zip(values, mask) if keep]

    correlation = _mean(pick("correlation"))
    acc = _mean(pick("anomalyCorrelation"))
    log_energy = [abs(value) for value in pick("logEnergyRatio") if value is not None]
    return {
        "vrmse": _mean(pick("vrmse")),
        "correlation": correlation,
        "decorrelation": None if correlation is None else 1 - correlation,
        "anomalyCorrelation": acc,
        "anomalyDecorrelation": None if acc is None else 1 - acc,
        "murphyBias": _mean(pick("murphyBias")),
        "murphyAmplitude": _mean(pick("murphyAmplitude")),
        "murphyPhase": _mean(pick("murphyPhase")),
        "energyError": _mean(log_energy),
        "wasserstein": _mean(pick("wasserstein")),
    }


def high_correlation_time(correlation: Sequence[float | None], times: Sequence[float]) -> float:
    """First time the correlation drops below HIGH_CORRELATION (the last time if it never does)."""
    for value, time in zip(correlation, times):
        if value is None or value < HIGH_CORRELATION:
            return float(time)
    return float(times[-1])


def _dominant_mode(coefficients: np.ndarray, dt: float) -> dict[str, float] | None:
    """Exact DMD on [time, mode] coefficients: frequency and growth rate of the strongest oscillating mode."""
    if len(coefficients) < 4:
        return None
    past, future = coefficients[:-1].T, coefficients[1:].T
    operator = future @ np.linalg.pinv(past)
    eigenvalues, eigenvectors = np.linalg.eig(operator)
    amplitudes = np.abs(np.linalg.pinv(eigenvectors) @ coefficients[0])
    oscillating = np.flatnonzero(np.angle(eigenvalues) > 1e-6)
    if not len(oscillating):
        return None
    best = oscillating[np.argmax(amplitudes[oscillating])]
    eigenvalue = complex(eigenvalues[best])
    return {
        "frequency": float(abs(np.angle(eigenvalue)) / (2 * math.pi * dt)),
        "growthRate": float(math.log(max(abs(eigenvalue), _EPS)) / dt),
    }


def modal_analysis(prediction: np.ndarray, target: np.ndarray, dt: float) -> dict[str, Any] | None:
    """POD of the true (settled) trajectory and DMD on its leading coefficients; None for a steady flow.

    The prediction is projected on the true POD basis, around the true mean field, so mean-field errors show
    up as uncaptured energy. Returns per-mode energies (for plotting) and scalars.
    """
    steps = len(target)
    if steps < 4:
        return None
    true = target.reshape(steps, -1)
    pred = prediction.reshape(steps, -1)
    mean = true.mean(axis=0)
    fluctuation = true - mean
    if np.square(fluctuation).mean() < _STEADY_FLUCTUATION * max(np.square(mean).mean(), _EPS):
        return None

    _, singular, modes = np.linalg.svd(fluctuation, full_matrices=False)
    energy = singular**2 / steps
    cumulative = np.cumsum(energy) / energy.sum()
    count = int(min(_MAX_MODES, steps - 1, np.searchsorted(cumulative, _POD_ENERGY) + 1))
    basis = modes[:count]  # [mode, space]

    pred_fluctuation = pred - mean
    true_coefficients = fluctuation @ basis.T
    pred_coefficients = pred_fluctuation @ basis.T
    true_energy = energy[:count]
    pred_energy = np.square(pred_coefficients).mean(axis=0)
    pred_total = float(np.square(pred_fluctuation).sum(axis=1).mean())

    with np.errstate(divide="ignore", invalid="ignore"):
        log_errors = np.abs(np.log10(pred_energy / true_energy))
    energy_error = _nanmean(np.where(np.isfinite(log_errors), log_errors, np.nan))

    # Largest principal angle between the leading true and predicted POD subspaces.
    own_fluctuation = pred - pred.mean(axis=0)
    if np.square(own_fluctuation).mean() <= _EPS * max(np.square(fluctuation).mean(), _EPS):
        angle = 90.0  # a (nearly) frozen prediction has no fluctuation subspace at all
    else:
        _, _, pred_modes = np.linalg.svd(own_fluctuation, full_matrices=False)
        cosines = np.linalg.svd(basis @ pred_modes[:count].T, compute_uv=False)
        angle = float(np.degrees(np.arccos(np.clip(cosines.min(), -1.0, 1.0))))

    dmd_count = min(_DMD_MODES, count)
    true_mode = _dominant_mode(true_coefficients[:, :dmd_count], dt)
    # A prediction whose modal coefficients barely move has no dynamics to fit (DMD would fit noise).
    pred_varying = pred_coefficients[:, :dmd_count] - pred_coefficients[:, :dmd_count].mean(axis=0)
    true_varying = true_coefficients[:, :dmd_count] - true_coefficients[:, :dmd_count].mean(axis=0)
    frozen = np.square(pred_varying).sum() <= 1e-6 * np.square(true_varying).sum()
    pred_mode = None if frozen else _dominant_mode(pred_coefficients[:, :dmd_count], dt)
    frequency_error = growth_error = None
    if true_mode and true_mode["frequency"] > _EPS:
        if pred_mode is None:
            frequency_error = 1.0  # the reference oscillates, the prediction does not: 100 % error
        else:
            frequency_error = abs(pred_mode["frequency"] - true_mode["frequency"]) / true_mode["frequency"]
            # Growth-rate error per reference period (dimensionless).
            growth_error = abs(pred_mode["growthRate"] - true_mode["growthRate"]) / true_mode["frequency"]

    return {
        "modeCount": count,
        "modeEnergy": {"target": true_energy.tolist(), "prediction": pred_energy.tolist()},
        "dominantMode": {"target": true_mode, "prediction": pred_mode},
        "scalars": {
            "podEnergyError": energy_error,
            "uncapturedEnergy": (
                None if pred_total <= _EPS else max(0.0, 1 - float(pred_energy.sum()) / pred_total)
            ),
            "subspaceAngle": angle,
            "frequencyError": frequency_error,
            "growthRateError": growth_error,
        },
    }


def first_nonfinite_step(prediction: np.ndarray) -> int:
    """Number of leading snapshots that are all finite (the rollout survival length)."""
    finite = np.isfinite(prediction.reshape(len(prediction), -1)).all(axis=1)
    bad = np.flatnonzero(~finite)
    return int(bad[0]) if len(bad) else len(prediction)
