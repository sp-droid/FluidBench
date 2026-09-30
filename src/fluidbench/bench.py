"""Training and submission workflow for FluidBench benchmarks."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from .cfdataset import CFDataset
from .downloader import download_dataset

_BATCH_SIZE = 8
_ID_PATTERN = re.compile(r"^[a-z0-9_-]+$", re.IGNORECASE)
_ROLLOUT_METRIC_KEYS = (
    "rmse",
    "rolloutRmse",
    "mae",
    "rolloutMae",
    "relativeL2",
    "rolloutRelativeL2",
)


def _validate_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise ValueError(f"{label} must contain only letters, numbers, '_' or '-'.")
    return value


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("Submission config must be a JSON object (mapping).")
    config = deepcopy(dict(config))

    for key in (
        "id",
        "datasetId",
        "model",
        "authors",
        "year",
        "representation",
    ):
        if key not in config:
            raise ValueError(f"Submission config is missing required field {key!r}.")

    _validate_id(config["id"], "Submission id")
    _validate_id(config["datasetId"], "Benchmark id")
    for key in ("model", "authors"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"Config field {key!r} must be a non-empty string.")
    if isinstance(config["year"], bool) or not isinstance(config["year"], int):
        raise ValueError("Config field 'year' must be an integer.")
    representation = config["representation"]
    if not isinstance(representation, str) or representation not in {"Graph", "Grid"}:
        raise ValueError("Config field 'representation' must be 'Graph' or 'Grid'.")

    return config


def _load_split(dataset_dir: Path, split: str) -> CFDataset:
    split_path = dataset_dir / f"{split}.h5"
    if not split_path.is_file():
        raise FileNotFoundError(f"Benchmark split file not found: {split_path}")

    try:
        with h5py.File(split_path, "r") as split_file:
            scalars = split_file["inputs/scalars"][:]
            fields = split_file["inputs/fields"][:]
            targets = split_file["targets/fields"][:]
    except KeyError as exc:
        raise ValueError(
            f"Benchmark split {split_path} must contain inputs/scalars, "
            "inputs/fields, and targets/fields."
        ) from exc
    except OSError as exc:
        raise ValueError(f"Could not read benchmark split file: {split_path}") from exc

    lengths = (len(scalars), len(fields), len(targets))
    if len(set(lengths)) != 1:
        raise ValueError(
            f"Benchmark split {split_path} has mismatched sample counts: "
            f"scalars={lengths[0]}, fields={lengths[1]}, targets={lengths[2]}."
        )
    if not targets.size:
        raise ValueError(f"Benchmark split {split_path} contains no samples.")

    return CFDataset(scalars, fields, targets, np.arange(len(targets)))


def _as_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _as_json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_as_json_value(item) for item in value]
    if hasattr(value, "item"):
        return _as_json_value(value.item())
    return value


def _benchmark_variables(benchmark_id: str) -> list[str]:
    metadata_path = (
        Path(__file__).resolve().parents[2] / "benchmarks" / f"{benchmark_id}.json"
    )
    try:
        with metadata_path.open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read benchmark metadata: {metadata_path}") from exc

    variables = metadata.get("flowVariables")
    if not isinstance(variables, list) or not all(isinstance(name, str) for name in variables):
        raise ValueError(
            "Benchmark metadata must define a 'flowVariables' string list: "
            f"{metadata_path}"
        )
    return variables


def _benchmark_scalar_names(benchmark_id: str) -> list[str]:
    metadata_path = (
        Path(__file__).resolve().parents[2] / "benchmarks" / f"{benchmark_id}.json"
    )
    try:
        with metadata_path.open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read benchmark metadata: {metadata_path}") from exc

    names = metadata.get("inputScalarNames", [])
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError(
            "Benchmark metadata must define an 'inputScalarNames' string list: "
            f"{metadata_path}"
        )
    return names


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _snapshot_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float | None]:
    error = prediction - target
    target_norm = torch.linalg.vector_norm(target)
    error_norm = torch.linalg.vector_norm(error)
    relative_l2 = (
        float(error_norm / target_norm)
        if target_norm.item() != 0
        else (0.0 if error_norm.item() == 0 else None)
    )
    return {
        "rmse": float(torch.square(error).mean().sqrt()),
        "mae": float(error.abs().mean()),
        "relativeL2": relative_l2,
    }


def _mean_metric(snapshots: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [snapshot[key] for snapshot in snapshots if snapshot.get(key) is not None]
    return float(np.mean(values)) if values else None


def radial_kinetic_energy_spectrum(
    fields: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute radial spectra for independent scalar snapshots shaped (..., H, W)."""
    fields = np.asarray(fields, dtype=np.float64)
    if fields.ndim < 2:
        raise ValueError("fields must have shape (..., height, width)")

    height, width = fields.shape[-2:]
    snapshots = fields.reshape(-1, height, width)
    if not len(snapshots):
        raise ValueError("fields must contain at least one snapshot")

    snapshots = snapshots - snapshots.mean(axis=(-2, -1), keepdims=True)
    velocity_hat = np.fft.fft2(snapshots, axes=(-2, -1))
    modal_energy = 0.5 * np.abs(velocity_hat) ** 2 / (height * width) ** 2

    ky = np.fft.fftfreq(height) * height
    kx = np.fft.fftfreq(width) * width
    radial_bins = np.floor(np.hypot(ky[:, None], kx[None, :]) + 0.5).astype(int)

    max_k = min(height, width) // 2
    wavenumbers = np.arange(1, max_k + 1)
    snapshot_spectra = np.zeros((len(snapshots), len(wavenumbers)), dtype=np.float64)
    for index, wavenumber in enumerate(wavenumbers):
        shell = radial_bins == wavenumber
        snapshot_spectra[:, index] = modal_energy[:, shell].sum(axis=1)

    return wavenumbers, snapshot_spectra.mean(axis=0), snapshot_spectra


def _channel_first_trajectory(
    fields: np.ndarray,
    variables: Sequence[str],
) -> np.ndarray | None:
    """Return [time, channel, height, width] fields when the layout is recognizable."""
    fields = np.asarray(fields, dtype=np.float64)
    if fields.ndim != 4:
        return None
    if fields.shape[1] == len(variables):
        return fields
    if fields.shape[-1] == len(variables):
        return np.moveaxis(fields, -1, 1)
    return None


def _trajectory_spectrum(
    fields: np.ndarray,
    variables: Sequence[str],
) -> dict[str, Any] | None:
    channel_first = _channel_first_trajectory(fields, variables)
    if channel_first is None:
        return None

    velocity_aliases = {
        "u", "v", "w", "ux", "uy", "uz", "vx", "vy", "vz",
        "velocity", "velocityx", "velocityy", "velocityz",
    }
    velocity_channels = [
        index for index, name in enumerate(variables)
        if (
            name.casefold().replace("_", "").replace("-", "").replace(" ", "")
            in velocity_aliases
        )
    ]
    if not velocity_channels:
        return None

    wavenumbers: np.ndarray | None = None
    component_spectra = []
    for channel in velocity_channels:
        current_wavenumbers, _, spectra = radial_kinetic_energy_spectrum(
            channel_first[:, channel, :, :]
        )
        if wavenumbers is None:
            wavenumbers = current_wavenumbers
        elif not np.array_equal(wavenumbers, current_wavenumbers):
            raise ValueError("Velocity components produced inconsistent wavenumber bins.")
        component_spectra.append(spectra)

    snapshot_spectra = np.sum(np.stack(component_spectra, axis=0), axis=0)
    return {
        "wavenumbers": wavenumbers.tolist() if wavenumbers is not None else [],
        "mean": snapshot_spectra.mean(axis=0).tolist(),
        "standardDeviation": snapshot_spectra.std(axis=0).tolist(),
        "snapshotCount": int(snapshot_spectra.shape[0]),
    }


def _test_case_conditions(scalars: Any, scalar_names: Sequence[str]) -> dict[str, Any]:
    values = np.asarray(scalars)
    if values.ndim > 1:
        values = values[0]
    values = values.reshape(-1)
    return {
        name: _as_json_value(values[index])
        for index, name in enumerate(scalar_names)
        if index < len(values)
    }


def _evaluate_test_split(
    model: Any,
    loader: DataLoader,
    rollout_loader: DataLoader,
    dataset: CFDataset,
    variables: list[str],
    scalar_names: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    device = _model_device(model)
    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    old_training_state = getattr(model, "training", None)
    if hasattr(model, "eval"):
        model.eval()

    try:
        predict = getattr(model, "predict", None)
        predict_rollout = getattr(model, "predict_rollout", None)
        if not callable(predict):
            raise ValueError("model must implement predict(test_dataloader).")
        if not callable(predict_rollout):
            raise ValueError("model must implement predict_rollout(test_dataloader).")

        if use_cuda:
            torch.cuda.synchronize(device)
        inference_start = time.perf_counter()
        with torch.inference_mode():
            predictions = predict(loader)
        if use_cuda:
            torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - inference_start

        if not isinstance(predictions, torch.Tensor):
            raise ValueError("model.predict(test_dataloader) must return a tensor.")
        predictions = predictions.detach().to(device="cpu", dtype=torch.float64)
        targets = torch.as_tensor(dataset.targets, dtype=torch.float64, device="cpu")
        if predictions.shape != targets.shape:
            raise ValueError(
                "Test predictions and targets must have matching shapes; "
                f"got {tuple(predictions.shape)} and {tuple(targets.shape)}."
            )
        if not torch.isfinite(predictions).all():
            raise ValueError("model.predict() returned non-finite test predictions.")
        if not torch.isfinite(targets).all():
            raise ValueError("Benchmark test targets contain non-finite values.")

        errors = predictions - targets
        squared_error_sum = torch.square(errors).sum().item()
        absolute_error_sum = torch.abs(errors).sum().item()
        target_square_sum = torch.square(targets).sum().item()
        element_count = errors.numel()
        sample_count = int(errors.shape[0])

        channel_square_sums: np.ndarray | None = None
        channel_element_counts: np.ndarray | None = None
        if errors.ndim >= 2 and variables:
            channel_axis = None
            if errors.ndim >= 5 and errors.shape[2] == len(variables):
                channel_axis = 2
            elif errors.shape[1] == len(variables):
                channel_axis = 1
            elif errors.shape[-1] == len(variables):
                channel_axis = errors.ndim - 1
            if channel_axis is not None:
                channel_errors = errors.movedim(channel_axis, 1)
                reduce_dims = tuple(axis for axis in range(channel_errors.ndim) if axis != 1)
                channel_square_sums = torch.square(channel_errors).sum(dim=reduce_dims).numpy()
                elements_per_channel = int(
                    np.prod([size for axis, size in enumerate(channel_errors.shape) if axis != 1])
                )
                channel_element_counts = np.full(
                    len(variables), elements_per_channel, dtype=np.int64
                )

        with torch.inference_mode():
            rollout_predictions = predict_rollout(rollout_loader)
    finally:
        if old_training_state is not None and hasattr(model, "train"):
            model.train(old_training_state)

    if not element_count or not sample_count:
        raise ValueError("Benchmark test split contains no evaluable predictions.")
    if target_square_sum == 0:
        relative_l2 = 0.0 if squared_error_sum == 0 else None
    else:
        relative_l2 = math.sqrt(squared_error_sum / target_square_sum)

    def variable_rmse(aliases: set[str]) -> float | None:
        if channel_square_sums is None or channel_element_counts is None:
            return None
        selected = [
            index
            for index, name in enumerate(variables)
            if name.casefold() in aliases
        ]
        if not selected:
            return None
        count = int(channel_element_counts[selected].sum())
        return math.sqrt(float(channel_square_sums[selected].sum()) / count) if count else None

    if isinstance(rollout_predictions, torch.Tensor):
        rollout_predictions = rollout_predictions.detach().to(device="cpu", dtype=torch.float64)
        if sample_count == 1 and rollout_predictions.shape == targets[0].shape:
            rollout_predictions = [rollout_predictions]
        elif rollout_predictions.ndim >= 2:
            rollout_predictions = list(rollout_predictions.unbind(dim=0))
        else:
            raise ValueError(
                "model.predict_rollout(test_dataloader) must return one trajectory per test case."
            )
    elif isinstance(rollout_predictions, np.ndarray):
        if sample_count == 1 and rollout_predictions.shape == tuple(targets[0].shape):
            rollout_predictions = [rollout_predictions]
        elif rollout_predictions.ndim >= 2:
            rollout_predictions = list(rollout_predictions)
        else:
            raise ValueError(
                "model.predict_rollout(test_dataloader) must return one trajectory per test case."
            )
    elif isinstance(rollout_predictions, Sequence) and not isinstance(
        rollout_predictions, (str, bytes, bytearray)
    ):
        rollout_predictions = list(rollout_predictions)
    else:
        raise ValueError(
            "model.predict_rollout(test_dataloader) must return one trajectory per test case."
        )

    if len(rollout_predictions) != sample_count:
        raise ValueError(
            "model.predict_rollout(test_dataloader) must return exactly one trajectory "
            f"for each of the {sample_count} test cases; got {len(rollout_predictions)}."
        )

    rollout_snapshot_metrics_by_case: list[list[dict[str, float | None]]] = []
    trajectory_records = []
    common_wavenumbers: list[int] | None = None
    for case_index, prediction in enumerate(rollout_predictions):
        if not isinstance(prediction, torch.Tensor):
            prediction = torch.as_tensor(prediction)
        prediction = prediction.detach().to(device="cpu", dtype=torch.float64)
        target = targets[case_index]
        if prediction.ndim == target.ndim + 1 and prediction.shape[0] == 1:
            prediction = prediction.squeeze(0)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Rollout trajectory {case_index + 1} and its test snapshots must match; "
                f"got {tuple(prediction.shape)} and {tuple(target.shape)}."
            )
        if target.ndim < 3 or target.shape[0] == 0:
            raise ValueError(
                f"Test case {case_index + 1} must contain a non-empty time sequence of snapshots."
            )
        if not torch.isfinite(prediction).all():
            raise ValueError(
                f"model.predict_rollout() returned non-finite values in test case {case_index + 1}."
            )

        snapshot_metrics = [
            _snapshot_metrics(prediction[time_index], target[time_index])
            for time_index in range(int(target.shape[0]))
        ]
        rollout_snapshot_metrics_by_case.append(snapshot_metrics)
        prediction_spectrum = _trajectory_spectrum(prediction.numpy(), variables)
        target_spectrum = _trajectory_spectrum(target.numpy(), variables)
        if prediction_spectrum is not None and target_spectrum is not None:
            if prediction_spectrum["wavenumbers"] != target_spectrum["wavenumbers"]:
                raise ValueError(
                    f"Prediction and reference spectra differ in test case {case_index + 1}."
                )
            if common_wavenumbers is None:
                common_wavenumbers = target_spectrum["wavenumbers"]
            elif common_wavenumbers != target_spectrum["wavenumbers"]:
                raise ValueError("Test cases produced inconsistent wavenumber bins.")

        trajectory_records.append(
            {
                "id": f"trajectory-{case_index + 1}",
                "label": f"Test case {case_index + 1}",
                "conditions": _test_case_conditions(dataset.scalars[case_index], scalar_names),
                "errors": {
                    key: [snapshot[key] for snapshot in snapshot_metrics]
                    for key in ("rmse", "mae", "relativeL2")
                },
                "kineticEnergySpectrum": (
                    {"target": target_spectrum, "prediction": prediction_spectrum}
                    if target_spectrum is not None and prediction_spectrum is not None
                    else None
                ),
            }
        )

    def rollout_metrics_at(horizon: int) -> dict[str, float | None]:
        point = [
            case_metrics[horizon - 1]
            for case_metrics in rollout_snapshot_metrics_by_case
            if len(case_metrics) >= horizon
        ]
        average = [
            snapshot
            for case_metrics in rollout_snapshot_metrics_by_case
            for snapshot in case_metrics[:horizon]
        ]
        return {
            "rmse": _mean_metric(point, "rmse"),
            "rolloutRmse": _mean_metric(average, "rmse"),
            "mae": _mean_metric(point, "mae"),
            "rolloutMae": _mean_metric(average, "mae"),
            "relativeL2": _mean_metric(point, "relativeL2"),
            "rolloutRelativeL2": _mean_metric(average, "relativeL2"),
        }

    inference_ms = inference_seconds * 1000 / sample_count
    if use_cuda:
        memory_gb = torch.cuda.max_memory_allocated(device) / 1_000_000_000
    else:
        model_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (*getattr(model, "parameters", lambda: [])(), *getattr(model, "buffers", lambda: [])())
        )
        memory_gb = model_bytes / 1_000_000_000

    one_step = {
        "rmse": math.sqrt(squared_error_sum / element_count),
        "mae": absolute_error_sum / element_count,
        "relativeL2": relative_l2,
    }
    all_rollout_snapshots = [
        snapshot
        for case_metrics in rollout_snapshot_metrics_by_case
        for snapshot in case_metrics
    ]
    rollout_average = {
        "rolloutRmse": _mean_metric(all_rollout_snapshots, "rmse"),
        "rolloutMae": _mean_metric(all_rollout_snapshots, "mae"),
        "rolloutRelativeL2": _mean_metric(all_rollout_snapshots, "relativeL2"),
    }
    rollout_analysis = {
        "schemaVersion": 1,
        "trajectoryCount": len(trajectory_records),
        "timeSteps": list(range(1, max(map(len, rollout_snapshot_metrics_by_case)) + 1)),
        "errorMetricIds": ["rmse", "mae", "relativeL2"],
        "wavenumbers": common_wavenumbers or [],
        "trajectories": trajectory_records,
    }
    metrics = {
        **one_step,
        "rollout": {
            "step1": rollout_metrics_at(1),
            "step25": rollout_metrics_at(25),
            "step100": rollout_metrics_at(100),
        },
        "efficiency": {
            "inferenceMs": inference_ms,
            "memoryGb": memory_gb,
        },
        "perVariable": {
            "velocityRmse": variable_rmse({"ux", "uy", "uz", "u", "v", "w", "velocity"}),
            "pressureRmse": variable_rmse({"p", "pressure"}),
        },
        "rolloutAnalysis": {
            "trajectoryCount": len(trajectory_records),
            "snapshotCountByTrajectory": [len(case_metrics) for case_metrics in rollout_snapshot_metrics_by_case],
            "errorMetricIds": ["rmse", "mae", "relativeL2"],
            "hasKineticEnergySpectrum": bool(common_wavenumbers),
        },
        **rollout_average,
    }
    return metrics, rollout_analysis


def _history_for_submission(history: Any) -> dict[str, list[Any]]:
    if not isinstance(history, Mapping):
        raise ValueError("model.fit() must return a mapping-like training history.")

    epochs = history.get("epochs", history.get("train_epochs"))
    training_losses = history.get(
        "trainingLosses",
        history.get("train_losses", history.get("training_losses")),
    )
    validation_losses = history.get(
        "validationLosses",
        history.get("validation_losses", history.get("val_losses")),
    )

    if training_losses is None:
        raise ValueError("model.fit() history must include training losses.")
    training_losses = list(training_losses)
    if epochs is None:
        epochs = list(range(len(training_losses)))
    else:
        epochs = list(epochs)
    if len(epochs) != len(training_losses):
        raise ValueError("model.fit() history epochs and training losses must align.")

    if validation_losses is None:
        validation_losses = [None] * len(epochs)
    else:
        validation_losses = list(validation_losses)
        if not validation_losses:
            validation_losses = [None] * len(epochs)
        elif len(validation_losses) != len(epochs):
            raise ValueError("model.fit() history validation losses must align with epochs.")

    return _as_json_value(
        {
            "epochs": epochs,
            "trainingLosses": training_losses,
            "validationLosses": validation_losses,
        }
    )


def _submission_root() -> Path:
    return Path.cwd() / "submission"


def run(model: Any, submission_config: Mapping[str, Any]) -> Any:
    """Train and evaluate ``model`` and write its submission record.

    Pass a decoded JSON object with submission metadata, including ``datasetId``.
    All supported metrics are calculated here from the benchmark test split;
    the input config does not need a ``metrics`` field. Artifacts are saved to
    ``./submission`` under the current working directory.
    """
    config = _validate_config(submission_config)
    benchmark_id = config["datasetId"]
    dataset_dir = download_dataset(benchmark_id)

    train_dataset = _load_split(dataset_dir, "train")
    validation_dataset = _load_split(dataset_dir, "validation")
    train_loader = DataLoader(train_dataset, batch_size=_BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=_BATCH_SIZE,
        shuffle=False,
    )
    test_dataset = _load_split(dataset_dir, "test")
    test_loader = DataLoader(test_dataset, batch_size=_BATCH_SIZE, shuffle=False)
    rollout_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    start_time = time.perf_counter()
    history = model.fit(train_loader, validation_loader)
    training_hours = (time.perf_counter() - start_time) / 3600
    submission_history = _history_for_submission(history)
    metrics, rollout_analysis = _evaluate_test_split(
        model,
        test_loader,
        rollout_loader,
        test_dataset,
        _benchmark_variables(benchmark_id),
        _benchmark_scalar_names(benchmark_id),
    )
    metrics["efficiency"]["parameters"] = int(model.total_parameters)
    metrics["efficiency"]["trainingHours"] = training_hours

    submission = deepcopy(config)
    submission.pop("metrics", None)
    submission.pop("artifacts", None)
    submission["datasetId"] = benchmark_id
    submission["submittedAt"] = date.today().isoformat()
    submission["metrics"] = metrics
    submission["artifacts"] = {
        "trainingHistory": "training-history.json",
        "rolloutAnalysis": "rollout-analysis.json",
    }

    root = _submission_root() / benchmark_id
    history_path = root / config["id"] / "training-history.json"
    rollout_analysis_path = root / config["id"] / "rollout-analysis.json"
    submission_path = root / f"{config['id']}.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with submission_path.open("w", encoding="utf-8") as submission_file:
        json.dump(submission, submission_file, indent=2, allow_nan=False)
        submission_file.write("\n")
    with history_path.open("w", encoding="utf-8") as history_file:
        json.dump(submission_history, history_file, indent=2, allow_nan=False)
        history_file.write("\n")
    with rollout_analysis_path.open("w", encoding="utf-8") as analysis_file:
        json.dump(rollout_analysis, analysis_file, indent=2, allow_nan=False)
        analysis_file.write("\n")

    print(f"submission written to 'submission/{benchmark_id}' folder")
    return history
