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
    if not isinstance(representation, str) or representation not in {"Graph", "Matrix"}:
        raise ValueError("Config field 'representation' must be 'Graph' or 'Matrix'.")

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


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _evaluate_test_split(model: Any, loader: DataLoader, variables: list[str]) -> dict[str, Any]:
    device = _model_device(model)
    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    old_training_state = getattr(model, "training", None)
    if hasattr(model, "eval"):
        model.eval()

    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    target_square_sum = 0.0
    element_count = 0
    sample_count = 0
    channel_square_sums: np.ndarray | None = None
    channel_element_counts: np.ndarray | None = None
    inference_seconds = 0.0
    try:
        with torch.inference_mode():
            for inputs, targets in loader:
                inputs = _move_to_device(inputs, device)
                targets = _move_to_device(targets, device)
                if use_cuda:
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                predictions = model(inputs)
                if use_cuda:
                    torch.cuda.synchronize(device)
                inference_seconds += time.perf_counter() - start

                if not isinstance(predictions, torch.Tensor):
                    raise ValueError("model(inputs) must return a tensor for test evaluation.")
                if predictions.shape != targets.shape:
                    raise ValueError(
                        "Test predictions and targets must have matching shapes; "
                        f"got {tuple(predictions.shape)} and {tuple(targets.shape)}."
                    )
                if not torch.isfinite(predictions).all():
                    raise ValueError("model produced non-finite predictions on the test split.")
                if not torch.isfinite(targets).all():
                    raise ValueError("Benchmark test targets contain non-finite values.")

                errors = predictions.detach().to(device="cpu", dtype=torch.float64) - targets.detach().to(
                    device="cpu", dtype=torch.float64
                )
                targets_cpu = targets.detach().to(device="cpu", dtype=torch.float64)
                squared_error_sum += torch.square(errors).sum().item()
                absolute_error_sum += torch.abs(errors).sum().item()
                target_square_sum += torch.square(targets_cpu).sum().item()
                element_count += errors.numel()
                sample_count += int(errors.shape[0])

                if errors.ndim >= 2 and variables:
                    channel_axis = 1 if errors.shape[1] == len(variables) else -1
                    if errors.shape[channel_axis] == len(variables):
                        channel_errors = errors.movedim(channel_axis, 1)
                        reduce_dims = tuple(axis for axis in range(channel_errors.ndim) if axis != 1)
                        batch_square_sums = torch.square(channel_errors).sum(dim=reduce_dims).numpy()
                        batch_element_counts = np.full(
                            len(variables),
                            int(np.prod([size for axis, size in enumerate(channel_errors.shape) if axis != 1])),
                            dtype=np.int64,
                        )
                        if channel_square_sums is None:
                            channel_square_sums = np.zeros(len(variables), dtype=np.float64)
                            channel_element_counts = np.zeros(len(variables), dtype=np.int64)
                        channel_square_sums += batch_square_sums
                        channel_element_counts += batch_element_counts
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
    return {
        **one_step,
        "rollout": {
            "step1": {
                **one_step,
                "rolloutRmse": None,
                "rolloutMae": None,
                "rolloutRelativeL2": None,
            },
            "step25": {key: None for key in _ROLLOUT_METRIC_KEYS},
            "step100": {key: None for key in _ROLLOUT_METRIC_KEYS},
        },
        "efficiency": {
            "inferenceMs": inference_ms,
            "memoryGb": memory_gb,
        },
        "perVariable": {
            "velocityRmse": variable_rmse({"ux", "uy", "uz", "u", "v", "w", "velocity"}),
            "pressureRmse": variable_rmse({"p", "pressure"}),
        },
        "rolloutRmse": None,
        "rolloutMae": None,
        "rolloutRelativeL2": None,
    }


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

    start_time = time.perf_counter()
    history = model.fit(train_loader, validation_loader)
    training_hours = (time.perf_counter() - start_time) / 3600
    submission_history = _history_for_submission(history)
    metrics = _evaluate_test_split(model, test_loader, _benchmark_variables(benchmark_id))
    metrics["efficiency"]["parameters"] = int(model.total_parameters)
    metrics["efficiency"]["trainingHours"] = training_hours

    submission = deepcopy(config)
    submission.pop("metrics", None)
    submission["datasetId"] = benchmark_id
    submission["submittedAt"] = date.today().isoformat()
    submission["metrics"] = metrics

    root = _submission_root() / benchmark_id
    history_path = root / config["id"] / "training-history.json"
    submission_path = root / f"{config['id']}.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with submission_path.open("w", encoding="utf-8") as submission_file:
        json.dump(submission, submission_file, indent=2, allow_nan=False)
        submission_file.write("\n")
    with history_path.open("w", encoding="utf-8") as history_file:
        json.dump(submission_history, history_file, indent=2, allow_nan=False)
        history_file.write("\n")

    print(f"Submission written to {submission_path}")
    print(f"Training history written to {history_path}")
    return history
