"""Benchmark dataset loaders selected from benchmark metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .huggingface import download_dataset as _download_huggingface

_BENCHMARKS_DIR = Path(__file__).resolve().parents[3] / "benchmarks"
_LOADERS: dict[str, Callable[[str, str, Path], Path]] = {
    "huggingface": _download_huggingface,
}


def load_dataset(dataset_id: str) -> Path:
    """Read benchmark metadata and load its data into ./<dataset_id>."""
    benchmark_json = _BENCHMARKS_DIR / f"{dataset_id}.json"
    if not benchmark_json.is_file():
        raise FileNotFoundError(f"Benchmark metadata not found: {benchmark_json}")

    with benchmark_json.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    source = metadata.get("source")
    loader_name = metadata.get("loader")
    if not isinstance(source, str):
        raise ValueError(f"Benchmark metadata has no string 'source': {benchmark_json}")
    if not isinstance(loader_name, str):
        raise ValueError(f"Benchmark metadata has no string 'loader': {benchmark_json}")

    try:
        loader = _LOADERS[loader_name.lower()]
    except KeyError:
        supported = ", ".join(sorted(_LOADERS))
        raise ValueError(
            f"Unsupported dataset loader {loader_name!r}. Supported loaders: {supported}"
        ) from None
    return loader(source, dataset_id, Path.cwd() / dataset_id)
