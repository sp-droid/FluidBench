"""Public FluidBench API."""

from pathlib import Path

from .loader import load_dataset


def load(dataset_id: str) -> Path:
    """Load a benchmark's dataset using its benchmark JSON configuration."""
    return load_dataset(dataset_id)


__all__ = ["load"]
