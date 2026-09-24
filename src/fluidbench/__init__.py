"""Public FluidBench API."""

from pathlib import Path

from .downloader import download_dataset


def download(dataset_id: str) -> Path:
    """Download a benchmark's dataset using its benchmark JSON configuration."""
    return download_dataset(dataset_id)


__all__ = ["download"]
