"""Download train, validation, and test splits from Hugging Face."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from huggingface_hub import snapshot_download

_SPLITS = ("train", "validation", "test")


def _hub_dataset_id(source: str) -> str:
    parsed = urlparse(source)
    if parsed.netloc.lower() not in {"huggingface.co", "www.huggingface.co"}:
        raise ValueError(f"Expected a Hugging Face dataset URL, got {source!r}")

    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 3 or parts[0] != "datasets":
        raise ValueError(f"Could not extract a dataset repo ID from {source!r}")
    return "/".join(parts[1:3])


def _split_complete(split_dir: Path) -> bool:
    """A split folder holds constants.h5 plus at least one case file."""
    return (split_dir / "constants.h5").is_file() and any(
        path.name != "constants.h5" for path in split_dir.glob("*.h5")
    )


def download_dataset(source: str, dataset_id: str, dataset_dir: Path) -> Path:
    """Download only the three split folders into the dataset's local folder."""
    hub_dataset_id = _hub_dataset_id(source)
    if all(_split_complete(dataset_dir / split) for split in _SPLITS):
        return dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=hub_dataset_id,
        repo_type="dataset",
        allow_patterns=[f"{split}/*.h5" for split in _SPLITS],
        local_dir=dataset_dir,
    )

    missing_splits = [
        split for split in _SPLITS if not _split_complete(dataset_dir / split)
    ]
    if missing_splits:
        raise FileNotFoundError(
            f"Dataset {hub_dataset_id} is missing expected split folders: "
            f"{', '.join(missing_splits)}"
        )
    return dataset_dir
