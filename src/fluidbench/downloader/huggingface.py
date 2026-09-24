"""Download train, validation, and test splits from Hugging Face."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from huggingface_hub import hf_hub_download


def _hub_dataset_id(source: str) -> str:
    parsed = urlparse(source)
    if parsed.netloc.lower() not in {"huggingface.co", "www.huggingface.co"}:
        raise ValueError(f"Expected a Hugging Face dataset URL, got {source!r}")

    parts = [part for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 3 or parts[0] != "datasets":
        raise ValueError(f"Could not extract a dataset repo ID from {source!r}")
    return "/".join(parts[1:3])


def download_dataset(source: str, dataset_id: str, dataset_dir: Path) -> Path:
    """Download only the three split files into the dataset's local folder."""
    hub_dataset_id = _hub_dataset_id(source)
    expected_files = {
        split: dataset_dir / f"{split}.h5"
        for split in ("train", "validation", "test")
    }
    missing_splits = [
        split for split, path in expected_files.items() if not path.is_file()
    ]
    if not missing_splits:
        return dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split in missing_splits:
        hf_hub_download(
            repo_id=hub_dataset_id,
            filename=f"{split}.h5",
            repo_type="dataset",
            local_dir=dataset_dir,
        )

    missing_files = [
        path.name for path in expected_files.values() if not path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Dataset {hub_dataset_id} is missing expected split files: "
            f"{', '.join(missing_files)}"
        )
    return dataset_dir
