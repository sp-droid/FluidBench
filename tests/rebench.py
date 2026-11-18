"""Re-run fluidbench.run on published submissions whose weights live in tests/weights (cheap models only).

    cd tests && uv run python rebench.py            # every entry in JOBS
    cd tests && uv run python rebench.py cylinder_interp/unet-classic

Each published record keeps its config, submission date and training-history.json; only metrics and
rollout-analysis.json are regenerated. Prints old vs new rollout RMSE as a sanity check.
"""

from __future__ import annotations

import json
import runpy
import shutil
import sys
from pathlib import Path

import h5py
import torch

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
sys.path.insert(0, str(TESTS))

import fluidbench  # noqa: E402
from models.minimal_models import BaselineSame, BaselineZero  # noqa: E402
from models.mlp import FieldMLP  # noqa: E402
from models.unet import UNetClassic  # noqa: E402


def field_mlp(benchmark: str):
    with h5py.File(TESTS / "downloads" / benchmark / "test" / "0.h5", "r") as file:
        return FieldMLP(file["fields"].shape[1:])


# (benchmark, submission id, model factory, weights file in tests/weights/{benchmark}/)
JOBS = [
    ("cylinder_Re41_interp", "baseline-same", lambda b: BaselineSame(), "BaselineSame.pt"),
    ("cylinder_Re41_interp", "baseline-zero", lambda b: BaselineZero(), "BaselineZero.pt"),
    ("cylinder_Re41_interp", "unet-classic", lambda b: UNetClassic(), "UNetClassic.pt"),
    ("cylinder_Re41_raw", "baseline-same", lambda b: BaselineSame(), "BaselineSame.pt"),
    ("cylinder_Re41_raw", "baseline-zero", lambda b: BaselineZero(), "BaselineZero.pt"),
    ("cylinder_Re41_raw", "mlp", field_mlp, "FieldMLP.pt"),
    ("cylinder_interp", "baseline-same", lambda b: BaselineSame(), "BaselineSame.pt"),
    ("cylinder_interp", "unet-classic", lambda b: UNetClassic(), "UNetClassic.pt"),
]
CONFIG_KEYS = ("id", "datasetId", "model", "authors", "year", "description", "representation")


def rebench(benchmark: str, submission_id: str, build, weights: str) -> None:
    published = ROOT / "submissions" / benchmark / submission_id
    old = json.loads((published / "submission.json").read_text(encoding="utf-8"))
    config = {key: old[key] for key in CONFIG_KEYS if key in old}

    model = build(benchmark)
    model.load_weights(TESTS / "weights" / benchmark / weights)
    model.to(model.device)
    torch.manual_seed(0)
    new = fluidbench.run(model, config)

    output = TESTS / "submission" / benchmark / submission_id
    new["submittedAt"] = old.get("submittedAt", new["submittedAt"])
    history = published / "training-history.json"
    if history.exists():
        shutil.copy2(history, output / "training-history.json")
        new["artifacts"]["trainingHistory"] = "training-history.json"
    (output / "submission.json").write_text(json.dumps(new, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    shutil.copytree(output, published, dirs_exist_ok=True)
    print(f"{benchmark}/{submission_id}: rolloutRmse {old['metrics'].get('rolloutRmse'):.5g} -> {new['metrics']['rolloutRmse']:.5g}")


if __name__ == "__main__":
    import os

    os.chdir(TESTS)  # fluidbench.run reads ./downloads and writes ./submission
    selected = set(sys.argv[1:])
    for benchmark, submission_id, build, weights in JOBS:
        if not selected or f"{benchmark}/{submission_id}" in selected:
            rebench(benchmark, submission_id, build, weights)
    runpy.run_path(str(ROOT / "submissions" / "update.py"), run_name="__main__")
