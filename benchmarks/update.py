"""Regenerate benchmarks.json from the benchmark JSON files in this folder."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "benchmarks.json"
VALID_ID = re.compile(r"[a-z0-9_-]+\Z", re.IGNORECASE)


def main():
    benchmark_ids = []
    for record in ROOT.glob("*.json"):
        if record == MANIFEST or not record.is_file():
            continue
        if not VALID_ID.fullmatch(record.stem):
            raise SystemExit(f"Invalid benchmark ID in filename: {record.name}")
        benchmark_ids.append(record.stem)

    benchmark_ids.sort(key=str.casefold)
    MANIFEST.write_text(
        json.dumps({"benchmarkIds": benchmark_ids}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Updated {MANIFEST.name}: {len(benchmark_ids)} benchmarks")


if __name__ == "__main__":
    main()
