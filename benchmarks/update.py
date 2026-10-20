"""Regenerate benchmarks.json from the benchmark JSON files in this folder, plus bundle.json with every benchmark
record so the site can load them all in a single request."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "benchmarks.json"
OPTIONS = ROOT / "options.json"
BUNDLE = ROOT / "bundle.json"
VALID_ID = re.compile(r"[a-z0-9_-]+\Z", re.IGNORECASE)


def main():
    benchmark_ids = []
    for record in ROOT.glob("*.json"):
        if record in (MANIFEST, OPTIONS, BUNDLE) or not record.is_file():
            continue
        if not VALID_ID.fullmatch(record.stem):
            raise SystemExit(f"Invalid benchmark ID in filename: {record.name}")
        benchmark_ids.append(record.stem)

    benchmark_ids.sort(key=str.casefold)
    MANIFEST.write_text(
        json.dumps({"benchmarkIds": benchmark_ids}, indent=2) + "\n",
        encoding="utf-8",
    )
    records = [json.loads((ROOT / f"{benchmark_id}.json").read_text(encoding="utf-8")) for benchmark_id in benchmark_ids]
    BUNDLE.write_text(json.dumps({"benchmarks": records}, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Updated {MANIFEST.name} and {BUNDLE.name}: {len(benchmark_ids)} benchmarks")


if __name__ == "__main__":
    main()
