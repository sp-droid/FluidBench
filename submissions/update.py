"""Regenerate submissions.json from benchmark folders and their submission files."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "submissions.json"
VALID_ID = re.compile(r"[a-z0-9_-]+\Z", re.IGNORECASE)


def main():
    submissions_by_benchmark = {}
    benchmark_dirs = sorted(
        (path for path in ROOT.iterdir() if path.is_dir() and not path.name.startswith((".", "_"))),
        key=lambda path: path.name.casefold(),
    )

    for benchmark_dir in benchmark_dirs:
        benchmark_id = benchmark_dir.name
        if not VALID_ID.fullmatch(benchmark_id):
            raise SystemExit(f"Invalid benchmark ID in folder name: {benchmark_id}")

        submission_ids = []
        for record in benchmark_dir.glob("*.json"):
            if not record.is_file():
                continue
            if not VALID_ID.fullmatch(record.stem):
                raise SystemExit(f"Invalid submission ID in filename: {record}")
            submission_ids.append(record.stem)
        submissions_by_benchmark[benchmark_id] = sorted(submission_ids, key=str.casefold)

    MANIFEST.write_text(
        json.dumps({"submissionsByBenchmark": submissions_by_benchmark}, indent=2) + "\n",
        encoding="utf-8",
    )
    count = sum(map(len, submissions_by_benchmark.values()))
    print(f"Updated {MANIFEST.name}: {count} submissions across {len(submissions_by_benchmark)} benchmark folders")


if __name__ == "__main__":
    main()
