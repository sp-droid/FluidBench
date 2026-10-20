"""Regenerate the submission manifests: one per benchmark folder, a top-level index of those folders, and a bundle
of every lightweight record so the site can load them all in a single request.

Layout:
    submissions/submissions.json                         {"benchmarkIds": [...]}    (written here: benchmark folders)
    submissions/bundle.json                              {"submissions": [...]}     (written here: all records)
    submissions/{benchmarkId}/submissions.json           {"submissionIds": [...]}   (written here)
    submissions/{benchmarkId}/{id}/submission.json       the submission record
    submissions/{benchmarkId}/{id}/*.json                its artifacts (rollout analysis, training history)
"""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST_NAME = "submissions.json"
BUNDLE_NAME = "bundle.json"
RECORD_NAME = "submission.json"
VALID_ID = re.compile(r"[a-z0-9_-]+\Z", re.IGNORECASE)


def main():
    benchmark_dirs = sorted(
        (path for path in ROOT.iterdir() if path.is_dir() and not path.name.startswith((".", "_"))),
        key=lambda path: path.name.casefold(),
    )

    benchmark_ids, bundle = [], []
    for benchmark_dir in benchmark_dirs:
        benchmark_id = benchmark_dir.name
        if not VALID_ID.fullmatch(benchmark_id):
            raise SystemExit(f"Invalid benchmark ID in folder name: {benchmark_id}")

        records = {}
        for record_path in benchmark_dir.glob(f"*/{RECORD_NAME}"):
            submission_id = record_path.parent.name
            if not VALID_ID.fullmatch(submission_id):
                raise SystemExit(f"Invalid submission ID in folder name: {record_path.parent}")
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("id") != submission_id or record.get("datasetId") != benchmark_id:
                raise SystemExit(
                    f"{record_path} must have id {submission_id!r} and datasetId {benchmark_id!r} "
                    f"(found {record.get('id')!r}, {record.get('datasetId')!r})."
                )
            records[submission_id] = record

        submission_ids = sorted(records, key=str.casefold)
        (benchmark_dir / MANIFEST_NAME).write_text(
            json.dumps({"submissionIds": submission_ids}, indent=2) + "\n", encoding="utf-8"
        )
        bundle.extend(records[submission_id] for submission_id in submission_ids)
        benchmark_ids.append(benchmark_id)
        print(f"Updated {benchmark_id}/{MANIFEST_NAME}: {len(submission_ids)} submissions")

    # Top-level index of the benchmark folders, so the site knows which per-benchmark manifests exist.
    (ROOT / MANIFEST_NAME).write_text(json.dumps({"benchmarkIds": benchmark_ids}, indent=2) + "\n", encoding="utf-8")
    # Every lightweight record in one compact file: the site's fast path (one request instead of a manifest chain).
    (ROOT / BUNDLE_NAME).write_text(
        json.dumps({"submissions": bundle}, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Updated {MANIFEST_NAME} and {BUNDLE_NAME}: {len(bundle)} submissions across {len(benchmark_ids)} benchmark folders")


if __name__ == "__main__":
    main()
