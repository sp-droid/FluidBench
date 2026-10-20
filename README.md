https://sp-droid.github.io/FluidBench/

# FluidBench website prototype

A static, multi-page benchmark site built with plain HTML, CSS, and JavaScript.

## Preview locally

Serve the repository root with any static HTTP server and open its local URL. The site should be served over HTTP rather than opened directly as a file URL.

## Sample data

`submissions/submissions.json` lists the benchmark folders that have submissions. Each benchmark has a folder under `submissions/` with its own manifest, `submissions/{benchmarkId}/submissions.json`, listing the submission IDs it contains. Every submission is a folder: the lightweight record lives at `submissions/{benchmarkId}/{id}/submission.json`, next to its heavier artifacts (`rollout-analysis.json`, optional `training-history.json`). Benchmark records are routed by `benchmarks/benchmarks.json` and stored in `benchmarks/{id}.json`. The sample metrics and training histories are illustrative and can be edited in their corresponding files.

## Pages

- /index.html — overview, featured benchmark, top-model preview, and benchmark summary.
- /pages/leaderboard.html — searchable and filterable sample rankings.
- /pages/compare.html — side-by-side model losses and efficiency details.
- /pages/benchmarks.html, /pages/models.html, /pages/metrics.html, and /pages/docs.html — focused reference pages.
