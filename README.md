https://sp-droid.github.io/FluidBench/

# FluidBench website prototype

A static, multi-page benchmark site built with plain HTML, CSS, and JavaScript.

## Preview locally

Serve the repository root with any static HTTP server and open its local URL. The site should be served over HTTP rather than opened directly as a file URL.

## Sample data

`submissions/submissions.json` is a lightweight manifest that groups submission IDs by benchmark. Each benchmark has a folder under `submissions/`; lightweight submission records live at `submissions/{benchmarkId}/{id}.json`, and heavier training histories live at `submissions/{benchmarkId}/{id}/training-history.json`. Benchmark records are routed by `benchmarks/benchmarks.json` and stored in `benchmarks/{id}.json`. The sample metrics and training histories are illustrative and can be edited in their corresponding files.

## Pages

- /index.html — overview, featured benchmark, top-model preview, and benchmark summary.
- /pages/leaderboard.html — searchable and filterable sample rankings.
- /pages/compare.html — side-by-side model losses and efficiency details.
- /pages/benchmarks.html, /pages/models.html, /pages/metrics.html, and /pages/docs.html — focused reference pages.
