# Changelog

## 0.2.0 — 2026-11-10

- Added `donorshift compare`: a repeated within-donor cell split alongside the leave-one-donor-out benchmark.
- Kept gene selection and all trainable preprocessing inside each training split.
- Recorded donor overlap, per-donor and per-class metrics, seeded splits, gene lists and cell-ID fingerprints.
- Added a local HTML comparison report and PNG plot, plus a donor-by-cell-type audit CSV.
- Added explicit dataset-provenance settings and a guide for preparing public scRNA-seq inputs.
- Streamed input-file SHA-256 hashing rather than loading entire large files for checksums.
- Expanded tests for split isolation, determinism, sparse/dense compatibility and leakage controls.

## 0.1.0 — 2026-10-31

- Leave-one-donor-out cell-type classification with training-fold-only feature selection.
- Raw-count validation for CSV and optional `.h5ad` input.
- Logistic regression, random forest, MLP and optional small Transformer baselines.
- Per-donor, per-class, prediction and fold-manifest outputs with a portable HTML report.
- Synthetic data, tests and GitHub Actions CI.
