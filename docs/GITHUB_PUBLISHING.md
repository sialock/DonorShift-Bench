# Development and release guide

This repository hosts **DonorShift-Bench**, a standalone, public Python package for leakage-aware donor-held-out single-cell benchmarking.

## Local development

```bash
git clone https://github.com/sialock/DonorShift-Bench.git
cd DonorShift-Bench
python -m pip install -e '.[dev]'
python -m pytest -q
python -m build
```

## Reproduce the synthetic demonstration

```bash
donorshift make-demo --output examples/data/synthetic_sc_counts.csv
donorshift validate --input examples/data/synthetic_sc_counts.csv
donorshift run --input examples/data/synthetic_sc_counts.csv \
  --output results/demo --models logreg random_forest --top-genes 48
```

The demonstration is fully synthetic, not evidence of biological or clinical validity. The committed example output under `examples/results/demo/` is provided for inspection.

## Release checklist

1. Run the complete test suite locally; confirm GitHub Actions checks pass on `main`.
2. Verify the example runs without access to nonpublic datasets, credentials, or private files.
3. Update `CHANGELOG.md`, documentation, and package version for every release.
4. Build source and wheel distributions with `python -m build`.
5. Tag a release only after a reviewed, passing commit: `git tag vX.Y.Z` and `git push origin vX.Y.Z`.
6. Document any public biological datasets with accession, license, provenance, splits, and reproducibility details.

## Data governance

Do not commit raw patient-identifying metadata, controlled-access expression matrices, credentials, or unlicensed research files. Maintain donor-disjoint validation and fit feature selection exclusively on training donors.
