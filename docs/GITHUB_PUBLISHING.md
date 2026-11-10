# Development and releases

This project is maintained at [sialock/DonorShift-Bench](https://github.com/sialock/DonorShift-Bench).

## Local checks

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m build

donorshift compare \
  --input examples/data/synthetic_sc_counts.csv \
  --output results/check --models logreg --top-genes 32 \
  --random-repeats 1 --bootstrap 100 --dataset-kind synthetic
```

GitHub Actions runs the tests on supported Python versions and checks both the `run` and `compare` CLI commands.

For a tagged release, verify that the tests and package build pass, update `CHANGELOG.md`, and tag a reviewed commit. The bundled example uses synthetic data only; it is not evidence about any clinical cohort. If adding public-data results in a later release, record the accession, the annotation provenance, the exact input and preprocessing procedure, and the software version. Do not commit controlled-access data, patient-identifying metadata or credentials.
