# Publishing DonorShift-Bench

**Recommended:** create an independent public repository named `DonorShift-Bench` after verifying third-party/data permissions. The zip contains no patient data; its bundled dataset is fully synthetic.

1. Unzip the archive and `cd DonorShift-Bench`.
2. Run `python -m pip install -e '.[dev]'` followed by `python -m pytest -q`.
3. Go to GitHub > New repository > Name: `DonorShift-Bench` > Public; do **not** initialize a README since this project supplies one.
4. Run:

```bash
git init
git add .
git commit -m "feat: initial leakage-aware donor-held-out scRNA benchmark"
git branch -M main
git remote add origin https://github.com/YOUR_LOGIN/DonorShift-Bench.git
git push -u origin main
```

5. Open Actions and check the CI jobs. Add tags `bioinformatics`, `single-cell`, `machine-learning`, `scRNA-seq`, `donor-generalization`.
6. Do not label the synthetic sample scores as experimental evidence. For a future real cohort, include a verified dataset accession, version, code release and clearly stated methods.

Publish to a dedicated repository or a reviewed staging branch. Do not overwrite unrelated research files.