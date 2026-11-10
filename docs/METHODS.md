# Methods

## The question

How does cell-type classification change when the evaluation uses unseen donors instead of cells from donors already in training? The two protocols estimate different things, so the comparison isn't a controlled experiment that isolates the effect of donor identity.

## Leave-one-donor-out

Every donor is held out once. For each fold, all cells from the held-out donor are test data. Feature detection, ranking and model fitting use the other donors only. No information from test-donor counts or labels is used to select genes or fit the model. The outputs include one accuracy per held-out donor, along with a record of the chosen genes and training donors.

## Within-donor cell split (added in 0.2)

For each repeat, the program splits cells within each donor (default: roughly 80% training, 20% test). It attempts cell-type stratification within a donor when class counts and split sizes allow; otherwise that donor is split without stratification. A donor with only one cell stays in training. The split is deterministic for a given seed. Training and test cells are disjoint, but the donor IDs normally **overlap**, as intended.

The program fits a *new* feature selector and model on the training cells of each repeat. It saves all selected genes, the training/test donor lists, the overlapping-donor list, per-repeat cell counts and SHA-256 fingerprints of cell IDs to `random_split_manifest.json`. Test-cell predictions are stored separately.

## Preprocessing and models

The input must be raw, nonnegative, integer-like counts. Per-cell total-count normalization to 10,000 and `log1p` are applied to training and test cells separately; they do not estimate a parameter from other cells. Gene detection (minimum training cells) and the simple dispersion score `variance / (mean + 1e-8)` are calculated on the training portion only. This is **not** the Seurat or Scanpy batch-aware highly variable gene method.

The baselines use fixed settings recorded in `src/donorshift/models.py`. The optional transformer starts from random weights. Hyperparameters are not selected using the held-out donors. A tuned comparison would need a donor-grouped *inner* validation scheme.

## Reporting

- **LODO accuracy** is the arithmetic mean of donor-wise test accuracies. The worst donor is also reported.
- **Random-split accuracy** is calculated for each tested donor, averaged across those donors within a repeat, then averaged over the repeats. Its replicate SD reflects repeated splitting, not independent-patient uncertainty.
- **Gap** is random-split mean minus LODO mean. We report the sign and magnitude but **do not** interpret it as a causal estimate of leakage or as a p-value.
- Both protocols also report per-class metrics and donor-wise macro F1 for classes present in the test donor.
- LODO's donor bootstrap interval is descriptive. Random-split repeats can retest the same cells and are not independent cohorts.

The protocols usually have different training-set sizes. In the bundled 8-donor example LODO trains on 336 cells per fold, while the within-donor split trains on about 304. The result can be higher or lower under either protocol. A true controlled sensitivity analysis would need additional experimental designs beyond this MVP.

## Input and interpretation checks

A patient contributing multiple specimens must remain grouped under one donor ID. Check cell-type annotation, missing labels, class support, technical batch effects and sample selection before interpreting any differences. Previously normalized or globally batch-corrected data may invalidate leakage claims. A successful test suite verifies software behavior, not biological validity or clinical use.

The bundled example contains only synthetic cells. No real breast-cancer benchmark has been executed for this release. See [public-data notes](PUBLIC_DATA.md) for a reproducible route using Wu et al.'s GSE176078 atlas.
