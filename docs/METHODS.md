# Methods and safeguards (v0.1)

## Question
How well can a cell-type classifier transfer its predictions to an entirely unseen donor? This is not equivalent to estimating diagnostic performance or proving biological feature causality.

## Protocol
Given donors D1..Dn, in each fold k all cells of Dk are held out. The remaining donors form the training set. On training counts alone: filter genes seen in at least `min_cells`, compute per-cell total-count normalization to 10,000 and log1p, score the dispersion `variance / (mean + 1e-8)` and select up to `top_genes`. Apply the identical selected gene indices and normalization to test counts. Training class labels are never taken from test donors. Classes missing from the training donors are flagged.

The baseline estimators use pre-registered hyperparameters in `models.py`. No search or tuning is performed on held-out donor performance. The small transformer, if opted into, uses fixed quantization thresholds and an epoch count provided by the user. Use nested group CV if comparing tuned models.

## Primary and secondary metrics
Primary: arithmetic mean of held-out donor accuracies and the lowest donor accuracy. Secondary: mean donor-wise macro F1 calculated on test-present classes, balanced accuracy, and overall pooled cell accuracy. Report per-class support, precision, recall and F1 for honest interpretation of rare types. The bootstrap samples donors, not individual cells, and is descriptive only.

## Why a simple HVG heuristic?
We deliberately avoid implying equivalence with batch-aware Scanpy/Seurat HVG routines; such equivalence requires a separate implementation and validation. This MVP proves the *leakage-safe algorithmic flow*, not an exact published cancer benchmark.

## Important limitations
Nonuniform donor sizes, imperfect reference annotation, technical batches, patient subtype variation, a missing label in training and assay protocols may affect benchmark outcomes. Test labels are used only to compute final evaluation metrics. Any cell-level or cohort-level processing learned from the entire dataset before calling this tool can still leak test information. Users must check their upstream pipeline independently.