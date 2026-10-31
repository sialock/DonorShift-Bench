"""Training-only feature selection; no test data may influence this transform."""
import numpy as np
from scipy import sparse


def normalize_log1p(X, target_sum=10000.0):
    """Per-cell library size normalization followed by log1p; preserves CSR sparsity."""
    if sparse.issparse(X):
        mat = X.tocsr().astype(np.float64, copy=True)
        totals = np.asarray(mat.sum(axis=1)).ravel()
        factors = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
        mat = sparse.diags(factors, format='csr') @ mat
        mat.data = np.log1p(mat.data)
        return mat
    mat = np.asarray(X, dtype=np.float64).copy()
    totals = mat.sum(axis=1, keepdims=True)
    np.divide(mat, totals, out=mat, where=totals > 0)
    mat *= target_sum
    return np.log1p(mat)


class TrainOnlyHVG:
    """Basic per-fold dispersion HVG heuristic, NOT Seurat's batch-aware HVG algorithm.

    Score is variance / (mean+epsilon) on training-only normalized/logged counts.
    Genes must be detected in at least min_cells training cells. Selected gene
    columns remain in original input order for reliable output interpretation.
    """
    def __init__(self, n_top=64, min_cells=3):
        if n_top < 2 or min_cells < 1:
            raise ValueError('n_top >=2 and min_cells >=1 required')
        self.n_top, self.min_cells = n_top, min_cells
        self.selected_indices_ = None

    def fit(self, X_train):
        if sparse.issparse(X_train):
            detections = np.asarray(X_train.getnnz(axis=0)).ravel()
        else:
            detections = np.count_nonzero(X_train, axis=0)
        eligible = np.flatnonzero(detections >= self.min_cells)
        if eligible.size < 2:
            raise ValueError('Fewer than two genes pass training-only min_cells filter')
        Y = normalize_log1p(X_train)
        mu = np.asarray(Y.mean(axis=0)).ravel()
        ex2 = np.asarray(Y.power(2).mean(axis=0)).ravel() if sparse.issparse(Y) else np.mean(Y**2, axis=0)
        var = np.maximum(ex2 - mu**2, 0)
        scores = np.divide(var, mu + 1e-8)
        order = np.lexsort((eligible, -scores[eligible]))
        self.selected_indices_ = np.sort(eligible[order[:min(self.n_top, len(eligible))]])
        return self

    def transform(self, X):
        if self.selected_indices_ is None:
            raise RuntimeError('TrainOnlyHVG.fit(train_only) must be called before transform')
        return normalize_log1p(X)[:, self.selected_indices_]

    def fit_transform(self, X_train):
        return self.fit(X_train).transform(X_train)