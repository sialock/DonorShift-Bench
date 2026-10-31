"""Data loading and validation. The input is RAW nonnegative expression counts."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

Matrix = np.ndarray | sparse.spmatrix


@dataclass
class CellData:
    X: Matrix
    donor_ids: np.ndarray
    labels: np.ndarray
    cell_ids: np.ndarray
    gene_names: np.ndarray

    def __post_init__(self):
        self.donor_ids = np.asarray(self.donor_ids, dtype=str)
        self.labels = np.asarray(self.labels, dtype=str)
        self.cell_ids = np.asarray(self.cell_ids, dtype=str)
        self.gene_names = np.asarray(self.gene_names, dtype=str)
        validate(self)


def validate(data: CellData) -> None:
    n, g = data.X.shape
    if len(data.X.shape) != 2 or n < 3 or g < 2:
        raise ValueError("Count matrix must be 2D with >=3 cells and >=2 genes")
    if any(len(x) != n for x in (data.donor_ids, data.labels, data.cell_ids)):
        raise ValueError("Metadata arrays must align with count matrix rows")
    if len(data.gene_names) != g:
        raise ValueError("gene_names must align with count matrix columns")
    if len(set(data.cell_ids)) != n or len(set(data.gene_names)) != g:
        raise ValueError("cell_id and gene names must be unique")
    if len(set(data.donor_ids)) < 3:
        raise ValueError("At least three donors required for a meaningful held-out benchmark")
    if len(set(data.labels)) < 2:
        raise ValueError("At least two cell-type labels required")
    if any(not s.strip() or s.lower() == 'nan' for col in (data.donor_ids, data.labels, data.cell_ids, data.gene_names) for s in col):
        raise ValueError("Missing or blank IDs, gene names, and labels are not allowed")
    values = data.X.data if sparse.issparse(data.X) else np.asarray(data.X)
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("Expression data must contain numeric raw counts")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Counts must be nonnegative and finite")
    if np.any(np.abs(values - np.rint(values)) > 1e-5):
        raise ValueError("Only unnormalized integer-like raw counts are accepted")


def from_csv(path: str | Path, donor_col='donor_id', label_col='cell_type', cell_col='cell_id') -> CellData:
    df = pd.read_csv(path)
    required = [donor_col, label_col]
    if not set(required).issubset(df.columns):
        raise ValueError(f"CSV needs columns: {required}")
    meta = {donor_col, label_col, cell_col}
    gene_cols = [c for c in df.columns if c not in meta]
    if len(gene_cols) < 2:
        raise ValueError("CSV needs >=2 gene expression columns")
    X = df[gene_cols].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float64)
    cell_ids = df[cell_col].to_numpy() if cell_col in df else np.array([f"cell_{i:06d}" for i in range(len(df))])
    return CellData(X, df[donor_col].to_numpy(), df[label_col].to_numpy(), cell_ids, np.array(gene_cols))


def from_h5ad(path: str | Path, donor_col='donor_id', label_col='cell_type', counts_layer=None) -> CellData:
    try:
        import anndata as ad
    except ImportError as exc:
        raise RuntimeError("To read .h5ad, install: pip install -e '.[h5ad]'") from exc
    adata = ad.read_h5ad(path)
    if donor_col not in adata.obs or label_col not in adata.obs:
        raise ValueError(f".h5ad obs must contain {donor_col!r} and {label_col!r}")
    if counts_layer and counts_layer not in adata.layers:
        raise ValueError(f"Count layer {counts_layer!r} does not exist")
    X = adata.layers[counts_layer] if counts_layer else adata.X
    X = X.copy() if sparse.issparse(X) else np.asarray(X).copy()
    return CellData(X, adata.obs[donor_col].to_numpy(), adata.obs[label_col].to_numpy(),
                    np.asarray(adata.obs_names), np.asarray(adata.var_names))


def load(path, donor_col='donor_id', label_col='cell_type', counts_layer=None):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        if counts_layer:
            raise ValueError("--counts-layer is only valid for .h5ad input")
        return from_csv(path, donor_col, label_col)
    if path.suffix.lower() == '.h5ad':
        return from_h5ad(path, donor_col, label_col, counts_layer)
    raise ValueError("Only .csv and .h5ad formats are supported")


def take_rows(X: Matrix, idx: np.ndarray) -> Matrix:
    return X[idx, :]