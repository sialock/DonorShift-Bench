"""Fully synthetic, non-patient demo count data with donor-specific shifts."""
from pathlib import Path
import numpy as np
import pandas as pd


def make_synthetic(n_donors=8, cells_per_donor=48, n_genes=64, seed=7):
    if n_donors < 3 or cells_per_donor < 8 or n_genes < 16:
        raise ValueError('Need >=3 donors, >=8 cells/donor, >=16 genes')
    rng = np.random.default_rng(seed)
    types = ['B_cell', 'CD4_T', 'Epithelial', 'Myeloid']
    rows = []
    marker_sets = [np.arange(i * 4, (i+1) * 4) for i in range(4)]
    for d in range(n_donors):
        donor = f'D{d+1:02d}'
        donor_scale = np.exp(rng.normal(0, 0.22, n_genes))
        donor_type_probs = rng.dirichlet(np.ones(4) * 5)
        for i in range(cells_per_donor):
            ci = rng.choice(4, p=donor_type_probs)
            rate = rng.uniform(0.35, 1.1, n_genes) * donor_scale
            rate[marker_sets[ci]] *= 10
            # Simulate overdispersion with Gamma-Poisson, without using real donors.
            rate *= rng.gamma(shape=3.0, scale=1/3.0, size=n_genes)
            counts = rng.poisson(rate)
            rows.append((f'{donor}_cell_{i:04d}', donor, types[ci], *counts.tolist()))
    return pd.DataFrame(rows, columns=['cell_id', 'donor_id', 'cell_type'] +
                        [f'GENE_{i+1:03d}' for i in range(n_genes)])


def save_synthetic(output, **kwargs):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = make_synthetic(**kwargs)
    df.to_csv(path, index=False)
    return path