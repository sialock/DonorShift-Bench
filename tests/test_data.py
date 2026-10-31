import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from donorshift.data import CellData, from_csv
from donorshift.synthetic import make_synthetic


def test_demo_reproducible():
    a = make_synthetic(4, 8, 24, seed=2)
    b = make_synthetic(4, 8, 24, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert a['donor_id'].nunique() == 4


def test_csv_loader_roundtrip(tmp_path):
    f = tmp_path / 'counts.csv'
    make_synthetic(4, 8, 24).to_csv(f, index=False)
    data = from_csv(f)
    assert data.X.shape == (32, 24)
    assert len(data.gene_names) == 24
    assert len(set(data.cell_ids)) == 32


def test_normalized_data_rejected(tmp_path):
    f = tmp_path / 'bad.csv'
    df = make_synthetic(4, 8, 24)
    df['GENE_001'] = df['GENE_001'].astype(float)
    df.loc[0,'GENE_001'] = 1.5
    df.to_csv(f,index=False)
    with pytest.raises(ValueError, match='raw counts'):
        from_csv(f)


def test_negative_data_rejected():
    with pytest.raises(ValueError, match='nonnegative'):
        CellData(np.array([[0,2],[-1,2],[3,4],[1,0]]),
                 ['D1','D2','D3','D3'], ['A','A','B','B'],
                 ['x','y','z','w'], ['g1','g2'])


def test_sparse_counts_accepted():
    data = CellData(sparse.csr_matrix([[0,2],[1,0],[3,0],[1,4]]),
                    ['D1','D2','D3','D3'], ['A','A','B','B'],
                    ['a','b','c','d'], ['g1','g2'])
    assert sparse.issparse(data.X)


def test_repeated_gene_names_rejected():
    with pytest.raises(ValueError, match='unique'):
        CellData(np.ones((4,2)), ['D1','D2','D3','D3'],
                 ['A','A','B','B'], ['x','y','z','w'], ['g','g'])