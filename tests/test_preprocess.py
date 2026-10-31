import numpy as np
from scipy import sparse
from donorshift.preprocess import TrainOnlyHVG, normalize_log1p


def test_train_only_hvg_unaffected_by_test_data():
    train = np.array([[2,0,3,1],[3,0,2,1],[0,1,1,2],[1,0,4,0]], dtype=int)
    selector = TrainOnlyHVG(n_top=2,min_cells=1).fit(train)
    original = selector.selected_indices_.copy()
    extreme_test = np.array([[999999,0,0,2000],[0,9999,0,0]])
    selector.transform(extreme_test)
    np.testing.assert_array_equal(original, selector.selected_indices_)


def test_sparse_dense_normalize_equivalence():
    X = np.array([[0,2,3],[1,0,7],[0,0,0]], dtype=int)
    dense = normalize_log1p(X)
    sp = normalize_log1p(sparse.csr_matrix(X))
    np.testing.assert_allclose(dense, sp.toarray())
    np.testing.assert_allclose(dense[-1], [0,0,0])


def test_sparse_dense_hvg_same_gene_indices():
    X = np.array([[0,1,4,7],[1,0,5,6],[2,1,8,9],[5,6,7,0]],dtype=int)
    a = TrainOnlyHVG(3,1).fit(X)
    b = TrainOnlyHVG(3,1).fit(sparse.csr_matrix(X))
    np.testing.assert_array_equal(a.selected_indices_, b.selected_indices_)