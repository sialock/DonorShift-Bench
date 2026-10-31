import numpy as np
import pytest
from donorshift.models import predict_model


def test_optional_transformer_small_data():
    pytest.importorskip('torch')
    X=np.tile(np.array([[0.,3.,2.,0.],[3.,0.,0.,2.]]), (6,1))
    y=np.tile(np.array(['A','B']),6)
    preds, confidence=predict_model('transformer',X,y,X[:4],seed=7,transformer_epochs=2)
    assert len(preds)==4
    assert set(preds).issubset({'A','B'})
    assert np.all((confidence>=0)&(confidence<=1))