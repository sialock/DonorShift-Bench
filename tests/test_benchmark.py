import json
import pandas as pd
import pytest
from donorshift.synthetic import make_synthetic
from donorshift.benchmark import benchmark


def _input(tmp_path):
    f = tmp_path/'synthetic_input.csv'
    make_synthetic(4, 15, 24, seed=19).to_csv(f, index=False)
    return f


def test_full_run_and_no_donor_leakage(tmp_path):
    inp = _input(tmp_path)
    dest = tmp_path/'results'
    result = benchmark(inp,dest,models=['logreg','random_forest'],
                       top_genes=12,n_bootstraps=100)
    assert result['n_donors'] == 4
    assert set(result['models']) == {'logreg','random_forest'}
    assert (dest/'report.html').exists()
    assert (dest/'donor_accuracy.png').stat().st_size > 1000
    assert (dest/'class_recall.png').stat().st_size > 1000
    manifest = json.loads((dest/'fold_manifest.json').read_text())
    assert len(manifest) == 4
    for fold in manifest:
        assert fold['held_out_donor'] not in fold['train_donors']
        assert fold['train_only_fit']
        assert len(fold['selected_genes']) == 12
    preds = pd.read_csv(dest/'predictions.csv')
    assert len(preds) == 2*4*15
    assert preds.groupby(['model','cell_id']).size().eq(1).all()
    assert preds['confidence'].between(0,1).all()


def test_reproducible_metrics(tmp_path):
    inp = _input(tmp_path)
    kwargs = dict(models=['logreg'],top_genes=12,n_bootstraps=100,seed=71,make_html=False)
    a = benchmark(inp,tmp_path/'a',**kwargs)
    b = benchmark(inp,tmp_path/'b',**kwargs)
    assert a['models'] == b['models']
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path/'a'/'per_donor_metrics.csv'),
                                  pd.read_csv(tmp_path/'b'/'per_donor_metrics.csv'))


def test_invalid_models_rejected(tmp_path):
    with pytest.raises(ValueError,match='Unknown model'):
        benchmark(_input(tmp_path),tmp_path/'results',models=['fake'])


def test_test_label_is_not_used_for_feature_selection(tmp_path):
    inp = _input(tmp_path)
    baseline = benchmark(inp,tmp_path/'baseline',models=['logreg'],top_genes=12,
                         n_bootstraps=100,make_html=False)
    df = pd.read_csv(inp)
    df.loc[df.donor_id=='D04','cell_type'] = 'A_completely_new_class'
    changed = tmp_path/'changed.csv'
    df.to_csv(changed,index=False)
    benchmark(changed,tmp_path/'changedout',models=['logreg'],top_genes=12,
              n_bootstraps=100,make_html=False)
    orig = json.loads((tmp_path/'baseline'/'fold_manifest.json').read_text())
    mod = json.loads((tmp_path/'changedout'/'fold_manifest.json').read_text())
    f0 = next(x for x in orig if x['held_out_donor']=='D04')
    f1 = next(x for x in mod if x['held_out_donor']=='D04')
    assert f0['selected_genes']==f1['selected_genes']
    assert 'A_completely_new_class' in f1['test_only_classes']


def test_heldout_counts_do_not_change_selected_genes(tmp_path):
    inp = _input(tmp_path)
    benchmark(inp, tmp_path/'baseline_counts', models=['logreg'],
              top_genes=12, n_bootstraps=100, make_html=False)
    data = pd.read_csv(inp)
    # Changing only the held-out donor's raw expression must not affect HVG fit.
    cols = [c for c in data if c.startswith('GENE_')]
    data.loc[data.donor_id == 'D04', cols] = 0
    mod_in = tmp_path/'changed_counts.csv'
    data.to_csv(mod_in, index=False)
    benchmark(mod_in, tmp_path/'after_counts_change', models=['logreg'],
              top_genes=12, n_bootstraps=100, make_html=False)
    old = json.loads((tmp_path/'baseline_counts'/'fold_manifest.json').read_text())
    new = json.loads((tmp_path/'after_counts_change'/'fold_manifest.json').read_text())
    old_d04 = next(x for x in old if x['held_out_donor']=='D04')
    new_d04 = next(x for x in new if x['held_out_donor']=='D04')
    assert old_d04['selected_genes'] == new_d04['selected_genes']