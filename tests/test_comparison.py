import json

import numpy as np
import pandas as pd
import pytest

from donorshift.cli import main
from donorshift.compare import compare_protocols, split_cells_within_donors
from donorshift.synthetic import make_synthetic


def _example(tmp_path):
    path = tmp_path / 'counts.csv'
    make_synthetic(n_donors=4, cells_per_donor=20, n_genes=24, seed=11).to_csv(path, index=False)
    return path


def test_random_split_is_per_donor_disjoint_deterministic():
    donors = np.repeat(['D1', 'D2', 'D3'], 12)
    labels = np.tile(['A', 'B'] * 6, 3)
    a = split_cells_within_donors(labels, donors, test_fraction=0.25, seed=123)
    b = split_cells_within_donors(labels, donors, test_fraction=0.25, seed=123)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert set(a[0]).isdisjoint(a[1])
    assert len(a[0]) + len(a[1]) == len(donors)
    assert set(donors[a[0]]) == set(donors[a[1]])
    assert set(a[2]) == set(donors)
    assert not a[3]
    assert not np.array_equal(a[1], split_cells_within_donors(labels, donors, test_fraction=0.25, seed=456)[1])


def test_random_split_handles_rare_celltypes_and_singleton_donors():
    donor = np.asarray(['a'] * 5 + ['b'] * 5 + ['c'])
    labels = np.asarray(['x'] * 4 + ['rare'] + ['x'] * 3 + ['y'] * 2 + ['x'])
    train, test, stratified, singleton = split_cells_within_donors(labels, donor, seed=17)
    assert set(train).isdisjoint(test)
    assert len(train) + len(test) == len(donor)
    assert 'c' in singleton
    assert (donor[test] != 'c').all()
    assert 'a' not in stratified


def test_random_split_invalid_input():
    with pytest.raises(ValueError, match='test_fraction'):
        split_cells_within_donors(['A'] * 5, ['x'] * 5, test_fraction=float('nan'))
    with pytest.raises(ValueError, match='aligned'):
        split_cells_within_donors(['A'], ['x', 'y'])


def test_comparison_outputs_donor_overlap_and_train_only_preprocessing(tmp_path):
    src = _example(tmp_path)
    out = tmp_path / 'comp'
    summary = compare_protocols(src, out, models=['logreg'], top_genes=12,
                                n_bootstraps=100, random_repeats=2,
                                dataset_kind='synthetic', make_html=True)
    assert summary['schema_version'] == '2'
    assert summary['dataset_kind'] == 'synthetic'
    assert summary['n_donors'] == 4
    assert len(summary['comparisons']) == 1
    for name in ['comparison.csv', 'random_split_metrics.csv', 'random_per_donor_metrics.csv',
                 'random_per_class_metrics.csv', 'random_predictions.csv',
                 'random_split_manifest.json', 'comparison_summary.json',
                 'comparison.png', 'comparison_report.html']:
        assert (out / name).is_file(), name
    assert (out / 'comparison.png').stat().st_size > 2000
    assert (out / 'donor_held_out' / 'summary.json').is_file()
    manifests = json.loads((out / 'random_split_manifest.json').read_text())
    assert len(manifests) == 2
    for manifest in manifests:
        assert set(manifest['train_donors']) == set(manifest['test_donors'])
        assert manifest['n_overlapping_donors'] == 4
        assert manifest['feature_selection_train_only'] is True
        assert len(manifest['selected_genes']) == 12
        assert manifest['train_ids_sha256'] != manifest['test_ids_sha256']
    data = pd.read_csv(out / 'random_split_metrics.csv')
    assert len(data) == 2
    assert data['mean_donor_accuracy'].between(0, 1).all()
    predictions = pd.read_csv(out / 'random_predictions.csv')
    assert predictions.groupby(['repeat', 'model', 'cell_id']).size().eq(1).all()
    assert 'within-donor cell split' in (out / 'comparison_report.html').read_text()


def test_comparison_is_seed_reproducible(tmp_path):
    src = _example(tmp_path)
    settings = dict(models=['logreg'], top_genes=12, n_bootstraps=100,
                    random_repeats=2, make_html=False)
    a = compare_protocols(src, tmp_path / 'a', **settings)
    b = compare_protocols(src, tmp_path / 'b', **settings)
    assert a['comparisons'] == b['comparisons']
    for filename in ['comparison.csv', 'random_split_metrics.csv',
                     'random_predictions.csv', 'random_split_manifest.json']:
        assert (tmp_path / 'a' / filename).read_bytes() == (tmp_path / 'b' / filename).read_bytes()
    assert not (tmp_path / 'a' / 'comparison_report.html').exists()


def test_random_feature_selection_ignores_test_cells(tmp_path):
    src = _example(tmp_path)
    df = pd.read_csv(src)
    train, test, _, _ = split_cells_within_donors(df.cell_type, df.donor_id, seed=42)
    a = compare_protocols(src, tmp_path / 'original', models=['logreg'], top_genes=12,
                          n_bootstraps=100, random_repeats=1, make_html=False)
    changed = df.copy()
    columns = [c for c in changed.columns if c.startswith('GENE_')]
    changed.loc[test, columns] = 0
    alt = tmp_path / 'changed.csv'
    changed.to_csv(alt, index=False)
    b = compare_protocols(alt, tmp_path / 'altered', models=['logreg'], top_genes=12,
                          n_bootstraps=100, random_repeats=1, make_html=False)
    manifest_a = json.loads((tmp_path / 'original' / 'random_split_manifest.json').read_text())[0]
    manifest_b = json.loads((tmp_path / 'altered' / 'random_split_manifest.json').read_text())[0]
    assert manifest_a['selected_genes'] == manifest_b['selected_genes']
    assert manifest_a['train_ids_sha256'] == manifest_b['train_ids_sha256']


def test_public_dataset_requires_accession(tmp_path):
    src = _example(tmp_path)
    with pytest.raises(ValueError, match='accession'):
        compare_protocols(src, tmp_path / 'out', models=['logreg'],
                          dataset_kind='public', random_repeats=1)


def test_cli_compare_smoke(tmp_path):
    src = _example(tmp_path)
    out = tmp_path / 'cli'
    assert main(['compare', '--input', str(src), '--output', str(out),
                 '--models', 'logreg', '--top-genes', '12', '--bootstrap', '100',
                 '--random-repeats', '1', '--dataset-kind', 'synthetic', '--no-html']) == 0
    assert (out / 'comparison.csv').is_file()
