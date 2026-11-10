"""Cell-level versus donor-held-out evaluation, with auditable split metadata.

The cell-level split is deliberately *not* donor-independent. No preprocessing
is fitted before the train/test split in either protocol.
"""
from __future__ import annotations

import base64
import hashlib
import json
from html import escape
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from . import __version__
from .benchmark import benchmark
from .data import load, take_rows
from .models import AVAILABLE, predict_model
from .preprocess import TrainOnlyHVG
from .utils import sha256_file


def split_cells_within_donors(labels, donor_ids, *, test_fraction=0.2, seed=42):
    """Split each donor's cells; stratify by cell type when sample counts allow.

    Donors with only one cell stay in training (no empty training-only donor).
    Returned indices are sorted and are disjoint by construction.
    """
    if not np.isfinite(test_fraction) or not 0 < test_fraction < 1:
        raise ValueError('test_fraction must be finite and between 0 and 1')
    labels, donors = np.asarray(labels), np.asarray(donor_ids)
    if labels.ndim != 1 or donors.ndim != 1 or len(labels) != len(donors):
        raise ValueError('Donor and cell-type arrays must be aligned and one-dimensional')

    train_parts, test_parts = [], []
    stratified_donors, singleton_donors = [], []
    sorted_donors = sorted(set(donors))
    for donor in sorted_donors:
        indices = np.flatnonzero(donors == donor)
        if len(indices) == 1:
            train_parts.append(indices)
            singleton_donors.append(str(donor))
            continue
        n_test = max(1, min(len(indices) - 1, int(round(len(indices) * test_fraction))))
        group_labels = labels[indices]
        _, counts = np.unique(group_labels, return_counts=True)
        stratify = (group_labels if counts.min() >= 2 and
                    n_test >= len(counts) and len(indices) - n_test >= len(counts)
                    else None)
        if stratify is not None:
            stratified_donors.append(str(donor))
        # Use a fixed per-donor seed from a *local* RNG (never Python's hash()).
        # A separate deterministic stream for each donor.
        donor_ix = sorted_donors.index(donor)
        local_seed = int(np.random.SeedSequence([seed, donor_ix]).generate_state(1)[0])
        tr, te = train_test_split(indices, test_size=n_test, random_state=local_seed,
                                  shuffle=True, stratify=stratify)
        train_parts.append(tr)
        test_parts.append(te)
    if not test_parts:
        raise ValueError('Need at least one donor with two or more cells for a cell split')
    train = np.sort(np.concatenate(train_parts))
    test = np.sort(np.concatenate(test_parts))
    if np.intersect1d(train, test).size or len(train) + len(test) != len(donors):
        raise AssertionError('Train/test index isolation failed')
    return train, test, stratified_donors, singleton_donors


def _sha_of_ids(ids):
    values = sorted(map(str, ids))
    return hashlib.sha256(('\n'.join(values) + '\n').encode('utf-8')).hexdigest()


def _save_comparison_html(out, comparison, summary):
    labels = comparison['model'].tolist()
    pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.9), 4.8))
    ax.bar(pos - .18, comparison['donor_held_out_accuracy'], width=.36,
           label='Unseen donor (LODO)', color='#3264a6')
    ax.bar(pos + .18, comparison['random_cell_accuracy'], width=.36,
           label='Within-donor cell split', color='#e18f42')
    ax.set_xticks(pos, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Equal-donor mean accuracy')
    ax.set_title('Two different evaluation questions')
    ax.grid(axis='y', alpha=.18)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'comparison.png', dpi=180, bbox_inches='tight')
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    img = base64.b64encode(buffer.getvalue()).decode('ascii')

    rows = ''.join(
        '<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in (
            record.model, f'{record.donor_held_out_accuracy:.3f}',
            f'{record.random_cell_accuracy:.3f}',
            f'{record.random_minus_held_out:.3f}',
        )) + '</tr>'
        for record in comparison.itertuples(index=False)
    )
    accession = escape(summary.get('accession') or 'Not supplied')
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DonorShift-Bench | Split comparison</title><style>
body{{margin:0;background:#f5f7fb;color:#17253b;font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1040px;margin:32px auto;padding:0 18px}}
section{{background:#fff;border:1px solid #e0e6f0;border-radius:14px;padding:22px;margin:20px 0}}
h1{{margin-bottom:0}}p{{max-width:85ch}}img{{display:block;width:100%;height:auto}}
table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:11px;border-bottom:1px solid #e2e7f0}}
.note{{background:#fff4df;border-left:4px solid #e49a46;padding:12px 16px}}
code{{background:#e9eff8;padding:2px 5px;border-radius:5px}}
</style></head><body><main><section><h1>DonorShift-Bench</h1>
<p>Cell-level versus donor-held-out evaluation</p><p>Input accession: {accession}</p>
<p class="note"><strong>How to read this:</strong> the within-donor cell split
allows the same donors in training and testing. It does not measure generalization
to a new donor. The two protocols also have different training sizes, so the
score difference is descriptive, not a causal estimate of donor leakage.</p></section>
<section><h2>Results</h2><table><thead><tr><th>Model</th>
<th>Held-out donor</th><th>Within-donor cell split</th>
<th>Random minus held-out</th></tr></thead><tbody>{rows}</tbody></table>
<p>Metrics are equal-donor mean accuracies. The cell-split value is averaged across
{summary['random_repeats']} random replicates. Each model uses the same fixed hyperparameters
in both protocols. Gene selection is fitted using training cells only.</p>
<img src="data:image/png;base64,{img}" alt="Comparison of the two evaluation protocols"></section>
<section><h2>Audit trail</h2><p>The 
<code>random_split_manifest.json</code> records training/test donor overlap,
selected genes, seeds and SHA-256 fingerprints of each split's cell identifiers.
See <code>random_split_metrics.csv</code> for replicate-wise results and
<code>donor_held_out/</code> for the full leave-one-donor-out report.</p></section>
<section><h2>Limits</h2><p>Reused test cells across random repetitions are not
independent observations. No hypothesis tests or confidence intervals for the
between-protocol difference are reported. Changes in donor composition, class
support, training size or annotation quality can affect the gap.</p>
<p>{'Synthetic data are used here only to demonstrate the software; no biological result is implied.' if summary['dataset_kind']=='synthetic' else 'Results depend on correct donor and cell-type annotations, raw counts and upstream processing.'}</p>
</section></main></body></html>'''
    (out / 'comparison_report.html').write_text(page, encoding='utf-8')


def compare_protocols(input_path, output_dir, *, models=('logreg', 'random_forest'),
                      top_genes=48, min_cells=3, seed=42, n_bootstraps=1000,
                      test_fraction=0.2, random_repeats=3,
                      donor_col='donor_id', label_col='cell_type', counts_layer=None,
                      transformer_epochs=10, make_html=True,
                      dataset_kind='unspecified', accession=None):
    """Compare two distinct estimation targets, without tuning on held-out cells.

    LODO evaluates unseen donors; random splitting deliberately shares donors.
    The observed score difference does NOT isolate the causal effect of leakage.
    """
    models = list(dict.fromkeys(models))
    if not models or any(model not in AVAILABLE for model in models):
        raise ValueError(f'Unknown model(s). Options: {AVAILABLE}')
    if random_repeats < 1 or random_repeats > 100:
        raise ValueError('random_repeats must be between 1 and 100')
    if not np.isfinite(test_fraction) or not 0 < test_fraction < 1:
        raise ValueError('test_fraction must be finite and between 0 and 1')
    if seed < 0 or top_genes < 2 or min_cells < 1 or n_bootstraps < 100 or transformer_epochs < 1:
        raise ValueError('Invalid seed/top_genes/min_cells/bootstrap/epochs setting')
    if dataset_kind not in {'unspecified', 'synthetic', 'public', 'other'}:
        raise ValueError('dataset_kind must be unspecified, synthetic, public or other')
    if dataset_kind == 'public' and not accession:
        raise ValueError('A public dataset needs --accession for provenance')

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = load(input_path, donor_col, label_col, counts_layer)
    donor_values = sorted(set(data.donor_ids))
    label_values = sorted(set(data.labels))
    held_out = benchmark(input_path, out / 'donor_held_out', models=models,
                         top_genes=top_genes, min_cells=min_cells, seed=seed,
                         n_bootstraps=n_bootstraps, donor_col=donor_col,
                         label_col=label_col, counts_layer=counts_layer,
                         transformer_epochs=transformer_epochs, make_html=make_html)

    donor_rows, class_rows, prediction_rows = [], [], []
    replicate_rows, manifests = [], []
    for repeat in range(random_repeats):
        split_seed = seed + repeat * 100003
        train_idx, test_idx, stratified, singletons = split_cells_within_donors(
            data.labels, data.donor_ids, test_fraction=test_fraction, seed=split_seed
        )
        y_train, y_test = data.labels[train_idx], data.labels[test_idx]
        if len(set(y_train)) < 2:
            raise ValueError('Random cell split has fewer than two training classes')
        selector = TrainOnlyHVG(top_genes, min_cells)
        X_train = selector.fit_transform(take_rows(data.X, train_idx))
        X_test = selector.transform(take_rows(data.X, test_idx))
        train_donors = sorted(set(data.donor_ids[train_idx]))
        test_donors = sorted(set(data.donor_ids[test_idx]))
        shared = sorted(set(train_donors).intersection(test_donors))
        manifests.append(dict(
            repeat=repeat, split_seed=split_seed,
            train_cell_count=int(len(train_idx)), test_cell_count=int(len(test_idx)),
            train_donors=train_donors, test_donors=test_donors,
            overlapping_donors=shared, n_overlapping_donors=len(shared),
            stratified_within_donors=stratified, train_only_singleton_donors=singletons,
            train_ids_sha256=_sha_of_ids(data.cell_ids[train_idx]),
            test_ids_sha256=_sha_of_ids(data.cell_ids[test_idx]),
            selected_genes=data.gene_names[selector.selected_indices_].tolist(),
            feature_selection_train_only=True,
            train_label_counts={str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
            test_label_counts={str(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
        ))
        for model_ix, model_name in enumerate(models):
            model_seed = split_seed + model_ix * 7919
            prediction, confidence = predict_model(model_name, X_train, y_train, X_test,
                                                   seed=model_seed, transformer_epochs=transformer_epochs)
            donor_accuracies, donor_f1s = [], []
            for donor in test_donors:
                donor_mask = data.donor_ids[test_idx] == donor
                yt, yp = y_test[donor_mask], prediction[donor_mask]
                accuracy = float(accuracy_score(yt, yp))
                f1 = float(f1_score(yt, yp, labels=sorted(set(yt)),
                                    average='macro', zero_division=0))
                balanced = float(recall_score(yt, yp, labels=sorted(set(yt)),
                                              average='macro', zero_division=0))
                donor_accuracies.append(accuracy)
                donor_f1s.append(f1)
                donor_rows.append(dict(repeat=repeat, model=model_name, donor=donor,
                                       n_test_cells=int(donor_mask.sum()), accuracy=accuracy,
                                       macro_f1_present=f1, balanced_accuracy=balanced,
                                       donor_in_training=donor in train_donors))
                pr, rc, f1_by_class, support = precision_recall_fscore_support(
                    yt, yp, labels=label_values, zero_division=0)
                for ix, label in enumerate(label_values):
                    if support[ix]:
                        class_rows.append(dict(repeat=repeat, model=model_name, donor=donor,
                                               cell_type=label, support=int(support[ix]),
                                               precision=float(pr[ix]), recall=float(rc[ix]),
                                               f1=float(f1_by_class[ix]),
                                               class_missing_from_train=label not in set(y_train)))
            replicate_rows.append(dict(
                repeat=repeat, model=model_name, split_seed=split_seed,
                n_train_cells=int(len(train_idx)), n_test_cells=int(len(test_idx)),
                n_train_donors=len(train_donors), n_test_donors=len(test_donors),
                n_overlapping_donors=len(shared),
                mean_donor_accuracy=float(np.mean(donor_accuracies)),
                worst_donor_accuracy=float(min(donor_accuracies)),
                mean_donor_macro_f1_present=float(np.mean(donor_f1s)),
                pooled_cell_accuracy=float(accuracy_score(y_test, prediction)),
            ))
            for i, row_idx in enumerate(test_idx):
                prediction_rows.append(dict(repeat=repeat, model=model_name,
                                            cell_id=str(data.cell_ids[row_idx]),
                                            donor=str(data.donor_ids[row_idx]),
                                            true_label=str(y_test[i]),
                                            predicted_label=str(prediction[i]),
                                            correct=bool(y_test[i] == prediction[i]),
                                            confidence=float(confidence[i])))

    rep = pd.DataFrame(replicate_rows)
    donor_label_counts = (pd.DataFrame({'donor': data.donor_ids, 'cell_type': data.labels})
                          .groupby(['donor', 'cell_type'], sort=True).size()
                          .rename('n_cells').reset_index())
    comparison_rows = []
    for model_name in models:
        lodo = held_out['models'][model_name]
        rs = rep.loc[rep.model == model_name]
        random_acc = float(rs['mean_donor_accuracy'].mean())
        lodo_acc = float(lodo['mean_donor_accuracy'])
        comparison_rows.append(dict(
            model=model_name,
            donor_held_out_accuracy=lodo_acc,
            random_cell_accuracy=random_acc,
            random_minus_held_out=random_acc - lodo_acc,
            random_replicate_sd=float(rs['mean_donor_accuracy'].std(ddof=1)) if len(rs) > 1 else 0.0,
            lodo_worst_donor_accuracy=float(lodo['worst_donor_accuracy']),
            random_worst_donor_accuracy=float(rs['worst_donor_accuracy'].mean()),
            lodo_mean_donor_macro_f1=float(lodo['mean_donor_macro_f1_present']),
            random_mean_donor_macro_f1=float(rs['mean_donor_macro_f1_present'].mean()),
        ))
    df = pd.DataFrame(comparison_rows)
    lodo_folds = json.loads((out / 'donor_held_out' / 'fold_manifest.json').read_text(encoding='utf-8'))
    summary = dict(
        schema_version='2', project='DonorShift-Bench', version=__version__,
        dataset_kind=dataset_kind, accession=accession,
        input_sha256=sha256_file(input_path),
        n_cells=int(len(data.labels)), n_genes=int(len(data.gene_names)),
        n_donors=len(donor_values), n_cell_types=len(label_values),
        average_lodo_train_cells=float(np.mean([f['n_train_cells'] for f in lodo_folds])),
        average_random_train_cells=float(rep.groupby('repeat')['n_train_cells'].first().mean()),
        random_actual_test_fraction=float(rep.groupby('repeat')['n_test_cells'].first().mean() / len(data.labels)),
        models=models, seed=seed, random_repeats=random_repeats,
        random_test_fraction_requested=test_fraction,
        lodo_evaluation='leave-one-donor-out',
        random_evaluation='within-donor cell split; train and test donors overlap by design',
        preprocessing='HVG detection and selection on training cells only, then independent total-count normalization',
        interpretation=('Random-minus-held-out difference is descriptive. Different training sizes, '
                        'test composition and donor overlap prevent causal interpretation.'),
        comparisons=comparison_rows,
    )
    donor_label_counts.to_csv(out / 'cohort_audit.csv', index=False)
    df.to_csv(out / 'comparison.csv', index=False, float_format='%.6f')
    rep.to_csv(out / 'random_split_metrics.csv', index=False, float_format='%.6f')
    pd.DataFrame(donor_rows).to_csv(out / 'random_per_donor_metrics.csv', index=False,
                                   float_format='%.6f')
    pd.DataFrame(class_rows).to_csv(out / 'random_per_class_metrics.csv', index=False,
                                   float_format='%.6f')
    pd.DataFrame(prediction_rows).to_csv(out / 'random_predictions.csv', index=False,
                                        float_format='%.6f')
    (out / 'random_split_manifest.json').write_text(
        json.dumps(manifests, indent=2), encoding='utf-8')
    (out / 'comparison_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    if make_html:
        _save_comparison_html(out, df, summary)
    return summary
