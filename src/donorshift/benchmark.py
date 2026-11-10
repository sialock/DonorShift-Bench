"""End-to-end evaluation with donor-separated folds and fit-on-train transforms."""
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import scipy
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_recall_fscore_support

from . import __version__
from .data import load, take_rows
from .models import AVAILABLE, predict_model
from .preprocess import TrainOnlyHVG
from .report import make_report
from .utils import sha256_file


def _bootstrap_mean_ci(values, rng, n_bootstraps=1000):
    """Percentile donor-cluster bootstrap; descriptive interval, not clinical CI."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    draws = rng.choice(arr, size=(n_bootstraps, len(arr)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def benchmark(input_path, output_dir, models=('logreg', 'random_forest'),
              top_genes=48, min_cells=3, seed=42, n_bootstraps=1000,
              donor_col='donor_id', label_col='cell_type', counts_layer=None,
              transformer_epochs=10, make_html=True):
    """Run strict leave-one-donor-out evaluation.

    The feature selector is instantiated and fit independently inside each fold
    using *only* the training donors. No hyperparameter selection uses test data.
    Model hyperparameters are fixed, as disclosed in models.py.
    """
    names = list(dict.fromkeys(models))
    if not names or any(name not in AVAILABLE for name in names):
        raise ValueError(f'Unknown model(s). Options: {AVAILABLE}')
    if top_genes < 2 or min_cells < 1 or n_bootstraps < 100 or transformer_epochs < 1:
        raise ValueError('Require top_genes>=2, min_cells>=1, bootstrap>=100, epochs>=1')
    input_path, out = Path(input_path), Path(output_dir)
    data = load(input_path, donor_col, label_col, counts_layer)
    donor_values = sorted(set(data.donor_ids))
    global_labels = sorted(set(data.labels))
    donor_rows, class_rows, prediction_rows, folds = [], [], [], []

    for fold_no, held_out in enumerate(donor_values):
        test_mask = data.donor_ids == held_out
        train_mask = ~test_mask
        train_idx, test_idx = np.flatnonzero(train_mask), np.flatnonzero(test_mask)
        train_donors = sorted(set(data.donor_ids[train_idx]))
        assert held_out not in train_donors and set(train_idx).isdisjoint(set(test_idx))
        y_train, y_test = data.labels[train_idx], data.labels[test_idx]
        if len(set(y_train)) < 2:
            raise ValueError(f'Fold {held_out}: training donors have only one cell type')
        selected = TrainOnlyHVG(top_genes, min_cells)
        Xtr = selected.fit_transform(take_rows(data.X, train_idx))
        Xte = selected.transform(take_rows(data.X, test_idx))
        unseen_labels = sorted(set(y_test) - set(y_train))
        folds.append(dict(held_out_donor=held_out, train_donors=train_donors,
                          n_train_cells=int(len(train_idx)), n_test_cells=int(len(test_idx)),
                          n_train_genes=int(Xtr.shape[1]),
                          selected_genes=data.gene_names[selected.selected_indices_].tolist(),
                          train_label_counts={str(k): int(v) for k,v in zip(*np.unique(y_train, return_counts=True))},
                          test_only_classes=unseen_labels,
                          train_only_fit=True))
        for model_idx, model_name in enumerate(names):
            # Different but deterministic seeds per model/fold.
            model_seed = seed + fold_no * 1009 + model_idx * 7919
            pred, confidence = predict_model(model_name, Xtr, y_train, Xte,
                                             seed=model_seed, transformer_epochs=transformer_epochs)
            acc = float(accuracy_score(y_test, pred))
            macro = float(f1_score(y_test, pred, labels=sorted(set(y_test)),
                                   average='macro', zero_division=0))
            bal = float(recall_score(y_test, pred, labels=sorted(set(y_test)),
                                     average="macro", zero_division=0))
            donor_rows.append(dict(model=model_name, held_out_donor=held_out,
                                   n_test_cells=len(test_idx), accuracy=acc,
                                   macro_f1_present=macro, balanced_accuracy=bal,
                                   n_test_classes=len(set(y_test)), n_unseen_test_classes=len(unseen_labels)))
            p, r, f, support = precision_recall_fscore_support(
                y_test, pred, labels=global_labels, zero_division=0)
            for ix, label in enumerate(global_labels):
                if support[ix] > 0:
                    class_rows.append(dict(model=model_name, held_out_donor=held_out,
                                           cell_type=label, support=int(support[ix]),
                                           precision=float(p[ix]), recall=float(r[ix]), f1=float(f[ix]),
                                           class_missing_from_train=label not in set(y_train)))
            for j, cell_idx in enumerate(test_idx):
                prediction_rows.append(dict(model=model_name, cell_id=str(data.cell_ids[cell_idx]),
                                            held_out_donor=held_out, true_label=str(y_test[j]),
                                            predicted_label=str(pred[j]), correct=bool(y_test[j] == pred[j]),
                                            confidence=float(confidence[j])))

    dm = pd.DataFrame(donor_rows)
    cm = pd.DataFrame(class_rows)
    pr = pd.DataFrame(prediction_rows)
    summary_models = {}
    for m in names:
        donor_subset = dm.loc[dm.model == m]
        pred_subset = pr.loc[pr.model == m]
        ci_lo, ci_hi = _bootstrap_mean_ci(donor_subset.accuracy.to_numpy(),
                                         np.random.default_rng(seed+names.index(m)), n_bootstraps)
        summary_models[m] = dict(
            n_donors=int(len(donor_subset)), n_test_predictions=int(len(pred_subset)),
            mean_donor_accuracy=float(donor_subset.accuracy.mean()),
            std_donor_accuracy=float(donor_subset.accuracy.std(ddof=1)),
            worst_donor_accuracy=float(donor_subset.accuracy.min()),
            mean_donor_macro_f1_present=float(donor_subset.macro_f1_present.mean()),
            pooled_cell_accuracy=float(pred_subset.correct.mean()),
            bootstrap_95pct_interval_mean_donor_accuracy=[ci_lo, ci_hi])
    sha = sha256_file(input_path)
    summary = dict(schema_version='1', project='DonorShift-Bench', version=__version__,
                   generated_utc=datetime.now(timezone.utc).isoformat(),
                   input_file=input_path.name, input_sha256=sha,
                   synthetic_demo=bool(input_path.name.startswith('synthetic_')),
                   n_cells=int(len(data.labels)), n_genes=int(len(data.gene_names)),
                   n_donors=len(donor_values), n_classes=len(global_labels),
                   donors=donor_values, classes=global_labels, models=summary_models,
                   evaluation='leave-one-donor-out', preprocessing='per-fold train-only dispersion HVG, total-count/10k log1p',
                   model_hyperparameters='Fixed; see src/donorshift/models.py; not tuned on held-out donors',
                   confidence_note='Maximum predicted class probability; uncalibrated, not clinical confidence',
                   bootstrap_note='Donor-cluster percentile bootstrap across held-out donor accuracies; descriptive',
                   limitations=['Synthetic demo is not a scientific benchmark',
                                'Single held-out donor folds do not establish prospective clinical validity',
                                'Macro F1 uses cell types present in each test donor',
                                'Prediction confidence is uncalibrated',
                                'HVG heuristic is not Seurat or Scanpy batch-aware HVG'])
    out.mkdir(parents=True, exist_ok=True)
    dm.to_csv(out / 'per_donor_metrics.csv', index=False, float_format='%.6f')
    cm.to_csv(out / 'per_class_metrics.csv', index=False, float_format='%.6f')
    pr.to_csv(out / 'predictions.csv', index=False, float_format='%.6f')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'fold_manifest.json').write_text(json.dumps(folds, indent=2, ensure_ascii=False), encoding='utf-8')
    env = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
               sklearn=sklearn.__version__, scipy=scipy.__version__, donorshift=__version__, seed=seed,
               models=names, top_genes=top_genes, min_cells=min_cells,
               bootstrap_draws=n_bootstraps, transformer_epochs=transformer_epochs)
    (out / 'run_config.json').write_text(json.dumps(env, indent=2), encoding='utf-8')
    if make_html:
        make_report(out, dm, cm, summary)
    return summary
