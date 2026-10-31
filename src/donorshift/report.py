"""Portable, offline HTML report and publication-ready PNG charts."""
import base64
from html import escape
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _fig_png(fig, filepath: Path):
    fig.savefig(filepath, dpi=180, bbox_inches='tight', facecolor='white')
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def make_report(out: Path, donor_df: pd.DataFrame, class_df: pd.DataFrame, summary: dict):
    names = sorted(summary['models'])
    donors = summary['donors']
    fig, ax = plt.subplots(figsize=(max(8, len(donors) * 0.85), 4.2))
    x = np.arange(len(donors))
    width = 0.76 / len(names)
    for i, name in enumerate(names):
        df = donor_df[donor_df.model == name].set_index('held_out_donor').reindex(donors)
        ax.bar(x + (i-(len(names)-1)/2)*width, df.accuracy, width, label=name)
    ax.set_xticks(x, donors, rotation=30)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Held-out donor accuracy')
    ax.set_title('Per-donor generalization: each donor withheld in turn')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    accuracy_img = _fig_png(fig, out / 'donor_accuracy.png')

    labels = sorted(summary['classes'])
    fig2, ax2 = plt.subplots(figsize=(max(7, len(labels) * 1.1), 4.4))
    mat = np.full((len(names), len(labels)), np.nan)
    for mi, name in enumerate(names):
        for li, label in enumerate(labels):
            r = class_df[(class_df.model == name) & (class_df.cell_type == label)]
            if not r.empty:
                mat[mi,li] = np.average(r.recall.to_numpy(), weights=r.support.to_numpy())
    image = ax2.imshow(mat, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax2.set_xticks(np.arange(len(labels)), labels, rotation=25, ha='right')
    ax2.set_yticks(np.arange(len(names)), names)
    for i in range(len(names)):
        for j in range(len(labels)):
            if np.isfinite(mat[i,j]):
                ax2.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',
                         color='white' if mat[i,j] < 0.7 else 'black',fontsize=9)
    fig2.colorbar(image, ax=ax2, label='Support-weighted per-class recall')
    ax2.set_title('Cell-type recall across test donors')
    fig2.tight_layout()
    recall_img = _fig_png(fig2, out / 'class_recall.png')

    html_rows = []
    for name in names:
        m = summary['models'][name]
        html_rows.append('<tr><td>'+escape(name)+'</td><td>'+
                         f"{m['mean_donor_accuracy']:.3f}</td><td>{m['worst_donor_accuracy']:.3f}</td>"+
                         f"<td>{m['mean_donor_macro_f1_present']:.3f}</td>"+
                         f"<td>{m['pooled_cell_accuracy']:.3f}</td></tr>")
    notes = ''.join(f'<li>{escape(s)}</li>' for s in summary['limitations'])
    page = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DonorShift-Bench | {escape(summary['input_file'])}</title>
<style>body{{background:#f6f8fc;color:#14243a;font:16px/1.6 system-ui,sans-serif;margin:0}}
main{{max-width:1080px;margin:32px auto;padding:0 20px}} h1{{font-size:2.2em;margin-bottom:0}} .muted{{color:#53657d}}
header,.panel{{background:white;border:1px solid #e3e8f2;border-radius:14px;padding:22px;margin:20px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:12px}}
.stat{{background:#edf3ff;border-radius:10px;padding:14px}} .stat strong{{font-size:1.5em;display:block}}
img{{width:100%;height:auto}} table{{border-collapse:collapse;width:100%;}}td,th{{padding:11px;border-bottom:1px solid #e4eaf1;text-align:left}}
code{{background:#edf3ff;padding:3px 7px;border-radius:5px}} .pill{{color:#0e634b;font-weight:700}}</style></head>
<body><main><header><span class="pill">DONOR-HELD-OUT BENCHMARK</span><h1>DonorShift-Bench</h1>
<p class="muted">Generated {escape(summary['generated_utc'])} · {escape(summary['input_file'])}</p>
<div class="grid"><div class="stat"><strong>{summary['n_donors']}</strong>donors</div>
<div class="stat"><strong>{summary['n_cells']}</strong>cells</div>
<div class="stat"><strong>{summary['n_genes']}</strong>input genes</div>
<div class="stat"><strong>{summary['n_classes']}</strong>cell types</div></div></header>
<section class="panel"><h2>Model comparison</h2><p class="muted">Macro average gives equal weight to each held-out donor.</p>
<table><thead><tr><th>Model</th><th>Mean donor accuracy</th><th>Worst donor accuracy</th>
<th>Mean test-present macro F1</th><th>Pooled cell accuracy</th></tr></thead>
<tbody>{''.join(html_rows)}</tbody></table></section>
<section class="panel"><h2>Performance by held-out donor</h2><img alt="Donor accuracy" src="data:image/png;base64,{accuracy_img}"></section>
<section class="panel"><h2>Cell-type performance</h2><img alt="Per-class recall" src="data:image/png;base64,{recall_img}"></section>
<section class="panel"><h2>Leakage controls</h2><p>Each donor appears only in testing for its outer fold.
Gene detection, selection and transformations are fit on training donors only. All hyperparameters are fixed, without
optimizing on the held-out donors. See <code>fold_manifest.json</code> for the selected genes in each fold.</p></section>
<section class="panel"><h2>Interpretation and limitations</h2><ul>{notes}</ul>
<p class="muted">Bootstrap intervals in summary.json are descriptive donor-resampling intervals. This report is not a clinical validation.</p></section>
</main></body></html>'''
    (out / 'report.html').write_text(page, encoding='utf-8')