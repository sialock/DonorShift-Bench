"""Command-line interface: donorshift make-demo / validate / run."""
import argparse
import json
import sys

from .benchmark import benchmark
from .data import load
from .models import AVAILABLE
from .synthetic import save_synthetic


def build_parser():
    parser = argparse.ArgumentParser(prog='donorshift',
        description='Patient/donor-held-out scRNA cell-type benchmark with train-only preprocessing')
    sub = parser.add_subparsers(dest='command', required=True)
    demo = sub.add_parser('make-demo', help='Generate entirely synthetic example raw counts')
    demo.add_argument('--output', default='examples/data/synthetic_sc_counts.csv')
    demo.add_argument('--donors', type=int, default=8)
    demo.add_argument('--cells-per-donor', type=int, default=48)
    demo.add_argument('--genes', type=int, default=64)
    demo.add_argument('--seed', type=int, default=7)
    for cmd in ('run', 'validate'):
        sp = sub.add_parser(cmd, help=('Run donor-held-out benchmark' if cmd == 'run' else 'Validate raw count input'))
        sp.add_argument('--input', required=True)
        sp.add_argument('--donor-col', default='donor_id')
        sp.add_argument('--label-col', default='cell_type')
        sp.add_argument('--counts-layer', default=None, help='h5ad raw counts layer; X used if omitted')
        if cmd == 'run':
            sp.add_argument('--output', default='results/benchmark')
            sp.add_argument('--models', nargs='+', choices=AVAILABLE, default=['logreg', 'random_forest'])
            sp.add_argument('--top-genes', type=int, default=48)
            sp.add_argument('--min-cells', type=int, default=3)
            sp.add_argument('--seed', type=int, default=42)
            sp.add_argument('--bootstrap', type=int, default=1000)
            sp.add_argument('--transformer-epochs', type=int, default=10)
            sp.add_argument('--no-html', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == 'make-demo':
            path = save_synthetic(args.output, n_donors=args.donors,
                                  cells_per_donor=args.cells_per_donor,
                                  n_genes=args.genes, seed=args.seed)
            print(f'Synthetic demo generated: {path}')
        elif args.command == 'validate':
            data = load(args.input, args.donor_col, args.label_col, args.counts_layer)
            print(json.dumps(dict(valid=True, n_cells=len(data.labels), n_genes=len(data.gene_names),
                                  n_donors=len(set(data.donor_ids)), n_cell_types=len(set(data.labels))), indent=2))
        elif args.command == 'run':
            s = benchmark(args.input, args.output, models=args.models,
                          top_genes=args.top_genes, min_cells=args.min_cells,
                          seed=args.seed, n_bootstraps=args.bootstrap,
                          donor_col=args.donor_col, label_col=args.label_col,
                          counts_layer=args.counts_layer,
                          transformer_epochs=args.transformer_epochs, make_html=not args.no_html)
            print(f"Benchmark complete: {args.output}/report.html")
            print(json.dumps(s['models'], indent=2))
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())