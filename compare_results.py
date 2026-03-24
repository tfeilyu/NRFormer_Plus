#!/usr/bin/env python
"""
Compare all experiment results across runs.

Usage:
    python compare_results.py                    # Show all results
    python compare_results.py --dataset 1D-data  # Filter by dataset
    python compare_results.py --top 5            # Show top 5 by test MAE
"""
import json
import os
import argparse
from pathlib import Path

def load_all_results(base_dir='logs', model_name='NRFormer_Plus', dataset=None):
    results = []
    base = Path(base_dir) / model_name

    if not base.exists():
        print(f"No results found in {base}")
        return results

    for dataset_dir in sorted(base.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if dataset and dataset_dir.name != dataset:
            continue

        for run_dir in sorted(dataset_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            results_file = run_dir / 'results.json'
            config_file = run_dir / 'config.json'

            if not results_file.exists():
                continue

            with open(results_file) as f:
                res = json.load(f)

            config = {}
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)

            entry = {
                'run_id': run_dir.name,
                'dataset': dataset_dir.name,
                'model_des': config.get('model_des', '?'),
                'hidden': config.get('hidden_channels', '?'),
                'temp_layers': config.get('num_temporal_att_layer', '?'),
                'spat_layers': config.get('num_spatial_att_layer', '?'),
                'batch_size': config.get('batch_size', '?'),
                'dropout': config.get('temporal_dropout', '?'),
                'ffn_ratio': config.get('ffn_ratio', '?'),
                'spatial_heads': config.get('spatial_heads', '?'),
                'num_params': config.get('num_params', '?'),
                'best_epoch': res.get('best_epoch', '?'),
                'valid_MAE': res['valid']['MAE'],
                'valid_RMSE': res['valid']['RMSE'],
                'test_MAE': res['test']['MAE'],
                'test_RMSE': res['test']['RMSE'],
                'test_MAPE': res['test']['MAPE'],
            }

            # Per-horizon results
            for step, metrics in res.get('per_horizon', {}).items():
                entry[f'{step}_MAE'] = metrics.get('MAE', '-')

            results.append(entry)

    return results


def print_comparison(results, top_n=None):
    if not results:
        print("No experiment results found.")
        return

    # Sort by test MAE
    results.sort(key=lambda x: x['test_MAE'])

    if top_n:
        results = results[:top_n]

    # Header
    print("\n" + "=" * 140)
    print(f"{'Experiment':<30} {'Dataset':<8} {'H':>3} {'TL':>2} {'SL':>2} {'BS':>3} {'Params':>8} "
          f"{'Epoch':>5} {'V-MAE':>7} {'T-MAE':>7} {'T-RMSE':>8} {'T-MAPE':>8}")
    print("=" * 140)

    for r in results:
        params_str = f"{r['num_params']/1e6:.2f}M" if isinstance(r['num_params'], (int, float)) else '?'
        mape_str = f"{r['test_MAPE']:.2f}%" if isinstance(r['test_MAPE'], (int, float)) else '?'
        print(f"{r['model_des']:<30} {r['dataset']:<8} {str(r['hidden']):>3} {str(r['temp_layers']):>2} "
              f"{str(r['spat_layers']):>2} {str(r['batch_size']):>3} {params_str:>8} "
              f"{str(r['best_epoch']):>5} {r['valid_MAE']:>7.4f} {r['test_MAE']:>7.4f} "
              f"{r['test_RMSE']:>8.4f} {mape_str:>8}")

    print("=" * 140)

    # Per-horizon breakdown for best run
    best = results[0]
    horizon_keys = [k for k in best.keys() if k.startswith('step_') and k.endswith('_MAE')]
    if horizon_keys:
        print(f"\nBest run [{best['model_des']}] per-horizon MAE:")
        for k in sorted(horizon_keys):
            print(f"  {k}: {best[k]:.4f}" if isinstance(best[k], float) else f"  {k}: {best[k]}")

    print(f"\nTotal experiments: {len(results)}")
    print(f"Best test MAE: {results[0]['test_MAE']:.4f} ({results[0]['model_des']})")


def save_comparison_csv(results, output_path='logs/experiment_comparison.csv'):
    """Save results to CSV for further analysis."""
    import csv
    if not results:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    keys = results[0].keys()
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nComparison CSV saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None, help='Filter by dataset (1D-data or 4H-data)')
    parser.add_argument('--top', type=int, default=None, help='Show top N results')
    args = parser.parse_args()

    results = load_all_results(dataset=args.dataset)
    print_comparison(results, top_n=args.top)
    save_comparison_csv(results)
