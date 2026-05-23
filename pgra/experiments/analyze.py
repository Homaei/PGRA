"""
Post-hoc analysis script: takes the latest main_<dataset>_<ts>/all_runs.csv
files, produces summary plots and a Markdown report suitable for the
manuscript's Results section.
"""
import os
import glob
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


RESULTS_ROOT = os.path.join(os.path.dirname(__file__), '..', 'results')


def _latest(prefix):
    paths = sorted(glob.glob(os.path.join(RESULTS_ROOT, f'{prefix}_*')))
    return paths[-1] if paths else None


def plot_main(df, dataset, out_dir):
    summary = (df.groupby('Aggregator')[['F1', 'ASR']]
                 .agg(['mean', 'std']).round(4))
    order = ['CentralizedOracle', 'FedAvg', 'PGRA', 'Krum', 'CoordMedian',
             'FLTrust', 'FLAME', 'FLAIR', 'Sine', 'FedRoLA', 'RFL-APIA']
    order = [a for a in order if a in summary.index]
    summary = summary.loc[order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(order))

    f1_mean = summary[('F1', 'mean')].values
    f1_std = summary[('F1', 'std')].values
    axes[0].bar(x, f1_mean, yerr=f1_std, capsize=4,
                 color=['#4c72b0' if a == 'PGRA' else '#aaaaaa' for a in order])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(order, rotation=45, ha='right')
    axes[0].set_ylabel('F1-Score')
    axes[0].set_title(f'{dataset.upper()} — F1 under physics-aware attack')
    axes[0].grid(axis='y', alpha=0.3)

    asr_mean = summary[('ASR', 'mean')].values
    asr_std = summary[('ASR', 'std')].values
    axes[1].bar(x, asr_mean, yerr=asr_std, capsize=4,
                 color=['#c44e52' if a == 'PGRA' else '#aaaaaa' for a in order])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order, rotation=45, ha='right')
    axes[1].set_ylabel('Attack Success Rate (ASR)')
    axes[1].set_title(f'{dataset.upper()} — ASR (lower is better)')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{dataset}_main.png'), dpi=130)
    plt.close()
    return summary


def plot_ablation_alpha(df, out_dir):
    if df is None or len(df) == 0:
        return
    summary = (df.groupby('alpha')[['F1', 'ASR']]
                 .agg(['mean', 'std']).round(4))
    fig, ax = plt.subplots(figsize=(8, 5))
    x = summary.index.values
    ax.errorbar(x, summary[('F1', 'mean')], yerr=summary[('F1', 'std')],
                 label='F1', marker='o', capsize=4)
    ax.errorbar(x, summary[('ASR', 'mean')], yerr=summary[('ASR', 'std')],
                 label='ASR', marker='s', capsize=4)
    ax.set_xscale('log')
    ax.set_xlabel(r'PGRA sensitivity $\alpha$')
    ax.set_ylabel('metric')
    ax.set_title(r'Ablation A: $\alpha$ sweep')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ablation_alpha.png'), dpi=130)
    plt.close()


def plot_ablation_eps_s(df, out_dir):
    if df is None or len(df) == 0:
        return
    summary = (df.groupby('epsilon_s')[['F1', 'ASR']]
                 .agg(['mean', 'std']).round(4))
    fig, ax = plt.subplots(figsize=(8, 5))
    x = summary.index.values
    ax.errorbar(x, summary[('F1', 'mean')], yerr=summary[('F1', 'std')],
                 label='F1', marker='o', capsize=4)
    ax.errorbar(x, summary[('ASR', 'mean')], yerr=summary[('ASR', 'std')],
                 label='ASR', marker='s', capsize=4)
    ax.set_xlabel(r'attacker $\varepsilon_s$ (stealthiness budget)')
    ax.set_ylabel('metric')
    ax.set_title(r'Ablation B: $\varepsilon_s$ sweep')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ablation_eps_s.png'), dpi=130)
    plt.close()


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(RESULTS_ROOT, f'analysis_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    md = [f"# PGRA Results — Analysis\n_{timestamp}_\n"]

    for ds in ['batadal', 'wadi']:
        latest = _latest(f'main_{ds}')
        if latest is None:
            continue
        csv = os.path.join(latest, 'all_runs.csv')
        if not os.path.exists(csv):
            continue
        df = pd.read_csv(csv)
        summary = plot_main(df, ds, out_dir)
        summary.to_csv(os.path.join(out_dir, f'main_{ds}_summary.csv'))
        md.append(f"\n## Main comparison — {ds.upper()}\n")
        md.append("Source: " + os.path.relpath(latest, RESULTS_ROOT) + "\n\n")
        md.append("```\n" + summary.to_string() + "\n```\n")

    # Ablation
    latest = _latest('ablation_batadal')
    if latest is not None:
        for csv_name, plot_fn in [
            ('alpha_sweep.csv', plot_ablation_alpha),
            ('epsilon_s_sweep.csv', plot_ablation_eps_s),
        ]:
            csv = os.path.join(latest, csv_name)
            if os.path.exists(csv):
                df = pd.read_csv(csv)
                plot_fn(df, out_dir)
                md.append(f"\n### Ablation: {csv_name}\n")
                key = 'alpha' if 'alpha' in csv_name else 'epsilon_s'
                agg = (df.groupby(key)[['F1', 'ASR']]
                         .agg(['mean', 'std']).round(4))
                agg.to_csv(os.path.join(out_dir, csv_name))
                md.append("```\n" + agg.to_string() + "\n```\n")

        # beta_mode
        csv = os.path.join(latest, 'beta_mode.csv')
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            md.append("\n### Ablation: beta_mode\n")
            agg = (df.groupby('beta_mode')[['F1', 'ASR']]
                     .agg(['mean', 'std']).round(4))
            agg.to_csv(os.path.join(out_dir, 'beta_mode.csv'))
            md.append("```\n" + agg.to_string() + "\n```\n")

    # Stealthiness
    for ds in ['batadal', 'wadi']:
        latest = _latest(f'stealthiness_{ds}')
        if latest is None:
            continue
        csv = os.path.join(latest, 'stealthiness.csv')
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            md.append(f"\n## Stealthiness — {ds.upper()}\n")
            md.append("```\n" + df.to_string(index=False) + "\n```\n")

    with open(os.path.join(out_dir, 'REPORT.md'), 'w') as f:
        f.write('\n'.join(md))
    print(f"Analysis written to {out_dir}")
    return out_dir


if __name__ == '__main__':
    main()
