"""Static exportable figures from saved measured artifacts."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='outputs/module1')
    args = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(args.root)
    out = root/'figures'; out.mkdir(exist_ok=True)
    profile = json.loads((root/'profile.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for method in ('normalized_gray', 'gradient'):
        rows = [r for r in profile['results'] if r['descriptor'] == method]
        axes[0].errorbar([r['anchor_limit'] for r in rows], [r['mean_cpu_wall_ms']['total'] for r in rows],
                         yerr=[r['std_total_ms'] for r in rows], marker='o', label=method)
        axes[1].plot([r['anchor_limit'] for r in rows],
                     [r['state_and_temporary_traced_peak_bytes']/1024**2 for r in rows], marker='o', label=method)
    axes[0].set(xlabel='Historical anchors', ylabel='M1 CPU overhead (ms)', title='Real two-frame microbenchmark')
    axes[1].set(xlabel='Historical anchors', ylabel='Traced peak memory (MiB)', title='Separate allocation pass')
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'overhead.png', dpi=180); fig.savefig(out/'overhead.pdf'); plt.close(fig)
    names = ['replay_strict', 'replay_appearance_ablation', 'replay_gradient']
    labels = ['Strict gray', 'Appearance ablation', 'Strict gradient']
    rates = []
    for name in names:
        metrics = json.loads((root/name/'validity_metrics.json').read_text())
        rates.append(metrics['methods']['geometry_observation']['valid_candidate_rate'])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(labels, rates, color=['#426ba5', '#cf9a42', '#548c67'])
    ax.set(ylabel='Physical candidate rate', ylim=(0, 1), title='Observation candidates only; no GPU work skipped')
    fig.tight_layout(); fig.savefig(out/'candidate_rates.png', dpi=180); fig.savefig(out/'candidate_rates.pdf'); plt.close(fig)
    print(out)


if __name__ == '__main__': main()
