"""Publication exports from measured receipts only; missing groups stay blank."""
from pathlib import Path
import numpy as np

from .common import LEVELS, METHODS


def plot_results(out, latency_rows, accuracy_rows, paired, smoke=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rc_context
    directory = Path(out) / 'figures'
    directory.mkdir(exist_ok=False)
    outputs = []
    labels = dict(native='Native VLM', worldstream='WorldStream')
    colors = dict(native='#6B7890', worldstream='#168C8C')
    with rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300}):
        def save(fig, name):
            if smoke:
                fig.suptitle('SMOKE TEST — execution validation only', fontsize=10)
            fig.tight_layout()
            for suffix in ('pdf', 'png'):
                path = directory / (name + '.' + suffix)
                fig.savefig(path, bbox_inches='tight')
                outputs.append(str(path.relative_to(out)))
            plt.close(fig)

        def bars(ax, rows, key, title, ylabel):
            lookup = {(r['method'], r['stability_level']): r for r in rows}
            any_data = False
            for i, method in enumerate(METHODS):
                positions, values = [], []
                for j, level in enumerate(LEVELS):
                    value = lookup.get((method, level), {}).get(key)
                    if value is not None:
                        positions.append(j + (i - .5) * .36)
                        values.append(value)
                if values:
                    any_data = True
                    ax.bar(positions, values, width=.34, color=colors[method], label=labels[method])
                else:
                    ax.plot([], [], color=colors[method], linewidth=6, label=labels[method])
            ax.set_xticks(np.arange(4), LEVELS)
            ax.set_xlabel('Environment stability (GT semantic change rate)')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(axis='y', alpha=.18)
            ax.set_axisbelow(True)
            if not any_data:
                ax.text(.5, .5, 'No evaluable measurements', ha='center', transform=ax.transAxes)
            ax.legend(frameon=False)

        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        bars(axes[0], latency_rows, 'average_e2e_ms', 'Average E2E latency', 'Latency (ms)')
        bars(axes[1], latency_rows, 'p95_e2e_ms', 'P95 E2E latency', 'Latency (ms)')
        save(fig, 'e2e_latency')
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars(ax, accuracy_rows, 'overall_accuracy_percent', 'Factual QA accuracy', 'Accuracy (%)')
        ax.set_ylim(0, 105)
        save(fig, 'factual_qa_accuracy')
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        bars(axes[0], accuracy_rows, 'stable_fact_accuracy_percent', 'Stable facts', 'Accuracy (%)')
        bars(axes[1], accuracy_rows, 'changed_fact_accuracy_percent', 'Changed facts', 'Accuracy (%)')
        for ax in axes:
            ax.set_ylim(0, 105)
        save(fig, 'stable_changed_accuracy')
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars(ax, accuracy_rows, 'stale_answer_rate_percent', 'Stale answers after GT changes', 'Stale answer rate (%)')
        ax.set_ylim(0, 105)
        save(fig, 'stale_answer_rate')
        fig, ax = plt.subplots(figsize=(5, 3.5))
        complete = [w for w in paired if w['complete_measured_window'] and w['stability_level'] in LEVELS
                    and w['change_rate'] is not None]
        for level, marker in zip(LEVELS, ('o', 's', '^', 'D')):
            part = [w for w in complete if w['stability_level'] == level]
            if part:
                ax.scatter([w['change_rate'] for w in part], [w['speedup'] for w in part],
                           label=level, marker=marker, s=24, alpha=.7)
        ax.axhline(1., color='#777777', linestyle='--', linewidth=.8)
        ax.set_xlabel('GT semantic change rate')
        ax.set_ylabel('Native / WorldStream mean E2E latency')
        if complete:
            ax.legend(frameon=False)
        else:
            ax.text(.5, .5, 'No complete paired windows', ha='center', transform=ax.transAxes)
        save(fig, 'semantic_change_vs_speedup')
    return outputs
