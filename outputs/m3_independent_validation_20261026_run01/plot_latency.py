"""Plot every paired round and the pooled scene-cluster interval."""
import csv
import json
from pathlib import Path


def plot(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root)
    s=json.loads((root/'comparison_summary.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),sharey=True)
    exported=[]
    for ax,key,title in zip(axes,('prefill','m3','end_to_end'),('Prefill / question','M3 generation / frame','Complete request / frame')):
        values=[(f"Round {r['repeat']+1}",r['paired_latency'][key]) for r in s['repeats_summary']]
        values.append(('All rounds',s['paired_latency'][key]))
        for y,(name,a) in enumerate(values):
            ci=a['ci95_percent'];color='#195c91' if name!='All rounds' else '#111111'
            if ci is not None:ax.hlines(y,ci[0],ci[1],color=color,linewidth=2)
            ax.plot(a['change_percent'],y,'o',color=color,markersize=5)
            exported.append(dict(metric=key,round=name,measurements=a['n'],scenes=a['scenes'],
                change_percent=a['change_percent'],ci95_lower_percent=ci[0] if ci else None,
                ci95_upper_percent=ci[1] if ci else None))
        ax.axvline(0,color='#b04440',linestyle='--',linewidth=1)
        ax.set_title(title,fontsize=10)
        ax.set_xlabel('WSDR vs native latency change (%)')
        ax.set_yticks(range(len(values)));ax.set_yticklabels([n for n,_ in values])
        ax.grid(axis='x',alpha=.2);ax.invert_yaxis()
    fig.suptitle(f"Frozen policy: {s['independent_scenes']} independent scenes, {s['unique_frames']} frames, "
        f"{s['unique_questions']} unique questions",fontsize=11)
    fig.text(.5,.01,'Negative = faster. Bars: paired scene-cluster bootstrap 95% intervals. Repeats do not multiply independent scenes.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,.93])
    fig.savefig(root/'paired_latency_stability.svg')
    fig.savefig(root/'paired_latency_stability.png',dpi=180)
    plt.close(fig)
    with (root/'paired_latency_stability.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(exported[0]));writer.writeheader();writer.writerows(exported)


if __name__=='__main__':plot(Path(__file__).resolve().parent)
