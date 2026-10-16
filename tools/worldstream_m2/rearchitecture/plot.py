"""Publication-style standalone PDF/PNG from measured JSON only."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT


def read(name):return json.loads((ROOT/name).read_text())


def save(fig,name):
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(ROOT/'figures'/f'{name}.{ext}',bbox_inches='tight',dpi=250)
    plt.close(fig)


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    bounds=read('cost_break_even.json');rows=bounds['rows']
    fig,ax=plt.subplots(figsize=(8,3.5))
    labels=[r['scope'].replace('skip_first_','Skip first ').replace('reuse_','Reuse ').replace('_cameras',' cameras') for r in rows]
    ax.barh(labels,[r['saving_upper_bound_wall_ms'] for r in rows]);ax.axvline(bounds['current_validation_medians_ms']['original'],color='firebrick',linestyle='--',label='Current real M1 median')
    ax.set_xlabel('Idealized saving upper bound (ms), excludes validation');ax.legend(frameon=False);save(fig,'feasibility')
    cases=[r for r in read('numerical_raw.json')['cases'] if r['case']=='patch']
    fig,ax=plt.subplots(figsize=(6,3.5))
    for r in cases:
        layers=sorted(r['layers'],key=int)
        ax.semilogy([int(i) for i in layers],[max(r['layers'][i]['C_vs_A']['max_abs'],1e-8) for i in layers],marker='.',label=f"Sparse vs full, seed {r['seed']}")
    ax.set_xlabel('EVA block');ax.set_ylabel('Maximum absolute feature difference');ax.legend(frameon=False);save(fig,'numerical_propagation')
    summary=read('summary.json');modes=['baseline','original','gate_maintain','gate_reset','gate_output','camera_auto']
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    labels=['Full','M1+M2','Gate+M1','Gate+reset','Output only','Camera Auto']
    for ax,key,label in [(axes[0],'complete_wall_ms','Complete VLM wall (ms)'),(axes[1],'m1_ms','M1 wall (ms)')]:
        median=[summary['real_'+m][key]['p50'] for m in modes];p95=[summary['real_'+m][key]['p95'] for m in modes]
        ax.bar(range(6),median,yerr=[np.zeros(6),np.asarray(p95)-median],capsize=3)
        ax.set_xticks(range(6));ax.set_xticklabels(labels,rotation=30,ha='right');ax.set_ylabel(label)
    save(fig,'real_ablation')
    quality=read('state_drift_results.json')['quality']
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    for ax,case in zip(axes,('mixed','repeat')):
        for mode in ('window_exact','camera_exact','camera_auto'):
            rows=[r for r in quality if r['case']==case and r['mode']==mode and r['repeat']==0]
            if not rows:continue
            ordinals=sorted({r['ordinal'] for r in rows})
            values=[max(r['detection'].get('max_center_difference_m') or 0 for r in rows if r['ordinal']==i) for i in ordinals]
            ax.plot(ordinals,values,marker='.',label=mode)
        ax.axhline(.01,color='gray',linestyle='--',label='Frozen 1 cm target')
        ax.set_yscale('symlog',linthresh=.005)
        ax.set_ylim(bottom=0)
        ax.set_title('Controlled '+case+' inputs');ax.set_xlabel('Chronological frame');ax.set_ylabel('Max matched detection shift (m)');ax.legend(fontsize=7,frameon=False)
    save(fig,'state_drift')
    camera=read('pre_numeric_policy_camera_costs.json')['records']
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    axes[0].plot([6-r['changed_cameras'] for r in camera],[r['reuse_wall_ms'] for r in camera],marker='o',label='Actual camera runtime')
    axes[0].plot([6-r['changed_cameras'] for r in camera],[r['full_wall_ms'] for r in camera],linestyle='--',label='Original full')
    axes[0].set_xlabel('Exactly repeated cameras (controlled)');axes[0].set_ylabel('EVA service wall (ms)');axes[0].legend(frameon=False,fontsize=8)
    balanced=read('balanced_camera_summary.json')
    modes=['baseline','camera_auto','camera_auto_m1'];vals=[balanced[m]['wall_ms']['p50'] for m in modes]
    p95=[balanced[m]['wall_ms']['p95'] for m in modes]
    axes[1].bar(['Full','Camera Auto','Auto + M1'],vals,yerr=[np.zeros(3),np.asarray(p95)-vals],capsize=3)
    axes[1].set_ylabel('Repeated-request complete VLM wall (ms)')
    save(fig,'camera_boundary')
    (ROOT/'figures/captions.md').write_text('''All values are read from executed experiment JSON; no theoretical FLOPs converted to latency.

feasibility: idealized current-oracle bypass, 10 measurements after 3 warmups, RTX 3090; not historical reuse evidence. M1 line is this round's six-scene median for the preserved prior M1+M2 code.

numerical_propagation: independent seeds 2801/3509, same patch change, original full vs dependency-sparse execution; no tolerance change. All-window packed comparison separately has zero difference in these inputs.

real_ablation: six independent real driving scenes, 24 unique frames, two measured replays after one warmup; bars P50 with upper P95. Reset/output-only alter WorldState history contract and do not represent continuity-preserving M1 optimizations. Complete heads/temporal state/greedy LLM execute.

state_drift: 16 chronological pose/time steps in each of two scenes. Curves show worst same-class matched centre error over scenes; mixed inputs are synthetic, repeated inputs preserve real varying pose/time. Neither is real camera reuse coverage or broad safety certification.

camera_boundary: left actual runtime cost includes exact validation/cache/assembly, but partial counts remain experimental unless quality qualified. Right final formal Auto and Auto with continuous M1, balanced mode order and two measured replays after one warmup; not real driving benefit.
''')

if __name__=='__main__':main()
