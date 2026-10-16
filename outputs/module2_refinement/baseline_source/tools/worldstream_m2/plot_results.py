"""Reproducible data-driven M2(a-g) figures; controlled and real evidence stay separate."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('outputs/module2')


def read(name):
    return json.loads((ROOT/name).read_text())


def draw(figure):
    plt.rcParams.update({'font.family':'serif','font.size':10,'figure.dpi':150,
        'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False,
        'savefig.bbox':'tight','savefig.pad_inches':.08,'mathtext.fontset':'stix'})
    rows=read('controlled_sparse_benchmark.json')['rows']
    ratio=np.array([r['ratio'] for r in rows])*100
    if figure=='a':
        trace=read('dependency_trace.json')
        values=[trace['input_patch_mask']]+[trace['stages'][i]['camera0_tokens'] for i in (0,1,2)]
        fig,axes=plt.subplots(1,4,figsize=(9,2.5))
        for ax,value,label in zip(axes,values,['Patch invalidation','Block 0: window','Block 1: window','Block 2: global']):
            ax.imshow(value,vmin=0,vmax=1,cmap='Blues',interpolation='nearest')
            ax.set_xlabel('Token column\n'+label); ax.set_xticks([0,16,32,39]); ax.set_yticks([0,16,32,39])
        axes[0].set_ylabel('Camera 0 token row')
    elif figure=='b':
        row=next(r for r in rows if r['updated']==8)
        ids=row['ids']; original=np.zeros(54); original[ids]=1
        fig,axes=plt.subplots(3,1,figsize=(8,4),gridspec_kw={'height_ratios':[2,1,2]})
        axes[0].imshow(original.reshape(6,9),vmin=0,vmax=1,cmap='Blues',aspect='auto')
        axes[0].set_ylabel('Camera'); axes[0].set_xlabel('Original window ID within each camera')
        axes[1].imshow(np.ones((1,len(ids))),vmin=0,vmax=1,cmap='Blues',aspect='auto')
        for i,ident in enumerate(ids): axes[1].text(i,0,str(ident),ha='center',va='center',color='white')
        axes[1].set_yticks([]); axes[1].set_xticks(range(len(ids)))
        axes[1].set_xlabel('Packed index; white labels are global window IDs')
        axes[2].imshow(original.reshape(6,9),vmin=0,vmax=1,cmap='Blues',aspect='auto')
        for j,ident in enumerate(ids):
            camera,window=divmod(ident,9)
            axes[2].text(window,camera,str(j),ha='center',va='center',color='white')
        axes[2].set_ylabel('Camera'); axes[2].set_xlabel('Scatter destinations; white labels are packed indices')
    elif figure in ('c','d'):
        fig,ax=plt.subplots(figsize=(3.5,2.9))
        full=np.array([r['full']['gpu_ms'] for r in rows])
        for method,style,label in [('full','--','Full dense'),('naive',':','Naive sparse'),('packed','-','Packed sparse')]:
            y=np.array([r[method]['gpu_ms'] for r in rows])
            if figure=='d': y=full/y
            ax.plot(ratio,y,style,marker='o',markersize=3,label=label)
        if figure=='d':
            ax.axhline(1,color='gray',linewidth=.8)
            good=[r['ratio']*100 for r in rows if r['packed']['gpu_ms']<r['full']['gpu_ms']]
            bad=[r['ratio']*100 for r in rows if r['packed']['gpu_ms']>=r['full']['gpu_ms']]
            if good and bad: ax.axvspan(max(good),min(bad),color='gray',alpha=.12,label='Packed crossing')
            ax.set_ylabel('Block speedup (dense / execution)')
            ax.set_yscale('log')
        else: ax.set_ylabel('Block GPU interval (ms)')
        ax.set_xlabel('Updated attention windows (%)'); ax.legend(frameon=False,fontsize=8)
    elif figure=='e':
        profile=read('packing_profile.json'); counts=[1,27,54]
        fig,ax=plt.subplots(figsize=(3.5,2.9)); bottom=np.zeros(3)
        for hatch,(key,label) in zip(['//','..','xx','--'],[
                          ('pack_ms','Pack + validity'),('sparse_ms','Attention + MLP'),
                          ('scatter_ms','Cache + scatter + layout'),
                          ('synchronization_and_host_residual_ms','Host / sync residual')]):
            values=np.array([profile[str(k)]['components'][key] for k in counts])
            ax.bar([str(k) for k in counts],values,bottom=bottom,label=label,hatch=hatch); bottom+=values
        ax.set_xlabel('Updated windows out of 54'); ax.set_ylabel('Component cost (ms)')
        ax.legend(frameon=False,fontsize=8,loc='lower left',bbox_to_anchor=(0,1.02))
    elif figure=='f':
        fig,axes=plt.subplots(1,2,figsize=(8,3.3))
        axes[0].plot(100-ratio,[r['error']['max_abs'] for r in rows],'-o',markersize=3)
        axes[0].set_xlabel('Attention sequences skipped in block (%)'); axes[0].set_ylabel('Block maximum absolute error')
        bb=read('backbone_benchmark.json')['rows']; check=read('correctness_report.json')['rows']
        for r,c in zip(bb,check):
            saved=r['diagnostics']['actual_skipped_windows']
            axes[1].scatter(saved,r['error']['relative_l2'],marker='o' if c['passed'] else 'x',color='tab:blue')
            axes[1].annotate(r['label'],(saved,r['error']['relative_l2']),fontsize=7,xytext=(3,3),textcoords='offset points')
        axes[1].set_xlabel('Early window evaluations skipped (blocks 0 + 1)')
        axes[1].set_ylabel('Complete EVA relative L2 error')
        axes[1].set_xlim(-5,130)
    elif figure=='g':
        real=read('real_m1_results.json')
        fs=[f for f in real['frames'] if f['mode']=='m1_m2_exact' and f['repeat']==1]
        fig,axes=plt.subplots(1,3,figsize=(9,3.3))
        labels=[str(i) for i in range(len(fs))]
        axes[0].bar(labels,[sum(v['complete_patch_candidates'] for v in f['m1_candidates'].values()) for f in fs])
        axes[0].set_ylabel('Complete 16-pixel patch candidates'); axes[0].set_xlabel('Continuous frame')
        axes[1].bar(labels,[sum(v['complete_attention_windows'] for v in f['m1_candidates'].values()) for f in fs])
        for i,f in enumerate(fs):
            count=sum(v['complete_attention_windows'] for v in f['m1_candidates'].values())
            axes[1].text(i,count+.5,str(count),ha='center',va='bottom')
        axes[1].set_ylim(0,54); axes[1].set_ylabel('16-token attention-window candidates'); axes[1].set_xlabel('Continuous frame')
        s=read('e2e_latency.json')['summary']; modes=list(s)
        values=[s[m]['complete_wall_ms'][0] for m in modes]
        upper=[s[m]['complete_wall_ms'][1]-s[m]['complete_wall_ms'][0] for m in modes]
        axes[2].bar(['Base','M1','Exact','Auto'],values,yerr=[np.zeros(len(values)),upper],capsize=3)
        axes[2].set_ylabel('Complete VLM wall P50 (ms)'); axes[2].tick_params(axis='x',rotation=20)
    else: raise ValueError(figure)
    fig.tight_layout()
    name=f'm2_{figure}'
    fig.savefig(ROOT/'figures'/f'{name}.pdf'); fig.savefig(ROOT/'figures'/f'{name}.png')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--figure',choices=list('abcdefg'))
    args=p.parse_args()
    for fig in [args.figure] if args.figure else 'abcdefg': draw(fig)


if __name__=='__main__':main()
