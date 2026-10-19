"""Plot only recorded Development results; undefined risk is never drawn as zero."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
data=json.loads((root/'development_summary.json').read_text())
out=root/'figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})


def save(fig,name):
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/(name+'.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


names=['selective_identity','temporal_identity','identity_no_history','identity_no_competition']
labels=['Selective','Finite history','No history','No competition\nresolution']
values=[data['identity'][n] for n in names];x=np.arange(len(names))
fig,ax=plt.subplots(figsize=(7.5,4))
good=np.array([v['independently_correct_proposals'] for v in values]);unknown=np.array([v['unscorable_proposals'] for v in values]);wrong=np.array([v['wrong_proposals'] for v in values])
ax.bar(x,good,label='Independently correct',color='#24796b')
ax.bar(x,unknown,bottom=good,label='Unscorable',color='#abb6c5')
ax.bar(x,wrong,bottom=good+unknown,label='Wrong',color='#c44e52')
for i,v in enumerate(values):ax.text(i,v['identity_proposals']+3,str(v['identity_proposals']),ha='center')
ax.set_ylim(0,max(v['identity_proposals'] for v in values)*1.18)
ax.set_xticks(x,labels);ax.set_ylabel('Proposed identity links');ax.legend(frameon=False)
ax.set_title('Development only; qualified certifications = 0\nFixed geometric denominator: '+str(values[0]['geometric_candidates']))
save(fig,'identity_coverage')

names=['conservative','dense_original','local_r12','surface_visibility','surface_local_plane','full_refresh']
labels=['Conservative','Dense','Local R12','Three view','Local plane','Full refresh']
values=[data['region'][n] for n in names];x=np.arange(len(names))
fig,axes=plt.subplots(1,2,figsize=(12,4))
axes[0].bar(x,[v['observation_stable_area']*100 for v in values],width=.75,color='#b7cad1',label='Observation')
axes[0].bar(x,[v['effective_stable_area']*100 for v in values],width=.45,color='#31798a',label='Effective after refresh')
for i,v in enumerate(values):axes[0].annotate(f"{v['effective_stable_area']*100:.4f}%",(i,v['observation_stable_area']*100),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
axes[0].set_ylabel('All-input stable area (%)');axes[0].set_xticks(x,labels,rotation=30,ha='right');axes[0].legend(frameon=False)
axes[1].axhline(5,color='#c44e52',ls='--',label='Frozen FAR threshold')
for i,v in enumerate(values):
    far=v['sparse_reference_far']
    if far is None:
        axes[1].text(i,1,'null\n0 accepted',ha='center',color='#666666',fontsize=8)
    else:
        axes[1].plot(i,far*100,'o',color='#31798a')
        ci=v['clustered_sparse_far_ci95']
        if ci:axes[1].vlines(i,ci[0]*100,ci[1]*100,color='#31798a')
        axes[1].plot(i,v['far_upper95_iid_descriptive']*100,'_',color='#c44e52')
        axes[1].annotate(f"{v['false_stable_points']}/{v['accepted_scorable_points']}",(i,far*100),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
axes[1].set_xticks(x,labels,rotation=30,ha='right');axes[1].set_ylabel('Sparse annotated FAR (%)')
axes[1].set_title('Dots: FAR; lines: log cluster bootstrap\nRed ticks: descriptive point CP95 upper bound');axes[1].legend(frameon=False)
fig.suptitle('Development region evidence; no independent confirmation')
save(fig,'region_risk_coverage')

diag=json.loads((root/'identity_rejection_diagnostics.json').read_text())['uncensored_RGB_probe']
values=diag['measured_reasons'];names=list(values);x=np.arange(len(names))
fig,ax=plt.subplots(figsize=(9,4));bottom=np.zeros(len(names))
for key,color,label in [('correct','#24796b','Correct reference'),('wrong','#c44e52','Wrong reference'),('unscorable','#abb6c5','Unscorable')]:
    v=np.array([values[n][key] for n in names]);ax.bar(x,v,bottom=bottom,color=color,label=label);bottom+=v
ax.set_xticks(x,[n.replace('_','\n') for n in names]);ax.set_ylabel('Original geometric candidate pairs')
ax.set_title('Pre-change uncensored RGB probe; overlapping reasons must not be added');ax.legend(frameon=False)
save(fig,'identity_root_causes')

funnel=json.loads((root/'surface_acceptance_funnel.json').read_text())['baseline']
names=list(funnel);counts=[funnel[n]['retained'] for n in names]
fig,ax=plt.subplots(figsize=(10,4));ax.bar(np.arange(len(names)),counts,color='#31798a');ax.set_yscale('log')
ax.set_xticks(np.arange(len(names)),[n.replace('_','\n') for n in names],rotation=40,ha='right',fontsize=8)
ax.set_ylabel('Retained pixel-camera pairs (log scale)')
ax.set_title('Pre-change full-image diagnostic gates; final two bars are actual manager outputs\nDynamic exclusion scheduling is separate from the uncensored dense probe')
save(fig,'surface_funnel')
