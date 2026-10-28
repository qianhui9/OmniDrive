"""Standalone descriptive figure, no service speedup claims."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
d=json.loads((root/'core_event_distribution.json').read_text())
b=json.loads((root/'kv_speedup_upper_bound.json').read_text())
rows=[json.loads(s) for s in (root/'controlled_micro/records.jsonl').read_text().splitlines()]
fig,axes=plt.subplots(1,3,figsize=(14,4.5))
for name,label in [('original_train','Original 6 TRAIN logs'),('expanded_train','Added 8 TRAIN logs'),('validation','Development validation')]:
    axes[0].plot([8,16,32],[d['windows'][str(n)][name]['anchor_extensions'] for n in (8,16,32)],'o-',label=label)
axes[0].set(xlabel='Causal window (frames)',ylabel='True core extensions',title='Data: legal positive-event enrichment')
axes[0].set_xticks([8,16,32])
axes[0].legend(fontsize=8)
kinds=['native','full','text','text_plus64','same_frame_visual']
axes[1].boxplot([[r['executor_wall_ms'] for r in rows if r['kind']==k] for k in kinds],labels=['Native','Full KV','Text','Text+64','Same frame'],showfliers=False)
axes[1].set(ylabel='Wall time (ms)',title='135 controlled prefill calls; zero decode')
axes[1].tick_params(axis='x',rotation=25)
values=[];low=[];high=[]
for workload,key,ci_key in [('1','estimated_net_ms_per_request','total_net_ms'),('1','incremental_world_prefix_net_ms_per_request','marginal_world_net_ms'),('3','incremental_world_prefix_net_ms_per_request','marginal_world_net_ms')]:
    w=b['workloads'][workload];v=w[key];a,c=w['timing_bootstrap_ci95'][ci_key]
    values.append(v);low.append(v-a);high.append(c-v)
axes[2].errorbar(range(3),values,yerr=[low,high],fmt='o',capsize=5)
axes[2].axhline(0,color='black',linewidth=.7)
axes[2].set(xticks=range(3),xticklabels=['1Q total','1Q world only','3Q world only'],ylabel='Estimated ms / request',title='Optimistic net: timing uncertainty only')
axes[2].tick_params(axis='x',rotation=20)
fig.suptitle('SF-WRCPS: training data ready; total system gain unproven',fontsize=13)
fig.text(.5,.01,'TRAIN pool is selected for positive events. Untrained structure is used for cost/diagnostics only. No trained service or E2E result.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.04,1,.92))
fig.savefig(root/'feasibility_summary.png',dpi=180)
fig.savefig(root/'feasibility_summary.svg')
svg=root/'feasibility_summary.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
plt.close(fig)
