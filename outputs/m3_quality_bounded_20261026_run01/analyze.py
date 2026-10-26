"""Rebuild report data from observed JSONL. Run from the repository root."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,json,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]))
from tools.worldstream_m123.benchmark.metrics import distribution,quality_summary
from tools.worldstream_m123.benchmark.quality_calibration import paired_quality

def read(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
seq=json.loads((ROOT/'sequence_extended.json').read_text())
base=read(ROOT/'native_reference/per_question_metrics.jsonl')
reference={(q['frame_id'],q['question']):q for q in base}
baseframes=read(ROOT/'native_reference/per_frame_metrics.jsonl')
frame_ref={f['frame_id']:f for f in baseframes}
summary={'runs':{},'quality':{},'calibration_controls':{},'limitations':[
    '24 unique frames, 72 questions; native reference combines two documented invocations',
    '12 calibration frames, 6 development frames, 6 fresh validation frames',
    'Final replay is after a rule rejection on service validation: regression, not a second untouched test',
    'Only map_relations actually used bounded approximation in final service',
    'Fresh final six frames had no actual approximate reuse in the selected service policy',
    'Small per-profile approximate validation samples; no broad task-quality guarantee',
    'No isolated end-to-end exact-only ablation; exact restriction available in paired calibration',
    'M2 refresh risk signal was constant full refresh; no causal attribution to M2 guidance']}
for name in ('native_reference','wsdr_kv','wsdr_kv_final'):
 q=read(ROOT/name/'per_question_metrics.jsonl');f=read(ROOT/name/'per_frame_metrics.jsonl')
 assert [(x['frame_id'],x['question']) for x in q]==[(x['frame_id'],x['question']) for x in base]
 details=[x['reuse_diagnostics'] for x in q if x.get('reuse_diagnostics')]
 if details:
  for x in details:
   assert x['effective_prefill_tokens']+x['reused_tokens']==x['input_tokens']
   assert sum(x['forward_prefill_lengths'])==x['effective_prefill_tokens']
   assert x['prefill_completed'] and x['decode_completed']
  m3=[x['m3']['runtime_host_elapsed_ms'] for x in f]
 else:m3=[sum(x['host_generate_call_ms'] for x in q if x['frame_id']==fr['frame_id']) for fr in f]
 policy_types=Counter(x.get('reuse_type','none') for x in details) if details else Counter(none=len(q))
 input_tokens=sum(x['input_tokens'] for x in q)
 reused=sum(x.get('reused_tokens',0) for x in details)
 stats=dict(frames=len(f),questions=len(q),input_tokens=input_tokens,reused_tokens=reused,
  effective_prefill_tokens=input_tokens-reused,reused_fraction=reused/input_tokens,
  reuse_requests=sum(x.get('reused_tokens',0)>0 for x in details),reuse_types=dict(policy_types),
  reused_tokens_by_type={t:sum(x['reused_tokens'] for x in details if x['reuse_type']==t) for t in ('exact','bounded_approx')},
  cross_frame_approx_requests=sum(x['reuse_type']=='bounded_approx' and x['cache_source_frame']!=row['frame_id'] for row,x in zip(q,details)),
  extra_tokens_over_exact_match=sum(max(0,x['reused_tokens']-x['matched_prefix_tokens']) for x in details),
  request_ms=distribution(x['request_ms'] for x in f),m3_runtime_ms=distribution(m3),
  prefill_ms=distribution(x['prefill_ms'] for x in q),decode_ms=distribution(x['decode_ms'] for x in q),
  frames_per_second=1000*len(f)/sum(x['request_ms'] for x in f),
  peak_allocated_bytes=max(x['peak_gpu_allocated_bytes'] for x in f),
  peak_reserved_bytes=max(x['peak_gpu_reserved_bytes'] for x in f),
  cache_bytes_max=max((x['cache_bytes'] for x in details),default=0),
  planning_ms=distribution(x['planning_ms'] for x in details),
  cache_branch_ms=distribution(x.get('cache_branch_ms') or 0 for x in details),
  layout_preparation_ms=distribution(x['layout_preparation_ms'] for x in details),
  cache_age_max=max((x['cache_age_frames'] for x in details),default=0),
  depth_max=max((x['approx_reuse_depth'] for x in details),default=0),
  m1_valid_requests=sum(x['m1_state_valid'] for x in details),
  m2_refresh_ratios=sorted(set(x['m2_refresh_ratio'] for x in details)),
  detection_map_digest_equal=sum(x['output_digest']==frame_ref[x['frame_id']]['output_digest'] for x in f),
  quality=quality_summary(q))
 summary['runs'][name]=stats
 paired=[]
 for b,a in zip(base,q):
  item=paired_quality(a['answer'],b['answer'],b['reference'])
  item.update(frame_id=a['frame_id'],question=a['question'],profile=a.get('reuse_diagnostics',{}).get('task_profile'),
   reuse_type=a.get('reuse_diagnostics',{}).get('reuse_type','none'),split=seq['frame_splits'][a['frame_id']],
   token_equal=a['token_ids']==b['token_ids'])
  paired.append(item)
 (ROOT/(name+'_paired_quality.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in paired))
 splits={}
 for split in ('calibration','development','validation','all'):
  a=[x for x in paired if split=='all' or x['split']==split]
  splits[split]=dict(n=len(a),f1_loss_mean=sum(x['f1_loss'] for x in a)/len(a),
    answer_equal=sum(x['answer_equal'] for x in a),token_equal=sum(x['token_equal'] for x in a),
    critical_requests=sum(x['critical_regressions']>0 for x in a),
    reuse_types=dict(Counter(x['reuse_type'] for x in a)))
 metrics={}
 for key in ('numeric_exact','numeric_value_precision','numeric_value_recall','spatial_label_match','spatial_label_f1','motion_label_match','motion_label_f1','safety_label_match','safety_label_f1'):
  values=[x['task_metrics'][key] for x in paired if x['task_metrics'].get(key) is not None]
  metrics[key]=dict(n=len(values),mean=sum(values)/len(values) if values else None)
 summary['quality'][name]=dict(splits=splits,task_metrics=metrics)
for name in ('calibration_final','calibration_short'):
 rows=read(ROOT/name/'paired_records.jsonl');by=defaultdict(list)
 for r in rows:
  if r['split']=='calibration':by[r['threshold']].append(r)
 summary['calibration_controls'][name]={str(t):dict(n=len(a),
  mean_m3_ms=sum(x['candidate_wall_ms'] for x in a)/len(a),
  mean_baseline_ms=sum(x['baseline_wall_ms'] for x in a)/len(a),
  f1_loss=sum(x['quality']['f1_loss'] for x in a)/len(a),
  critical_requests=sum(x['quality']['critical_regressions']>0 for x in a),
  reused_tokens=sum(x['diagnostics']['reused_tokens'] for x in a)) for t,a in by.items()}
base_stats=summary['runs']['native_reference'];new=summary['runs']['wsdr_kv_final']
summary['final_changes']=dict(latency_reduction_fraction=1-new['request_ms']['mean']/base_stats['request_ms']['mean'],
  end_to_end_speedup=base_stats['request_ms']['mean']/new['request_ms']['mean'],
  prefill_reduction_fraction=1-new['prefill_ms']['total']/base_stats['prefill_ms']['total'],
  m3_wall_reduction_fraction=1-new['m3_runtime_ms']['total']/base_stats['m3_runtime_ms']['total'])
(ROOT/'comparison_summary.json').write_text(json.dumps(summary,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(8,5))
with (ROOT/'quality_latency_curve.csv').open('w') as f:
 writer=csv.writer(f);writer.writerow(['visual_cap','threshold','mean_f1_loss','paired_m3_latency_ratio','critical_requests'])
 for name,label,marker in [('calibration_final','unrestricted visual prefix','o'),('calibration_short','visual prefix <= 128','s')]:
  rows=summary['calibration_controls'][name];points=sorted(rows.items(),key=lambda p:float(p[0]))
  xs=[];ys=[]
  for t,p in points:
   ratio=p['mean_m3_ms']/p['mean_baseline_ms'];xs.append(p['f1_loss']);ys.append(ratio)
   writer.writerow([label,t,p['f1_loss'],ratio,p['critical_requests']])
   ax.annotate(f"risk={float(t):.3g}, critical={p['critical_requests']}",(p['f1_loss'],ratio),xytext=(6,6),textcoords='offset points',fontsize=7)
  ax.plot(xs,ys,marker+'-',label=label)
 ax.axvline(0,color='grey',linestyle=':',alpha=.5)
 ax.text(.98,.04,'Initial F1 budget = 0.02 (outside zoom)\nCritical regressions also disqualify a candidate',transform=ax.transAxes,ha='right',va='bottom',fontsize=8)
 ax.ticklabel_format(axis='x',style='sci',scilimits=(-3,3),useOffset=False)
 ax.set(xlabel='Calibration GT Token F1 loss (absolute)',ylabel='Paired complete M3 wall / full native wall',title='Natural inputs: quality / latency candidates')
 ax.legend(loc='lower right',bbox_to_anchor=(1.,.18));ax.margins(x=.25,y=.15);fig.tight_layout();fig.savefig(ROOT/'quality_latency_curve.svg');fig.savefig(ROOT/'quality_latency_curve.png',dpi=160);plt.close(fig)
print(json.dumps({'final':summary['runs']['wsdr_kv_final'],'changes':summary['final_changes'],'quality':summary['quality']['wsdr_kv_final']},indent=2))
