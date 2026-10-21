"""Read recorded results only. No tracker, inference or association is run."""
import json,hashlib,ast
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parent
MAIN=ROOT/'main_method'
OLD=Path('outputs/m1_dqtrack_policy_20261021')
PRIOR=Path('outputs/m1_world_identity_20261021_run01/main_method')
summary=json.loads((MAIN/'summary.json').read_text())
alltimes=[];stats=[];fast=[];recover=[];opportunities=[];by_scene={};differences={}
newcorrect=set();oldcorrect=set();priorcorrect=set();newwrong=set();oldwrong=set();priorwrong=set();saved_metrics={}
byreason=Counter();byclass={};measured_reasons=Counter();view_reasons=Counter();occlusion=Counter();full_log_bytes=[]
def correct(traces):return {(t['frame_id'],e['current_instance']) for t in traces for e in t['edges'] if e['status']=='correct'}
def wrong(traces):return {(t['frame_id'],e['previous_instance'],e['current_instance']) for t in traces for e in t['edges'] if e['status']=='wrong'}
for scene in summary['scenes']:
    current=json.loads((MAIN/(scene+'.json')).read_text())
    old=json.loads((OLD/'paired_owned'/scene/'geometry.json').read_text())
    prior=json.loads((PRIOR/(scene+'.json')).read_text())
    assert current['metrics']['eligible']==old['metrics']['eligible']==prior['metrics']['eligible']
    assert current['metrics']['candidates']==old['metrics']['candidates']==prior['metrics']['candidates']
    newcorrect|=correct(current['traces']);oldcorrect|=correct(old['traces']);priorcorrect|=correct(prior['traces'])
    newwrong|=wrong(current['traces']);oldwrong|=wrong(old['traces']);priorwrong|=wrong(prior['traces'])
    fast+=current['fast_diagnosis'];recover+=current['recoveries'];opportunities+=current['recovery_opportunities']
    alltimes += [t['m1_ms'] for t in current['traces']]
    per=[]
    for t in current['traces']:
        assert not t['certified'] and t['dqtrack'] is None
        d=t['association_tracks'];stats.append(d);per.append(d)
        assert d['audit']['sample_bytes']+d['measurement_receipt_bytes']<=65536
        assert d['tracks']<=256 and d['accounted_memory_bytes']<=4194304 and d['serialized_bytes']<=4194304
        full_log_bytes.append(len(json.dumps(d).encode()))
        byreason.update(d['audit']['counts'])
        for cls,counts in d['audit']['by_class'].items():byclass.setdefault(cls,Counter()).update(counts)
        for m in d['measurements']:measured_reasons[m['reason']]+=1
        for sample in d['audit']['samples']:
            for v in sample['appearance']['views']:
                view_reasons[v['reason']]+=1;occlusion[str(v.get('occluded'))]+=1
    saved_metrics[scene]=current['saved_continuity_metrics']
    by_scene[scene]=dict(metrics=current['metrics'],m1=current['m1'],
        supported=sum(d['supported'] for d in per),contradicted=sum(d['mismatches'] for d in per),
        unknown=sum(d['unknown'] for d in per),bootstrap=sum(d['bootstrap_supported'] for d in per),
        fast=current['strata'].get('motion:fast'),recovered=len(current['recoveries']),
        recovery_opportunities=len(current['recovery_opportunities']))
assert len(fast)==115
for name,oldset in [('geometry',oldcorrect),('prior',priorcorrect)]:
    differences[name]=dict(lost_correct=[list(x) for x in sorted(oldset-newcorrect)],
        gained_correct=[list(x) for x in sorted(newcorrect-oldset)])
saved_totals={name:{key:sum(v[name].get(key,0) for v in saved_metrics.values())
                   for key in next(iter(saved_metrics.values()))[name]} for name in ('geometry','prior')}
fast_counts=Counter(row['measurement']['reason'] for row in fast)
fast_gates=Counter(k for row in fast for k,passed in row.get('gates',{}).items() if not passed)
visibility={}
for scene,row in summary['scenes'].items():
    for key,value in row['strata'].items():
        if key.startswith('visibility:'):
            target=visibility.setdefault(key,Counter())
            for k,v in value.items():
                if isinstance(v,int):target[k]+=v
frozen=json.loads((ROOT/'frozen_manifest.json').read_text())
assert all(hashlib.sha256(Path(n).read_bytes()).hexdigest()==h for n,h in frozen.items())
analysis=dict(decision=summary['decision'],metrics=summary['metrics'],saved_new_metrics=saved_totals,
    scene_metrics=by_scene,new_wrong_relations=[list(x) for x in sorted(newwrong-oldwrong)],
    same_wrong_relations=newwrong==oldwrong,wrong_relations=[list(x) for x in sorted(newwrong)],
    differences=differences,fast=dict(opportunities=len(fast),correct=sum(row['correct'] for row in fast),
        measurement_reasons=dict(fast_counts),failed_independent_gates=dict(fast_gates),
        source_image_available=sum(row.get('source_image_available',False) for row in fast),
        motion_prediction_available=sum(row.get('motion_prediction_available',False) for row in fast),
        all_cases='main_method per-scene fast_diagnosis; all 115 retained'),
    recovery=dict(actual=len(recover),statuses=dict(Counter(v['status'] for v in recover)),
        candidates=sum(d['dormant_candidates'] for d in stats),supported=sum(d['dormant_supported'] for d in stats),
        appearance_conflicts=sum(d['dormant_appearance_conflicts'] for d in stats),
        opportunities=len(opportunities),opportunity_types=dict(Counter(v['kind'] for v in opportunities)),
        limitation='GT absence/support gaps are observation proxies, not visually established occlusion episodes.'),
    roi=dict(pair_measurements=sum(d['measured_edges'] for d in stats),supported=sum(d['supported'] for d in stats),
        contradicted=sum(d['mismatches'] for d in stats),unknown=sum(d['unknown'] for d in stats),
        bootstrap_supported=sum(d['bootstrap_supported'] for d in stats),
        measurement_reasons=dict(measured_reasons),sampled_camera_reasons=dict(view_reasons),
        sampled_occlusion=dict(occlusion)),
    diagnostics=dict(reasons=dict(byreason),by_class={k:dict(v) for k,v in byclass.items()},
        offline_visibility={k:dict(v) for k,v in visibility.items()},
        pair_log_p50_bytes=float(np.median(full_log_bytes)),pair_log_p95_bytes=float(np.percentile(full_log_bytes,95)),
        pair_log_max_bytes=max(full_log_bytes),note='Whole diagnostic object includes bounded aggregate counters in addition to the 64KiB detailed/receipt budget.'),
    resources=dict(m1_p50_ms=float(np.median(alltimes)),m1_p95_ms=float(np.percentile(alltimes,95)),
        max_tracks=max(d['tracks'] for d in stats),max_track_bytes=max(d['accounted_memory_bytes'] for d in stats),
        max_serialized_bytes=max(d['serialized_bytes'] for d in stats),
        max_image_history_bytes=max(d['image_history_bytes'] for d in stats),
        max_image_history_frames=max(d['image_history_frames'] for d in stats)),
    source_frozen_verified=True,physical_certifications=0,frames=len(alltimes),main_method_runs=1)
(ROOT/'analysis.json').write_text(json.dumps(analysis,indent=2))
(ROOT/'relation_differences.json').write_text(json.dumps(differences,indent=2))
(ROOT/'fast_115_current_diagnosis.json').write_text(json.dumps(fast,indent=2))
(ROOT/'recovery_cases.json').write_text(json.dumps(dict(recoveries=recover,opportunities=opportunities),indent=2))
print(json.dumps({k:v for k,v in analysis.items() if k not in ('diagnostics','scene_metrics','differences')},indent=2))
