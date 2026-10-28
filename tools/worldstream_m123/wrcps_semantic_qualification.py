"""Offline EQ-SSC fitting and one-shot scene/log-disjoint validation.

Partial-oracle unknowns stay unknown. Zero observed errors/cluster bootstrap
intervals are finite-data descriptions, not a physical or population guarantee.
"""
from collections import Counter,defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from tools.worldstream_m123.benchmark.common import write,sha256
from tools.worldstream_m123.wrcps_semantic_experiment import rows
from worldstream.m3.semantic_core import SemanticQualifier,QUALIFICATION_VERSION,SEMANTIC_CLAIM
from worldstream.m3.stable_world import StableWorldSnapshot,StableAnchorManager


def attach_labels(frames,labels,split):
    result=[]
    for f in frames:
        scene=f['context']['scene_id'];fid=f['context']['frame_id']
        edge={e['anchor_id']:e for e in labels[fid]['edges']}
        for r in f['records']:
            if not r['eligible_observation']:continue
            e=edge.get(r['anchor_id'],{})
            source,target=e.get('previous_gt'),e.get('current_gt')
            status='unknown' if source is None or target is None else 'correct' if source==target else 'wrong'
            result.append(dict(record=r,scene_id=scene,frame_id=fid,split=split[scene],status=status))
    return result


def summarize(items):
    counts=Counter(r['status'] for r in items);n=len(items);scored=counts['correct']+counts['wrong']
    scene=defaultdict(list)
    for r in items:scene[r['scene_id']].append(r)
    means=np.array([sum(r['status']=='wrong' for r in rr)/len(rr) for rr in scene.values()])
    if len(means):
        rng=np.random.default_rng(20261028)
        sample=rng.choice(means,(10000,len(means)),replace=True).mean(1)
        ci=np.quantile(sample,[.025,.975]).tolist()
    else:ci=[None,None]
    return dict(candidates=n,scored=scored,correct=counts['correct'],wrong=counts['wrong'],unknown=counts['unknown'],
        scenes=len(scene),unique_tracks=len({(r['scene_id'],r['record']['anchor_id']) for r in items}),
        identity_retention=counts['correct']/scored if scored else None,
        unknown_fraction=counts['unknown']/n if n else None,
        scene_mean_wrong_fraction_ci95=ci,
        interval_scope='descriptive scene bootstrap; unknowns separate; zero observed errors do not guarantee unseen risk')


def acceptable(s,limits):
    return (s['scored']>=limits['min_scored_edges'] and s['scenes']>=limits['min_scenes']
        and s['wrong']<=limits['max_observed_wrong_edges']
        and s['unknown_fraction']<=limits['max_unknown_fraction'])


def subset(items,rule):
    return [r for r in items if SemanticQualifier.rejection(r['record'],rule,r['record']['evidence']['timestamp']) is None]


def calibration_anchor_hits(frames,cls,rule):
    manager=StableAnchorManager();hits=0
    for frame in frames:
        ctx=frame['context']
        records=[r for r in frame['records'] if r['semantic_class']==cls
            and SemanticQualifier.rejection(r,rule,ctx['timestamp']) is None]
        facts=tuple(sorted((r['anchor_id'],r['anchor_type'],cls) for r in records))
        priority=[r['anchor_id'] for r in sorted(records,key=lambda r:(-r['score'],r['anchor_id']))]
        snapshot=StableWorldSnapshot(ctx['scene_id'],ctx['frame_id'],ctx['timestamp'],ctx['state_epoch'],
            ctx['state_version'],ctx['state_execution_signature'],facts,json.dumps(dict(priority_ids=priority)),
            qualification_id='calibration_trial')
        _,hit=manager.update(snapshot,ctx);hits+=int(hit)
    return hits


def unadmitted_rule_diagnostics(root,frames,labels,split,rules):
    """Real heldout observations under FROZEN rules, explicitly not admission.

    Shows scarce facts and lifetimes rather than hiding them behind a zero
    formal hit count. No embeddings, KV, or deployment decisions are produced.
    """
    output=[];summary={}
    for cls,rule in rules.items():
        if rule['admitted']:continue
        manager=StableAnchorManager();counts=Counter();birth={}
        for frame in frames:
            ctx=frame['context'];fid=ctx['frame_id']
            if split[ctx['scene_id']]!='qualification_validation':continue
            accepted=[r for r in frame['records'] if r['semantic_class']==cls
                and SemanticQualifier.rejection(r,rule,ctx['timestamp']) is None]
            facts=tuple(sorted((r['anchor_id'],r['anchor_type'],cls) for r in accepted))
            priority=[r['anchor_id'] for r in sorted(accepted,key=lambda r:(-r['score'],r['anchor_id']))]
            snap=StableWorldSnapshot(ctx['scene_id'],fid,ctx['timestamp'],ctx['state_epoch'],ctx['state_version'],
                ctx['state_execution_signature'],facts,json.dumps(dict(priority_ids=priority)),
                qualification_id='UNADMITTED_DIAGNOSTIC_ONLY')
            anchor,hit=manager.update(snap,ctx)
            if not hit:birth={f[0]:labels[fid]['current_labels'].get(f[0]) for f in anchor.facts}
            checks=[]
            for fact in anchor.facts:
                a,b=birth.get(fact[0]),labels[fid]['current_labels'].get(fact[0])
                status='unknown' if a is None or b is None else 'correct' if a==b else 'wrong'
                if hit:counts[status]+=1
                checks.append(dict(anchor_id=fact[0],status=status))
            counts['observation_facts']+=len(accepted);counts['potential_hits']+=int(hit)
            if accepted or hit:
                output.append(dict(scene_id=ctx['scene_id'],frame_id=fid,semantic_class=cls,
                    source_frame=anchor.source_frame,facts=anchor.facts,potential_hit=hit,
                    checks_offline=checks,qualified=False,KV_executed=False,
                    reason=rule['rejection_reason']))
        summary[cls]=dict(counts)
    with (root/'unadmitted_rule_observations.jsonl').open('w') as f:
        for r in output:f.write(json.dumps(r)+'\n')
    return summary


def qualify(root):
    run=root/'qualification_capture'
    if json.loads((run/'status.json').read_text())['status']!='COMPLETED':raise ValueError('Capture incomplete')
    if (root/'semantic_qualification.json').exists():raise FileExistsError('Never retune a published policy on validation')
    plan=json.loads((root/'experiment_plan.json').read_text());seq=json.loads((root/'qualification_sequence.json').read_text())
    frames=rows(run/'semantic_candidates.jsonl');labels={r['frame_id']:r for r in rows(run/'semantic_labels_offline.jsonl')}
    if [f['context']['frame_id'] for f in frames]!=seq['tokens']:raise ValueError('Frame coverage/order mismatch')
    extension_sources=[]
    if (root/'qualification_extensions.json').exists():
        extension=json.loads((root/'qualification_extensions.json').read_text())
        if sha256(root/'calibration_frozen_rules.json')!=extension['frozen_calibration_sha256']:
            raise ValueError('Frozen calibration rule changed during validation extension')
        extra_seq=json.loads((root/extension['sequence']).read_text());extra_run=root/extension['capture']
        if sha256(root/extension['sequence'])!=extension['sequence_sha256']:raise ValueError('Extension sequence changed')
        if json.loads((extra_run/'status.json').read_text())['status']!='COMPLETED':raise ValueError('Validation extension incomplete')
        extra=rows(extra_run/'semantic_candidates.jsonl')
        if [f['context']['frame_id'] for f in extra]!=extra_seq['tokens']:raise ValueError('Extension coverage mismatch')
        if set(seq['scene_logs'].values()) & set(extra_seq['scene_logs'].values()):raise ValueError('Validation extension log overlap')
        frames+=extra;labels.update({r['frame_id']:r for r in rows(extra_run/'semantic_labels_offline.jsonl')})
        seq['scene_splits'].update(extra_seq['scene_splits']);seq['scene_logs'].update(extra_seq['scene_logs'])
        extension_sources=[dict(path=str(extra_run/'semantic_candidates.jsonl'),sha256=sha256(extra_run/'semantic_candidates.jsonl'),
            labels_sha256=sha256(extra_run/'semantic_labels_offline.jsonl'))]
    protocols={f['protocol_signature'] for f in frames}
    if len(protocols)!=1:raise ValueError('M1 semantic protocol changed during capture')
    split=seq['scene_splits'];limits=plan['calibration']
    items=attach_labels(frames,labels,split)
    calibration=[r for r in items if r['split']=='calibration']
    validation=[r for r in items if r['split']=='qualification_validation']
    classes=sorted({r['record']['semantic_class'] for r in calibration})
    selected={};trials=[]
    calibration_frames=[f for f in frames if split[f['context']['scene_id']]=='calibration']
    # Only three predeclared quantile rules per class. No validation rows are
    # consulted while defining cutoffs, selecting a trial, or freezing rules.
    frozen_path=root/'calibration_frozen_rules.json'
    if frozen_path.exists():
        frozen=json.loads(frozen_path.read_text());selected=frozen['rules'];trials=frozen['trials']
    for cls in ([] if frozen_path.exists() else classes):
        sample=[r for r in calibration if r['record']['semantic_class']==cls]
        scores=[r['record']['evidence']['score'] for r in sample]
        residuals=[r['record']['evidence']['residual_m'] for r in sample]
        feasible=[]
        for trial,(sq,rq,obs) in enumerate(((.50,.75,2),(.75,.50,3),(.90,.25,3))):
            rule=dict(min_score=float(np.quantile(scores,sq)),max_residual_m=float(np.quantile(residuals,rq)),
                min_observations=obs,admitted=True,trial=trial)
            metrics=summarize(subset(sample,rule))
            core_hits=calibration_anchor_hits(calibration_frames,cls,rule)
            trials.append(dict(semantic_class=cls,rule=rule,metrics=metrics,calibration_anchor_hits=core_hits))
            if acceptable(metrics,limits):feasible.append((core_hits,metrics['candidates'],rule,metrics))
        if feasible:
            _,_,rule,metrics=max(feasible,key=lambda v:(v[0],v[1],-v[2]['trial']))
            selected[cls]=dict(rule,calibration=metrics)
    if not frozen_path.exists():
        write(frozen_path,dict(rules=selected,trials=trials,
            calibration_scenes=[s for s,p in split.items() if p=='calibration'],validation_consulted=False))
    rules={}
    for cls,rule in selected.items():
        metrics=summarize(subset([r for r in validation if r['record']['semantic_class']==cls],rule))
        rules[cls]=dict(rule,validation=metrics,admitted=acceptable(metrics,limits),
            rejection_reason=None if acceptable(metrics,limits) else 'independent_validation_failed_no_retuning')
    policy=dict(version=QUALIFICATION_VERSION,claim=SEMANTIC_CLAIM,protocol_signature=next(iter(protocols)),
        selector_implementation=SemanticQualifier.implementation_signature(),
        independent_validation_completed=True,rules=rules,qualification_limits=limits,
        calibration_scene_ids=[s for s,p in split.items() if p=='calibration'],
        validation_scene_ids=[s for s,p in split.items() if p=='qualification_validation'],
        calibration_log_ids=[seq['scene_logs'][s] for s,p in split.items() if p=='calibration'],
        validation_log_ids=[seq['scene_logs'][s] for s,p in split.items() if p=='qualification_validation'],
        calibration_frozen_sha256=sha256(root/'calibration_frozen_rules.json'),
        candidates_sha256=sha256(run/'semantic_candidates.jsonl'),labels_sha256=sha256(run/'semantic_labels_offline.jsonl'),
        validation_extension_sources=extension_sources,
        absence_grace_frames=0,physical_certification=False,
        limitation='Empirical claim on observed identity/class only; partial oracle unknowns explicit; no population risk guarantee')
    write(root/'semantic_qualification.json',policy)
    qualifier=SemanticQualifier(root/'semantic_qualification.json')
    unadmitted=unadmitted_rule_diagnostics(root,frames,labels,split,rules)
    manager=StableAnchorManager(max_core_facts=plan['selector']['core_size'])
    timeline=[];lifetimes=Counter();previous_scene=None;birth_labels={};hits=transitions=wrong=unknown=0
    per_split=defaultdict(Counter);refresh=Counter();new_object_kept=0;prev_ids=set()
    for frame in frames:
        ctx=frame['context'];fid=ctx['frame_id'];scene=ctx['scene_id'];which=split[scene]
        tick=perf_counter();selection=qualifier.select(frame);select_ms=(perf_counter()-tick)*1000
        snapshot=StableWorldSnapshot(scene,fid,ctx['timestamp'],ctx['state_epoch'],ctx['state_version'],
            ctx['state_execution_signature'],selection['facts'],json.dumps(selection,sort_keys=True),
            qualification_id=qualifier.sha256)
        tick=perf_counter();anchor,hit=manager.update(snapshot,ctx);anchor_ms=(perf_counter()-tick)*1000
        transition=scene==previous_scene
        if transition:per_split[which]['transitions']+=1
        current_ids={r['anchor_id'] for r in frame['records']}
        new_ids=current_ids-prev_ids if transition else set()
        if hit and new_ids:new_object_kept+=1;per_split[which]['new_objects_core_kept']+=1
        checks=[]
        if not hit:
            refresh[anchor.update_reason]+=1
            birth_labels={fact[0]:labels[fid]['current_labels'].get(fact[0]) for fact in anchor.facts}
        for fact in anchor.facts:
            source=birth_labels.get(fact[0]);target=labels[fid]['current_labels'].get(fact[0])
            status='unknown' if source is None or target is None else 'correct' if source==target else 'wrong'
            checks.append(dict(anchor_id=fact[0],semantic_class=fact[2],status=status))
            if hit:
                per_split[which]['core_fact_'+status]+=1
                wrong+=int(status=='wrong');unknown+=int(status=='unknown')
        per_split[which]['frames']+=1;per_split[which]['raw_candidates']+=len(frame['records'])
        per_split[which]['history_candidates']+=sum(r['has_history'] for r in frame['records'])
        per_split[which]['qualified_facts']+=len(selection['facts']);per_split[which]['selected_core_facts']+=len(anchor.facts)
        per_split[which]['anchor_hits']+=int(hit and transition)
        lifetimes[anchor.anchor_id]+=1;hits+=int(hit and transition);transitions+=int(transition)
        timeline.append(dict(scene_id=scene,frame_id=fid,split=which,source_frame=anchor.source_frame,
            anchor=asdict(anchor),anchor_hit=hit,raw_candidates=len(frame['records']),
            history_candidates=sum(r['has_history'] for r in frame['records']),qualified_facts=len(selection['facts']),
            qualified_ids=[f[0] for f in selection['facts']],new_objects=len(new_ids),
            core_fact_checks_offline=checks,selection_wall_ms=select_ms,anchor_wall_ms=anchor_ms,
            snapshot=asdict(snapshot)))
        previous_scene=scene;prev_ids=current_ids
    with (root/'stable_anchor_timeline.jsonl').open('w') as f:
        for r in timeline:f.write(json.dumps(r,allow_nan=False)+'\n')
    independent=per_split['qualification_validation']
    valid_anchor_ids={r['anchor']['anchor_id'] for r in timeline if r['anchor']['stable_state_valid']}
    valid_lifetimes=[lifetimes[a] for a in valid_anchor_ids]
    stage_pass=(independent['anchor_hits']>0 and independent['core_fact_wrong']==0 and independent['core_fact_unknown']==0)
    result=dict(stage_one_pass=stage_pass,frames=len(frames),scenes=len(split),
        qualified_semantic_types=sorted(qualifier.rules),anchor_hits=hits,crossframe_transitions=transitions,
        anchor_hit_rate=hits/transitions,valid_anchor_mean_frames=float(np.mean(valid_lifetimes)) if valid_lifetimes else 0.,
        all_anchor_mean_frames=float(np.mean(list(lifetimes.values()))),new_objects_without_refresh=new_object_kept,
        core_fact_wrong_extensions=wrong,core_fact_unknown_extensions=unknown,refresh_reasons=dict(refresh),
        per_split={k:dict(v,anchor_hit_rate=v['anchor_hits']/v['transitions']) for k,v in per_split.items()},
        mean_selection_replay_ms=float(np.mean([r['selection_wall_ms'] for r in timeline])),
        mean_anchor_replay_ms=float(np.mean([r['anchor_wall_ms'] for r in timeline])),
        timing_scope='real M1 candidate CPU replay, excludes image/model inference; not E2E performance',
        policy_sha256=qualifier.sha256,calibration_summary=summarize(calibration),validation_summary=summarize(validation),
        GT_online=False,no_observation_grace_frames=0,
        unadmitted_frozen_rule_observations=unadmitted,
        limitation='Native M1 class-gating prevents observing same-ID class flips directly. Missing/changed identity is withdrawn; unseen/occluded entities never extend a core.')
    write(root/'stable_core_analysis.json',result)
    print(json.dumps(result,indent=2),flush=True)
