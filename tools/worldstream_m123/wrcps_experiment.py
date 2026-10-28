"""WRCPS stage-one real-world feasibility gate and honest staged delivery.

The gate is intentionally ahead of adapter training or an A/B/C performance
campaign. An empty certified state never becomes a manufactured stable scene.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from time import perf_counter
from tools.worldstream_m123.benchmark.common import sha256, write
from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
from tools.worldstream_m123.benchmark.independent_validation import historical_audit
from tools.worldstream_m123.benchmark.sequence import validate_sequence
from tools.worldstream_m123.crossframe_e2e import native_pairs
from tools.worldstream_m2.global_residual_benchmark import execute, wait_gpu
from worldstream.m3.stable_world import StableAnchorManager, StableWorldSnapshot


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def prepare(root):
    from mmcv import Config
    root.mkdir(parents=True,exist_ok=False)
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py')
    infos=pickle.load(open(cfg.data.test.ann_file,'rb'))['infos']
    audit=historical_audit('outputs',root,infos)
    write(root/'historical_data_audit.json',audit)
    grouped=defaultdict(list)
    for info in infos:
        if info['scene_token'] not in audit['excluded_scenes']:grouped[info['scene_token']].append(info)
    eligible=[]
    for scene,ss in sorted(grouped.items()):
        segment=sorted(ss,key=lambda x:x['timestamp'])[:6]
        if len(segment)!=6 or any(len(native_pairs(i['token']))<3 for i in segment):continue
        validate_sequence(segment,scene)
        if all(a.get('next')==b['token'] and b.get('prev')==a['token'] for a,b in zip(segment,segment[1:])):
            eligible.append((scene,segment))
    if len(eligible)<8:raise ValueError('Need 2 development and 6 independent six-frame scenes')
    manifest=dict(tokens=[],questions={},scene_splits={})
    for j,(scene,segment) in enumerate(eligible[:8]):
        manifest['scene_splits'][scene]='development' if j<2 else 'independent_acceptance'
        for info in segment:
            token=info['token'];manifest['tokens'].append(token)
            manifest['questions'][token]=[p['question'] for p in native_pairs(token)[:3]]
    write(root/'anchor_sequence.json',manifest)
    write(root/'experiment_plan.json',dict(stage='world_state_feasibility_first',K=128,total_tokens=513,
        initial_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        quality_budgets=dict(mAP=.1,NDS=.1,DET_l=.1,Token_F1=.1),
        selection='first 8 eligible unseen scenes; 6 real consecutive frames each',
        scene_splits=manifest['scene_splits'],training_scene_ids=[],seed=20261028,
        sequence_sha256=sha256(root/'anchor_sequence.json'),
        gates='Stable facts -> train -> ZeroKV quality -> exact equivalence -> paired E2E'))


def audit(root):
    plan=json.loads((root/'experiment_plan.json').read_text())
    if plan.get('sequence_sha256') and sha256(root/'anchor_sequence.json')!=plan['sequence_sha256']:
        raise ValueError('Frozen sequence changed')
    if (root/'anchor_audit_language').exists():raise FileExistsError('Preserve the previous audit')
    with gpu_job_lock():
        wait_gpu()
        execute(root,'anchor_audit_language',['-m','tools.worldstream_m123.benchmark.worker',
            '--output',str(root/'anchor_audit_language'),'--run-root',str(root),
            '--condition','B3','--workload','language','--limit','48','--warmup','1',
            '--sequence-file',str(root/'anchor_sequence.json'),'--capture-stable-world'])


def analyze(root):
    run=root/'anchor_audit_language'
    if json.loads((run/'status.json').read_text())['status']!='COMPLETED':
        raise ValueError('Only completed real audits may enter the feasibility conclusion')
    raw=rows(run/'stable_world_snapshots.jsonl')
    sequence=json.loads((root/'anchor_sequence.json').read_text())
    if [r['frame_id'] for r in raw]!=sequence['tokens']:raise ValueError('Snapshot coverage/order mismatch')
    manager=StableAnchorManager();updates=Counter();scene_rows=defaultdict(list);timeline=[]
    qualified=hits=transitions=0;lifetimes=Counter();overhead=[];previous_scene=None
    for item in raw:
        s=StableWorldSnapshot(**dict(item,facts=tuple(tuple(f) for f in item['facts'])))
        if s.content_sha256!=item['content_sha256']:raise ValueError('Snapshot digest changed')
        context=dict(scene_id=s.scene_id,frame_id=s.frame_id,timestamp=s.timestamp,
            state_epoch=s.state_epoch,state_version=s.state_version,state_execution_signature=s.execution_signature)
        tick=perf_counter();anchor,hit=manager.update(s,context);wall=(perf_counter()-tick)*1000
        overhead.append(wall)
        transition=previous_scene==s.scene_id
        transitions+=int(transition);hits+=int(hit and transition);qualified+=int(anchor.stable_state_valid)
        previous_scene=s.scene_id
        if not hit:updates[anchor.update_reason]+=1
        lifetimes[anchor.anchor_id]+=1
        provenance=json.loads(s.provenance_json)
        row=dict(scene_id=s.scene_id,frame_id=s.frame_id,timestamp=s.timestamp,
            anchor=asdict(anchor),anchor_hit=hit,qualified_stable_facts=len(s.facts),
            geometric_candidates=provenance['candidates'],
            producer_qualification_sha256=provenance['producer_qualification_sha256'],
            anchor_management_replay_wall_ms=wall)
        scene_rows[s.scene_id].append(row);timeline.append(row)
    per_scene={scene:dict(frames=len(rr),split=sequence['scene_splits'][scene],
        qualified_frames=sum(r['anchor']['stable_state_valid'] for r in rr),
        anchor_hits=sum(r['anchor_hit'] for r in rr),
        anchor_hit_rate=sum(r['anchor_hit'] for r in rr)/max(1,len(rr)-1),
        candidate_counts=[r['geometric_candidates'] for r in rr]) for scene,rr in scene_rows.items()}
    stable_ids={r['anchor']['anchor_id'] for r in timeline if r['anchor']['stable_state_valid']}
    stable_lifetimes=[lifetimes[k] for k in stable_ids]
    result=dict(status='PASS' if hits else 'FAILED_NO_STABLE_WORLD_CONTINUITY',
        frames=len(raw),scenes=len(scene_rows),independent_acceptance_scenes=sum(
            s['split']=='independent_acceptance' for s in per_scene.values()),
        qualified_frames=qualified,anchor_hits=hits,crossframe_transitions=transitions,
        anchor_hit_rate=hits/transitions if transitions else None,
        valid_anchor_mean_frames=sum(stable_lifetimes)/len(stable_lifetimes) if stable_lifetimes else 0.,
        all_anchor_mean_frames=sum(lifetimes.values())/len(lifetimes),
        stable_state_update_frequency=sum(updates.values())/len(raw),
        invalidation_reasons=dict(updates),per_scene=per_scene,
        replay_anchor_management_mean_ms=sum(overhead)/len(overhead),
        timing_scope='CPU replay of real immutable M1 snapshots, not an end-to-end service measurement',
        source=str(run/'stable_world_snapshots.jsonl'),source_sha256=sha256(run/'stable_world_snapshots.jsonl'),
        producer_qualification_available=any(r['producer_qualification_sha256'] for r in timeline))
    write(root/'stable_anchor_analysis.json',result)
    with (root/'stable_anchor_timeline.jsonl').open('w') as f:
        for r in timeline:f.write(json.dumps(r,allow_nan=False)+'\n')
    # Existing per-request executor observations remain labelled as a native
    # layout feasibility audit. They are not WRCPS A/B/C performance evidence.
    requests=rows(run/'per_question_metrics.jsonl')
    with (root/'per_request_metrics.jsonl').open('w') as f:
        for r in requests:
            d=dict(r,experiment_role='native_layout_stage_one_audit',wrcps_executed=False)
            f.write(json.dumps(d,allow_nan=False)+'\n')
    write(root/'crossframe_reuse_summary.json',dict(wrcps_executed=False,
        qualified_world_prefix_frames=qualified,legal_crossframe_anchor_hits=hits,
        actual_wrcps_exact_visual_kv_tokens=0 if not hits else None,
        wrcps_prefill_forward_tokens_saved=0 if not hits else None,
        actual_wrcps_mean_visual_kv_tokens_per_request=None,
        reason='No admitted stable world anchor; no trained layout or KV campaign' if not hits else 'Requires training and validation',
        native_audit_requests=len(requests)))
    decision=dict(recommended=False,replace_formal_configuration=False,
        stage_one_pass=bool(hits),stage_two_quality_pass=None,stage_three_exact_kv_pass=None,
        stage_four_e2e_pass=None,adapter_trained=False,
        reason='M1 has no qualified stable continuity on real scenes' if not hits else 'Further stages required',
        formal_configuration='projects/configs/OmniDrive/worldstream_m123_auto.py',
        quality_budgets=dict(mAP=.1,NDS=.1,DET_l=.1,Token_F1=.1))
    write(root/'deployment_decision.json',decision)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--stage',choices=['prepare','audit','analyze'],required=True);args=p.parse_args()
    if set(os.sched_getaffinity(0)) & {4,5}:p.error('Use taskset -c 0-3,6-31')
    root=Path(args.root)
    result={'prepare':prepare,'audit':audit,'analyze':analyze}[args.stage](root)
    if result:print(json.dumps(result,indent=2))


if __name__=='__main__':main()
