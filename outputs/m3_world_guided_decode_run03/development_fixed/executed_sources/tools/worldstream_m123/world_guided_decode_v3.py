"""v3 orchestration; reuse the real full-model v2 A/B/C measurement framework."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

from tools.worldstream_m123.benchmark.common import write,sha256
from tools.worldstream_m123.world_guided_decode_v2 import prepare,run
from worldstream.m3.answer_memory import AnswerMemory
from worldstream.m3.world_state import CommittedWorldState,CommittedAnchor

ROOT=Path('outputs/m3_world_guided_decode_run03')
V2=Path('outputs/m3_world_guided_decode_run02')


def load_world(raw):
    return CommittedWorldState(**dict(raw,anchors=tuple(CommittedAnchor(**dict(a,
        center_ego=tuple(a['center_ego']),velocity_ego=tuple(a['velocity_ego']),cameras=tuple(a['cameras'])))
        for a in raw['anchors']),ego_to_world=tuple(raw['ego_to_world'])))


def prepare_v3(root):
    prior=json.loads((V2/'experiment_plan.json').read_text())
    prepare(root,extra_excluded_scenes=[s['scene_id'] for s in prior['scenes']])
    plan=json.loads((root/'experiment_plan.json').read_text())
    plan.update(version='v3',development_source=str(V2/'p3/e2e_trials.jsonl'),
        development_sha256=sha256(V2/'p3/e2e_trials.jsonl'),
        excluded_v2_scene_ids=[s['scene_id'] for s in prior['scenes']],
        policy='One K8 verifier. Preserve complete exact drafts; M1-supported historical positive facts form <=4 candidates. Current processed prefill distribution ranks them without RNG or extra forward. Two prior unproductive verifications suppress repeated candidates.',
        offline_analysis='Only v2 development scenes before primary execution. Native current answers are diagnostic LCP/oracle reference ONLY, never online truth or runtime inputs.')
    write(root/'experiment_plan.json',plan)


def lcp(a,b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))


def offline(root,source=None,destination='offline_acceptance_upper_bound.json',development=True):
    """No target forward. Causal previous NATIVE outputs, current tokens ONLY
    compared after candidate construction. This is an optimistic diagnostic,
    not a sampling acceptance measurement or an online algorithm.
    """
    from transformers import AutoTokenizer
    source=Path(source or V2/'p3/e2e_trials.jsonl')
    rows=[json.loads(s) for s in source.read_text().splitlines()]
    native=[r for r in rows if r['condition']=='A' and r['workload']=='natural' and r['question_count']==3 and r['round']==0]
    worlds={r['frame_id']:load_world(r['committed_world']) for r in rows
        if r['condition']=='C' and r['workload']=='natural' and r['question_count']==3 and r['round']==0}
    tokenizer=AutoTokenizer.from_pretrained('ckpts/pretrain_qformer/',use_fast=False,local_files_only=True)
    memory=AnswerMemory();scene=None;events=[];binding=('offline-fixed-model-tokenizer',)
    for r in native:
        if scene!=r['scene_id']:memory.reset();scene=r['scene_id']
        world=worlds[r['frame_id']]
        ctx=dict(scene_id=scene,frame_id=r['frame_id'],frame_index=r['frame_index'],timestamp=world.timestamp)
        for request,answer in zip(r['requests'],r['answers']):
            q=request['question'];prompt=tuple(tokenizer.encode(q));tick=perf_counter()
            selected=memory.retrieve(q,prompt,ctx,binding,world,True)
            selected=memory.construct(selected,q,prompt,ctx,binding,world,tokenizer)
            cost=(perf_counter()-tick)*1000
            candidates=[]
            # The current target sequence is read AFTER the proposals are fixed.
            current=request['tokens']
            for c in selected.candidates:
                common=lcp(c.tokens,current);available=min(len(c.tokens),len(current))
                # Optimistic teacher-forced trajectory reference. FP16 batch
                # and stochastic rejection can only invalidate this proxy.
                saved=max(0,common-1-(common//9))
                candidates.append(dict(c.metadata(),lcp_tokens=common,first_token_match=bool(common),
                    acceptable_prefix_length=common,first_rejection_position=common if common<available else None,
                    estimated_saved_forwards=saved,optimistic_net_ms=saved*25.-cost,
                    fact_reached=bool(c.fact_token_start>=0 and common>c.fact_token_start)))
            events.append(dict(scene_id=scene,frame_id=r['frame_id'],question=q,construction_selection_wall_ms=cost,
                candidates=candidates,current_native_tokens=current,
                scope='OFFLINE diagnostic reference, unavailable online'))
            memory.remember(q,prompt,current,ctx,binding,world,answer['A'][0])
    summaries=[]
    for label,kinds,filter_first in [('raw_history',{'exact','semantic_raw'},False),
        ('question_conditioned',{'question_conditioned'},False),('oracle_first_token_filtered',{'exact','semantic_raw','question_conditioned'},True)]:
        parts=[]
        for e in events:
            cs=[c for c in e['candidates'] if c['kind'] in kinds and (not filter_first or c['first_token_match'])]
            best=max(cs,key=lambda c:c['lcp_tokens']) if cs else None
            parts.append((e,best))
        present=[c for _,c in parts if c is not None]
        summaries.append(dict(source=label,requests=len(events),available=len(present),
            first_token_matches=sum(c['first_token_match'] for c in present),
            at_least_two_tokens=sum(c['lcp_tokens']>=2 for c in present),
            lcp_tokens=sum(c['lcp_tokens'] for c in present),
            estimated_saved_forwards=sum(c['estimated_saved_forwards'] for c in present),
            optimistic_net_positive=sum(c['lcp_tokens']>=2 and c['optimistic_net_ms']>0 for c in present),
            actual_prefill_screen='NOT_RUN: only an offline current-token oracle, not online processed logits'))
    write(root/destination,dict(status='OFFLINE_ORACLE_DIAGNOSTIC',development=development,
        source=str(source),source_sha256=sha256(source),scene_ids=sorted({e['scene_id'] for e in events}),
        reference='Existing native sampled trajectories, not greedy ground truth or an upper bound on all stochastic target paths. Optimistic current-trajectory LCP upper bound only. 25ms/forward is prior measured cost, NOT a new speed measurement.',
        candidate_policy_frozen_before_current_reference=True,summary=summaries,events=events))
    print('OFFLINE',summaries,flush=True)


def development(root):
    """One bounded targeted B/C fixed-stream run on already-used v2 scenes."""
    dest=root/'development_fixed'
    dest.mkdir(exist_ok=False)
    p=json.loads((V2/'experiment_plan.json').read_text());p['version']='v3'
    write(dest/'experiment_plan.json',p)
    run(dest,three_questions_only=True,skip_controls=True,skip_development=True,
        rounds=1,conditions=('B','C'),workloads=('fixed',),skip_greedy=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--stage',choices=['prepare','offline','development','run','offline-heldout'],required=True)
    a=p.parse_args()
    if a.stage=='prepare':prepare_v3(a.root)
    elif a.stage=='offline':offline(a.root)
    elif a.stage=='development':development(a.root)
    elif a.stage=='offline-heldout':offline(a.root,a.root/'e2e_trials.jsonl','heldout_acceptance_diagnostic.json',False)
    else:run(a.root,skip_controls=True,skip_development=True)
