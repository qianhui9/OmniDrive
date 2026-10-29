"""WRSAR v2 protocol and diagnostics, using the existing full-M123 worker."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
from . import selective_answer_reuse as worker
from .benchmark.common import write,sha256

ROOT=Path('outputs/m3_temporal_selective_bypass_run01')
OLD=Path('outputs/m3_selective_answer_reuse_run01')
RISK_CONFIGS=[dict(min_confidence=.45,max_binding_risk=.35),
              dict(min_confidence=.45,max_binding_risk=.75),
              dict(min_confidence=.65,max_binding_risk=.35)]


def prepare(root):
    import pickle
    from collections import defaultdict
    from .world_guided_decode_v2 import source_questions
    from tools.worldstream_m1.dataset import resolve_image
    root.mkdir(parents=True,exist_ok=False)
    old=json.loads((OLD/'experiment_plan.json').read_text())
    tables={s['token']:s for s in json.loads(Path('data/nuscenes/v1.0-trainval/scene.json').read_text())}
    excluded=set()
    for path in [Path('outputs/m3_world_guided_decode_run02/experiment_plan.json'),
                 Path('outputs/m3_world_guided_decode_run03/experiment_plan.json'),
                 Path('outputs/m3_world_guided_decode_run03/initial_scope_mismatch/experiment_plan.json'),OLD/'experiment_plan.json']:
        if not path.exists():continue
        p=json.loads(path.read_text())
        for key in ('scenes','development_scenes','test_scenes'):
            excluded.update(tables[s['scene_id']]['log_token'] for s in p.get(key,[]))
    grouped=defaultdict(list)
    for f in pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb'))['infos']:
        grouped[f['scene_token']].append(f)
    scenes=[];logs=set();questions={}
    for sid,frames in sorted(grouped.items()):
        log=tables[sid]['log_token']
        if log in excluded or log in logs:continue
        frames=sorted(frames,key=lambda f:f['timestamp'])[:4]
        if len(frames)!=4 or any(len(source_questions(f['token']))<3 for f in frames):continue
        if not all(resolve_image(c['data_path'],'data/nuscenes').exists() for f in frames for c in f['cams'].values()):continue
        scenes.append(dict(scene_id=sid,log_id=log,tokens=[f['token'] for f in frames]));logs.add(log)
        questions.update({f['token']:source_questions(f['token'])[:3] for f in frames})
        if len(scenes)==3:break
    assert len(scenes)==3,'No substitution of previously examined TEST scenes allowed'
    diagnostic=[s for s in old['test_scenes'] if '79070dca796f4ffb868d84f1b067f9c2' in s['tokens']]
    diagnostic.append(dict(old['development_scenes'][0],tokens=old['development_scenes'][0]['tokens'][:2]))
    indices={t:i for i,t in enumerate(t for s in old['test_scenes'] for t in s['tokens'])}
    plan=dict(old,version='wrsar2',test_scenes=scenes,natural_questions=questions,
        diagnostic_scenes=diagnostic,legacy_frame_indices=indices,experiment_seed=20261029,
        risk_configs=RISK_CONFIGS,prior_excluded_log_ids=sorted(excluded),
        selection='First three sorted image-complete four-frame scenes on disjoint logs, excluding previous WRSAR and target-aware DEV/diagnostic/TEST logs; no GT/detector/outcome selection.',
        B='Existing K8 exact Target-Aware, stable question RNG, no bypass',
        C='Same K8 fallback and question RNG; current binding plus temporal competitor risk; current facts only',
        baseline_commit='b6c11e0876c9da7fa82fbe8cc10edf708f971659')
    write(root/'experiment_plan.json',plan)
    write(root/'protocol_freeze.json',dict(plan_sha256=sha256(root/'experiment_plan.json'),
        candidates=RISK_CONFIGS,threshold_tuning='DEV only, before TEST',
        verdict='GO needs positive scene-paired E2E, <=2pp task loss, no new severe errors and temporal contribution; scoped partial otherwise; missing evidence INCOMPLETE',
        quality_scope='Existing independently annotated current visible task facts; unknown retained; no Native-answer truth',
        no_test_threshold_or_seed_tuning=True))
    # Read and retain a structural audit of every requested previous artifact.
    names=['implementation_summary.md','result_answers.md','abc_comparison.json','e2e_performance_report.md',
        'task_quality.json','fast_hit_paired_quality.json','perception_grounding_audit.json','severe_error_audit.json',
        'fallback_sampling_diagnostic.json','quality_latency_pareto.json','quality_budget_bounds.json','go_no_go.json']
    write(root/'prior_evidence_read.json',{n:dict(bytes=len((OLD/n).read_bytes()),sha256=sha256(OLD/n)) for n in names})
    print('PREPARED',json.dumps(scenes),flush=True)


def stage_jobs(root,stage,plan):
    if stage=='diagnostic':
        return [('addressable',3,2,c,'online' if c.startswith('C') else 'off',
                 OLD/'selected_policy.json' if c.startswith('C') else None)
                for c in ('B-legacy','C-legacy','B-isolated','C-isolated')]
    if stage=='pareto':
        return [('addressable',3,r,c,'online' if c!='B' else 'off',p)
                for r in range(2) for c,p in ([('B',None)]+[(f'C{i}',root/f'candidate_{i}.json')
                    for i in range(len(RISK_CONFIGS))] if r==0 else list(reversed([('B',None)]+[
                    (f'C{i}',root/f'candidate_{i}.json') for i in range(len(RISK_CONFIGS))])))]
    return [('addressable',n,r,c,'online',root/'selected_policy.json')
            for n in (1,3) for r in range(plan['rounds']) for c in ('C-current',)]


class DiagnosticCapture:
    saved={}
    def __init__(self,model,condition,out):
        self.model=model;self.condition=condition;self.out=out;self.frame_id='';self.rows=[]
        self.original=model.m3_runtime.executor.generate
        def wrapped(*args,**kwargs):
            import torch
            from worldstream.m3.request_rng import rng_digest
            qi=model.m3_runtime.last_diagnostics['active_request_index']
            rng=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone())
            expanded=args[1]
            signature=hashlib.sha256(expanded.embeddings.detach().cpu().numpy().tobytes()).hexdigest()
            result,diag=self.original(*args,**kwargs)
            if self.frame_id=='79070dca796f4ffb868d84f1b067f9c2' and qi==2:
                self.saved[self.condition]=(args,kwargs,rng)
            self.rows.append(dict(question_index=qi,rng_before=hashlib.sha256(rng[0].numpy().tobytes()+rng[1].numpy().tobytes()).hexdigest(),
                expanded_current_input_sha256=signature,input_shape=list(expanded.embeddings.shape),
                draft_candidates=[c.metadata() for c in kwargs.get('draft_candidates',[])],
                selected_candidate_index=diag.get('selected_candidate_index'),
                tokens=(result.sequences if hasattr(result,'sequences') else result)[0,1:].tolist()))
            return result,diag
        model.m3_runtime.executor.generate=wrapped
    def finish_frame(self,row):
        worker.append(self.out/'diagnostic_inputs.jsonl',dict(condition=self.condition,frame_id=self.frame_id,
            scene_id=row['scene_id'],questions=row['questions'],requests=self.rows,
            question_rng=[r.get('question_rng') for r in row['requests']]))
        self.rows=[]
    def close(self):self.model.m3_runtime.executor.generate=self.original


def diagnostic_replay(model,out):
    import torch
    from worldstream.m3.prefill_executor import PrefillExecutor
    from worldstream.m3.prefix_kv_cache import PrefixKVCache
    rows=[]
    # Same current embeddings, same draft candidates, only RNG is crossed.
    for args_name in ('B-legacy','C-legacy','B-isolated','C-isolated'):
        args,kwargs,_=DiagnosticCapture.saved[args_name]
        for state_name in ('B-legacy','C-legacy') if 'legacy' in args_name else ('B-isolated','C-isolated'):
            cpu,cuda=DiagnosticCapture.saved[state_name][2]
            with torch.random.fork_rng(devices=[0]):
                torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda)
                executor=PrefillExecutor(PrefixKVCache(0))
                answer,diag=executor.generate(*args,**kwargs)
            tokens=(answer.sequences if hasattr(answer,'sequences') else answer)[0,1:].tolist()
            rows.append(dict(input_and_draft_source=args_name,rng_source=state_name,tokens=tokens,
                answer=model.tokenizer.decode(tokens,skip_special_tokens=True),
                llm_forward_calls=1+diag.get('decode_forward_calls',0),selected_candidate_index=diag.get('selected_candidate_index')))
    write(out/'rng_crossover_replay.json',dict(scope='8 finite same-current-input M3-only diagnostic generations; outside formal E2E, never used to tune seeds',rows=rows))
    DiagnosticCapture.saved.clear()


def calibrate(root):
    from .selective_answer_reuse_report import lines,load_labels,scored_requests,mean
    rows=lines(root/'shadow/requests.jsonl');events=scored_requests(rows,load_labels(rows))
    primary=[e for e in events if e['workload']=='addressable'];known=[e for e in primary if e['quality']['known']]
    configs=[]
    for i,cfg in enumerate(RISK_CONFIGS):
        hits=[e for e in primary if e.get('candidate_quality') and
              e['proposal']['support_score']>=cfg['min_confidence'] and
              e['proposal'].get('target_binding',{}).get('total_risk',0)<=cfg['max_binding_risk']]
        scored=[e for e in hits if e['candidate_quality']['known'] and e['quality']['known']]
        configs.append(dict(index=i,**cfg,hits=len(hits),scored_hits=len(scored),
            correct=sum(e['candidate_quality']['score'] for e in scored),unknown=len(hits)-len(scored),
            loss=sum(e['quality']['score']-e['candidate_quality']['score'] for e in scored)/len(known) if known else None,
            new_severe=sum(e['candidate_quality']['severe_error'] and not e['quality']['severe_error'] for e in scored),
            estimated_saved_ms=sum(e['request_wall_ms'] for e in hits),
            correct_frames=len({e['frame_id'] for e in scored if e['candidate_quality']['score']==1})))
    write(root/'answer_validity_calibration.json',dict(stage='DEV_SHADOW',configs=configs,
        requests=len(primary),known=len(known),normal_m3_ms=mean([e['request_wall_ms'] for e in primary]),
        score_kind='Empirical risk bins, NOT error probabilities',candidate_source_sha256=sha256(root/'shadow/requests.jsonl')))
    expected=mean([e['request_wall_ms'] for e in primary])
    for c in configs:
        write(root/f"candidate_{c['index']}.json",dict(status='FROZEN_RESEARCH_POLICY',policy_id=f"temporal_dev_{c['index']}",
            budget=.02,min_confidence=c['min_confidence'],max_binding_risk=c['max_binding_risk'],min_net_gain_ms=0.,
            expected_normal_m3_ms=expected,calibration_sha256=sha256(root/'answer_validity_calibration.json'),
            calibration_metrics=c,automatic_deployment=False))
    print('CALIBRATION',json.dumps(configs),flush=True)


def freeze(root):
    from .selective_answer_reuse_report import lines,load_labels,scored_requests,mean
    rows=lines(root/'pareto/requests.jsonl');events=scored_requests(rows,load_labels(rows))
    bases={(e['scene_id'],e['frame_id'],e['round'],e['request_index']):e for e in events if e['condition']=='B'}
    configs=[]
    for i,cfg in enumerate(RISK_CONFIGS):
        part=[e for e in events if e['condition']==f'C{i}'];pairs=[(bases[(e['scene_id'],e['frame_id'],e['round'],e['request_index'])],e) for e in part]
        known=[(b,e) for b,e in pairs if b['quality']['known'] and e['quality']['known']]
        configs.append(dict(index=i,**cfg,stage='DEV_ONLINE_GPU',requests=len(part),
            hits=sum(e['llm_skipped'] for e in part),loss=mean([b['quality']['score']-e['quality']['score'] for b,e in known]),
            new_severe=sum(e['quality']['severe_error'] and not b['quality']['severe_error'] for b,e in known),
            mean_e2e_ms=mean([r['e2e_wall_ms'] for r in rows if r['condition']==f'C{i}'])))
    policies=[]
    for budget in (.01,.02,.03):
        usable=[c for c in configs if c['loss'] is not None and c['loss']<=budget and not c['new_severe'] and c['hits']>=3]
        if not usable:policies.append(dict(budget=budget,status='NO_QUALIFIED_DEV_POLICY'));continue
        best=min(usable,key=lambda c:c['mean_e2e_ms'])
        p=json.loads((root/f"candidate_{best['index']}.json").read_text());p.update(budget=budget,online_dev_metrics=best)
        write(root/f'policy_{int(budget*100)}pp.json',p);policies.append(p)
        if budget==.02:write(root/'selected_policy.json',p)
    write(root/'quality_latency_pareto.json',dict(stage='DEV_ONLY_REAL_GPU',configs=configs,policies=policies,test_tuning=False))
    assert (root/'selected_policy.json').exists(),'DEV gate did not qualify a policy; stop online TEST'
    write(root/'POLICY_FREEZE.json',dict(selected_policy_sha256=sha256(root/'selected_policy.json'),
        grader_sha256=sha256('tools/worldstream_m123/selective_answer_reuse_report.py'),
        calibration_sha256=sha256(root/'answer_validity_calibration.json'),before_test=True))
    print('FROZEN',json.dumps(json.loads((root/'selected_policy.json').read_text())),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--stage',required=True,choices=['prepare','diagnostic','shadow','calibrate','pareto','freeze','run','ablation','report'])
    a=parser.parse_args()
    if a.stage=='prepare':prepare(a.root)
    elif a.stage=='calibrate':calibrate(a.root)
    elif a.stage=='freeze':freeze(a.root)
    elif a.stage=='report':
        from .temporal_selective_report import report
        report(a.root)
    else:
        with (a.root/f'{a.stage}_execution.log').open('w') as log,redirect_stdout(log),redirect_stderr(log):worker.run(a.root,a.stage)


if __name__=='__main__':main()
