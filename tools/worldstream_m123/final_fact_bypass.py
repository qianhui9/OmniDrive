"""Current-only cleanup: reproducible offline audit, no synthetic GPU results.

Historical DEV/TEST remain diagnostics. New TEST is blocked unless both model
training isolation and historical scene/log isolation can be established.
"""
import argparse
from collections import Counter,defaultdict,OrderedDict
import io
import json
import math
import os
from pathlib import Path
import pickle
import re
import subprocess
from time import perf_counter
import zipfile

from .benchmark.common import sha256,write
from .selective_answer_reuse_report import lines,mean,wilson
from .target_complete_report import world,context
from .fact_bypass_identity import match_instances,question_target,evaluate,request_identity,coverage,UNKNOWN_REASONS

ROOT=Path('outputs/m3_final_fact_bypass_run01')
PRIOR=Path('outputs/m3_target_complete_bypass_run01')


def historical_manifest(root):
    """Bind every tracked historical M3 output, before modifying any code."""
    paths=subprocess.check_output(['git','ls-files','outputs'],text=True).splitlines()
    return {p:sha256(p) for p in paths if Path(p).is_file() and
        ('m3_' in p or '/final_route/' in p) and not Path(p).is_relative_to(root)}


def checkpoint_metadata(path):
    """Read pickle metadata without allocating/loading checkpoint tensors."""
    class Metadata(pickle.Unpickler):
        def persistent_load(self,pid):return None
        def find_class(self,module,name):
            if module=='collections' and name=='OrderedDict':return OrderedDict
            if module=='torch._utils' and name.startswith('_rebuild_'):return lambda *a,**kw:None
            if module=='torch' and (name.endswith('Storage') or name.endswith('Tensor')):return type(name,(),{})
            raise pickle.UnpicklingError('Unsupported metadata global '+module+'.'+name)
    with zipfile.ZipFile(path) as z:
        name=next(n for n in z.namelist() if n.endswith('/data.pkl'))
        obj=Metadata(io.BytesIO(z.read(name))).load()
    return obj.get('meta',{})


def audit_dataset(root):
    from nuscenes.utils.splits import create_splits_scenes
    from tools.worldstream_m1.dataset import resolve_image
    from tools.worldstream_m1.io_utils import iter_json_array
    table=Path('data/nuscenes/v1.0-trainval')
    scenes={s['token']:s for s in iter_json_array(table/'scene.json')}
    logs={s['log_token'] for s in scenes.values()}
    frames={s['token']:s['scene_token'] for s in iter_json_array(table/'sample.json')}
    seen=set();execution_seen=set();sources=[]
    pattern=re.compile(r'(?<![a-z0-9])[0-9a-f]{32}(?![a-z0-9])')
    # Include frozen plans, DEV, TEST, diagnostic and natural-language records,
    # not just the most recent experiment_plan files.
    for p in sorted(Path('outputs').rglob('*')):
        if not p.is_file() or root in p.parents or p.suffix not in ('.json','.jsonl'):continue
        if not any(part.startswith(('m3_','m123_')) or part=='final_route' for part in p.parts):continue
        local=set()
        with p.open(errors='replace') as f:
            for line in f:
                for token in pattern.findall(line):
                    if token in scenes:local.add(scenes[token]['log_token'])
                    elif token in logs:local.add(token)
                    elif token in frames:local.add(scenes[frames[token]]['log_token'])
        if local:sources.append(dict(path=str(p),sha256=sha256(p),log_ids=sorted(local)))
        seen.update(local)
        if p.name in ('requests.jsonl','per_question_metrics.jsonl','paired_records.jsonl','service_evidence.jsonl'):
            execution_seen.update(local)
    official=create_splits_scenes();partitions={};available=[]
    train_scenes=set();val_scenes=set()
    for split in ('train','val'):
        info_path=Path(f'data/nuscenes/nuscenes2d_ego_temporal_infos_{split}.pkl')
        infos=pickle.load(info_path.open('rb'))['infos'];grouped=defaultdict(list)
        for i in infos:grouped[i['scene_token']].append(i)
        if split=='train':train_scenes=set(grouped)
        else:val_scenes=set(grouped)
        rows=[]
        for sid,items in sorted(grouped.items()):
            s=scenes[sid];seg=sorted(items,key=lambda i:i['timestamp'])[:8]
            camera_ok=len(seg)==8 and all(len(i['cams'])==6 and all(
                resolve_image(c['data_path'],'data/nuscenes').is_file() for c in i['cams'].values()) for i in seg)
            linked=len(seg)==8 and all(a.get('next')==b['token'] and b.get('prev')==a['token'] for a,b in zip(seg,seg[1:]))
            qa_ok=all(any((Path('data/nuscenes')/folder/split/(i['token']+'.json')).is_file()
                for folder in ('vqa','conv')) for i in seg)
            row=dict(scene_id=sid,scene_name=s['name'],log_id=s['log_token'],
                official_split=split if s['name'] in official[split] else 'mismatch',
                frames=len(items),six_camera_first8=camera_ok,linked_first8=linked,qa_first8=qa_ok,
                previously_used_log=s['log_token'] in seen,
                compatible_unused_validation=split=='val' and s['name'] in official['val'] and
                    camera_ok and linked and qa_ok and s['log_token'] not in seen)
            rows.append(row)
            if row['compatible_unused_validation']:available.append(row)
        partitions[split]=dict(path=str(info_path),sha256=sha256(info_path),frames=len(infos),
            scenes=len(grouped),logs=len({s['log_id'] for s in rows}),
            compatible_scenes=sum(s['six_camera_first8'] and s['linked_first8'] and s['qa_first8'] for s in rows),
            previously_used_scenes=sum(s['previously_used_log'] for s in rows),scene_manifest=rows)
        del infos
    cp=Path('ckpts/iter_10548.pth')
    try:meta=checkpoint_metadata(cp);meta_error=None
    except Exception as e:meta={};meta_error=repr(e)
    config=meta.get('config','');(root/'checkpoint_training_config.txt').write_text(config)
    write(root/'checkpoint_metadata.json',dict(path=str(cp),bytes=cp.stat().st_size,sha256=sha256(cp),
        metadata={k:v for k,v in meta.items() if k!='config'},metadata_error=meta_error,
        training_config_file='checkpoint_training_config.txt',tensor_data_loaded=False))
    overlap={}
    for p in sorted(Path('outputs').glob('m3*/experiment_plan.json')):
        text=p.read_text();ids=set(pattern.findall(text))
        membership=ids & set(scenes)
        overlap[str(p)]=dict(scene_ids=sorted(membership),train_scene_overlap=sorted(membership&train_scenes),
            val_scene_overlap=sorted(membership&val_scenes))
    audit=dict(status='INCOMPLETE_INDEPENDENT_VALIDATION' if not available else 'CANDIDATE_REQUIRES_TRAINING_ATTESTATION',
        partitions=partitions,historical_sources=sources,historical_log_ids=sorted(seen),
        unused_compatible_validation_scenes=available,unused_compatible_validation_logs=sorted({s['log_id'] for s in available}),
        validation_logs_with_actual_execution_records=sorted({scenes[s]['log_token'] for s in val_scenes}&execution_seen),
        historical_plan_training_overlap=overlap,official_train_val_scene_disjoint=not train_scenes&val_scenes,
        official_train_val_log_overlap=sorted({scenes[s]['log_token'] for s in train_scenes}&
            {scenes[s]['log_token'] for s in val_scenes}),
        checkpoint_training_config_available=bool(config),exact_per_scene_training_ledger_available=False,
        checkpoint_source='Local existing checkpoint; upstream download URL not authenticated by an acquisition manifest',
        official_validation_isolation='Official split evidence, not proof of all pretraining exposure; historical M3 reuse also excluded',
        no_download=True)
    write(root/'dataset_split_provenance.json',audit)
    (root/'dataset_training_overlap_audit.md').write_text(
        '# Training exposure audit\n\nLocal original training configs and the embedded checkpoint configuration are inspected separately from WRSAR training. '
        'The original pipeline uses nuScenes train image/Conv/VQA inputs, so physical-train WRSAR scenes cannot be called model-unseen. '
        'The local checkpoint metadata reports exp_name=mask_eva_lane_det_vlm.py, iter=10547, epoch=1, time=Fri Apr 18 04:41:08 2025. '
        'Its embedded config directly specifies nuscenes2d_ego_temporal_infos_train.pkl and train Conv/VQA. '
        'An exact scene-by-scene training execution ledger and authenticated acquisition manifest are unavailable.\n\n'+
        '\n'.join(f"- {k}: {v['frames']} frames / {v['scenes']} scenes; compatible first8 six-camera linked segments {v['compatible_scenes']}; historical used-log scenes {v['previously_used_scenes']}." for k,v in partitions.items())+
        f'\n\nUnused compatible official validation scenes: {len(available)}. Metadata and per-source SHA256 receipts are in dataset_split_provenance.json. '
        'Exclusion uses entire logs across all historical M3/M123 plans and records, including diagnostics. Trainval variants are alternate indexes, not new datasets. '
        'The official unlabeled test split is not an eligible substitute. No new TEST or GPU baseline is scheduled without an eligible independent pool.\n\n'
        'Upstream primary configuration also specifies train metadata and train QA: '
        '[NVlabs configuration](https://raw.githubusercontent.com/NVlabs/OmniDrive/main/projects/configs/OmniDrive/mask_eva_lane_det_vlm.py). '
        'The supplied GitHub fork page was unavailable through browsing; local HEAD is the execution authority.\n')
    return audit


def audit_prior(root):
    names=['implementation_summary.md','final_research_assessment.md','result_answers.md','go_no_go.json',
        'task_quality.json','quality_by_task.json','quality_by_scene.json','severe_error_audit.json',
        'nearest_target_correctness.json','nearest_target_failure_audit.json','observation_source_quality.json',
        'current_vs_target_complete_ablation.json','ablation_execution_equivalence.json','quality_latency_pareto.json',
        'workload_coverage.json','experiment_plan.json','protocol_freeze.json','REPRODUCE.md']
    receipts={n:dict(sha256=sha256(PRIOR/n),bytes=(PRIOR/n).stat().st_size) for n in names}
    eq=json.loads((PRIOR/'ablation_execution_equivalence.json').read_text())
    policy=json.loads((PRIOR/'selected_policy.json').read_text())
    write(root/'prior_evidence_audit.json',dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources=receipts,policy=policy,ablation_summary={k:v for k,v in eq.items() if k!='pairs'},
        historical_verdict=json.loads((PRIOR/'go_no_go.json').read_text()),
        attribution='Current-only fact bypass; no independent completeness compute or quality contribution demonstrated'))


def load_objects(tokens):
    from .answer_validity_quality import load_current_annotations
    result={}
    for split in ('train','val'):
        path=f'data/nuscenes/nuscenes2d_ego_temporal_infos_{split}.pkl'
        infos=pickle.load(open(path,'rb'))['infos'];ids={i['token'] for i in infos}&tokens;del infos
        if ids:result.update(load_current_annotations(path,ids,'data/nuscenes'))
    return result


def diagnostic_quality(root):
    rows=lines(PRIOR/'run/requests.jsonl')
    # All three tasks on each of 32 historical frames; keep one execution per
    # condition to avoid inflating denominators with workloads/rounds.
    rows=[r for r in rows if r['workload']=='addressable' and r['question_count']==3]
    objects=load_objects({r['frame_id'] for r in rows});records=[]
    old=lines(PRIOR/'quality_requests.jsonl')
    old=[r for r in old if r['stage']=='run' and r['workload']=='addressable' and r['question_count']==3]
    original={(r['condition'],r['frame_id'],r['question']):r['quality'] for r in old}
    for row in rows:
        w=row.get('committed_world');matches=match_instances(w['anchors'],objects[row['frame_id']],
            row['frame_id'],row['timestamp']) if w else {}
        for i,(req,ans) in enumerate(zip(row['requests'],row['answers'])):
            q=req['question'];target=question_target(q,objects.get(row['frame_id']))
            identity=request_identity(req.get('fast_answer'),target,matches)
            records.append(dict(scene_id=row['scene_id'],log_id=row['log_id'],frame_id=row['frame_id'],
                question=q,request_index=i,condition=row['condition'],answer=ans['A'][0],
                task=__import__('worldstream.m3.task_conditioner',fromlist=['question_facts']).question_facts(q).intent,
                llm_skipped=req['llm_skipped'],target_reference=target,**identity,
                quality=evaluate(q,ans['A'][0],objects[row['frame_id']],target,row['frame_id']),
                original_quality=original[(row['condition'],row['frame_id'],q)],proposal=req.get('fast_answer')))
    with (root/'historical_quality_requests.jsonl').open('w') as f:
        for e in records:f.write(json.dumps(e)+'\n')
    c=[r for r in records if r['condition']=='C'];fast=[r for r in c if r['llm_skipped']]
    write(root/'nearest_target_identity_audit.json',dict(scope='Historical v4 object3Q C proposals, NOT new TEST',
        all_proposals=dict(object=coverage(c,'object'),target=coverage(c,'target')),
        actual_fast=dict(object=coverage(fast,'object'),target=coverage(fast,'target')),
        by_task=[dict(task=t,object=coverage([r for r in c if r['task']==t],'object'),
            target=coverage([r for r in c if r['task']==t],'target'),
            reference_identity_known=sum(r['target_reference']['known'] for r in c if r['task']==t))
            for t in ('category','existence','position','motion')],
        matching='Class-independent one-to-one 3D center assignment; 2m radius, .5m ambiguity; duplicate competitors UNKNOWN',
        visibility='nuScenes visibility>=3, sensor support, center projection with camera exposure pose; proxy, not image-certified',
        source='historical_quality_requests.jsonl'))
    tasks=[]
    for condition in ('A','B','C'):
        for task in ('category','existence','position','motion'):
            part=[r for r in records if r['condition']==condition and r['task']==task]
            tasks.append(quality_summary(part,condition=condition,task=task))
    known=[r for r in c if r['quality']['known']]
    old=[r for r in old if r['condition']=='C']
    oldknown={(r['frame_id'],r['question']) for r in old if r['quality']['known']}
    newknown={(r['frame_id'],r['question']) for r in known}
    write(root/'quality_label_coverage.json',dict(scope='Historical v4 relabelling only; no new independent samples',
        requests=len(c),old_answer_scored=len(oldknown),answer_scored=len(known),UNKNOWN=len(c)-len(known),
        unknown_reasons={reason:sum(r['target_reference'].get('reason')==reason for r in c
            if not r['quality']['known']) for reason in UNKNOWN_REASONS},
        gained_answer_labels=sorted(newknown-oldknown),lost_answer_labels=sorted(oldknown-newknown),
        object=coverage(c,'object'),question_target=coverage(c,'target'),
        explanation='Common-sector/common-class ambiguity permits an answer-level label only; nearest instance remains UNKNOWN. Closer visibility-ambiguous competitors now veto the label. Matching standards are unchanged or stricter.'))
    write(root/'quality_by_task.json',dict(status='HISTORICAL_DIAGNOSTIC_ONLY',groups=tasks))
    write(root/'quality_by_scene.json',dict(status='HISTORICAL_DIAGNOSTIC_ONLY',groups=[quality_summary(
        [r for r in records if r['condition']==cond and r['scene_id']==sid],condition=cond,scene_id=sid)
        for sid in sorted({r['scene_id'] for r in records}) for cond in ('A','B','C')]))
    paired=[];lookup={(r['frame_id'],r['question'],r['condition']):r for r in records}
    for r in c:
        b=lookup[(r['frame_id'],r['question'],'B')]
        if not (r['quality']['known'] and b['quality']['known']):continue
        paired.append(dict(scene_id=r['log_id'],frame_id=r['frame_id'],question=r['question'],
            loss=b['quality']['score']-r['quality']['score'],
            new_severe=int(bool(r['quality']['severe_error']) and not b['quality']['severe_error']),
            severe_delta=int(bool(r['quality']['severe_error']))-int(bool(b['quality']['severe_error'])),
            fast=r['llm_skipped']))
    from .world_guided_decode_v2_report import scene_ci
    write(root/'task_quality.json',dict(status='HISTORICAL_DIAGNOSTIC_ONLY',groups=[quality_summary(
        [r for r in records if r['condition']==cond],condition=cond) for cond in ('A','B','C')],
        paired_scored=len(paired),paired_loss=mean([r['loss'] for r in paired]),
        paired_scene_CI95=scene_ci(paired,'loss') if paired else None,
        full_driving_quality=False,new_independent_test=False))
    write(root/'fast_answer_quality.json',dict(status='HISTORICAL_DIAGNOSTIC_ONLY',
        fast=quality_summary(fast),fallback=quality_summary([r for r in c if not r['llm_skipped']]),
        current_run_actual_skips=0))
    frozen_severe=json.loads((PRIOR/'severe_error_audit.json').read_text())
    write(root/'severe_error_audit.json',dict(status='HISTORICAL_DIAGNOSTIC_ONLY',
        paired_denominator=len(paired),new_severe=sum(r['new_severe'] for r in paired),
        net_severe_change=sum(r['severe_delta'] for r in paired),
        scene_CI95=scene_ci(paired,'new_severe') if paired else None,
        descriptive_request_wilson95=wilson(sum(r['new_severe'] for r in paired),len(paired)),
        independent_logs=len({r['scene_id'] for r in paired}),cluster_unit='log_id',
        cases=[r for r in paired if r['new_severe']],
        fast_new_severe=sum(r['new_severe'] for r in paired if r['fast']),
        fallback_new_severe=sum(r['new_severe'] for r in paired if not r['fast']),
        nearest_identity_errors=coverage(fast,'target'),strict_2percent_certification=False,
        wrong_visible_witness_cases=[dict(frame_id=r['frame_id'],question=r['question'],task=r['task'],
            target=r['target'],object=r['object']) for r in fast if r['target']['known'] and not r['target']['correct']],
        identity_severe_vs_B='UNKNOWN: B generated answers have no independently matched instance receipt',
        frozen_v4_new_severe_cases=frozen_severe['new_severe_vs_B'],
        frozen_v4_verdict='NO_GO_UNCHANGED: 1/41 new severe, empirical scene CI upper 8.333%',
        caveat='Small log-cluster empirical intervals cannot establish rare-error safety. UNKNOWN target identity is excluded, never counted correct. Historical v4 NO_GO unchanged.'))
    return records


def quality_summary(part,**extra):
    known=[r for r in part if r['quality']['known']]
    return dict(**extra,requests=len(part),scored=len(known),UNKNOWN=len(part)-len(known),
        score=mean([r['quality']['score'] for r in known]),severe=sum(bool(r['quality']['severe_error']) for r in known),
        actual_skips=sum(r['llm_skipped'] for r in part),independent_logs=len({r['log_id'] for r in part}),
        scored_logs=len({r['log_id'] for r in known}))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['prepare','diagnose','report','verify'])
    p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args();root=a.root
    if {4,5}&os.sched_getaffinity(0):p.error('Use taskset -c 0-3,6-31')
    root.mkdir(parents=True,exist_ok=True)
    if a.stage=='prepare':
        if (root/'historical_manifest.json').exists():raise FileExistsError('Do not replace frozen evidence')
        write(root/'historical_manifest.json',historical_manifest(root));audit_prior(root);audit_dataset(root)
    elif a.stage=='diagnose':
        diagnostic_quality(root)
        from .final_fact_bypass_report import diagnose
        diagnose(root)
    elif a.stage=='report':
        from .final_fact_bypass_report import report
        report(root)
    else:
        manifest=json.loads((root/'historical_manifest.json').read_text())
        changed=[f for f,h in manifest.items() if not Path(f).exists() or sha256(f)!=h]
        assert not changed,changed
        sources=root/'execution_sources.json'
        if sources.exists():
            source=json.loads(sources.read_text())
            for kind in ('source_hashes','diagnostic_input_hashes'):
                assert all(Path(f).exists() and sha256(f)==h for f,h in source[kind].items()),kind
            d=json.loads((root/'go_no_go.json').read_text())
            assert d['research_status']=='INCOMPLETE' and not d['independent_test_qualified']
            assert d['current_full_M123_GPU_calls']==0 and d['new_TEST_e2e_speedup'] is None
            for key in ('validation_cpu','validation_gpu'):
                summary=json.loads((root/key/'summary.json').read_text())
                assert summary['exit_code']==0 and not {4,5}&set(summary['cpu_affinity'])
            assert json.loads((root/'validation_gpu/summary.json').read_text())['CUDA_available']
            records=lines(root/'historical_quality_requests.jsonl')
            for cond in ('A','B','C'):
                assert len([r for r in records if r['condition']==cond])==96
            for key in ('object','target'):
                c=[r for r in records if r['condition']=='C'];c_cov=coverage(c,key)
                assert sum(c_cov['unknown_reasons'].values())==c_cov['UNKNOWN']
        print(json.dumps(dict(status='PASS',historical_files_unchanged=len(manifest),
            current_sources_verified=sources.exists())))


if __name__=='__main__':main()
