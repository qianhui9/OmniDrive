"""Frozen calibration/holdout comparison for risk, coverage and CPU overhead.

Historical full-model outputs are consumed only by commit; labels stay offline.
The reference implementation must be an immutable pre-change source snapshot.
"""
import argparse
from dataclasses import fields, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter
import cv2
import numpy as np
from worldstream.m1 import M1Config, WorldStateManager
from .dataset import frame_from_info, prediction_outputs, load_oracle
from .io_utils import read_prediction_subset, write_json
from .refinement_common import summarize_rows, ratio
from . import run_refinement

POLICIES = {'exact_parallel': {}, 'sparse_parallel': {'flow_max_points': 800}}
QUALITY = [(0.,1,0.), (.5,1,0.), (.7,1,0.), (0.,2,0.), (.5,2,.5)]


def reference(root):
    manifest=json.loads((root/'baseline_manifest.json').read_text())
    for name,expected in manifest['source_sha256'].items():
        path=root/'baseline_source'/name if name.startswith('worldstream/') else Path(name)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Reference source/config hash mismatch: '+str(path))
    source=root/'baseline_source/worldstream/m1/__init__.py'
    spec=importlib.util.spec_from_file_location('m1_optimization_reference',source,
                                               submodule_search_locations=[str(source.parent)])
    mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
    keys={f.name for f in fields(mod.M1Config)}
    return lambda cfg: mod.WorldStateManager({k:v for k,v in cfg.to_dict().items() if k in keys})


def fingerprint(manager, result):
    return dict(valid=result.valid_anchor_ids,invalid=result.invalid_anchor_ids,
        features=result.valid_feature_candidates,versions=result.feature_versions,
        masks={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in manager.last_masks.items()},
        association=manager.last_commit_diagnostics['association_pairs'])


def benchmark(root, frames, predictions, cfg, reference_factory):
    outputs=[prediction_outputs(predictions[f.frame_id],cfg) for f in frames]
    variants={'reference': (reference_factory,cfg), **{f'workers_{w}':(WorldStateManager,replace(cfg,temporal_workers=w)) for w in (1,2,3,6)}}
    records={}; expected=None
    # Interleaved repetitions reduce order/thermal confounding; first repetition warmup.
    for repeat in range(5):
        for name in (list(variants) if repeat%2==0 else list(variants)[::-1]):
            factory,config=variants[name]; manager=factory(config); signs=[]; samples=[]
            for index,(frame,output) in enumerate(zip(frames,outputs)):
                start=perf_counter(); r=manager.precheck(frame); manager.commit(frame,output,r)
                if index: samples.append((perf_counter()-start)*1000)
                signs.append(fingerprint(manager,r))
            if expected is None: expected=signs
            if signs!=expected: raise AssertionError('Decision/mask/state/association mismatch: '+name)
            rec=records.setdefault(name,dict(samples_ms=[],exact_reference_parity=True))
            if repeat: rec['samples_ms']+=samples
    for rec in records.values(): rec['p50_p95_ms']=np.percentile(rec['samples_ms'],[50,95]).tolist()
    chosen=min(range(1,7),key=lambda w: records.get(f'workers_{w}',{'p50_p95_ms':[float('inf')]})['p50_p95_ms'][0])
    write_json(root/'cpu_benchmark.json',dict(records=records,selected_workers=chosen,
        warmup=1,repeats=4,sample_ids=[f.frame_id for f in frames],opencv_threads=cv2.getNumThreads(),
        scope='Preloaded real six-camera frames; excludes IO, model and offline oracle. Interleaved modes.'))
    return chosen


def collect(root, split, method):
    paths=sorted((root/'scene_results'/split).glob('*/'+method))
    rows=[]; pixels=[]; novel=[]; times=[]
    for path in paths:
        for dest,name in [(rows,'oracle_records'),(pixels,'pixels'),(novel,'new_regions'),(times,'timings')]:
            dest.extend(json.loads((path/(name+'.json')).read_text()))
    summary=summarize_rows(rows)
    summary.update(scenes=len(paths),invalidation_area_ratio=float(np.mean([p['invalidation_area_ratio'] for p in pixels])),
        new_regions=len(novel),new_region_recall=ratio(sum(r['recalled'] for r in novel),len(novel)),
        m1_wall_p50_ms=float(np.median([t['m1_overall_wall_ms'] for t in times])),
        accepted_unique_instances=len({r['identity'] for r in rows if r['accepted'] and r['identity']}),
        accepted_scenes=len({r['scene_id'] for r in rows if r['accepted'] and r['oracle_valid'] is not None}))
    return summary,rows


def quality_accept(row, q):
    score,history,innovation=q
    return bool(row['accepted'] and (score==0 or row['source_score'] is not None and row['source_score']>=score)
        and row['history_observations']>=history
        and (innovation==0 or row['last_innovation_m'] is not None and row['last_innovation_m']<=innovation))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/module1_optimization')
    parser.add_argument('--selected-config',default='configs/worldstream_m1_balanced.yaml')
    parser.add_argument('--predictions',default='test/mask_eva_lane_det_vlm/Tue_Sep_29_20_12_56_2026/pts_bbox/results_nusc.json')
    args=parser.parse_args(); root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists(): raise ValueError('Protocol exists: choose fresh output; do not silently reuse results')
    cv2.setNumThreads(1)
    cfg=replace(M1Config.load('configs/worldstream_m1_refinement.yaml'),logging=False,visualization=False)
    infos=pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb'))['infos']
    scenes=sorted({i['scene_token'] for i in infos})
    specs={split:{f'scene_{idx:02d}':sorted([i for i in infos if i['scene_token']==scenes[idx]],key=lambda i:i['timestamp'])[:12]
                 for idx in indices} for split,indices in [('calibration',range(12,28)),('holdout',range(32,48))]}
    protocol=dict(scene_splits={split:{name:[i['token'] for i in items] for name,items in names.items()} for split,names in specs.items()},
        image_policies=POLICIES,quality_candidates=QUALITY,baseline_config=cfg.to_dict(),
        selection='Calibration only. At least 10 scorable accepted instances in 3 scenes to select a quality gate; otherwise retain unfiltered exact policy. Feasible: risk <= reference, candidate coverage >= 75% reference, valid recovery >= 90% reference, new-region recall >= reference - 0.005, invalidation <= reference + 0.02. Pareto then minimum median cost; ties prefer recovery.',
        source='Historical Full-Model Prediction Replay',official_test=False,
        limitation='All scenes are local nuScenes validation predictions; new to this optimization, not official held-out test. Per-trial intervals are descriptive because repeated identities/frames correlate.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('worldstream/m1').glob('*.py'))})
    write_json(root/'protocol.json',protocol) # before opening any labels or metrics
    all_infos=[i for names in specs.values() for items in names.values() for i in items]
    bench_infos=sorted([i for i in infos if i['scene_token']==scenes[0]],key=lambda i:i['timestamp'])[:4]
    predictions=read_prediction_subset(args.predictions,{i['token'] for i in all_infos+bench_infos})
    ref=reference(root)
    workers=benchmark(root,[frame_from_info(i,'data/nuscenes') for i in bench_infos],predictions,cfg,ref)
    cfg=replace(cfg,temporal_workers=workers)
    oracle=load_oracle(all_infos,'data/nuscenes')
    print('Selected workers:',workers,flush=True)
    for name,items in specs['calibration'].items():
        frames=[frame_from_info(i,'data/nuscenes') for i in items]
        for method,changes in POLICIES.items():
            run_refinement.run_scene(root/'scene_results/calibration'/name,frames,predictions,oracle,replace(cfg,**changes),method,'calibration')
    candidates=[]
    for method in POLICIES:
        summary,rows=collect(root,'calibration',method)
        for q in QUALITY:
            filtered=[dict(r,accepted=quality_accept(r,q)) for r in rows]
            metrics={**summary,**summarize_rows(filtered)}
            candidates.append(dict(method=method,quality=q,metrics=metrics))
    base=candidates[0]['metrics']; enough=base['accepted_unique_instances']>=10 and base['accepted_scenes']>=3
    def feasible(c):
        m=c['metrics']
        return (m['physical_false_acceptance_rate'] is not None and base['physical_false_acceptance_rate'] is not None
            and m['physical_false_acceptance_rate']<=base['physical_false_acceptance_rate']
            and m['valid_candidate_rate']>=.75*base['valid_candidate_rate']
            and m['valid_state_recovery_rate']>=.9*base['valid_state_recovery_rate']
            and (base['new_region_recall'] is None or m['new_region_recall'] is not None and m['new_region_recall']>=base['new_region_recall']-.005)
            and m['invalidation_area_ratio']<=base['invalidation_area_ratio']+.02)
    feasible_set=[c for c in candidates if feasible(c)] if enough else [candidates[0]]
    if not feasible_set: feasible_set=[candidates[0]]
    def axes(c):
        m=c['metrics'];return (m['physical_false_acceptance_rate'] or 0,-(m['valid_state_recovery_rate'] or 0),m['invalidation_area_ratio'],m['m1_wall_p50_ms'])
    pareto=[c for c in feasible_set if not any(all(x<=y for x,y in zip(axes(o),axes(c))) and any(x<y for x,y in zip(axes(o),axes(c))) for o in feasible_set)]
    best=min(pareto,key=lambda c:(c['metrics']['m1_wall_p50_ms'],-(c['metrics']['valid_state_recovery_rate'] or 0),sum(c['quality'])))
    score,history,innovation=best['quality']
    selected=replace(cfg,**POLICIES[best['method']],certification_min_source_score=score,
        certification_min_history=history,certification_max_innovation_m=innovation)
    write_json(root/'selection.json',dict(calibration_candidates=candidates,adequate_independent_support=enough,
        pareto=pareto,selected=best,selected_config=selected.to_dict(),frozen_before_holdout=True))
    import yaml
    deployment=replace(selected,logging=True,log_path=str(root/'online_state.jsonl'))
    Path(args.selected_config).write_text(yaml.safe_dump(deployment.to_dict(),sort_keys=False))
    # No selection changes after this point.
    for name,items in specs['holdout'].items():
        frames=[frame_from_info(i,'data/nuscenes') for i in items]
        for method,config,factory in [('reference',cfg,ref),('balanced',selected,WorldStateManager)]:
            original=run_refinement.WorldStateManager
            try:
                run_refinement.WorldStateManager=factory
                run_refinement.run_scene(root/'scene_results/holdout'/name,frames,predictions,oracle,config,method,'holdout')
            finally: run_refinement.WorldStateManager=original
    write_json(root/'holdout_summary.json',{m:collect(root,'holdout',m)[0] for m in ['reference','balanced']})
    print('Complete:',root,flush=True)

if __name__=='__main__': main()
