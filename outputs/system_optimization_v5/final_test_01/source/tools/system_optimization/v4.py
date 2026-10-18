"""Gate-controlled v4: native Planning, fresh exact cost evidence, final Auto.

No v3 certificates are imported. Full original sampling/weights are retained.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess

from .audit import write

ROOT = Path('outputs/system_optimization_v4')


def freeze():
    """Select by scene/location metadata only, before new model outcomes."""
    path=ROOT/'protocol.json'
    if path.exists():
        raise FileExistsError('Frozen protocol exists; never overwrite outcomes/protocol')
    info_path=Path('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl')
    infos=pickle.load(info_path.open('rb'))['infos']
    groups={}
    for i in infos:groups.setdefault(i['scene_token'],[]).append(i)
    # Exclude every v3 executed scene and all old exploratory false accepts.
    excluded=set()
    for p in Path('outputs/system_optimization_v3').glob('*/protocol.json'):
        d=json.loads(p.read_text())
        excluded.update(s['scene'] for s in d.get('scenes',[]))
    diagnosis=json.loads(Path('outputs/system_optimization_v3/m1_false_accept_diagnosis.json').read_text())
    for case in diagnosis['cases']:
        excluded.add(case.get('scene',case.get('scene_id',case.get('oracle',{}).get('scene'))))
    pools={}
    for scene, rows in sorted(groups.items()):
        if scene in excluded or len(rows)<8:continue
        city='boston' if rows[0]['location'].startswith('boston') else 'singapore'
        pools.setdefault(city,[]).append(scene)
    selections={}
    for offset,split in enumerate(('calibration','heldout','final_test')):
        chosen=sorted(s for city in sorted(pools) for s in pools[city][3*offset:3*offset+3])
        if len(chosen)!=6:raise ValueError('Need three scenes per city per split')
        selections[split]=[dict(scene=s,location=groups[s][0]['location'],
            tokens=[i['token'] for i in sorted(groups[s],key=lambda i:i['timestamp'])[:8]]) for s in chosen]
    previous=json.loads(Path('outputs/system_optimization_v3/protocol.json').read_text())
    write(path,dict(frozen_before_v4_gpu_outcomes=True,gate_a='task_state_feasibility.json',
        online_bc_allowed=False,split_selection='sorted unused-v3 scenes, three per city per split; no outcome selection',
        selections=selections,excluded_scenes=sorted(s for s in excluded if s),
        performance=previous['performance'],physical_risk=previous['physical_risk'],
        task_quality=previous['task_quality'],generation=previous['generation'],
        cpu_affinity='0-3,6-31',checkpoint='ckpts/iter_10548.pth',
        calibration_frames_per_scene=8,final_frames_per_scene=8,
        calibration_modes=['m1','overlap'], final_modes=['original','m1','overlap','unified_auto'],
        warmup=0,repeats=1,
        cost_domain='exact question lengths, cold/warm; warm observed calibration min/max within frozen [.49,.51]; no expansion',
        certificate_scope='this exact instrumented implementation and CPU resources; not uninstrumented deployment',
        streaming=dict(policy='FCFS, every request retained, one persistent inference worker per scene',
            deadline_s=.5,observation_horizon='last real sensor arrival; also report drain',
            execution_cost_source='measured per-frame E2E incl input preparation, no fabricated service times',
            limitations='trace-driven replay; extra queueing wait may stale historical state; no claim of live serving validation'),
        final_test='local validation-scene holdout, not official hidden nuScenes test'))
    for name in ('logs','traces','figures'): (ROOT/name).mkdir(parents=True,exist_ok=True)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def gpu_main():
    import cv2
    import torch
    import transformers
    if set(os.sched_getaffinity(0)) & {4,5}:raise RuntimeError('Forbidden CPUs 4/5')
    if json.loads((ROOT/'task_state_feasibility.json').read_text())['gate_a']['online_bc_allowed']:
        raise RuntimeError('This runner implements the Gate A No-Go branch only')
    from . import v2
    from .v3 import quality, environment
    from .v3_costs import calibrate
    from tools.worldstream_m2.rearchitecture.integration import load
    torch.set_num_threads(1);cv2.setNumThreads(1);torch.backends.cuda.matmul.allow_tf32=False
    protocol=json.loads((ROOT/'protocol.json').read_text())
    manifest=dict(checkpoint=protocol['checkpoint'],sha256=sha256(protocol['checkpoint']),
        lora='embedded original checkpoint; no merge/retraining',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__,
        cpu_affinity=sorted(os.sched_getaffinity(0)))
    expected=json.loads(Path('outputs/system_optimization_v3/model_manifest.json').read_text())['sha256']
    if manifest['sha256']!=expected:raise AssertionError('Original checkpoint changed')
    write(ROOT/'model_manifest.json',manifest);write(ROOT/'environment_before.json',environment())
    v2.ROOT=ROOT
    model,dataset=load()
    class Driver(v2.Driver):
        def setup(self,model,mode,*args):
            super().setup(model,mode,*args)
            # All v4 reference/overlap/auto instances use identical trace and
            # cost-key capture scope. No imported instrumented v3 permission.
            model.stream_runtime.config=replace(model.stream_runtime.config,
                capture_cost_key=mode in ('overlap','unified_auto'))
    driver=Driver(model,dataset)
    scenes=sorted({i['scene_token'] for i in dataset.data_infos})
    try:
        for split in ('calibration','heldout','final_test'):
            for ordinal,spec in enumerate(protocol['selections'][split]):
                name=f'{split}_{ordinal:02d}'
                modes=protocol['final_modes'] if split=='final_test' else protocol['calibration_modes']
                frames=protocol['final_frames_per_scene'] if split=='final_test' else protocol['calibration_frames_per_scene']
                # Rotate method order between scenes using the existing runner.
                ordered=modes[ordinal%len(modes):]+modes[:ordinal%len(modes)]
                args=v2.arguments(name,'single',scenes.index(spec['scene']),1,frames,ordered,warmup=0)
                # Original is required first by the native reference writer;
                # remaining modes still rotate, and all paired states reset.
                if 'original' in ordered:
                    args.modes=['original']+[m for m in ordered if m!='original']
                args.scene_splits={spec['scene']:split}
                print(f'RUN {name}: {args.modes}, {spec["scene"]}',flush=True)
                driver.run(args);quality(ROOT/name)
                write(ROOT/'progress.json',dict(last_completed=name,split=split))
            if split=='heldout':
                runs=tuple(f'{s}_{i:02d}' for s in ('calibration','heldout') for i in range(6))
                calibrate(ROOT,runs)
                print('COST CERTIFICATE',json.loads((ROOT/'cost_calibration.json').read_text())['approved_keys'],flush=True)
        write(ROOT/'environment_after.json',environment())
        manifest['checkpoint_sha256_after']=sha256(protocol['checkpoint'])
        manifest['weights_unchanged']=manifest['checkpoint_sha256_after']==manifest['sha256']
        write(ROOT/'model_manifest.json',manifest)
    finally:driver.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');args=p.parse_args()
    if args.freeze:freeze()
    else:
        if (ROOT/'gpu_status.json').exists():raise FileExistsError('Use existing progress for recovery; never overwrite a run')
        write(ROOT/'gpu_status.json',dict(status='starting'))
        try:gpu_main()
        except BaseException as e:
            write(ROOT/'gpu_status.json',dict(status='failed',error=repr(e)));raise
        else:write(ROOT/'gpu_status.json',dict(status='completed'))
