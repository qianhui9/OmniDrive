"""Scene-isolated full Planning comparisons through the existing model_smoke tool.

Each scene runs in a fresh process/model/M1 instance. Weights, generation and
M2/M3 mechanisms are unchanged. This is an experiment driver, not a runtime.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from .effective_acceptance import sha, verify
from .scene_protocol import select_scenes,aggregate
from .scene_metrics import summarize_scene
from .dataset import log_context_from_root


def source_manifest():
    paths=[*Path('worldstream/m1').glob('*.py'),
           Path('projects/mmdet3d_plugin/models/detectors/petr3d.py'),
           *[Path('configs')/n for n in ('worldstream_m1_effective.yaml','worldstream_m11_diagnostics.yaml')],
           *[Path('tools/worldstream_m1')/n for n in ('model_smoke.py','effective_acceptance.py',
             'effective_gpu_validation.py','scene_protocol.py','scene_metrics.py','dataset.py','observation_diagnostics.py')]]
    return {str(p):sha(p) for p in paths}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--frames',type=int,default=12)
    p.add_argument('--scene');p.add_argument('--scene-ids',help='Comma-separated fixed target scene IDs')
    p.add_argument('--scenes',type=int,default=2,help='0: all configured dataset scenes')
    p.add_argument('--config',default='projects/configs/OmniDrive/worldstream_m11_diagnostics.py')
    p.add_argument('--m1-config',default='configs/worldstream_m11_diagnostics.yaml')
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    write=lambda name,value:(root/name).write_text(json.dumps(value,indent=2)+'\n')
    manifest=source_manifest();write('source_before.json',manifest)
    from mmcv import Config
    cfg=Config.fromfile(a.config)
    assert cfg.model.get('m2_enable') is False and cfg.model.get('m3_enable') is False
    with open(cfg.data.test.ann_file,'rb') as f:infos=pickle.load(f)['infos']
    context=log_context_from_root('data/nuscenes')
    sequences,selection=select_scenes(infos,None,a.scenes,a.frames,context)
    explicit=a.scene_ids.split(',') if a.scene_ids else [a.scene] if a.scene else None
    if explicit:
        if len(set(explicit))!=len(explicit):raise ValueError('Duplicate target scene IDs')
        sequences,selection=select_scenes(infos,None,0,a.frames,context)
        if set(explicit)-set(sequences):raise ValueError('Target scene absent from configured dataset')
        sequences={s:sequences[s] for s in explicit}
        selection['sampling_rule']='Explicit frozen scene IDs; timestamp-sorted available keyframe prefixes'
        for spec in selection['scenes']:
            spec['scheduled']=spec['scene_id'] in sequences;spec['status']='Scheduled' if spec['scheduled'] else 'Not Run'
        selection['scheduled_scene_count']=len(sequences)
        selection['unexecuted_scene_ids']=[s['scene_id'] for s in selection['scenes'] if not s['scheduled']]
    write('selection.json',selection)
    checkpoints={}
    for path in [Path('ckpts/iter_10548.pth'),*sorted(Path('ckpts/pretrain_qformer').glob('*'))]:
        if not path.is_file():continue
        print('Hashing',str(path),flush=True)
        checkpoints[str(path)]=dict(bytes=path.stat().st_size,sha256=sha(path))
        write('checkpoint_digests.json',checkpoints)
    write('provenance.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        affinity=sorted(os.sched_getaffinity(0)),python=sys.executable,checkpoint=checkpoints,
        dataset=cfg.data.test.ann_file,dataset_sha256=sha(cfg.data.test.ann_file),
        log_context_sha256=context.digest,population='Exploratory; no independent qualification',
        m2_enabled=False,m3_enabled=False,planning_task_unchanged=True))
    summaries=[];parity=[];(root/'scenes').mkdir()
    for scene,sequence in sequences.items():
        folder=root/'scenes'/scene
        cmd=[sys.executable,'-m','tools.worldstream_m1.model_smoke','--config',a.config,
             '--m1-config',a.m1_config,'--compare','--frames',str(a.frames),'--scene',scene,'--output',str(folder)]
        with (root/'scenes'/f'{scene}.stdout.log').open('w') as out:
            result=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT)
        if result.returncode:
            write('failure.json',dict(scene=scene,returncode=result.returncode,completed_scenes=[s['scene_id'] for s in summaries]))
            raise RuntimeError('Scene model validation failed: '+scene)
        rows=[json.loads(line) for line in (folder/'m1_state.jsonl').read_text().splitlines()]
        model_result=json.loads((folder/'results.json').read_text())
        measurements={x['frame_id']:x for x in model_result['observe_only']['frame_measurements']}
        for index,row in enumerate(rows):row['experiment']=dict(latency=measurements[row['precheck']['frame_id']],dataset_sequence_index=index)
        summary=summarize_scene(rows,verify(folder/'m1_state.jsonl'))
        summary['baseline_latency']=[x['end_to_end_wall_ms'] for x in model_result['baseline']['frame_measurements']]
        baseline=folder/'baseline/language';observed=folder/'observe_only/language'
        names=sorted(p.name for p in baseline.iterdir())
        comparisons=[dict(frame=n,equal=(baseline/n).read_bytes()==(observed/n).read_bytes(),
                          baseline_sha256=sha(baseline/n),observed_sha256=sha(observed/n)) for n in names]
        equal=names==sorted(p.name for p in observed.iterdir()) and all(c['equal'] for c in comparisons)
        (folder/'language_comparison.json').write_text(json.dumps(dict(equal=equal,files=comparisons),indent=2)+'\n')
        (folder/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
        summaries.append(summary);parity.append(dict(scene_id=scene,language_equal=equal,native_model_parity=model_result['observe_only_equal']))
        write('summary.json',dict(aggregate(summaries),parity=parity,executed_scene_ids=[s['scene_id'] for s in summaries],
            unexecuted_scene_ids=[s['scene_id'] for s in selection['scenes'] if s['scene_id'] not in {x['scene_id'] for x in summaries}],
            full_dataset_validated=False,independent_scene_execution=True))
        if not equal or not model_result['observe_only_equal']:raise AssertionError('Native parity failed')
        print('Completed independent model scene',scene,flush=True)
    after=source_manifest();write('source_after.json',after)
    write('final_verification.json',dict(sources_unchanged=manifest==after,
        all_scene_native_parity=all(x['native_model_parity'] and x['language_equal'] for x in parity)))
    assert manifest==after


if __name__=='__main__':main()
