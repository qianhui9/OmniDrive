"""Freeze scene acquisition before new model predictions or quality labels."""
import json
from pathlib import Path
import re
import subprocess
from .io_utils import write_json


def main():
    root=Path('outputs/module1_evidence_optimization')
    available=json.loads((root/'train_data_availability.json').read_text())['accessible_endpoint_scenes']
    scene_table=json.loads(Path('data/nuscenes/v1.0-trainval/scene.json').read_text())
    logs={s['token']:s['log_token'] for s in scene_table}
    # Audit only recorded M1 outcomes/protocols, not model training membership.
    pattern=re.compile(r'"(?:scene|scene_id|scene_token)"\s*:\s*"([0-9a-f]{32})"')
    seen=set();audited=[]
    for directory in sorted(Path('outputs').iterdir()):
        if not directory.is_dir() or directory==root:continue
        if not (directory.name.startswith('module1') or directory.name.startswith('system_optimization')):continue
        for path in directory.rglob('*.json'):
            if any(p in path.parts for p in ('source','baseline_source','delivery_source')):continue
            # Protocols and metrics record explicit scene membership. Large raw
            # predictions/annotations are not evidence of prior M1 development.
            if path.stat().st_size>32*1024*1024:continue
            seen.update(pattern.findall(path.read_text(errors='replace')));audited.append(str(path))
        for path in directory.rglob('*.jsonl'):
            if any(p in path.parts for p in ('source','baseline_source','delivery_source')):continue
            # Each stream directory is scene-local; sample head and tail to
            # audit membership without reparsing all dense mask payloads.
            with path.open(errors='replace') as handle:
                for _,line in zip(range(2),handle):seen.update(pattern.findall(line))
            audited.append(str(path))
    candidates=[s for s in available if s['scene_id'] not in seen]
    used_logs=set();chosen=[]
    import hashlib
    for spec in sorted(candidates,key=lambda s:hashlib.sha256(('M1-evidence-v1:'+s['scene_id']).encode()).hexdigest()):
        log=logs[spec['scene_id']]
        if log in used_logs:continue
        used_logs.add(log);chosen.append(dict(spec,log_token=log))
        if len(chosen)==38:break
    if len(chosen)<38:raise RuntimeError('Insufficient new-to-M1 scenes with distinct logs')
    splits=dict(calibration=chosen[:20],heldout=chosen[20:26],final_test=chosen[26:])
    for split,specs in splits.items():
        for spec in specs:spec['split']=split
    write_json(root/'independent_data_audit.json',dict(audited_files=audited,previously_recorded_scenes=sorted(seen),
        train_total_scenes=len(available),M1_unexposed_train_scenes=len(candidates),selected_scenes=len(chosen),
        selected_distinct_logs=len(used_logs),model_training_exposure=True,
        independence_scope='New to locally recorded M1 development, disjoint scenes and logs; NOT model-heldout or official hidden test',
        test_split=dict(images_available=True,annotation_available=False,quality_evaluation='Not Run: missing labels'),
        prior_val_population='All 150 scenes remain development/reproduction'))
    write_json(root/'data_selection.json',dict(frozen_before_new_predictions=True,info_file='data/nuscenes/nuscenes2d_ego_temporal_infos_train.pkl',
        splits=splits,model_training_exposure=True,actual_command=['taskset','-c','0-3,6-31',*__import__('sys').argv],
        checkpoint='ckpts/iter_10548.pth',model_processing='Unchanged original test pipeline, detector/map branches only; no language study',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    print('frozen acquisition:',{s:len(v) for s,v in splits.items()},'new to local M1 records; model train population',flush=True)


if __name__=='__main__':main()
