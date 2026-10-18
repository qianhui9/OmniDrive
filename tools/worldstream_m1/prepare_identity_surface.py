"""Freeze fresh M1 scene/log acquisition, without reading quality labels."""
import hashlib
import json
from pathlib import Path
import subprocess
from .io_utils import write_json


def main():
    root=Path('outputs/module1_identity_surface_refinement');root.mkdir(exist_ok=True)
    if (root/'data_selection.json').exists():raise RuntimeError('Preserve frozen selection')
    prior=Path('outputs/module1_evidence_optimization')
    audit=json.loads((prior/'independent_data_audit.json').read_text())
    previous=json.loads((prior/'data_selection.json').read_text())
    seen=set(audit['previously_recorded_scenes'])|{s['scene_id'] for specs in previous['splits'].values() for s in specs}
    table=json.loads(Path('data/nuscenes/v1.0-trainval/scene.json').read_text())
    logs={s['token']:s['log_token'] for s in table};seen_logs={logs[s] for s in seen if s in logs}
    available=json.loads((prior/'train_data_availability.json').read_text())['accessible_endpoint_scenes']
    options=[s for s in available if s['scene_id'] not in seen and logs[s['scene_id']] not in seen_logs]
    chosen=[];used=set()
    for spec in sorted(options,key=lambda s:hashlib.sha256(('identity-surface-v1:'+s['scene_id']).encode()).hexdigest()):
        log=logs[spec['scene_id']]
        if log in used:continue
        used.add(log);chosen.append(dict(spec,log_token=log))
    if len(chosen)<12:raise RuntimeError('Expected 12 untouched scene/log groups; audit changed')
    chosen=chosen[:12];splits=dict(calibration=chosen[:4],heldout=chosen[4:8],final_test=chosen[8:12])
    for stage,specs in splits.items():
        for spec in specs:spec['split']=stage
    selection=dict(splits=splits,info_file=previous['info_file'],frozen_before_new_predictions=True,
        checkpoint=previous['checkpoint'],model_training_exposure=True,
        scope='New to recorded M1 scene AND acquisition log, original model train population',
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    write_json(root/'data_selection.json',selection)
    write_json(root/'independent_reference_audit.json',dict(selected_scenes=len(chosen),distinct_new_logs=len(used),
        splits={s:len(v) for s,v in splits.items()},prior_seen_scenes=sorted(seen),prior_seen_logs=sorted(seen_logs),
        selection_without_quality_labels=True,model_training_exposure=True,model_generalization_test=False,
        identity_reference='Offline unique same-class 2m + .25m competition annotation, censored/unscorable explicitly counted',
        surface_reference='Offline synchronized annotation-owned LiDAR, not instance pixel masks',
        calibration_for_selection='Fresh four-scene calibration; prior 20 calibration scenes are development only',
        minimum_zero_error_independent_validation_units=59,
        maximum_available_fresh_logs=len(used),deployment_sample_size_sufficient=False))
    print({s:len(v) for s,v in splits.items()},'new scene/log groups, model-train population')


if __name__=='__main__':main()
