"""Local nuScenes-only availability, exposure and cache audit. No adapter."""
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import pickle
import numpy as np
from .dataset import resolve_image
from .io_utils import iter_json_array,write_json

ROOT=Path('outputs/module1_physical_observability_optimization')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    data=Path('data/nuscenes');tables=data/'v1.0-trainval'
    scenes=json.loads((tables/'scene.json').read_text());logs={s['token']:s['log_token'] for s in scenes}
    sample_scene={s['token']:s['scene_token'] for s in json.loads((tables/'sample.json').read_text())}
    inventory_path=Path('outputs/module1_quality_coverage_optimization/historical_exposure_inventory.json')
    history=json.loads(inventory_path.read_text());witness=defaultdict(list)
    for r in history:
        # Availability inventories and split declarations never establish exposure.
        if not any(s in r['path'] for s in ('oracle_records.jsonl','pixels.jsonl','identity.jsonl','state.jsonl','frame_metrics.jsonl','evaluations.jsonl')):continue
        observed=set(r.get('logs',[]));observed.update(logs[s] for s in r.get('scenes',[]) if s in logs)
        observed.update(logs[sample_scene[s]] for s in r.get('samples',[]) if s in sample_scene)
        for log in observed:witness[log].append(r['path'])
    exposure={}
    for log,paths in sorted(witness.items()):
        p=Path(sorted(paths,key=lambda s:('pixels.jsonl' not in s,'identity.jsonl' not in s,s))[0])
        if not p.is_file():raise RuntimeError('Historical exposure witness missing: '+str(p))
        exposure[log]=dict(path=str(p),sha256=sha(p),historical_state_or_quality_trace=True)
    annotations=Counter(a['sample_token'] for a in iter_json_array(tables/'sample_annotation.json'))
    cache_records=[];cached=set()
    for root in ('module1_evidence_optimization','module1_identity_surface_refinement'):
        p=Path('outputs')/root/'model_observation_provenance.json';provenance=json.loads(p.read_text())
        prediction=p.with_name('model_observations.pkl');digest=sha(prediction)
        assert digest==provenance['prediction_sha256']
        cached.update(r['sample'] for r in provenance['frames'])
        cache_records.append(dict(path=str(prediction),sha256=digest,provenance_path=str(p),provenance_sha256=sha(p),
            frames=len(provenance['frames']),scenes=len({r['scene'] for r in provenance['frames']}),
            original_model=True,label_inputs_used=provenance['label_inputs_used']))
    selections=json.loads((ROOT/'data_selection.json').read_text());wanted={t for s in selections['development'] for t in s['samples']}
    info_inventory=[];scene_stats=defaultdict(Counter);used_files={};training_scenes=set();frame_logs={}
    for split in ('train','val'):
        path=data/('nuscenes2d_ego_temporal_infos_'+split+'.pkl');infos=pickle.load(path.open('rb'))['infos'];counts=Counter()
        for info in infos:
            scene=info['scene_token'];log=logs[scene];frame_logs[info['token']]=log
            if split=='train':training_scenes.add(scene)
            counts['frames']+=1;local=scene_stats[scene];local['info_frames']+=1
            counts['annotated_frames']+=annotations[info['token']]>0
            local['annotated_frames']+=annotations[info['token']]>0
            cameras=info['cams'];available=0
            for cid,cam in cameras.items():
                try:rgb=resolve_image(cam['data_path'],data);exists=rgb.is_file()
                except FileNotFoundError:exists=False
                counts['camera_files']+=1;counts['available_camera_files']+=exists;available+=exists
                finite=all(np.isfinite(cam[k]).all() for k in ('cam_intrinsic','sensor2ego_rotation','sensor2ego_translation','ego2global_rotation','ego2global_translation'))
                counts['finite_camera_calibration_pose']+=finite
                if exists and info['token'] in wanted:used_files[str(rgb)]=sha(rgb)
            six=len(cameras)==6 and available==6
            counts['complete_six_camera_frames']+=six;local['complete_six_camera_frames']+=six
            try:lidar=resolve_image(info['lidar_path'],data);exists=lidar.is_file()
            except FileNotFoundError:exists=False
            counts['available_lidar_frames']+=exists;local['available_lidar_frames']+=exists
            if exists and info['token'] in wanted:used_files[str(lidar)]=sha(lidar)
        info_inventory.append(dict(split=split,path=str(path),sha256=sha(path),bytes=path.stat().st_size,
            scenes=len({i['scene_token'] for i in infos}),logs=len({logs[i['scene_token']] for i in infos}),**counts))
        del infos
    scene_inventory=[dict(scene=s['token'],log_token=s['log_token'],historically_exposed=s['log_token'] in exposure,
        model_training_population=s['token'] in training_scenes,**scene_stats[s['token']]) for s in scenes]
    unseen={s['log_token'] for s in scene_inventory if not s['historically_exposed'] and s.get('complete_six_camera_frames',0) and s.get('annotated_frames',0)}
    other=[]
    for name,path in [('test',Path('/data/qh/nuscenes/v1.0-test')),('mini',Path('/data/qh/nuscenes/v1.0-mini/v1.0-mini'))]:
        if not (path/'scene.json').exists():continue
        ss=json.loads((path/'scene.json').read_text());ann=path/'sample_annotation.json'
        n=sum(1 for _ in iter_json_array(ann)) if ann.exists() else 0
        other.append(dict(split=name,metadata_path=str(path),scenes=len(ss),logs=len({s['log_token'] for s in ss}),
            new_logs=sorted({s['log_token'] for s in ss}-set(exposure)),annotation_records=n,
            metadata_sha256={p.name:sha(p) for p in (path/'scene.json',ann) if p.is_file()},
            eligible_independent_quality=False,reason='No labeled unexposed acquisition log' if name=='mini' else 'Official local test annotations are empty; camera-only Test cannot score physical quality'))
    table_pins={str(tables/name):sha(tables/name) for name in ('scene.json','sample.json','log.json','calibrated_sensor.json','ego_pose.json','sample_annotation.json')}
    write_json(ROOT/'nuscenes_scene_inventory.json',scene_inventory)
    write_json(ROOT/'historical_exposure_audit.json',dict(acquisition_log_witnesses=exposure,
        prior_inventory_path=str(inventory_path),prior_inventory_sha256=sha(inventory_path),
        availability_inventories_count_as_exposure=False,all_68_trainval_logs_have_actual_historical_traces=set(logs.values())<=set(exposure)))
    result=dict(dataset='nuScenes',experiment_environment='nuScenes + original OmniDrive + M1',
        audit_kind='same_dataset_availability_and_exposure; no cross-dataset adapter',trainval_scenes=len(scenes),trainval_logs=len(set(logs.values())),
        infos=info_inventory,other_nuscenes_splits=other,original_prediction_caches=cache_records,
        cache_scope='Two audited M1 full-model output caches; feature pickles are not prediction caches',
        raw_availability_audit_scope='Every frame in both actual Train/Val info files; six actual camera files and offline LiDAR checked',
        raw_unexposed_labeled_logs=sorted(unseen),new_independent_quality_possible=bool(unseen),
        metadata_sha256=table_pins,exposure_audit_sha256=sha(ROOT/'historical_exposure_audit.json'),
        qualified_certifications=0,population='exploratory_reused_population',
        statistical_limitation=dict(unit='acquisition log, any wrong acceptance is a failure',
            zero_error_independent_fully_scorable_accepted_logs_required=math.ceil(math.log(.05)/math.log(.95)),
            new_available=0,repeat_scenes_frames_objects_do_not_add_units=True),
        reference_scope='LiDAR/Annotation strictly offline; normal M1 sees RGB, real calibration, ego pose, and original predictions')
    write_json(ROOT/'independent_data_adapter_audit.json',result)
    write_json(ROOT/'data_manifest.json',dict(used_development_files_sha256=used_files,metadata_sha256=table_pins,
        checkpoint=dict(path='ckpts/iter_10548.pth',sha256=sha('ckpts/iter_10548.pth')),
        cache_provenance=cache_records,original_model_training_exposure=True,model_generalization_test=False))
    print(json.dumps(dict(info_inventory=info_inventory,trainval_logs=len(set(logs.values())),historically_exposed_logs=len(exposure),unseen_labeled_logs=len(unseen),used_files=len(used_files)),indent=2))


if __name__=='__main__':main()
