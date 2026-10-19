"""Inventory actual local references and RGB availability, independent of cache selection."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import pickle
from .io_utils import write_json,iter_json_array
from .dataset import resolve_image

ROOT=Path('outputs/module1_quality_coverage_refinement')
OLD=Path('outputs/module1_quality_coverage_optimization')


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    old=json.loads((OLD/'data_selection.json').read_text());seen=set(old['previously_exposed_logs'])
    previous=json.loads(Path('outputs/module1_evidence_optimization/data_selection.json').read_text())
    cached=set()
    for root in [Path('outputs/module1_evidence_optimization'),Path('outputs/module1_identity_surface_refinement')]:
        p=root/'model_observation_provenance.json'
        if p.exists():cached.update(r['sample'] for r in json.loads(p.read_text())['frames'])
    infofiles=[Path('data/nuscenes/nuscenes2d_ego_temporal_infos_train.pkl'),Path('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl')]
    infos={};info_inventory=[];training_scenes=set()
    for path in infofiles:
        data=pickle.load(path.open('rb'))['infos'];infos.update({i['token']:i for i in data})
        if path.name.endswith('_train.pkl'):training_scenes.update(i['scene_token'] for i in data)
        info_inventory.append(dict(path=str(path),frames=len(data),scene_count=len({i['scene_token'] for i in data}),bytes=path.stat().st_size))
    scene_table=json.loads(Path('data/nuscenes/v1.0-trainval/scene.json').read_text())
    annotations=Counter(a['sample_token'] for a in iter_json_array('data/nuscenes/v1.0-trainval/sample_annotation.json'))
    grouped={s['token']:[] for s in scene_table}
    for token,info in infos.items():grouped[info['scene_token']].append(info)
    inventory=[]
    for scene in scene_table:
        selected=sorted(grouped[scene['token']],key=lambda i:i['timestamp'])[:12];samples=[]
        for info in selected:
            files=[]
            for cam in info['cams'].values():
                try:p=resolve_image(cam['data_path'],'data/nuscenes');files.append(str(p) if p.is_file() else None)
                except FileNotFoundError:files.append(None)
            samples.append(dict(sample_token=info['token'],timestamp=info['timestamp']/1e6,
                annotation_count=annotations[info['token']],camera_files=files,
                raw_camera_count=sum(f is not None for f in files),cached_original_model_output=info['token'] in cached))
        inventory.append(dict(scene_token=scene['token'],log_token=scene['log_token'],
            log_exposed=scene['log_token'] in seen,info_frames=len(grouped[scene['token']]),samples=samples,
            model_training_population=scene['token'] in training_scenes))
    # Camera-only test logs are distinct raw acquisitions, but have no official
    # identity/geometry labels. Do not confuse them with eligible quality units.
    tables=[]
    for path in [Path('/data/qh/nuscenes/v1.0-test/scene.json'),Path('/data/qh/nuscenes/v1.0-mini/v1.0-mini/scene.json')]:
        if not path.exists():continue
        data=json.loads(path.read_text());annotation=path.with_name('sample_annotation.json')
        tables.append(dict(path=str(path),scenes=len(data),distinct_logs=len({s['log_token'] for s in data}),
            unseen_logs=sorted({s['log_token'] for s in data}-seen),annotation_table_available=annotation.exists(),
            annotation_records=sum(1 for _ in iter_json_array(annotation)) if annotation.exists() else 0,
            quality_eligible=False if 'test/' in str(path) else None))
    external=[]
    roots=[Path('/data/qh/V2X-Seq-SPD'),Path('/data/qh/DAIR-V2X-C')]
    for base in roots:
        for path in base.rglob('data_info.json'):
            if path.parent.name not in ('vehicle-side','infrastructure-side'):continue
            data=json.loads(path.read_text());directory=path.parent;counts=Counter();examples=[]
            for item in data:
                for key in ('image_path','pointcloud_path','calib_camera_intrinsic_path','calib_lidar_to_camera_path','label_camera_std_path','label_lidar_std_path'):
                    value=item.get(key)
                    if value and (directory/value).is_file():counts[key]+=1
                if len(examples)<2:
                    examples.append(dict(frame_id=item.get('frame_id'),sequence_id=item.get('sequence_id'),
                        camera_timestamp=item.get('image_timestamp'),lidar_timestamp=item.get('pointcloud_timestamp'),
                        paths={k:str(directory/v) for k,v in item.items() if 'path' in k and isinstance(v,str)}))
            example=data[0] if data else {};intrinsic=directory/example.get('calib_camera_intrinsic_path','__missing__')
            distortion=json.loads(intrinsic.read_text()).get('cam_D') if intrinsic.is_file() else None
            external.append(dict(metadata=str(path),frames=len(data),available=dict(counts),examples=examples,
                sequence_ids=sorted({str(i.get('sequence_id')) for i in data}),
                calibration_distortion_example=distortion,original_omnidrive_output_cache=False,
                eligible_same_protocol=False,
                blockers=['Single vehicle/infrastructure camera, not the nuScenes camera rig',
                    'Camera/LiDAR timestamps differ; original reference is synchronized annotation-owned nuScenes LiDAR',
                    'Nonzero raw lens distortion needs a dataset-specific rectified coordinate adapter',
                    'Sequence ID alone does not establish independent acquisition logs',
                    'Missing RGB in full releases when image_path count is zero; examples are not independent full logs']))
    lidar_sets=[]
    for name in ['OPV2V','v2v4real','V2XSet','OPV2V_Hetero','SEE-V2X']:
        path=Path('/data/qh/dataset')/name
        if not path.is_dir():continue
        examples=[]
        for p in path.glob('*/*/*'):
            if p.is_dir():examples.append(p)
            if len(examples)>=3:break
        lidar_sets.append(dict(path=str(path),sampled_agent_directories=[dict(path=str(p),
            rgb_files=len(list(p.glob('*.jpg')))+len(list(p.glob('*.png'))),
            pointclouds=len(list(p.glob('*.pcd'))),metadata_yaml=len(list(p.glob('*.yaml')))) for p in examples],
            same_nuscenes_protocol_compatible=False,
            status='Not Run: sampled agents provide LiDAR/metadata; no validated synchronized RGB/instance surface adapter'))
    write_json(ROOT/'dataset_scene_inventory.json',inventory)
    write_json(ROOT/'historical_exposure_audit.json',dict(previously_exposed_scenes=old['previously_exposed_scenes'],
        previously_exposed_logs=sorted(seen),basis='Prior frozen exposure inventory plus all current round Development sources',
        prior_inventory_sha256=digest(OLD/'historical_exposure_inventory.json'),all_trainval_logs_seen={s['log_token'] for s in scene_table}<=seen))
    new=[s for s in inventory if not s['log_exposed'] and any(p['raw_camera_count']==6 and p['annotation_count'] for p in s['samples'])]
    write_json(ROOT/'independent_data_availability.json',dict(trainval_scene_count=len(scene_table),
        trainval_logs=len({s['log_token'] for s in scene_table}),info_sources=info_inventory,
        raw_unseen_labeled_trainval_logs=len({s['log_token'] for s in new}),other_nuscenes_metadata=tables,
        other_local_data=external,lidar_only_or_unvalidated_data=lidar_sets,
        raw_images_missing_cache_not_confused_with_missing_labels=True,
        conclusion='No new log with the unchanged nuScenes reference protocol. Other local data exist but are not merged or silently rectified.',
        no_downloads=True))
    # All old M1 Calibration/Heldout/Final groups remain development/exploration.
    write_json(ROOT/'data_selection.json',dict(development=json.loads((OLD/'data_selection.json').read_text())['splits']['development'],
        calibration=[],heldout=[],final_test=[],selection_population='exploratory_reused_population',
        new_independent_logs=len({s['log_token'] for s in new}),
        other_datasets_not_merged=True,info_sources=[str(p) for p in infofiles],original_model_training_exposure=True))
    write_json(ROOT/'experiment_protocol.json',dict(version='m1-visible-surface-refinement-v1',
        fixed_before_new_algorithm_quality=True,critical_event_recall_min=.998,critical_point_recall_min=.998,
        critical_roi_refresh_min=.9,sparse_annotated_far_max=.05,sparse_far_upper95_max=.05,
        accepted_scene_min=3,identity_log_event_upper95_max=.05,zero_acceptance_far=None,
        independent_unit='Acquisition log; any false acceptance is a unit failure; incomplete scoring does not qualify',
        development_population='exploratory_reused_population',original_geometric_denominator=5798,
        region_denominator='All camera pixels, including startup and forced refresh',
        oracle_changed=False,GT_online=False,latency_optimized=False,
        proposed_surface='Connected visible support, multiple plane hypotheses, spatially withheld independent measurements, selective measured pixels',
        proposed_identity='Existing finite history plus motion-compensated box-surface/ring observations; unqualified links remain shadow',
        selection='Null unless new Calibration passes all frozen constraints; no test reselection',
        source_baseline_sha256=digest(ROOT/'baseline_source_manifest.json'),data_selection_sha256=digest(ROOT/'data_selection.json')))
    print('eligible unseen labeled logs',len({s['log_token'] for s in new}),'external metadata inventories',len(external))


if __name__=='__main__':main()
