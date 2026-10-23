"""Posthoc attribution of the predeclared anomaly using complete actual runs."""
import json
import pickle
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.worldstream_m2.task_diagnostics import diagnose_detection,diagnose_map,line_distances
from tools.worldstream_m1.io_utils import read_prediction_subset
from worldstream.m1.omnidrive_adapter import quaternion_transform
from worldstream.m1.geometry import transform_points

ROOT=Path(__file__).resolve().parent
OLD=ROOT/'references/wsgrrr_20261022_followup'
TOKEN='a7795dcadad44c1c85fa27eef18366f6'
def read(path):return json.loads(path.read_text())
def row(path):
    with path.open() as f:
        for line in f:
            r=json.loads(line)
            if r['frame_id']==TOKEN:return r
    raise ValueError('Missing predeclared frame')
trace={}
for name,path,key in [('before_full',OLD/'D3_full/per_frame_metrics.jsonl','m2'),
        ('after_full',ROOT/'main_full/per_frame_metrics.jsonl','m2'),
        ('after_visual',ROOT/'visual40_main/frames.jsonl',None)]:
    r=row(path);d=r[key] if key else r;f=d['region_refresh']
    trace[name]=dict(frame_id=TOKEN,camera_id='CAM_FRONT',window_id=4,
        selected=4 in d['stages'][0]['executed_window_ids'],mandatory=4 in f['mandatory_window_ids'],
        age=f.get('ages',[None]*54)[4],drift=f.get('drift',[None]*54)[4],
        uncertainty=f.get('uncertainty',[None]*54)[4],score=f.get('scores',[None]*54)[4],
        age_after=r.get('window_ages_after',{}).get('0',[None]*54)[4],
        uncertainty_regions=[a for a in f.get('uncertainty_regions',[])
            if a['camera_id']=='CAM_FRONT' and 4 in a['windows']])
(ROOT/'anomaly_window_trace.json').write_text(json.dumps(trace,indent=2))
if '--windows-only' in sys.argv:
    print('Saved actual window trace; task quality remains pending complete predictions.')
    raise SystemExit(0)

assert read(ROOT/'main_full/official_eval/status.json')['status']=='COMPLETED'
predictions={}
for name,path in [('native',OLD/'Native_full'),('before',OLD/'D3_full'),('after',ROOT/'main_full')]:
    tokens=read(path/'sequence.json')['tokens']
    with (path/'predictions.pkl').open('rb') as f:all_predictions=pickle.load(f)
    assert len(tokens)==len(all_predictions)==6019
    predictions[name]=all_predictions[tokens.index(TOKEN)]
    del all_predictions

det={};maps={}
for name in ['before','after']:
    det[name]=diagnose_detection(predictions['native'],predictions[name],TOKEN)
    maps[name]=diagnose_map(predictions['native']['lane_results'][0],predictions[name]['lane_results'][0],TOKEN)

with open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb') as f:
    infos=pickle.load(f)['infos']
info=next(i for i in infos if i['token']==TOKEN)
pose=quaternion_transform(info['ego2global_rotation'],info['ego2global_translation'])
del infos
tables=Path('data/nuscenes/v1.0-trainval')
categories={c['token']:c['name'] for c in read(tables/'category.json')}
instances={i['token']:categories[i['category_token']] for i in read(tables/'instance.json')}
# Reverse annotation->sample links avoid the known broken raw sample.anns
# lists. These are actual annotation centers, not predictions or pseudo-GT.
annotations=[a for a in read(tables/'sample_annotation.json')
    if a['sample_token']==TOKEN and instances[a['instance_token']]=='vehicle.car']
native_serialized=read_prediction_subset(OLD/'Native_full/official_eval/detection/pts_bbox/results_nusc.json',{TOKEN})[TOKEN]
gt={}
for name in ['native','before','after']:
    native_match=(None if name=='native' else next((m for m in det[name]['legacy'] if m['native_index']==4),None))
    index=4 if name=='native' else native_match['candidate_index'] if native_match else None
    if index is None:
        gt[name]=dict(status='UNMATCHED_PREDICTION');continue
    box=predictions[name]['pts_bbox'];center=box['boxes_3d'].gravity_center.numpy()[index]
    world=transform_points(center,pose)
    if name=='native':
        # Independently check the ego->world contract against official formatting.
        error=min(np.linalg.norm(world-np.asarray(p['translation'])) for p in native_serialized
                  if p['detection_name']=='car')
        assert error<1e-4, ('Coordinate contract differs from official serialization',error)
    nearest=sorted(annotations,key=lambda a:np.linalg.norm(np.asarray(a['translation'])[:2]-world[:2]))[:3]
    gt[name]=dict(raw_returned_prediction_index=index,score=float(box['scores_3d'][index]),
        ego_center=center.tolist(),world_center=world.tolist(),
        nearest_actual_car_annotations=[dict(annotation_token=a['token'],instance_token=a['instance_token'],
            translation=a['translation'],lidar_points=a['num_lidar_pts'],radar_points=a['num_radar_pts'],
            xy_distance_m=float(np.linalg.norm(np.asarray(a['translation'])[:2]-world[:2])),
            center_distance_m=float(np.linalg.norm(np.asarray(a['translation'])-world))) for a in nearest],
        caveat='Nearest GT attribution only, not score-ordered official detection matching or a certified object identity.')

original_map={}
nlines,nscores=predictions['native']['lane_results'][0]
for name in ['before','after']:
    lines,scores=predictions[name]['lane_results'][0]
    original_map[name]=dict(native_index=95,original_forced_candidate_index=87,
        original_pair_geometry=line_distances(nlines[95],lines[87]),
        same_returned_index_95_geometry=line_distances(nlines[95],lines[95]),
        native_score=float(np.asarray(nscores[95]).max()),
        original_candidate_87_score=float(np.asarray(scores[87]).max()),
        same_returned_index_95_score=float(np.asarray(scores[95]).max()),
        caveat='Same returned index is not an identity certificate; original wrong-pair geometry is deliberately retained.')
result=dict(frame_id=TOKEN,detection=det,map=maps,actual_car_gt_attribution=gt,
    original_map_indices=original_map,
    diagnostic_scope='Existing score>=0.3, 2m detection and 3m symmetric map gates unchanged; diagnostic consistency is not task accuracy. Full official metrics remain authoritative.')
(ROOT/'anomaly_prediction_diagnostics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print('Final anomaly diagnostic saved',ROOT/'anomaly_prediction_diagnostics.json')
