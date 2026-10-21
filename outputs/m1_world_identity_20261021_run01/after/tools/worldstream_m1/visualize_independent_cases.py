"""Real source/current failure cases selected from saved OFFLINE scores."""
from collections import defaultdict
import json
from pathlib import Path
import pickle
import cv2
import numpy as np
from worldstream.m1 import M1Config, WorldAnchor
from worldstream.m1.geometry import project_anchor
from worldstream.m1.mask_codec import decode_mask
from .dataset import frame_from_info, prediction_outputs
from .io_utils import read_prediction_subset, write_json


def main():
    root = Path('outputs/module1_independent_refinement')
    config = M1Config.load('configs/worldstream_m1_independent.yaml')
    chosen = {}; unavailable = []
    for path in sorted((root/'traces').glob('*/*/current/metrics.json')):
        d = json.loads(path.read_text())
        for row in d['associations']:
            kind = ('identity_switch' if row['correct'] is False and row['continued'] else
                    'correct_candidate_association' if row['correct'] is True and row['continued'] else
                    'correctly_rejected_wrong_association' if row['correct'] is False else None)
            if kind and kind not in chosen:
                chosen[kind] = dict(row, trace=str(path.parent/'state.jsonl'))
        for row in d['physical']:
            kind = ('observation_supported_but_physical_oracle_false' if row['accepted'] and row['oracle_valid'] is False else
                    'oracle_valid_but_identity_proof_missing' if row['oracle_valid'] is True and not row['certified'] else None)
            if kind and kind not in chosen:
                chosen[kind] = dict(row, trace=str(path.parent/'state.jsonl'))
        for event in d['events']:
            kind = 'newly_visible' if 'newly_visible' in event['reasons'] else None
            if kind and kind not in chosen:
                chosen[kind] = dict(event, trace=str(path.parent/'state.jsonl'))
            if not event['recalled'] and 'missed_critical_change' not in chosen:
                chosen['missed_critical_change'] = dict(event, trace=str(path.parent/'state.jsonl'))
    cache = {}; sources = {}; required = set()
    for row in chosen.values():
        path = row['trace']
        if path not in cache:
            cache[path] = {d['precheck']['frame_id']:d for d in map(json.loads,Path(path).read_text().splitlines())}
            for frame, record in cache[path].items():
                for aid, index in zip(record['commit']['committed_anchor_ids'],record['commit']['selected_prediction_indices']):
                    sources[frame,aid] = index
        record = cache[path][row['frame']]['precheck']; aid = row.get('anchor')
        source = record['anchor_provenance'].get(aid,{}).get('source_frame_id')
        row['source_frame'] = source
        required.add(row['frame'])
        if source: required.add(source)
    infos = pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb'))['infos']
    by_token = {i['token']:i for i in infos if i['token'] in required}
    predictions = read_prediction_subset('test/mask_eva_lane_det_vlm/Tue_Sep_29_20_12_56_2026/pts_bbox/results_nusc.json',required)
    index = []
    folder = root/'visualizations/cases'; folder.mkdir(exist_ok=True)
    for kind,row in chosen.items():
        record = cache[row['trace']][row['frame']]['precheck']
        current = frame_from_info(by_token[row['frame']],'data/nuscenes')
        aid = row.get('anchor'); check = record['diagnostics']['anchors'].get(aid)
        if check:
            cid = max(check['cameras'],key=lambda cid: np.prod(np.subtract(
                check['cameras'][cid]['predicted_region'][2:],check['cameras'][cid]['predicted_region'][:2]))
                if check['cameras'][cid]['predicted_region'] else -1)
            region = check['cameras'][cid]['predicted_region']
        else: cid,region = row['camera'],row['region']
        target = current.cameras[cid].image.copy()
        source_image = np.zeros_like(target); source_region = None
        source_frame = row['source_frame']
        if source_frame:
            source = frame_from_info(by_token[source_frame],'data/nuscenes')
            source_image = source.cameras[cid].image.copy()
            idx = sources.get((source_frame,aid))
            if idx is not None:
                pred = prediction_outputs(predictions[source_frame],config).anchors[idx]
                p = project_anchor(WorldAnchor('source',source.scene_id,pred.anchor_type,pred.semantic_class,
                    pred.geometry_world,source.timestamp),source.cameras[cid],config)
                source_region = p.region
        def rectangle(image, box, color, text):
            if box is None:return
            a,b = tuple(map(int,box[:2])),tuple(map(int,box[2:]))
            cv2.rectangle(image,a,b,color,2)
            cv2.putText(image,text,(a[0],max(30,a[1]-3)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
        rectangle(source_image,source_region,(50,255,80),'actual source prediction')
        rectangle(target,region,(255,170,30),'motion-compensated historical projection')
        j = row.get('prediction')
        if j is not None:
            pred = prediction_outputs(predictions[row['frame']],config).anchors[j]
            p = project_anchor(WorldAnchor('current',current.scene_id,pred.anchor_type,pred.semantic_class,
                pred.geometry_world,current.timestamp),current.cameras[cid],config)
            rectangle(target,p.region,(255,50,50),'current predicted candidate')
        masks = record['invalidation_regions'][cid]
        stable,changed,unknown = [decode_mask(masks[k]) for k in ('observation_stable_mask','observed_changed_mask','unknown_mask')]
        partition = current.cameras[cid].image.copy()
        colors = np.zeros_like(partition);colors[stable]=(40,230,70);colors[changed]=(240,40,40);colors[unknown]=(110,110,150)
        partition = (.6*partition+.4*colors).astype(np.uint8)
        refresh = current.cameras[cid].image.copy();refresh[decode_mask(masks['refresh_required_mask'])] //= 4
        tiles = [source_image,target,partition,refresh]
        for tile,label in zip(tiles,['actual source RGB','current RGB: geometric candidate only','green stable / red changed / grey unknown','refresh dark; observation evidence kept separate']):
            cv2.putText(tile,label,(8,16),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
        panel = np.concatenate([np.concatenate(tiles[:2],axis=1),np.concatenate(tiles[2:],axis=1)],axis=0)
        header = np.zeros((75,panel.shape[1],3),np.uint8)
        cv2.putText(header,kind+' | '+cid,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
        cv2.putText(header,'Online physical certification: REJECTED, independent foreground/identity missing',(8,43),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,190,60),1)
        identity = 'offline source='+str(row.get('source_instance'))+' current='+str(row.get('current_instance'))
        cv2.putText(header,identity,(8,63),cv2.FONT_HERSHEY_SIMPLEX,.4,(220,220,220),1)
        path = folder/(kind+'.jpg')
        cv2.imwrite(str(path),cv2.cvtColor(np.concatenate([header,panel]),cv2.COLOR_RGB2BGR))
        index.append(dict(case=kind,path=str(path),**row))
    for kind in ('correct_physical_certification','independently_proven_background_texture_correspondence'):
        unavailable.append(dict(case=kind,status='Not available',reason='No certified physical objects / no independent pixel ownership reference; do not fabricate examples'))
    write_json(folder/'index.json',dict(real_cases=index,unavailable=unavailable,all_scenes_previously_exposed=True))
    print('real source/current case figures',len(index))


if __name__ == '__main__':
    main()
