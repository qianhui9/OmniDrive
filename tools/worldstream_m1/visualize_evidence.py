"""Real source/current evidence proposals and event-only failure figures."""
import json
from pathlib import Path
import pickle
import cv2
import numpy as np
from .dataset import frame_from_info
from .io_utils import write_json
from worldstream.m1.mask_codec import decode_mask


def main():
    root=Path('outputs/module1_evidence_optimization');chosen=[];events=[]
    for path in sorted((root/'traces').glob('*/*/complete.json')):
        folder=path.parent;rows=[json.loads(line) for line in (folder/'foreground.jsonl').read_text().splitlines()] if (folder/'foreground.jsonl').exists() else []
        for row in rows:
            if row['status']=='candidate' and not any(r['scene']==row['scene'] for r in chosen):
                chosen.append(dict(row,trace=str(folder/'state.jsonl')))
        for line in (folder/'pixels.jsonl').open():
            row=json.loads(line)
            if row['method']!='conservative':continue
            for event in row.get('events',[]):
                if not event['recalled'] and len(events)<12:events.append(dict(row,event=event,trace=str(folder/'state.jsonl')))
    needed={r['frame'] for r in chosen+events}|{r['source_frame_id'] for r in chosen}
    infos={}
    for file in ['data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','data/nuscenes/nuscenes2d_ego_temporal_infos_train.pkl']:
        data=pickle.load(open(file,'rb'))['infos']
        infos.update({i['token']:i for i in data if i['token'] in needed})
        del data
    folder=root/'visualizations/evidence_cases';folder.mkdir(parents=True,exist_ok=True);index=[]
    for number,row in enumerate(chosen):
        source=frame_from_info(infos[row['source_frame_id']],'data/nuscenes')
        current=frame_from_info(infos[row['frame']],'data/nuscenes')
        points=row['selected_points'];cid=max({p['camera'] for p in points},key=lambda c:sum(p['camera']==c for p in points))
        a=source.cameras[cid].image.copy();b=current.cameras[cid].image.copy()
        for point in points:
            if point['camera']!=cid:continue
            cv2.circle(a,tuple(np.rint(point['source_uv']).astype(int)),3,(50,240,70),1)
            cv2.circle(b,tuple(np.rint(point['current_uv']).astype(int)),3,(255,180,30),1)
        panel=np.concatenate([a,b],axis=1);header=np.zeros((55,panel.shape[1],3),np.uint8)
        cv2.putText(header,'Measured proposal: source / current. Same RGB; NOT certified instance foreground.',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1)
        cv2.putText(header,'Physical certification REJECTED: independent producer qualification missing.',(8,43),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,190,40),1)
        path=folder/f'foreground_candidate_{number:02d}.jpg'
        cv2.imwrite(str(path),cv2.cvtColor(np.concatenate([header,panel]),cv2.COLOR_RGB2BGR))
        index.append(dict(kind='measured_candidate_not_certified',source_frame=source.frame_id,current_frame=current.frame_id,
            scene=row['scene'],camera=cid,path=str(path),offline_reference=row['offline_points']))
    records={}
    for number,row in enumerate(events):
        path=row['trace']
        if path not in records:records[path]={r['precheck']['frame_id']:r['precheck'] for r in map(json.loads,Path(path).read_text().splitlines())}
        frame=frame_from_info(infos[row['frame']],'data/nuscenes');cid=row['camera'];event=row['event']
        image=frame.cameras[cid].image.copy();mask=decode_mask(records[path][frame.frame_id]['invalidation_regions'][cid]['refresh_required_mask'])
        stable=~mask;overlay=image.copy();overlay[stable]=(.4*overlay[stable]+.6*np.array([50,230,80])).astype(np.uint8)
        region=event['region'];p=tuple(np.floor(region[:2]).astype(int));q=tuple(np.ceil(region[2:]).astype(int))
        for im in (image,overlay):cv2.rectangle(im,p,q,(255,30,30),2)
        panel=np.concatenate([image,overlay],axis=1)
        cv2.putText(panel,f"ROI event missed: {event['reasons']} refresh={event['invalidation_coverage']:.3f}; green=accepted stable",
                    (8,20),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
        path=folder/f'critical_roi_event_{number:02d}.jpg';cv2.imwrite(str(path),cv2.cvtColor(panel,cv2.COLOR_RGB2BGR))
        index.append(dict(kind='critical_ROI_event_failure',scene=row['scene'],frame=row['frame'],camera=cid,path=str(path),event=event))
    write_json(folder/'index.json',dict(real_cases=index,GT_mask_used=False,
        absent_examples=['No real certified physical object: do not fabricate positive certification image'],
        interpretation='Point/ROI visualizations do not supply online labels or a dense pixel ownership mask'))
    print('real evidence candidates',len(chosen),'ROI failure figures',len(events))


if __name__=='__main__':main()
