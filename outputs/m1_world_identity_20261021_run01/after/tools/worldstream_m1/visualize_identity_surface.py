"""Real paired identity-candidate RGB, explicitly without qualified certificates."""
import json
import pickle
from pathlib import Path
import cv2
import numpy as np
from .identity_surface_audit import ROOT
from .dataset import frame_from_info
from .io_utils import write_json


def main():
    entries=[];wanted=set();rows=[]
    for stage in ('development','calibration','heldout','final_test'):
        used=set()
        for p in sorted((ROOT/'traces'/stage).glob('*/identity.jsonl')):
            for row in map(json.loads,p.read_text().splitlines()):
                if (row['method']=='selective_identity' and row['candidate'] and row['correct'] is True
                        and row['scene'] not in used):
                    rows.append((stage,row));used.add(row['scene']);wanted.update([row['source_frame'],row['frame']]);break
                if len(used)>=3:break
            if len(used)>=3:break
    infos={}
    for split in ('train','val'):
        data=pickle.load(open(f'data/nuscenes/nuscenes2d_ego_temporal_infos_{split}.pkl','rb'))['infos']
        infos.update({i['token']:i for i in data if i['token'] in wanted})
    folder=ROOT/'visualizations/identity';folder.mkdir(parents=True,exist_ok=True)
    for index,(stage,row) in enumerate(rows):
        source=frame_from_info(infos[row['source_frame']],'data/nuscenes');current=frame_from_info(infos[row['frame']],'data/nuscenes')
        v=next(v for v in row['measurements']['views'] if v['reason']=='measured_identity_population_candidate');cid=v['camera']
        a=source.cameras[cid].image.copy();b=current.cameras[cid].image.copy()
        for pic,region,pts,color in ((a,v['source_region'],v['source_points'],(40,255,70)),(b,v['current_region'],v['current_points'],(255,200,40))):
            x0,y0,x1,y1=map(int,region);cv2.rectangle(pic,(x0,y0),(x1,y1),color,1)
            for p in pts:cv2.circle(pic,tuple(np.rint(p).astype(int)),2,color,1)
        top=np.zeros((60,a.shape[1]*2,3),np.uint8)
        cv2.putText(top,'Measured identity candidate; offline unique instance matches; NOT physically certified.',(7,18),0,.45,(255,255,255),1)
        cv2.putText(top,'Qualification insufficient. Tracks/box are not an instance foreground mask.',(7,39),0,.45,(255,210,40),1)
        panel=np.concatenate([top,np.concatenate([a,b],axis=1)],axis=0)
        path=folder/f'{index:02d}_{stage}_{cid}.jpg';cv2.imwrite(str(path),cv2.cvtColor(panel,cv2.COLOR_RGB2BGR))
        entries.append(dict(stage=stage,scene=row['scene'],source_frame=row['source_frame'],current_frame=row['frame'],
            source_instance=row['source_instance'],current_instance=row['current_instance'],camera=cid,path=str(path),qualified=False))
    write_json(folder/'index.json',dict(real_cases=entries,synthetic_or_certified_images=False,
        scope='Independent annotation identity correspondence under unique2m gate; no pixel ownership truth'))
    print('Saved',len(entries),'real paired candidate examples')


if __name__=='__main__':main()
