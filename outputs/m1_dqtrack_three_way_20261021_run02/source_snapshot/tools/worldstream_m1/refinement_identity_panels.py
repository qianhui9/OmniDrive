"""Show the four real additional correct identity links and measured supports."""
from pathlib import Path
import json
import cv2
import numpy as np
from .refinement_report import ROOT
from .io_utils import write_json


def main():
    manifest=json.loads((ROOT/'data_manifest.json').read_text());frames={r['sample_token']:r for r in manifest['frames']}
    recovery=json.loads((ROOT/'identity_unique_instance_recovery.json').read_text())['added_vs_same_shadow_tracking']['records']
    output=[];folder=ROOT/'visualizations/identity_recovery';folder.mkdir(parents=True,exist_ok=True)
    for index,row in enumerate(recovery):
        if row['correct'] is not True:continue
        for view in row['measurements']['views']:
            if view.get('status')!='supported' or view.get('observation_model')!='translation_compensated_box_surface_ring':continue
            panels=[]
            for sample,side in ((row['source_frame'],'source'),(row['frame'],'current')):
                path=next(Path(p) for p in frames[sample]['source_files'] if Path(p).parent.name==view['camera'])
                image=cv2.imread(str(path));h,w=image.shape[:2];scale=min(1.,640/w)
                image=cv2.resize(image,(int(round(w*scale)),int(round(h*scale))),interpolation=cv2.INTER_AREA)
                region=np.asarray(view[side+'_region']);corners=np.rint(region).astype(int)
                cv2.rectangle(image,tuple(corners[:2]),tuple(corners[2:]),(220,220,220),1)
                for p in view[side+'_points']:cv2.circle(image,tuple(np.rint(p).astype(int)),2,(20,230,20),-1)
                for p in view['ring_'+side+'_points']:cv2.circle(image,tuple(np.rint(p).astype(int)),2,(240,170,0),1)
                margin=np.maximum((region[2:]-region[:2])*.75,18)
                x0,y0=np.maximum(np.floor(region[:2]-margin),0).astype(int)
                x1,y1=np.minimum(np.ceil(region[2:]+margin),[image.shape[1],image.shape[0]]).astype(int)
                panels.append(cv2.resize(image[y0:y1,x0:x1],(360,280),interpolation=cv2.INTER_NEAREST))
            canvas=np.zeros((330,720,3),np.uint8);canvas[:280]=np.concatenate(panels,axis=1)
            cv2.putText(canvas,'Actual source | current RGB; green: conditional object surface, blue: measured ring',(5,300),0,.4,(240,240,240),1)
            cv2.putText(canvas,'Independent identity reference: correct. No dense ownership truth or formal certification.',(5,320),0,.4,(240,240,240),1)
            file=folder/f"{index:02d}_{row['frame']}_{view['camera']}.png";cv2.imwrite(str(file),canvas)
            output.append(dict(frame=row['frame'],source_frame=row['source_frame'],prediction=row['prediction'],camera=view['camera'],
                correct_identity=True,formal_certification=False,image=str(file),view=view))
    write_json(ROOT/'identity_recovery_visualizations.json',dict(links=len(recovery),panels=output))
    print('actual identity recovery panels',len(output))


if __name__=='__main__':main()
