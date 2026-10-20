"""Measured diagnostic panels; green tracks are not identity truth."""
from pathlib import Path
import json
import cv2
import numpy as np


def visualize_case(manager,frame,folder,index):
    # Preserve one deterministic early/late example per scene, including all-rejected cases.
    if index not in (1,5):return
    cid=next(iter(frame.cameras));camera=frame.cameras[cid];e=manager.last_evidence.get(cid)
    current=camera.image.copy();old=manager._previous_cameras.get(cid)
    # Commit has replaced previous cameras. Draw the measured source points on a
    # separately saved precheck source image supplied by the runner.
    source=getattr(manager,'_diagnostic_source_images',{}).get(cid)
    if source is None:return
    source=cv2.cvtColor(source,cv2.COLOR_GRAY2RGB) if source.ndim==2 else source.copy()
    if e is not None:
        for p,q,ok in zip(e.source_points[::8],e.current_points[::8],e.supported[::8]):
            color=(30,200,30) if ok else (230,60,40)
            cv2.circle(source,tuple(np.rint(p).astype(int)),2,color,-1)
            cv2.circle(current,tuple(np.rint(q).astype(int)),2,color,-1)
    panel=np.concatenate([source,current],axis=1)
    cv2.putText(panel,'Actual source / current RGB tracks; green=local support, NOT identity truth',
                (8,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
    path=Path(folder)/'cases';path.mkdir(exist_ok=True)
    cv2.imwrite(str(path/f'{index:03d}_{cid}.png'),cv2.cvtColor(panel,cv2.COLOR_RGB2BGR))
    checks=manager.last_result.diagnostics.get('anchors',{})
    samples=[]
    for aid,a in checks.items():
        if cid in a['cameras']:
            v=a['cameras'][cid]
            if v.get('predicted_region'):
                samples.append(dict(anchor_id=aid,accepted=a['valid'],camera=cid,**v))
    from worldstream.m1.state_types import json_value
    (path/f'{index:03d}_{cid}.json').write_text(json.dumps(json_value(samples),indent=2)+'\n')
