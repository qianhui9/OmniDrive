"""Archive exact RGB samples for every frozen diagnostic reference point."""
from collections import defaultdict
import json
import pickle
import cv2
import numpy as np
from .correspondence_diagnostics import ROOT, sha
from .dataset import frame_from_info
from worldstream.m1.surface_observability import project
from .io_utils import write_json


def main():
    cv2.setNumThreads(1)
    records=json.loads((ROOT/'patch_rejection_root_cause.json').read_text())['records']
    selection=json.loads(open('outputs/module1_evidence_optimization/data_selection.json').read())
    needed={r[k] for r in records for k in ('frame','source_frame')}
    with open(selection['info_file'],'rb') as f:infos={i['token']:i for i in pickle.load(f)['infos'] if i['token'] in needed}
    groups=defaultdict(list)
    for i,r in enumerate(records):groups[r['frame'],r['source_frame'],r['camera']].append((i,r))
    rad=3;yy,xx=np.mgrid[-rad:rad+1,-rad:rad+1];offsets=np.c_[xx.ravel(),yy.ravel()]
    arrays={k:np.zeros((len(records),7,7,3),np.uint8) for k in ('current_rgb','source_center_rgb','frontoparallel_rgb','measured_local_rgb')}
    valid=np.zeros((len(records),7,7),bool);cache={}
    def sample(cam,uv):
        uv=np.where(np.isfinite(uv),uv,-10).astype(np.float32)
        return cv2.remap(cam.image,uv[:,0].reshape(7,7),uv[:,1].reshape(7,7),cv2.INTER_LINEAR)
    for (token,oldtoken,cid),values in groups.items():
        for t in (token,oldtoken):
            if t not in cache:cache[t]=frame_from_info(infos[t],'data/nuscenes')
        cur=cache[token].cameras[cid];old=cache[oldtoken].cameras[cid]
        for i,r in values:
            q=np.asarray(r['pixel']);grid=q+offsets;p=np.asarray(r['measured_source_uv'])
            world=np.asarray(r['world'],float);_,depth=project(world[None],cur)
            rays=np.c_[grid,np.ones(len(grid))]@np.linalg.inv(cur.intrinsic).T
            xyz=(rays*depth[0]-cur.world_to_camera[:3,3])@cur.world_to_camera[:3,:3]
            uv,z=project(xyz,old);h,w=old.image.shape[:2]
            ch,cw=cur.image.shape[:2]
            valid[i]=(np.isfinite(uv).all(1)&(uv[:,0]>=1)&(uv[:,0]<w-1)&(uv[:,1]>=1)&(uv[:,1]<h-1)
                &(grid[:,0]>=1)&(grid[:,0]<cw-1)&(grid[:,1]>=1)&(grid[:,1]<ch-1)
                &(z>.5)&(z<200)&(depth[0]>.5)&(depth[0]<200)).reshape(7,7)
            arrays['current_rgb'][i]=sample(cur,grid)
            arrays['source_center_rgb'][i]=sample(old,p+offsets)
            arrays['frontoparallel_rgb'][i]=sample(old,uv)
            arrays['measured_local_rgb'][i]=sample(old,np.c_[offsets,np.ones(len(offsets))]@np.asarray(r['local_affine']))
        cache={t:cache[t] for t in (token,oldtoken)}
    directory=ROOT/'diagnostics';directory.mkdir(exist_ok=True)
    path=directory/'patch_rgb.npz';np.savez_compressed(path,**arrays,frontoparallel_observed=valid)
    write_json(ROOT/'patch_rgb_archive.json',dict(file=str(path),sha256=sha(path),shape=[len(records),7,7,3],
        indexing='Array row i is patch_rejection_root_cause.json records[i]; exact OpenCV bilinear RGB samples at archived calibrated coordinates',
        invalid_samples='Explicit frontoparallel_observed bit; out-of-image zeros are never measurements',
        note='Measured local affine samples may lack tracking support; masks and FB in frozen records determine availability, not their RGB appearance.'))


if __name__=='__main__':main()
