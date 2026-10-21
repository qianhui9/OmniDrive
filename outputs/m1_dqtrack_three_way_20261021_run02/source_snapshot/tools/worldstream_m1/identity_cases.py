"""Draw actual measured candidate/rejection examples; never a qualification plot."""
import argparse
import json
from pathlib import Path
import pickle
import cv2
import numpy as np
from .dataset import frame_from_info


def measured_tracks(record):
    """Do not draw failed LK placeholder coordinates as measured matches."""
    p=np.asarray(record.get('source_points',[]),float);q=np.asarray(record.get('current_points',[]),float)
    if p.ndim!=2 or p.shape[1:]!=(2,) or q.shape!=p.shape:return np.empty((0,2)),np.empty((0,2))
    valid=np.asarray(record.get('correspondence_valid',np.ones(len(p),bool)),bool)
    if valid.shape!=(len(p),):return np.empty((0,2)),np.empty((0,2))
    valid=valid & np.isfinite(p).all(1) & np.isfinite(q).all(1)
    return p[valid],q[valid]


def render(root,output):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((root/('protocol.json' if (root/'protocol.json').exists() else 'provenance.json')).read_text())
    with open(protocol.get('infos_path',protocol.get('dataset')),'rb') as f:infos={x['token']:x for x in pickle.load(f)['infos']}
    raw=(root/'protocol.json').exists()
    cfg=None
    if not raw:
        from mmcv import Config
        from .state_interface_replay import preprocessed_frame
        from worldstream.m1 import M1Config
        cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m11_diagnostics.py')
        settings=M1Config.load('configs/worldstream_m11_diagnostics.yaml')
    traces=sorted((root/'scenes').glob('*/repeat_0.jsonl')) or sorted((root/'scenes').glob('*/m1_state.jsonl'))
    manifest=[]
    for trace in traces:
        rows=[json.loads(x) for x in trace.read_text().splitlines()]
        proposals=[p for row in rows for p in row['commit']['identity_continuity_proposals']]
        for status in ('candidate','unknown'):
            chosen=None
            for p in proposals:
                if p['status']!=status:continue
                for v in p['measurements'].get('views',[]):
                    d=v.get('track_diagnostics',v)
                    if len(measured_tracks(d)[0])>=2:
                        chosen=(p,v,d);break
                if chosen:break
            if chosen is None:
                manifest.append(dict(scene_id=trace.parent.name,status=status,visualization='Not Available: no actual measured view'))
                continue
            p,v,d=chosen
            def get_frame(token):
                if raw:return frame_from_info(infos[token],'data/nuscenes')
                return preprocessed_frame(infos[token],cfg,settings)[0]
            old=get_frame(p['source_frame_id']).cameras[v['camera']].image.copy()
            current=get_frame(p['current_frame_id']).cameras[v['camera']].image.copy()
            source_points,current_points=measured_tracks(d)
            for x,y in zip(source_points,current_points):
                cv2.circle(old,tuple(np.rint(x).astype(int)),2,(30,210,70),-1)
                cv2.circle(current,tuple(np.rint(y).astype(int)),2,(30,210,70),-1)
            for image,key in ((old,'source_region'),(current,'current_region')):
                r=v.get(key)
                if r:cv2.rectangle(image,tuple(np.rint(r[:2]).astype(int)),tuple(np.rint(r[2:]).astype(int)),(250,180,20),1)
            panel=np.concatenate([old,current],axis=1)
            label=f'Measured {status}; source/current; NOT qualified identity or pixel ownership'
            cv2.putText(panel,label,(6,18),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
            name=f'{trace.parent.name}_{status}.png';cv2.imwrite(str(out/name),cv2.cvtColor(panel,cv2.COLOR_RGB2BGR))
            manifest.append(dict(scene_id=trace.parent.name,status=status,source_frame=p['source_frame_id'],
                current_frame=p['current_frame_id'],camera=v['camera'],measured_points=len(source_points),drawn_scope='Finite valid RGB correspondences, not ownership',
                formal_certified=False,foreground_ownership='Unknown',file=name))
    (out/'case_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();cv2.setNumThreads(1);render(a.root,a.output)


if __name__=='__main__':main()
