"""Draw actual candidate LK measurements; prediction markers are not truth."""
import argparse
import json
from pathlib import Path
import pickle
import cv2
import numpy as np
from .dataset import frame_from_info
from .effective_acceptance import sha


def run(root,output):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((root/'protocol.json').read_text())
    with open(protocol['infos_path'],'rb') as f:infos={i['token']:i for i in pickle.load(f)['infos']}
    manifest=[]
    for trace in sorted((root/'scenes').glob('*/repeat_0.jsonl')):
        previous=None;selected=set();folder=out/trace.parent.name;folder.mkdir()
        with trace.open() as stream:
            for line in stream:
                row=json.loads(line);token=row['precheck']['frame_id']
                if previous is not None and len(selected)<3:
                    old,new=frame_from_info(infos[previous],'data/nuscenes'),frame_from_info(infos[token],'data/nuscenes')
                    for cid,e in row['precheck']['diagnostics'].get('temporal_evidence',{}).items():
                        m=e.get('point_measurements',{});candidates=m.get('candidates',{})
                        if not candidates:continue
                        p=np.asarray(m['source_points'],float);chosen=np.asarray(m['selected_candidate'],object)
                        for kind in ('recovered','ambiguous','quality_rejected'):
                            if kind in selected:continue
                            raw=candidates['raw']
                            mask=(np.asarray(m['seed_recovered'],bool) if kind=='recovered' else chosen=='ambiguous' if kind=='ambiguous'
                                  else np.asarray(raw['lk_valid'],bool)&~np.asarray(raw['fully_supported'],bool))
                            indices=np.flatnonzero(mask)
                            if not len(indices):continue
                            image=np.concatenate([old.cameras[cid].image,new.cameras[cid].image],axis=1);w=old.cameras[cid].image.shape[1]
                            entries=[]
                            for j in indices[:12]:
                                source=tuple(np.rint(p[j]).astype(int));cv2.circle(image,source,3,(255,255,255),1)
                                for number,(name,c) in enumerate(candidates.items()):
                                    if not c['lk_valid'][j]:continue
                                    q=np.asarray(c['points'][j],float)
                                    if not np.isfinite(q).all():continue
                                    target=tuple(np.rint(q+[w,0]).astype(int));color=((0,240,0) if c['fully_supported'][j] else (255,100,0))
                                    cv2.line(image,source,target,color,1);cv2.circle(image,target,3+number,color,1)
                                    entries.append(dict(index=int(j),candidate=name,fully_supported=c['fully_supported'][j],
                                        measured_point=q.tolist(),FB_error_px=c['fb_error_px'][j],patch_residual=c['patch_residual'][j],
                                        geometry_valid=c['geometry_valid'][j]))
                            cv2.putText(image,kind+'; actual LK measurements only; green full gates/orange rejected; correctness Unknown',(8,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                            path=folder/(kind+'.png');cv2.imwrite(str(path),cv2.cvtColor(image,cv2.COLOR_RGB2BGR));selected.add(kind)
                            manifest.append(dict(scene_id=trace.parent.name,source_frame=previous,current_frame=token,camera=cid,
                                kind=kind,path=str(path),sha256=sha(path),measurements=entries,identity_or_correctness_claim=False))
                previous=token
        for kind in set(('recovered','ambiguous','quality_rejected'))-selected:
            manifest.append(dict(scene_id=trace.parent.name,kind=kind,status='No qualifying measured case'))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.root,a.output)


if __name__=='__main__':main()
