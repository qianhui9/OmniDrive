"""Render measured supported/rejected views without failed-LK placeholders.

Scores are read from post-run scoring artifacts; no annotations are consulted.
No rejected/raw-invalid coordinate is portrayed as an actual accepted match.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import pickle
import numpy as np
from .dataset import frame_from_info
from .motion_scoring import panel
from .effective_acceptance import sha
from worldstream.m1 import M1Config


def render(root,scoring,output):
    root=Path(root);scoring=Path(scoring);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    raw=(root/'protocol.json').exists()
    protocol=json.loads((root/('protocol.json' if raw else 'provenance.json')).read_text())
    score_protocol=json.loads((scoring/'protocol.json').read_text())
    with open(protocol.get('infos_path',protocol.get('dataset')),'rb') as f:infos={i['token']:i for i in pickle.load(f)['infos']}
    if not raw:
        from mmcv import Config
        from .state_interface_replay import preprocessed_frame
        cfg=Config.fromfile(score_protocol['runtime_config']);settings=M1Config.load(score_protocol['m1_config'])
    def get_frame(token):
        return frame_from_info(infos[token],'data/nuscenes') if raw else preprocessed_frame(infos[token],cfg,settings)[0]
    traces=sorted((root/'scenes').glob('*/repeat_0.jsonl')) if raw else sorted((root/'scenes').glob('*/m1_state.jsonl'))
    manifest=[]
    for trace in traces:
        scored=defaultdict(list)
        for line in (scoring/trace.parent.name/'scored_tracks.jsonl').open():
            r=json.loads(line);scored[r['current_frame'],r['camera']].append(r)
        choices={};rows=[json.loads(x) for x in trace.read_text().splitlines()]
        for index,row in enumerate(rows):
            if not index:continue
            for cid,e in row['precheck']['diagnostics'].get('temporal_evidence',{}).items():
                m=e.get('point_measurements',{});rs=scored.get((row['precheck']['frame_id'],cid),[])
                if not rs:continue
                fb=np.asarray(m['correspondence_valid'],bool);measured=np.asarray([r['measured'] for r in rs],bool)
                for kind,mask in [('supported',fb&measured),('quality_rejected',fb&~measured)]:
                    if kind not in choices and mask.any():choices[kind]=(index,cid,m,rs,int(mask.sum()))
        for kind in ('supported','quality_rejected'):
            if kind not in choices:
                manifest.append(dict(scene_id=trace.parent.name,kind=kind,status='Not Available: no FB-valid measured view'));continue
            index,cid,m,rs,n=choices[kind];source=rows[index-1]['precheck']['frame_id'];target=rows[index]['precheck']['frame_id']
            old,current=get_frame(source),get_frame(target);name=f'{trace.parent.name}_{kind}.png'
            panel(old,current,cid,m,rs,out/name,rejected=kind=='quality_rejected')
            manifest.append(dict(scene_id=trace.parent.name,source_frame=source,current_frame=target,camera=cid,
                kind=kind,FB_valid_points=n,file=name,formal_identity_certified=False,
                statement='Actual RGB tracks and conditional search predictions only; raw invalid points omitted'))
    (out/'manifest.json').write_text(json.dumps(dict(renderer_sha256=sha(__file__),plotter_sha256=sha('tools/worldstream_m1/motion_scoring.py'),
        source_trace_sha256={str(p):sha(p) for p in traces},cases=manifest),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--scoring',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();render(a.root,a.scoring,a.output)


if __name__=='__main__':main()
