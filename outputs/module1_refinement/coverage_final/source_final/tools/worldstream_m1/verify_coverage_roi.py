"""Actual replay proving the ROI scheduling optimization preserves every output mask."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
from time import perf_counter
import cv2
import numpy as np
from worldstream.m1 import M1Config,WorldStateManager
from .dataset import frame_from_info,prediction_outputs
from .io_utils import read_prediction_subset,write_json


def main():
    root=Path('outputs/module1_refinement/coverage_final');cv2.setNumThreads(1)
    protocol=json.loads((root/'protocol.json').read_text());selected=json.loads((root/'selection.json').read_text())['selected']
    cfg=M1Config.load(protocol['config']);names=['sparse_m1','dense_guard_2','dense_guard_4','dense_guard_8']
    infos=pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb'))['infos'];by_id={i['token']:i for i in infos}
    tokens={t for seq in protocol['splits']['holdout'].values() for t in seq}
    predictions=read_prediction_subset('test/mask_eva_lane_det_vlm/Tue_Sep_29_20_12_56_2026/pts_bbox/results_nusc.json',tokens)
    rows=[];count=0
    for scene,tokens in protocol['splits']['holdout'].items():
        frames=[frame_from_info(by_id[t],'data/nuscenes') for t in tokens]
        for name in names:
            config=replace(cfg,**protocol['methods'][name]);manager=WorldStateManager(config)
            saved=[json.loads(line) for line in (root/'holdout'/scene/(name+'_state.jsonl')).open()]
            for frame,expected in zip(frames,saved):
                outputs=prediction_outputs(predictions[frame.frame_id],config)
                tick=perf_counter();result=manager.precheck(frame);manager.commit(frame,outputs,result);elapsed=(perf_counter()-tick)*1000
                for k in ['valid_anchor_ids','invalid_anchor_ids','invalidation_regions','feature_versions','provenance','full_refresh_hint','valid_feature_candidates']:
                    from worldstream.m1.state_types import json_value
                    assert json_value(getattr(result,k))==expected[k],(scene,name,frame.frame_id,k)
                count+=1
                rows.append(dict(scene=scene,method=name,frame=frame.frame_id,wall_ms=elapsed,
                    precheck=result.diagnostics['timings_ms'],commit=manager.last_commit_diagnostics))
        print(scene,'exact parity',flush=True)
    summary={name:np.percentile([r['wall_ms'] for r in rows if r['method']==name],[50,95]).tolist() for name in names}
    write_json(root/'roi_verification.json',dict(frames_checked=count,all_exact=True,selected=selected,
        preserved=['anchor decisions','exact masks','feature candidates','feature versions','provenance','refresh hints'],
        latency_p50_p95_ms=summary,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('worldstream/m1').glob('*.py')}))
    write_json(root/'roi_timings.json',rows)

if __name__=='__main__':main()
