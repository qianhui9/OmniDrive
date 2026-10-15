"""Rerun frozen confirmation after separating local pixel veto from anchor evidence.

No threshold or scene selection changes. Compare every final mask against the
pre-correction run and every anchor decision against the existing sparse path.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import cv2
from worldstream.m1 import M1Config
from .refine_coverage import run_scene,collect,VARIANTS
from .dataset import frame_from_info,load_oracle
from .io_utils import write_json,read_prediction_subset


def main():
    root=Path('outputs/module1_refinement/coverage_final');protocol=json.loads((root/'protocol.json').read_text())
    selected=json.loads((root/'selection.json').read_text())['selected'];cv2.setNumThreads(1)
    if (root/'holdout_before_scope_fix').exists():raise ValueError('Already started')
    (root/'holdout').rename(root/'holdout_before_scope_fix')
    (root/'holdout_summary.json').rename(root/'holdout_summary_before_scope_fix.json')
    write_json(root/'scope_fix_protocol.json',dict(selected=selected,parameter_selection_unchanged=True,
        reason='Calibration exposed anchor FAR change from pixel-only motion veto. Separate dense pixel exclusion from the existing sparse anchor evidence, preserving anchor decisions. Confirm identical pixel masks and rerun frozen scenes; no selection on confirmation results.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('worldstream/m1').glob('*.py')}))
    cfg=M1Config.load(protocol['config'])
    configs={name:replace(cfg,**v) for name,v in protocol['methods'].items()}
    infos=pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb'))['infos'];by_id={i['token']:i for i in infos}
    all_infos=[by_id[t] for tokens in protocol['splits']['holdout'].values() for t in tokens]
    predictions=read_prediction_subset('test/mask_eva_lane_det_vlm/Tue_Sep_29_20_12_56_2026/pts_bbox/results_nusc.json',{i['token'] for i in all_infos});oracle=load_oracle(all_infos,'data/nuscenes')
    audited=0
    for name,tokens in protocol['splits']['holdout'].items():
        chosen=[by_id[t] for t in tokens]
        run_scene(root/'holdout'/name,[frame_from_info(i,'data/nuscenes') for i in chosen],chosen,predictions,oracle,configs)
        for method in configs:
            prior=[json.loads(s) for s in (root/'holdout_before_scope_fix'/name/(method+'_state.jsonl')).open()]
            actual=[json.loads(s) for s in (root/'holdout'/name/(method+'_state.jsonl')).open()]
            base=[json.loads(s) for s in (root/'holdout'/name/'sparse_m1_state.jsonl').open()]
            for old,new,reference in zip(prior,actual,base):
                assert old['invalidation_regions']==new['invalidation_regions'],(name,method,'pixel masks changed')
                if method!='original_m1':
                    assert new['valid_anchor_ids']==reference['valid_anchor_ids'],(name,method,'anchor scope changed')
                    assert new['invalid_anchor_ids']==reference['invalid_anchor_ids']
                audited+=1
    write_json(root/'holdout_summary.json',{n:collect(root,'holdout',n)[0] for n in configs})
    write_json(root/'scope_fix_verification.json',dict(frames_checked=audited,exact_invalidation_metadata_parity=True,
        exact_sparse_anchor_decision_parity=True,threshold_selection_unchanged=True))

if __name__=='__main__':main()
