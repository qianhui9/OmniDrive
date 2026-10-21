"""Frozen 96-frame online replay; annotations only select offline diagnostics."""
import copy
import cProfile
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import cv2
import numpy as np
import torch
from worldstream.m1 import M1Config, WorldStateManager
from worldstream.m1.geometry import project_anchor, region_slices, intersect_area, region_area
from worldstream.m1.mask_codec import decode_mask
from .source_state_refinement_replay import preprocessed_frame
from .evidence_experiments import load_inputs
from .io_utils import write_json

ROOT=Path('outputs/module1_final_validation_refinement')


def main():
    ROOT.mkdir(exist_ok=True);(ROOT/'baseline').mkdir(exist_ok=True)
    (ROOT/'profile_output').mkdir(exist_ok=True);dest=ROOT/'critical_event_overlays';dest.mkdir(exist_ok=True)
    files=[*Path('worldstream').rglob('*.py'),Path('configs/worldstream_m1_source_state_refined.yaml')]
    pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    write_json(ROOT/'baseline/manifest.json',dict(HEAD=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=pins))
    with tarfile.open(ROOT/'baseline/source.tar.gz','w:gz') as tar:
        for path in files:tar.add(path)
    source=Path('outputs/module1_source_state_refinement/change_first_refresh.json')
    failures=[e for e in json.loads(source.read_text())['methods']['optimized']['failures'] if not e['recalled']]
    assert len(failures)==14
    torch.set_num_threads(1);cv2.setNumThreads(1)
    from mmcv import Config
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m1_observe.py')
    specs,infos,outputs,oracle=load_inputs('development');settings=M1Config.load('configs/worldstream_m1_source_state_refined.yaml')
    # Diagnostic arrays alone do not change any acceptance gate.
    settings=replace(settings,surface_diagnostics=True)
    import worldstream.m1.temporal_visibility as temporal
    original=temporal.build_evidence;raw={}
    def traced(*a,**kw):
        e=original(*a,**kw)
        raw[e.camera_id]=dict(changed=e.changed_mask.copy(),stable=e.stable_mask.copy(),supported=e.supported.copy())
        return e
    temporal.build_evidence=traced
    rows=[];timings=[];prof=cProfile.Profile();prof.enable()
    try:
        for spec in specs[:8]:
            manager=WorldStateManager(settings)
            for token in spec['samples'][:12]:
                frame,_=preprocessed_frame(infos[token],cfg,settings);raw.clear()
                result=manager.precheck(frame)
                for event in (e for e in failures if e['frame']==token):
                    cid=event['camera'];e=manager.last_evidence[cid];cam=frame.cameras[cid]
                    region=region_slices(event['region'],cam.image.shape)
                    b=result.computation_state;contract=b.refresh_contract[cid]
                    masks={k:decode_mask(contract[k]) for k in ('dirty_hint_mask','reuse_candidate_mask','unknown_mask')}
                    masks.update(stable_mask=e.stable_mask,changed_mask=e.changed_mask,
                        observation_exclusion_mask=e.observation_exclusion_mask,
                        measured_change_risk_mask=e.measured_change_risk_mask if e.measured_change_risk_mask is not None else np.zeros(cam.image.shape[:2],bool),
                        raw_changed_mask=raw[cid]['changed'],raw_stable_mask=raw[cid]['stable'])
                    anchors=[]
                    for a in manager.store.anchors.values():
                        if a.anchor_type!='object':continue
                        p=project_anchor(a,cam,settings)
                        if p.region is None:continue
                        inter=intersect_area(p.region,event['region'])
                        if inter<=0:continue
                        from worldstream.m1.dense_support import swept_dynamic_region
                        swept=swept_dynamic_region(a,cam,p,settings)
                        obs=a.observation_support.get(cid)
                        anchors.append(dict(anchor_id=a.anchor_id,semantic_class=a.semantic_class,
                            committed_source_frame=a.source_frame_id,source_timestamp=a.source_timestamp,
                            source_state_epoch=a.source_state_epoch,source_execution_signature=a.source_execution_signature,
                            previous_region=obs.region if obs else None,current_predicted_region=p.region,swept_region=swept,
                            velocity_world=a.velocity_world,velocity_reliable=a.velocity_reliable,
                            intersection_fraction=inter/max(1,region_area(event['region'])),
                            candidate_pixels_in_sweep=int(masks['reuse_candidate_mask'][region_slices(swept,cam.image.shape)].sum())))
                    idx=len(rows);path=dest/f'{idx:02d}_{token}_{cid}'
                    np.savez_compressed(str(path)+'.npz',image=cam.image,source_points=e.source_points,current_points=e.current_points,
                        fb=e.fb_error_px,geometry=e.geometry_valid,correspondence=e.correspondence_valid,
                        supported=e.supported,photo=e.patch_residual,**masks,
                        **({} if e.dense_measurements is None else {f'dense_{k}':v for k,v in e.dense_measurements.items()}))
                    stats={k:int(v[region].sum()) for k,v in masks.items()}
                    lost=int((raw[cid]['changed']&e.observation_exclusion_mask)[region].sum())
                    row=dict(event,scene=frame.scene_id,online_anchors=anchors,mask_pixels_in_offline_ROI=stats,
                        changed_evidence_cleared_by_exclusion=lost,
                        online_evidence_diagnostics=e.diagnostics,
                        root_cause=('measured_change_discarded_by_dynamic_exclusion' if lost else
                            'committed_projection_overlaps_but_risk_not_forwarded' if anchors else
                            'no_overlapping_committed_anchor_local_flow_self_consistent'),
                        GT_used_online=False,arrays=str(path)+'.npz')
                    rows.append(row)
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    fig,axes=plt.subplots(2,4,figsize=(16,8))
                    for ax,(key,value) in zip(axes.ravel(),[('online image + offline GT',cam.image),*list(masks.items())[:7]]):
                        ax.imshow(value,cmap=None if value.ndim==3 else 'gray');x0,y0,x1,y1=event['region']
                        ax.add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color='red',linewidth=1))
                        for a in anchors:
                            u,v,s,t=a['current_predicted_region'];ax.add_patch(plt.Rectangle((u,v),s-u,t-v,fill=False,color='cyan',linewidth=.5))
                        ax.set_xlim(max(0,x0-50),min(cam.image.shape[1],x1+50));ax.set_ylim(min(cam.image.shape[0],y1+50),max(0,y0-50))
                        ax.set_title(key,fontsize=8)
                    fig.suptitle(f'{cid} {event["semantic_class"]}: red=offline GT; cyan=online committed projection')
                    fig.tight_layout();fig.savefig(str(path)+'.png',dpi=120);plt.close(fig)
                manager.commit(frame,copy.deepcopy(outputs[token]),result)
                timings.append(dict(frame=token,scene=frame.scene_id,anchors=len(manager.store.anchors),
                    precheck=result.diagnostics.get('timings_ms'),commit=manager.last_commit_diagnostics))
                print(token,'diagnosed',len(rows),flush=True)
    finally:
        prof.disable();prof.dump_stats(str(ROOT/'profile_output/baseline.prof'));temporal.build_evidence=original
    assert len(rows)==14
    write_json(ROOT/'critical_event_root_cause.json',dict(events=rows,population='frozen_96_frames_exploratory_reused_population',
        baseline_source_sha256=pins,annotation_use='offline diagnostic selection and red overlay only'))
    write_json(ROOT/'profile_output/baseline_timings.json',timings)


if __name__=='__main__':main()
