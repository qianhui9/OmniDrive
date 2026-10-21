"""Fresh CPU nuScenes replay in the original test pipeline coordinates.

Historical original-model detections are Commit inputs. GT is used after
Precheck only for sparse scoring, never to produce candidates or refresh masks.
This tool does not run EVA, heads, memory or the LLM; their results remain Not Run.
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter
import resource
import cv2
import numpy as np
import torch
from worldstream.m1 import M1Config,WorldStateManager
from worldstream.m1.mask_codec import decode_mask
from worldstream.m1.novelty_detector import validate_refresh_partition
from worldstream.m1.omnidrive_adapter import frame_from_model
from .evidence_experiments import load_inputs,append_json,summary
from .coverage_oracle import world_lidar,prepare_pair,score_camera
from .io_utils import write_json


def preprocessed_frame(info,cfg,settings):
    from mmdet3d.datasets.pipelines import Compose
    from projects.mmdet3d_plugin.datasets.nuscenes_dataset import CustomNuScenesDataset
    obj=object.__new__(CustomNuScenesDataset)
    obj.data_infos=[info];obj.modality=dict(use_camera=True);obj.test_mode=True
    x=Compose(cfg.test_pipeline[:5])(obj.get_data_info(0))
    data={k:np.array([x[k]]) for k in ("intrinsics","extrinsics","img_timestamp","ego_pose")}
    data["img"]=torch.from_numpy(np.stack(x["img"]).transpose(0,3,1,2))[None]
    data["timestamp"]=np.array([x["timestamp"]])
    return frame_from_model([x],data,settings),data["img"][0]


def main():
    p=argparse.ArgumentParser();p.add_argument("--scenes",type=int,default=2)
    p.add_argument("--frames",type=int,default=12)
    p.add_argument("--output",default="outputs/module1_state_interface_rearchitecture/replay")
    args=p.parse_args();dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);cv2.setNumThreads(1)
    from mmcv import Config
    cfg=Config.fromfile("projects/configs/OmniDrive/worldstream_m1_observe.py")
    specs,infos,outputs,oracle=load_inputs("development")
    specs=specs[:args.scenes]
    base=M1Config.load("configs/worldstream_m1_physical_observability.yaml")
    linked=M1Config.load("configs/worldstream_m1_source_linked_state.yaml")
    methods=dict(strict_surface=base,
        conservative_observation=replace(base,dense_region_policy="uniform",dense_guard_radius_px=4),
        dense_observation=replace(base,dense_region_policy="uniform",dense_motion_guard=False),
        source_linked=replace(linked,validity_lease=False),source_linked_lease=linked,
        source_linked_refresh=linked,source_linked_on_demand=linked)
    rows=[];timings=[];sources=[];leases=[];funnel=[];identity=[]
    pins={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
        [*Path("worldstream/m1").glob("*.py"),Path(__file__),Path("configs/worldstream_m1_source_linked_state.yaml")]}
    write_json(dest/"protocol.json",dict(population="exploratory_reused_population",scenes=specs,
        frames_per_scene=args.frames,coordinate_space="preprocessed_image",pipeline=cfg.test_pipeline[:5],
        source_sha256=pins,original_perception="historical_full_model_detection_replay",
        online_GT=False,CUDA_execution=False,lease_ablation="hard maximum history age remains mandatory in every arm; optional lease export only",
        refresh_ablation="candidate population unchanged; refresh contract exports policy independently"))
    for spec in specs:
        managers={name:WorldStateManager(settings) for name,settings in methods.items()}
        from worldstream.m2.cheap_admission import CheapAdmission
        gate=CheapAdmission();previous=None;previous_points=None
        for token in spec["samples"][:args.frames]:
            t=perf_counter();frame,images=preprocessed_frame(infos[token],cfg,linked);adapter_ms=(perf_counter()-t)*1000
            # Rejection-only gate on actual preprocessed inputs, no GT.
            patch=cfg.model.img_backbone.get("patch_size",16)
            window=cfg.model.img_backbone.get("window_size",16)
            admission=gate.check(images,dict(scene_id=frame.scene_id,camera_ids=list(frame.cameras),
                execution_signature=frame.execution_signature),(patch*window,patch*window))
            current_points=world_lidar(infos[token])
            refs=prepare_pair(previous,frame,previous_points,current_points,
                oracle[previous.frame_id],oracle[token],base) if previous else None
            for name,m in managers.items():
                start=perf_counter()
                result=m.precheck_with_observation(frame,not admission["reject"]) if name=="source_linked_on_demand" else m.precheck(frame)
                pre_ms=(perf_counter()-start)*1000
                b=result.computation_state
                for cid,cam in frame.cameras.items():
                    e=m.last_evidence.get(cid);strict=m.last_masks[cid]
                    candidate=(decode_mask(b.refresh_contract[cid]["reuse_candidate_mask"]) if b else ~strict)
                    dirty=decode_mask(b.refresh_contract[cid]["dirty_hint_mask"]) if b else decode_mask(result.invalidation_regions[cid]["observed_changed_mask"])
                    unknown=~(candidate|dirty)
                    assert validate_refresh_partition(dirty,candidate,unknown)
                    row=dict(method=name,scene=frame.scene_id,frame=token,camera=cid,pixels=candidate.size,
                        stable_pixels=int(candidate.sum()),effective_pixels=int(candidate.sum()),
                        changed_pixels=int(dirty.sum()),unknown_pixels=int(unknown.sum()),
                        strict_pixels=int((~strict).sum()),forced_refresh=result.full_refresh_hint)
                    if refs:
                        ref=dict(refs[cid]);ref["scorable"]=ref["scorable"] & ref["annotated"]
                        row.update(score_camera(ref,~candidate,e,m.config))
                        strict_score=score_camera(ref,strict,e,m.config)
                        row["strict_refresh_score"]=strict_score
                    append_json(dest/"pixels.jsonl",row);rows.append(row)
                    if b:
                        link=b.observation_links[cid]
                        leases.append(dict(method=name,frame=token,camera=cid,**link["validity_lease"]))
                        for r in b.source_records:
                            if r["source_camera_id"]==cid:
                                assert r["computation_frame_id"] != frame.frame_id
                                assert r["source_frame_id"] in (None,m._previous_frame_id)
                                sources.append(dict(method=name,frame=token,**r))
                        # Full input comparison is an opportunity measurement only.
                        # No M2 execution or task-quality certificate is inferred.
                        exact=np.zeros_like(candidate) if previous is None else (cam.image==previous.cameras[cid].image).all(-1)
                        candidates=verified=0
                        for y in range(0,640,patch*window):
                            for x in range(0,640,patch*window):
                                candidates+=bool(candidate[y:y+patch*window,x:x+patch*window].any())
                                verified+=bool(exact[y:y+patch*window,x:x+patch*window].all())
                        funnel.append(dict(method=name,frame=token,camera=cid,candidate_windows=candidates,
                            exact_RGB_window_opportunities=verified,actual_skipped_windows=None,
                            scope="RGB_input_only_no_EVA_execution"))
                        if name=="source_linked_refresh" and candidate.any() and not (dest/"spatial_masks.npz").exists():
                            np.savez_compressed(dest/"spatial_masks.npz",image=cam.image,candidate=candidate,dirty=dirty,unknown=unknown)
                start=perf_counter();m.commit(frame,copy.deepcopy(outputs[token]),result);commit_ms=(perf_counter()-start)*1000
                identity.append(dict(method=name,frame=token,anchor_ids=list(m.store.anchors),
                    certified=len(m.get_current_certifications().certified_anchor_ids)))
                timings.append(dict(method=name,frame=token,M1_precheck_ms=pre_ms,M1_commit_ms=commit_ms,
                    input_adapter_ms=adapter_ms,M1_wall_ms=pre_ms+commit_ms,
                    validation_requested=not admission["reject"] if name=="source_linked_on_demand" else True))
            print(spec["scene_id"],token,"completed",flush=True)
            previous,previous_points=frame,current_points
    results={name:summary([r for r in rows if r["method"]==name]) for name in methods}
    for name in methods:
        selected=[r for r in rows if r["method"]==name];events=[e for r in selected for e in r.get("events",[])]
        results[name]["minimum_critical_ROI_refresh"]=min([e["invalidation_coverage"] for e in events],default=None)
        results[name]["strict_observation_area"]=sum(r["strict_pixels"] for r in selected)/sum(r["pixels"] for r in selected)
        results[name]["candidate_dense_correct_false_unscorable_area"]=None
    write_json(dest/"method_comparison.json",results)
    write_json(dest/"timings.json",timings);write_json(dest/"source_audit.json",sources)
    write_json(dest/"leases.json",leases);write_json(dest/"funnel.json",funnel)
    write_json(dest/"identity.json",identity)
    write_json(dest/"completion.json",dict(status="completed",frames=len({r["frame"] for r in rows}),
        camera_observations=len({(r["frame"],r["camera"]) for r in rows}),
        rss_high_water_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        CUDA_execution=False,source_sha256=pins))


if __name__=="__main__":main()
