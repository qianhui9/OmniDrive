"""Matched-camera-boundary final M1 validation full-task CUDA comparison.

Uses existing frozen quality and cost gates. Natural and repeated inputs are
separate invocations. This is an audited workload including diagnostic copies.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter
import random
import resource
import numpy as np
import torch
from worldstream.m1 import M1Config, WorldStateManager
from worldstream.m2 import IncrementalVisionRuntime
from tools.worldstream_m1.io_utils import write_json
from .rearchitecture.integration import load
from .rearchitecture.common import arrays, memory, compare, quality_pass
from .refine_integration import planning_metric


@torch.no_grad()
def window_scope_probe(model, dataset, ids, dest):
    """Real EVA prefix receipts and real RGB support; no forced partial updates."""
    from mmcv import Config
    from tools.worldstream_m1.source_state_refinement_replay import preprocessed_frame
    from tools.worldstream_m1.evidence_experiments import load_inputs
    settings=M1Config.load('configs/worldstream_m1_final_validation.yaml')
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m1_observe.py')
    _,_,outputs,_=load_inputs('development')
    manager=WorldStateManager(settings)
    runtime=IncrementalVisionRuntime(model.img_backbone,dict(mode='auto',reuse_boundary='window',
        cache_capture='always',cost_table='outputs/module2/break_even_results.json'))
    info=dataset.data_infos[ids[0]];frame,images=preprocessed_frame(info,cfg,settings)
    pre=manager.precheck(frame)
    _,receipt=runtime.forward(images.cuda(),execution_context=dict(scene_id=frame.scene_id,
        frame_id=frame.frame_id,timestamp=frame.timestamp,camera_ids=list(frame.cameras),
        coordinate_space='preprocessed_image',execution_signature=frame.execution_signature))
    out=copy.deepcopy(outputs[frame.frame_id]);out.execution_receipt=receipt
    manager.commit(frame,out,pre)
    frame,_=preprocessed_frame(dataset.data_infos[ids[1]],cfg,settings)
    pre=manager.precheck(frame)
    records=[r for r in pre.computation_state.source_records if str(r.get('feature_id','')).startswith('eva_prefix:')]
    write_json(dest/'window_scope_probe.json',dict(status='completed' if records else 'no_cache',
        records=records,receipt=receipt,population='exploratory_reused_population',
        scope='Real original EVA prefix cache, actual nuScenes RGB; Auto quality gates unchanged'))
    manager.abort_pending();runtime.reset()


@torch.no_grad()
def profile_zero_camera_cost(backbone, images, path):
    """Measure this exact signature; task qualification is initially withheld."""
    from worldstream.m2.camera_executor import camera_cost_key
    from worldstream.m2.profiler import measure, errors
    runtime = IncrementalVisionRuntime(backbone, dict(mode="exact", reuse_boundary="camera"))
    context = dict(scene_id="zero_cost_probe", frame_id="0", camera_ids=list(range(len(images))))
    runtime.forward(images, execution_context=context)
    cache = runtime.camera_executor.cache
    saved = copy.copy(vars(cache))
    def reuse():
        cache.__dict__.update(saved)
        return runtime.forward(images, execution_context=dict(context, frame_id="1"))[0][0]
    reference = backbone(images)[0]
    actual = reuse()
    if runtime.last_diagnostics.get("actual_reused_cameras") != len(images):
        raise AssertionError("Zero-update profiler did not reuse every camera")
    error = errors(actual, reference)
    if not torch.equal(actual, reference):
        raise AssertionError("Zero-update feature copy differs from the full backbone")
    reuse_cost = measure(reuse, warmup=3, repeats=10)
    full_cost = measure(lambda: backbone(images), warmup=3, repeats=10)
    table = dict(schema_version=1, records=[dict(
        key=camera_cost_key(backbone, images), changed_cameras=0,
        quality_certified=False, reuse_wall_ms=reuse_cost['wall_ms'],
        full_wall_ms=full_cost['wall_ms'], reuse_gpu_ms=reuse_cost['gpu_ms'],
        full_gpu_ms=full_cost['gpu_ms'], reuse=reuse_cost, full=full_cost,
        feature_error=error, qualification="Pending fresh full-task zero-update probe")])
    write_json(path, table)
    runtime.reset()
    return table


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=12)
    p.add_argument("--scenes", type=int, default=2)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--case", choices=["natural", "repeat"], default="natural")
    p.add_argument("--replay-protocol", help="Use the existing replay scenes and first samples")
    p.add_argument("--camera-cost-table", default="outputs/module1_state_interface_rearchitecture/gpu_validation_zero_cost/repeat/camera_costs.json")
    p.add_argument("--lease-experiment", action="store_true")
    p.add_argument("--selective-probe", action="store_true")
    p.add_argument("--qualify-zero-camera-cost", action="store_true",
        help="Fresh zero-update-only feature/cost/full-task probe before Auto; never qualifies partial updates")
    p.add_argument("--output", default="outputs/module1_source_state_refinement/cuda_integration")
    args = p.parse_args()
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        write_json(dest/"status.json", dict(status="Not Run", reason="CUDA unavailable in this execution context"))
        raise SystemExit("CUDA unavailable; no full-model results generated")
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)
    model, dataset = load()
    fresh_costs = None
    groups = {}
    for i, info in enumerate(dataset.data_infos):
        groups.setdefault(info["scene_token"], []).append(i)
    if args.replay_protocol:
        selected = json.loads(Path(args.replay_protocol).read_text())["scenes"][:args.scenes]
        chosen = {}
        for spec in selected:
            scene = spec["scene_id"]
            ids = sorted(groups[scene], key=lambda i: dataset.data_infos[i]["timestamp"])
            start = next(j for j,i in enumerate(ids) if dataset.data_infos[i]["token"] == spec["samples"][0])
            chosen[scene] = ids[start:start+args.frames]
        groups = chosen
    else:
        groups = {scene: sorted(ids, key=lambda i: dataset.data_infos[i]["timestamp"])[:args.frames]
                  for scene, ids in sorted(groups.items())[:args.scenes]}
    if not groups or any(len(ids) != args.frames for ids in groups.values()):
        raise ValueError("Requested complete continuous sequences are unavailable")
    modes = ["original", "full_maintain", "old_source_exact", "current_optimized", "measurement_only", "optimized_exact", "optimized_on_demand"]
    if args.lease_experiment:
        if args.case != 'repeat':raise ValueError('Lease separation probe uses controlled inputs only')
        modes.append('optimized_uncoupled')
    if args.selective_probe:
        if args.case != 'repeat':raise ValueError('Injected Dirty probe uses controlled inputs only')
        modes.append('optimized_dirty_front')
    if args.qualify_zero_camera_cost:
        if args.case != "repeat":
            raise ValueError("Zero-update qualification requires controlled repeated images")
        # The old strict arm was already measured; add a fresh experimental probe.
        modes = ["original", "source_linked_zero_probe", "source_linked_exact",
                 "source_linked_on_demand", "full_maintain"]
        args.camera_cost_table = str(dest/"camera_costs.json")
    write_json(dest/"protocol.json", dict(args=vars(args), groups=groups, modes=modes,
        matched_comparison=dict(boundary='camera',cost_table=args.camera_cost_table,
            cost_table_sha256=hashlib.sha256(Path(args.camera_cost_table).read_bytes()).hexdigest(),
            old_source='Legacy M1 flags and coupled/global-invalidating contract on the shared instrumented runtime',
            new_source='Refined scope/links/guard/cache and selective camera contract',
            partial_updates='Auto remains restricted to existing signature-specific task/cost qualifications'),
        population="exploratory_reused_population",
        service="persistent_state for all M1 arms; original=current_output",
        quality_thresholds="existing frozen module2_rearchitecture targets; unchanged",
        language="original greedy generation, seed 0",
        timing="scatter outside wall; M1/M2/cache/heads/memory/LLM and diagnostic CPU feature copies inside"))
    sources = [*Path("worldstream/m1").glob("*.py"), *Path("worldstream/m2").glob("*.py"),
        Path(__file__), Path("projects/mmdet3d_plugin/models/detectors/petr3d.py")]
    write_json(dest/"source_manifest.json",{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    if args.case == 'natural':
        window_scope_probe(model,dataset,next(iter(groups.values())),dest)
    runs, quality, refs = [], [], {}
    captured = []
    original_extract = model.extract_img_feat
    def extract(*a, **kw):
        values = original_extract(*a, **kw)
        captured.append(arrays(values))
        return values
    model.extract_img_feat = extract
    raw_stages = {}
    backbone_events=[];backbone_forward=model.img_backbone.forward
    def timed_backbone(*a,**kw):
        first,last=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        first.record();value=backbone_forward(*a,**kw);last.record()
        backbone_events.append((first,last))
        return value
    model.img_backbone.forward=timed_backbone
    handles = [model.img_backbone.blocks[i].register_forward_hook(
        lambda module, inputs, output, i=i: raw_stages.update({i:output.detach()}))
        for i in (0,1,2,5,11,23) if i < len(model.img_backbone.blocks)]
    for rep in range(args.warmup + args.repeats):
        for scene, ids in groups.items():
            # Collect the current baseline oracle first; rotate the remaining arms.
            arms = modes[1:]
            order = (["original", "source_linked_zero_probe"] + [m for m in arms if m != "source_linked_zero_probe"]
                if args.qualify_zero_camera_cost else
                ["original"] + arms[rep % len(arms):] + arms[:rep % len(arms)])
            for mode in order:
                random.seed(0)
                np.random.seed(0)
                torch.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                model.test_flag = False
                model.m2_runtime = None
                model.m1_manager = None
                if mode != "original":
                    settings = M1Config.load("configs/worldstream_m1_physical_observability.yaml"
                        if mode == "strict_exact" else ("configs/worldstream_m1_source_linked_state.yaml"
                        if mode == "old_source_exact" else ("configs/worldstream_m1_source_state_refined.yaml" if mode=="current_optimized" else ("configs/worldstream_m1_engineering_frozen.yaml" if mode=="measurement_only" else "configs/worldstream_m1_final_validation.yaml"))))
                    model.m1_manager = WorldStateManager(settings)
                    if mode != "full_maintain":
                        model.m2_runtime = IncrementalVisionRuntime(model.img_backbone, dict(
                            mode="exact" if mode == "source_linked_zero_probe" else "auto",
                            reuse_boundary="camera",
                            camera_selective_invalidation=mode != "old_source_exact",
                            experimental_exact_lease_separation=mode == "optimized_uncoupled",
                            state_interface="legacy" if mode == "strict_exact" else "source_linked",
                            cheap_gate=mode == "optimized_on_demand", m1_service="maintain",
                            cost_table="outputs/module2/break_even_results.json",
                            camera_cost_table=args.camera_cost_table))
                if mode == 'optimized_dirty_front':
                    precheck=model.m1_manager.precheck_with_observation
                    def injected(frame, validate_observation=True):
                        from worldstream.m1.mask_codec import decode_mask,encode_mask
                        result=precheck(frame,validate_observation)
                        cid=next((c for c in frame.cameras if c=='CAM_FRONT'),next(iter(frame.cameras)))
                        contract=result.computation_state.refresh_contract[cid]
                        d=decode_mask(contract['dirty_hint_mask']);c=decode_mask(contract['reuse_candidate_mask'])
                        d[1,1]=True;c[d]=False
                        contract.update(dirty_hint_mask=encode_mask(d),reuse_candidate_mask=encode_mask(c),
                            unknown_mask=encode_mask(~(d|c)))
                        result.diagnostics['controlled_dirty_injection']=cid
                        return result
                    model.m1_manager.precheck_with_observation=injected
                model.save_path = str(dest/f"{args.case}_{scene}_{mode}_{rep}"/"language") + "/"
                baseline_image, frames = None, []
                torch.cuda.reset_peak_memory_stats()
                for ordinal, index in enumerate(ids):
                    from mmcv.parallel import collate, scatter
                    batch = scatter(collate([dataset[index]], samples_per_gpu=1), [torch.cuda.current_device()])[0]
                    if args.case == "repeat":
                        if baseline_image is None:
                            baseline_image = batch["img"][0].clone()
                        batch["img"][0].copy_(baseline_image)
                    if args.qualify_zero_camera_cost and fresh_costs is None:
                        fresh_costs = profile_zero_camera_cost(model.img_backbone,
                            batch['img'][0].reshape(-1, *batch['img'][0].shape[-3:]),
                            Path(args.camera_cost_table))
                    if mode == "source_linked_zero_probe" and not torch.equal(batch['img'][0], baseline_image):
                        raise AssertionError("Qualification probe requires identical full-camera inputs")
                    captured.clear()
                    raw_stages.clear()
                    backbone_events.clear()
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    tick = perf_counter()
                    a.record()
                    out = model(return_loss=False, rescale=True, **batch)
                    b.record()
                    torch.cuda.synchronize()
                    wall = (perf_counter() - tick) * 1000
                    current = dict(output=arrays(out), feature=captured[-1][0], memory=memory(model),
                        text=out[0].get("text_out"),
                        planning=planning_metric(out[0].get("text_out"), dataset.data_infos[index]),
                        stages={i:arrays(x) for i,x in raw_stages.items()})
                    key = scene, ordinal, rep
                    if mode == "original":
                        refs[key] = current
                    else:
                        q = compare(current, refs[key])
                        q.update(mode=mode, scene=scene, ordinal=ordinal, repeat=rep,
                                 frozen_targets_pass=quality_pass(q))
                        from .refinement_common import error_location
                        q['executed_intermediate_EVA'] = {str(i):error_location(
                            torch.from_numpy(x),torch.from_numpy(refs[key]['stages'][i]))
                            for i,x in current['stages'].items()}
                        q['intermediate_scope'] = 'Executed blocks only; whole-camera cache hits produce no new intermediate tensors'
                        quality.append(q)
                    diag = copy.deepcopy(model.m2_runtime.last_diagnostics) if model.m2_runtime else {}
                    if mode == "source_linked_zero_probe" and diag.get('recomputed_cameras', 0) not in (0, 6):
                        raise AssertionError("Partial camera updates are forbidden in zero-update qualification")
                    bundle = (model.m1_manager.last_result.computation_state
                        if model.m1_manager and model.m1_manager.last_result else None)
                    source_state = None if bundle is None else dict(
                        scene_id=bundle.scene_id,frame_id=bundle.frame_id,state_epoch=bundle.state_epoch,
                        diagnostics=copy.deepcopy(bundle.diagnostics),
                        source_records=copy.deepcopy([s for s in bundle.source_records
                            if s.get('feature_id')=='backbone' or str(s.get('feature_id','')).startswith('eva_prefix:')]))
                    frames.append(dict(frame=dataset.data_infos[index]["token"],
                        E2E_wall_ms=wall, E2E_GPU_ms=a.elapsed_time(b),
                        backbone_GPU_interval_ms=sum(first.elapsed_time(last) for first,last in backbone_events),
                        M1_time_ms=model.m1_manager.last_commit_diagnostics.get("m1_total_including_adapters_ms")
                            if model.m1_manager else 0., runtime=diag,
                        state_epoch=model.m1_manager.store.generation if model.m1_manager else None,
                        m1_commit=copy.deepcopy(model.m1_manager.last_commit_diagnostics) if model.m1_manager else None,
                        m1_precheck_diagnostics=copy.deepcopy(model.m1_manager.last_result.diagnostics) if model.m1_manager else None,
                        m1_anchor_ids=list(model.m1_manager.store.anchors) if model.m1_manager else None,
                        m1_computation_state=source_state,
                        current_model_outputs=dict(text=current['text'],planning=current['planning'])))
                    # Native dtype memory snapshots are written outside the timed request.
                    folder=dest/'memory'/f'{scene}_{mode}_{rep}'
                    folder.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(folder/f'{ordinal}.npz',**{
                        f'{branch}.{name}':value for branch,values in current['memory'].items()
                        for name,value in values.items() if isinstance(value,np.ndarray)})
                    print(scene,mode,rep,ordinal,round(wall,2),diag.get('fallback_reason'),flush=True)
                runs.append(dict(mode=mode, scene=scene, repeat=rep, warmup=rep < args.warmup,
                    frames=frames, peak_gpu_allocated=torch.cuda.max_memory_allocated(),
                    peak_gpu_reserved=torch.cuda.max_memory_reserved(),
                    cpu_rss_high_water_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
                write_json(dest/"runs.json", runs)
                write_json(dest/"quality.json", quality)
                if mode == "source_linked_zero_probe":
                    checks = [q for q in quality if q['mode'] == mode]
                    hits = sum(f['runtime'].get('actual_reused_cameras', 0) for f in frames)
                    qualified = bool(checks) and hits > 0 and all(q['frozen_targets_pass'] for q in checks)
                    fresh_costs['records'][0].update(quality_certified=qualified,
                        qualification=dict(population="exploratory_reused_population",
                            source=str(dest/'quality.json'), checks=len(checks),
                            scope="All-six-camera zero updates only; unchanged full-task quality gates"))
                    write_json(Path(args.camera_cost_table), fresh_costs)
                    if not qualified:
                        raise AssertionError("Fresh full-task zero-update qualification failed")
                print(scene, mode, rep, "completed", flush=True)
    model.extract_img_feat = original_extract
    model.img_backbone.forward = backbone_forward
    for handle in handles:handle.remove()
    write_json(dest/"status.json", dict(status="completed",
        quality_pass=all(q["frozen_targets_pass"] for q in quality)))


if __name__ == "__main__":
    main()
