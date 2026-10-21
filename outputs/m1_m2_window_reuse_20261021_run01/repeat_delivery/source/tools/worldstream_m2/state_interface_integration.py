"""Five-arm original OmniDrive full-task CUDA regression with fresh language paths.

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
    p.add_argument("--warmup-frames",type=int,default=None,
        help="Optional shorter warmup per scene/arm; measured sequences keep all --frames")
    p.add_argument("--case", choices=["natural", "repeat", "local"], default="natural")
    p.add_argument("--replay-protocol", help="Use the existing replay scenes and first samples")
    p.add_argument("--window-study", action="store_true",
        help="Pair same-M1 full execution with strict source-linked window Exact and Auto")
    p.add_argument("--window-cost-table", default="outputs/module2_refinement/break_even_results.json")
    p.add_argument("--camera-cost-table", default="outputs/module2_rearchitecture/camera_costs.json")
    p.add_argument("--qualify-zero-camera-cost", action="store_true",
        help="Fresh zero-update-only feature/cost/full-task probe before Auto; never qualifies partial updates")
    p.add_argument("--output", default="outputs/module1_state_interface_rearchitecture/cuda_integration")
    args = p.parse_args()
    if args.warmup_frames is not None and not 1<=args.warmup_frames<=args.frames:
        p.error("--warmup-frames must be between 1 and --frames")
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        write_json(dest/"status.json", dict(status="Not Run", reason="CUDA unavailable in this execution context"))
        raise SystemExit("CUDA unavailable; no full-model results generated")
    torch.set_num_threads(1)
    if args.window_study:
        # Compare both arms under the same FP32 math policy. TF32 GEMM changes
        # with packed batch shape and failed the frozen full-task quality gate.
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
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
    modes = ["original", "strict_exact", "source_linked_exact", "source_linked_on_demand", "full_maintain"]
    if args.window_study:
        modes = ["original", "source_linked_window_exact", "source_linked_window_auto", "full_maintain"]
        if args.qualify_zero_camera_cost:
            raise ValueError("Window study cannot qualify camera costs")
    if args.qualify_zero_camera_cost:
        if args.case != "repeat":
            raise ValueError("Zero-update qualification requires controlled repeated images")
        # The old strict arm was already measured; add a fresh experimental probe.
        modes = ["original", "source_linked_zero_probe", "source_linked_exact",
                 "source_linked_on_demand", "full_maintain"]
        args.camera_cost_table = str(dest/"camera_costs.json")
    write_json(dest/"protocol.json", dict(args=vars(args), groups=groups, modes=modes,
        population="exploratory_reused_population",
        service="persistent_state for all M1 arms; original=current_output",
        quality_thresholds="existing frozen module2_rearchitecture targets; unchanged",
        language="original greedy generation, seed 0",
        tf32=dict(matmul=torch.backends.cuda.matmul.allow_tf32,cudnn=torch.backends.cudnn.allow_tf32),
        timing="scatter outside wall; M1/M2/cache/heads/memory/LLM and diagnostic CPU feature copies inside"))
    sources = [*Path("worldstream/m1").glob("*.py"), *Path("worldstream/m2").glob("*.py"),
        Path(__file__), Path("projects/mmdet3d_plugin/models/detectors/petr3d.py")]
    write_json(dest/"source_manifest.json",{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    for source in sources:
        target=dest/'source'/source
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source.read_bytes())
    runs, quality, refs = [], [], {}
    captured = []
    original_extract = model.extract_img_feat
    def extract(*a, **kw):
        values = original_extract(*a, **kw)
        captured.append(arrays(values))
        return values
    model.extract_img_feat = extract
    raw_stages = {}
    handles = [model.img_backbone.blocks[i].register_forward_hook(
        lambda module, inputs, output, i=i: raw_stages.update({i:output.detach()}))
        for i in (0,1,2,5,11,23) if i < len(model.img_backbone.blocks)]
    for rep in range(args.warmup + args.repeats):
        for scene, ids in groups.items():
            refs.clear()  # Only the paired current sequence is needed; bound host memory.
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
                        if mode == "strict_exact" else "configs/worldstream_m1_source_linked_state.yaml")
                    model.m1_manager = WorldStateManager(settings)
                    if mode != "full_maintain":
                        model.m2_runtime = IncrementalVisionRuntime(model.img_backbone, dict(
                            mode="exact" if mode in ("source_linked_zero_probe", "source_linked_window_exact") else "auto",
                            cache_capture="always" if mode == "source_linked_window_exact" else "adaptive",
                            reuse_boundary="window" if mode == "strict_exact" or mode.startswith("source_linked_window_") else "camera",
                            state_interface="legacy" if mode == "strict_exact" else "source_linked",
                            cheap_gate=mode == "source_linked_on_demand", m1_service="maintain",
                            cost_table=args.window_cost_table if args.window_study else "outputs/module2/break_even_results.json",
                            camera_cost_table=args.camera_cost_table))
                        if args.window_study:
                            model.m2_runtime.stage_observer=lambda i,x:raw_stages.update({i:x.detach()}) if i in (0,1,2,5,11,23) else None
                model.save_path = str(dest/f"{args.case}_{scene}_{mode}_{rep}"/"language") + "/"
                baseline_image, frames = None, []
                torch.cuda.reset_peak_memory_stats()
                sequence=ids[:args.warmup_frames] if rep<args.warmup and args.warmup_frames else ids
                for ordinal, index in enumerate(sequence):
                    from mmcv.parallel import collate, scatter
                    batch = scatter(collate([dataset[index]], samples_per_gpu=1), [torch.cuda.current_device()])[0]
                    if args.case in ("repeat", "local"):
                        if baseline_image is None:
                            baseline_image = batch["img"][0].clone()
                        batch["img"][0].copy_(baseline_image)
                        if args.case == "local" and ordinal:
                            # A single current-input pixel in camera zero changes;
                            # all metadata and all downstream tasks still execute.
                            batch["img"][0].reshape(-1,*batch["img"][0].shape[-3:])[0,:,0,0] += ordinal*.125
                    if args.qualify_zero_camera_cost and fresh_costs is None:
                        fresh_costs = profile_zero_camera_cost(model.img_backbone,
                            batch['img'][0].reshape(-1, *batch['img'][0].shape[-3:]),
                            Path(args.camera_cost_table))
                    if mode == "source_linked_zero_probe" and not torch.equal(batch['img'][0], baseline_image):
                        raise AssertionError("Qualification probe requires identical full-camera inputs")
                    captured.clear()
                    raw_stages.clear()
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
                        q['intermediate_scope'] = 'Observed executed/reconstructed prefix and executed full blocks; camera hits have no new intermediate tensors'
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
                        M1_time_ms=model.m1_manager.last_commit_diagnostics.get("m1_total_including_adapters_ms")
                            if model.m1_manager else 0., runtime=diag,
                        state_epoch=model.m1_manager.store.generation if model.m1_manager else None,
                        m1_commit=copy.deepcopy(model.m1_manager.last_commit_diagnostics) if model.m1_manager else None,
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
    for handle in handles:handle.remove()
    write_json(dest/"status.json", dict(status="completed",
        quality_pass=all(q["frozen_targets_pass"] for q in quality)))


if __name__ == "__main__":
    main()
