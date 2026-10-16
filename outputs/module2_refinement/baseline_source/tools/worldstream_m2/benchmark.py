"""Production-weight CUDA block/backbone benchmark, correctness oracle and profiler traces."""
import argparse
from pathlib import Path
import torch
import numpy as np
from time import perf_counter
from .common import backbone, write
from worldstream.m2 import IncrementalVisionRuntime
from worldstream.m2.cost_model import cost_key
from worldstream.m2.packing import partition, unpartition
from worldstream.m2.window_executor import WindowExecutor, packed_forward
from worldstream.m2.profiler import measure, errors


def kernel_profile(fn, path):
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                          torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
        fn(); torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return dict(kernel_count=len(events), kernel_active_ms=sum(e.self_cuda_time_total for e in events)/1000,
                h2d_d2h_events=[e.name for e in events if 'Memcpy' in e.name],
                linear_input_shapes=[e.input_shapes for e in prof.events() if e.name=='aten::linear'],
                cpu_cuda_calls=[dict(name=e.key, calls=e.count, cpu_us=e.self_cpu_time_total)
                                for e in prof.key_averages() if 'cuda' in e.key.lower()])


@torch.no_grad()
def breakdown(block, x, cache, ids, executor, repeats=20):
    samples=[]
    for _ in range(repeats):
        ev=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
        tick=perf_counter(); ev[0].record()
        windows,padded=partition(x,block.window_size)
        valid,_=partition(torch.ones_like(x[...,:1]),block.window_size)
        packed=executor.buffer.gather(windows,ids); valid=valid.index_select(0,ids)
        ev[1].record()
        output=packed_forward(block,packed,valid)
        ev[2].record()
        target=cache.clone(); target.index_copy_(0,ids,output)
        target=unpartition(target,block.window_size,padded,x.shape[1:3])
        ev[3].record(); ev[3].synchronize()
        parts=[ev[i].elapsed_time(ev[i+1]) for i in range(3)]
        samples.append(dict(pack_ms=parts[0], sparse_ms=parts[1], scatter_ms=parts[2],
            synchronization_and_host_residual_ms=max(0.,(perf_counter()-tick)*1000-sum(parts))))
    return {k:float(np.median([r[k] for r in samples])) for k in samples[0]}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', default='outputs/module2')
    p.add_argument('--repeats', type=int, default=20)
    args = p.parse_args(); root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True); (root/'traces').mkdir(exist_ok=True)
    torch.set_num_threads(1); torch.manual_seed(2026); np.random.seed(2026)
    torch.backends.cuda.matmul.allow_tf32=False
    model = backbone(); block = model.blocks[0]
    images = torch.randn(6, 3, 640, 640, device='cuda', dtype=torch.float32)
    # Independent all-dense precision calibration: original EVA at camera batch 6
    # versus original EVA at camera batch 1, repeated before any sparse execution.
    # No cache, mask or sparse implementation is used to establish this envelope.
    dense_layers={}
    hh=[b.register_forward_hook(lambda m,inp,out,i=i: dense_layers.update({i:out.detach().clone()}))
        for i,b in enumerate(model.blocks)]
    dense_reference=model(images)[0]
    for handle in hh: handle.remove()
    calibration_runs=[]; calibration_layers={i:[] for i in range(24)}
    for repeat in range(3):
        chunks={i:[] for i in range(24)}
        hh=[b.register_forward_hook(lambda m,inp,out,i=i: chunks[i].append(out.detach().clone()))
            for i,b in enumerate(model.blocks)]
        micro=torch.cat([model(camera[None])[0] for camera in images])
        for handle in hh: handle.remove()
        calibration_runs.append(errors(micro,dense_reference))
        for i in range(24): calibration_layers[i].append(errors(torch.cat(chunks[i]),dense_layers[i]))
    # A separate original full-forward FP32 attention oracle quantifies the
    # deployed FlashAttention FP16 arithmetic floor, independently of sparse execution.
    precision_layers={}
    hh=[b.register_forward_hook(lambda m,inp,out,i=i: precision_layers.update({i:out.detach().clone()}))
        for i,b in enumerate(model.blocks)]
    for b in model.blocks: b.attn.flash_attn=False
    fp32_reference=model(images)[0]
    for b in model.blocks: b.attn.flash_attn=True
    for handle in hh: handle.remove()
    precision_error=errors(fp32_reference,dense_reference)
    for i in range(24): calibration_layers[i].append(errors(precision_layers[i],dense_layers[i]))
    # Envelope equals measured independent dense execution variation, without tuning
    # against sparse results or multiplying it by a post-hoc safety factor.
    tolerances={i:{k:max(r[k] for r in calibration_layers[i]) for k in ('max_abs','relative_l2')}
                for i in range(24)}
    write(root/'precision_calibration.json',dict(protocol='3 repeats: original full EVA batch6 vs six original full batch1 forwards, plus independent original full FP32 non-flash attention oracle; no reuse',
        final=calibration_runs, fp32_attention=precision_error, per_layer=calibration_layers, tolerance=tolerances))
    del dense_layers, dense_reference, chunks, micro, precision_layers, fp32_reference
    from projects.mmdet3d_plugin.models.backbones.eva_vit import get_abs_pos
    x = model.patch_embed(images)
    x += get_abs_pos(model.pos_embed, model.pretrain_use_cls_token, x.shape[1:3])
    cached, _ = partition(block(x), 16)
    dense_time = measure(lambda: block(x), repeats=args.repeats)
    ex = WindowExecutor(); rows=[]; profiles={}; cost_records=[]; graph_rows=[]
    # Fixed tolerance set before evaluating sparse accuracy. Dense-repeat noise is recorded.
    # FP16 comparisons use 0.02 absolute / 0.002 relative-L2 as experiment acceptance;
    # failures are recorded, not silently used to calibrate a larger tolerance.
    calibration = errors(block(x), block(x))
    for count in [0, 1, 2, 4, 8, 16, 27, 40, 48, 53, 54]:
        ids = torch.randperm(54, device='cuda')[:count]
        current_windows, padded = partition(x, 16)
        current_windows[ids] += .03125
        current = unpartition(current_windows, 16, padded, (40, 40))
        oracle = block(current)
        fn = lambda: ex.execute(block, current, cached, ids)
        naive = lambda: ex.execute(block, current, cached, ids, naive=True)
        a, _ = fn(); e = errors(a, oracle)
        packed_time = measure(fn, repeats=args.repeats)
        naive_time = measure(naive, warmup=3, repeats=args.repeats)
        row = dict(updated=count, total=54, ratio=count/54, ids=ids.cpu().tolist(),
            full=dense_time, packed=packed_time, naive=naive_time, error=e,
            correctness=e['max_abs'] <= .02 and e['relative_l2'] <= .002,
            executed_attention_sequences=count, dense_attention_sequences=54)
        rows.append(row)
        cost_records.append(dict(key=cost_key(block, x), updated=count,
            incremental_gpu_ms=packed_time['gpu_ms'], full_gpu_ms=dense_time['gpu_ms'],
            incremental_wall_ms=packed_time['wall_ms'], full_wall_ms=dense_time['wall_ms']))
        if count in (1, 27, 54):
            profiles[str(count)] = dict(packed=kernel_profile(fn, root/'traces'/f'packed_{count}.json'),
                naive=kernel_profile(naive, root/'traces'/f'naive_{count}.json'),
                components=breakdown(block,current,cached,ids,ex,args.repeats))
        # Capture whole packed path, including gather, cache copy, scatter and reconstruction.
        try:
            capture_start=perf_counter()
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3): captured = fn()
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(graph): captured = fn()
            graph.replay(); torch.cuda.synchronize()
            capture_ms=(perf_counter()-capture_start)*1000
            ge = errors(captured[0], oracle)
            gt = measure(graph.replay, repeats=args.repeats)
            graph_rows.append(dict(updated=count, status='supported', timing=gt, error=ge,
                capture_and_warmup_ms=capture_ms,
                faster_in_microbenchmark=gt['gpu_ms'] < packed_time['gpu_ms'],
                retained_in_runtime=False, scope='fixed addresses/count, benchmark only'))
            del graph, captured
        except Exception as error:
            graph_rows.append(dict(updated=count, status='unsupported', reason=repr(error)))
        print('block', count, 'packed/full', packed_time['gpu_ms'], dense_time['gpu_ms'], 'error', e, flush=True)
    profiles['full'] = kernel_profile(lambda: block(x), root/'traces'/'full.json')
    write(root/'controlled_sparse_benchmark.json', dict(status='measured', checkpoint='ckpts/iter_10548.pth',
        dtype='float32 weights/activations, original FP16 FlashAttention', shape=list(x.shape), seed=2026, warmup=5, repeats=args.repeats,
        calibration_dense_repeat=calibration, tolerance=dict(max_abs=.02, relative_l2=.002), rows=rows))
    write(root/'break_even_results.json', dict(records=cost_records,
        beneficial_counts=[r['updated'] for r in rows if r['packed']['gpu_ms']<dense_time['gpu_ms']
                           and r['packed']['wall_ms']<dense_time['wall_ms']],
        scope='block 0 only; includes packing/cache/scatter; backbone input comparison excluded'))
    write(root/'packing_profile.json', profiles); write(root/'cuda_graph_profile.json', graph_rows)
    # Separately time the actual complete 24-block backbone and input certification overhead.
    runtime = IncrementalVisionRuntime(model, dict(mode='exact'))
    context = dict(scene_id='controlled', frame_id='0', camera_ids=[str(i) for i in range(6)])
    cold = measure(lambda: (runtime.reset(), runtime.forward(images, execution_context=context)), warmup=1, repeats=3)
    full = measure(lambda: model(images), repeats=max(5, args.repeats//2))
    backbone_rows=[]; correctness=[]
    dependency_trace=None
    cases={}
    for label in ['repeat', 'one_patch', 'one_window', 'scattered', 'most', 'all']:
        runtime.reset(); runtime.forward(images, execution_context=context)
        current = images.clone()
        if label == 'one_patch': current[0, :, 0:16, 0:16] += .03
        if label == 'one_window': current[0, :, :256, :256] += .03
        if label == 'scattered': current[:, :, :16, :16] += .03
        if label == 'most': current[:, :, :512, :] += .03
        if label == 'all': current += .03
        cases[label]=current
        if label == 'one_patch':
            from worldstream.m2.dependency_planner import exact_pixel_changes,patch_updates
            token_updates=patch_updates(exact_pixel_changes(current,images), model.patch_embed.proj)
            plans=runtime.planner.plan(token_updates)
            dependency_trace=dict(input_patch_mask=token_updates[0].cpu().int().tolist(),
                stages=[dict(block=s.block, reason=s.reason, window_size=s.window_size,
                    camera0_tokens=s.tokens[0].cpu().int().tolist(),
                    updated_units=int(s.updated.sum())) for s in plans])
        c = dict(context, frame_id='1')
        oracle = model(current)[0]
        layer_oracles={}; handles=[]
        for i, b in enumerate(model.blocks):
            handles.append(b.register_forward_hook(lambda m, inp, out, i=i: layer_oracles.update({i: out.detach().clone()})))
        model(current)
        for h in handles: h.remove()
        stage_errors={}
        # Sparse block bypasses Block.forward; validate prefix separately below through executor.
        runtime.stage_observer = lambda i, out: stage_errors.update({i: errors(out, layer_oracles[i])})
        actual, diagnostics = runtime.forward(current, execution_context=c)
        runtime.stage_observer = None
        error = errors(actual[0], oracle)
        # Restore the SAME historical state for each timing sample, outside the timed interval.
        runtime.reset(); runtime.forward(images, execution_context=context)
        saved = (runtime.cache.images, runtime.cache.outputs, runtime.cache.context,
                 runtime.cache.signature, runtime.cache.version)
        def execute():
            runtime.cache.images, runtime.cache.outputs, runtime.cache.context, runtime.cache.signature, runtime.cache.version = saved
            return runtime.forward(current, execution_context=c)
        timing = measure(execute, warmup=3, repeats=max(5,args.repeats//2))
        backbone_rows.append(dict(label=label, full=full, incremental=timing,
            speedup=full['gpu_ms']/timing['gpu_ms'], error=error, diagnostics=diagnostics))
        layer_pass={i:stage_errors[i]['max_abs']<=max(tolerances[i]['max_abs'],1e-6)
                    and stage_errors[i]['relative_l2']<=max(tolerances[i]['relative_l2'],1e-7)
                    for i in stage_errors}
        correctness.append(dict(label=label, final=error, downstream_blocks=stage_errors,
            per_layer_pass=layer_pass, passed=all(layer_pass.values())))
        print('backbone',label,backbone_rows[-1]['speedup'],error,flush=True)
    write(root/'backbone_benchmark.json',dict(cold=cold, rows=backbone_rows))
    write(root/'dependency_trace.json',dependency_trace)
    write(root/'correctness_report.json',dict(block_all_passed=all(r['correctness'] for r in rows),
        backbone_all_passed=all(r['passed'] for r in correctness), rows=correctness))
    certified_counts=set()
    for row, check in zip(backbone_rows,correctness):
        if check['passed']:
            certified_counts.add(row['diagnostics']['stages'][0]['updated_windows'])
    for record in cost_records:
        record['quality_certified']=record['updated'] in certified_counts
    write(root/'break_even_results.json',dict(records=cost_records,
        beneficial_counts=[r['updated'] for r in rows if r['packed']['gpu_ms']<dense_time['gpu_ms']
                           and r['packed']['wall_ms']<dense_time['wall_ms']],
        calibrated_quality_counts=sorted(certified_counts),
        scope='block costs; automatic sparse requires separately verified whole-backbone numerical envelope'))
    automatic=IncrementalVisionRuntime(model,dict(mode='auto',cost_table=str(root/'break_even_results.json')))
    auto_rows=[]
    for label,current in cases.items():
        automatic.reset(); automatic.forward(images,execution_context=context)
        saved=(automatic.cache.images,automatic.cache.outputs,automatic.cache.context,
               automatic.cache.signature,automatic.cache.version)
        def auto_execute():
            automatic.cache.images,automatic.cache.outputs,automatic.cache.context,automatic.cache.signature,automatic.cache.version=saved
            return automatic.forward(current,execution_context=dict(context,frame_id='1'))
        y,d=auto_execute()
        auto_rows.append(dict(label=label,timing=measure(auto_execute,warmup=3,repeats=5),
            error=errors(y[0],model(current)[0]),diagnostics=d))
    write(root/'auto_policy_benchmark.json',auto_rows)
    write(root/'memory_profile.json',dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), live_cache_bytes=runtime.cache.bytes,
        budget_bytes=runtime.cache.budget_bytes, scope='controlled benchmark including oracle/graphs'))


if __name__ == '__main__': main()
