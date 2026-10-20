"""One bounded diagnostic of actual Batch=1 dense GEMM and commit critical path."""
from collections import Counter
import torch
from worldstream.m3.visual_state import base_model,expand_input
from .common import OLD,read,write_json,measure,stats


@torch.no_grad()
def run(model,root):
    import hashlib
    from pathlib import Path
    sources=[Path(__file__),Path(__file__).with_name('full_parity.py'),Path(__file__).with_name('greedy_engineering.py'),
        Path(__file__).with_name('final_source.py')]
    manifest={str(p.relative_to(Path.cwd())) if p.is_absolute() else str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    write_json(root/'fallback_execution_sources.json',manifest)
    for p in sources:
        out=root/'fallback_execution_source'/p.name;out.parent.mkdir(exist_ok=True);out.write_bytes(p.read_bytes())
    from .full_parity import run as check_prefixes
    check_prefixes(model,root)
    from .greedy_engineering import run as greedy_control
    greedy_control(model,root)
    from .final_source import run as confirm_final_source
    confirm_final_source(model,root)
    # Private RNG alone cannot certify native FP16 target probabilities.
    parity=read(root/'microbatch_numerical_parity.json')
    if parity['all_targets_equal']:
        quality=read(root/'service/status.json')
        if quality['microbatch_quality_pass']:return
    dest=root/'fallback';dest.mkdir(exist_ok=False)
    snapshots=torch.load(OLD/'phase1/carrier_snapshots.pt',map_location='cpu');s=snapshots[0]
    model.m3_runtime=None;model.m2_runtime=None;model.m1_manager=None
    base=base_model(model.lm_head)
    x=expand_input(model.lm_head,torch.tensor(s['prompt_ids'],device='cuda'),s['carriers'].cuda())
    out=base(inputs_embeds=x.embeddings,attention_mask=x.attention_mask,position_ids=x.position_ids,use_cache=True,return_dict=True)
    kv=out.past_key_values;token=torch.tensor([[s['native_tokens'][0]]],device='cuda');del out
    def step():
        return base(input_ids=token,past_key_values=kv,attention_mask=x.attention_mask.new_ones((1,x.length+1)),
            position_ids=x.position_ids.new_tensor([[x.length]]),use_cache=True,return_dict=True)
    captured={};hooks=[];modules={}
    # Actual first-layer dense/LoRA GEMM shapes and LM head, no random surrogate.
    for name,module in base.named_modules():
        if isinstance(module,torch.nn.Linear) and (name.startswith('model.layers.0.') or name=='lm_head'):
            def hook(m,args,name=name):captured[name]=args[0].detach().clone()
            hooks.append(module.register_forward_pre_hook(hook));modules[name]=module
    out=step();del out
    for h in hooks:h.remove()
    individual=[]
    for name,module in modules.items():
        if name not in captured:continue
        values=[]
        for rep in range(12):
            out,timing=measure(lambda:module(captured[name]));del out
            values.append(dict(repeat=rep,warmup=rep<3,**timing))
        p50=stats([r['gpu_ms'] for r in values if not r['warmup']])['p50']
        weight_bytes=module.weight.numel()*module.weight.element_size()
        individual.append(dict(module=name,input_shape=list(captured[name].shape),weight_shape=list(module.weight.shape),
            weight_dtype=str(module.weight.dtype),weight_bytes=weight_bytes,rows=values,
            weight_bytes_per_gpu_interval_gb_s=weight_bytes/(p50*1e6),
            flop_per_weight_byte=2*captured[name].numel()/captured[name].shape[-1]/module.weight.element_size()))
    full=[]
    for rep in range(16):
        out,timing=measure(step);del out;full.append(dict(repeat=rep,warmup=rep<3,**timing))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
        for _ in range(4):out=step();del out
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(dest/'single_request_decode_trace.json'))
    operators=[dict(name=e.key,calls=e.count,self_cpu_ms=e.self_cpu_time_total/1000,self_cuda_ms=e.self_cuda_time_total/1000,
        input_shapes=e.input_shapes) for e in prof.key_averages(group_by_input_shape=True)]
    kernels=[dict(name=e.name,gpu_us=e.time_range.elapsed_us()) for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
    write_json(dest/'operators.json',operators);write_json(dest/'kernels.json',kernels)
    categorized=Counter()
    for k in kernels:
        label='gemm_gemv' if any(s in k['name'].lower() for s in ('gemm','gemv')) else 'other'
        categorized[label]+=k['gpu_us']/1000
    result=dict(production_optimization_implemented=False,full_token=full,individual_real_module_gemm=individual,
        leaf_kernel_totals_ms=dict(categorized),decode_trace_calls=4,
        inference='Batch=1 projections have approximately one/two FLOPs per weight byte per output row, '
            'and measured batch throughput improves sharply as shared weight reads serve more rows. '
            'This is consistent with weight-traffic/shape efficiency limits; no unmeasured DRAM counter or compute utilization is claimed.',
        chosen_finite_direction='Batch=1 GEMM diagnosis, with commit dependency census only',
        decision='No certified minimal replacement identified; original dense FP16/LoRA math retained. '
            'No CUDA Graph recapture, precision change, quantization or network rewrite was attempted.',
        timings='GPU event intervals include host gaps. Operator CUDA attribution is not added to leaf kernels.')
    write_json(root/'single_request_gemm_analysis.json',result)
    import json
    runs=[json.loads(line) for line in (root/'service/runs.jsonl').read_text().splitlines()]
    native=[r for r in runs if r['mode']=='m3_native']
    commit=[next(t['wall_ms'] for t in r['timeline'] if t['stage']=='commit_end') for r in native]
    overlap=dict(implemented=False,opportunity='bounded_commit_only',commit_wall_ms=stats(commit),
        per_frame=[dict(scene=r['scene'],frame=r['frame'],population=r['population'],timeline=r['timeline'],m1=r['m1'],
            optimistic_overlap_ms=next(t['wall_ms'] for t in r['timeline'] if t['stage']=='commit_end'),
            optimistic_service_speedup=r['wall_ms']/(r['wall_ms']-next(t['wall_ms'] for t in r['timeline'] if t['stage']=='commit_end')))
            for r in native],
        dependencies=[['vision','detection/map temporal update'],['detection/map','CPU output adapter snapshot'],
            ['CPU adapter','M1 atomic commit'],['detection/map carriers','current native LLM'],
            ['M1 atomic commit','next frame precheck/M2 admission']],
        current_native_llm_reads_commit_result=False,
        caveat='M3 context currently reads M1 metadata even in native full mode; a future overlap path must bind immutable metadata '
            'and exclusively lease the manager, join before the next precheck/reset, and drain on failure. '
            'Existing M1Worker supports precheck only and cannot safely be repurposed without lifecycle changes.',
        decision='Diagnostic only: do not introduce a second async state system for a small commit-only upper bound. '
            'Adapter/precheck/maintenance are not silently removed and cannot all overlap the current decode.',
        actual_overlap_ms=0,measured_overlap_speedup=None)
    write_json(root/'m1_decode_overlap_analysis.json',overlap)
