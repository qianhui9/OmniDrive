"""Bounded diagnosis on an existing failed carrier; exploratory, not holdout."""
import ast,hashlib,inspect,json,subprocess,types
from pathlib import Path
from time import perf_counter
import torch
from .audit import write
from .experiment import seed
from worldstream.m3 import expand_input
from worldstream.m3.visual_state import base_model
from worldstream.m3.prefix_kv_cache import PrefixKVCache
from worldstream.m3.batched_executor import BatchedExecutor,AdaptiveBatchExecutor


def old_executor():
    import worldstream.m3.batched_executor as module
    source=subprocess.check_output(['git','show','b4782abfa9d8f1e5ff5980a1ce484777ea55ce3f:worldstream/m3/batched_executor.py'],text=True)
    cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='BatchedExecutor')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='try_generate')
    scope=dict(vars(module));exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'frozen_original_batched_executor','exec'),scope)
    return scope['try_generate']


@torch.no_grad()
def run(driver,dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    path=Path('outputs/module3/clean_multi_20261016_220113/carriers/e12ae352f31d4268a7f71c4d27aa0886.pt')
    snap=torch.load(path,map_location='cpu');lm=driver.model.lm_head
    lm.generate=inspect.getclosurevars(driver.native_generate).nonlocals['generate'];base=base_model(lm)
    versions=[(n,p._version,p.dtype) for n,p in base.named_parameters()]
    from tools.worldstream_m3.numerical_refinement import diagnose
    diagnose(base,snap,'native',dest)
    ids=[snap['ids'][i].cuda() for i in snap['selected_indices']];vision=snap['carriers'].cuda()
    xs=[expand_input(lm,x,vision) for x in ids]
    # Native raw logits captured outside performance runs. No patched attention.
    references=[];reference_logits=[]
    for i,x in enumerate(ids):
        captured=[];handle=base.register_forward_hook(lambda m,a,o:captured.append(o.logits[:,-1].detach().cpu()))
        try:
            seed(16000+i);out=lm.generate(inputs=x[None],images=vision,do_sample=False,max_new_tokens=320,use_cache=True)
            references.append(out.cpu());reference_logits.append(captured)
        finally:handle.remove()
    cases=[]
    for count,maximum,adaptive in ((2,2,False),(3,3,False),(5,5,False),(5,3,True)):
        selected=xs[:count];cache=PrefixKVCache(512*1024**2)
        executor=AdaptiveBatchExecutor(cache,6144*1024**2,maximum) if adaptive else BatchedExecutor(cache,6144*1024**2)
        # The direct batch probes supply an unambiguous B-row raw-logit trace.
        captured=[];handle=None
        if not adaptive:
            handle=base.register_forward_hook(lambda m,a,o:captured.append(o.logits[:,-1].detach().cpu()) if o.logits.shape[0]==count else None)
        try:answers,diag=executor.try_generate(base,selected,do_sample=False,max_new_tokens=320,use_cache=True)
        finally:
            if handle:handle.remove()
        quality=[]
        if answers is not None:
            for i,answer in enumerate(answers):
                actual=answer.cpu();ref=references[i]
                first=next((j for j in range(min(actual.shape[1],ref.shape[1])) if actual[0,j]!=ref[0,j]),None)
                # Step zero predicts generated token at position one. Include
                # the first divergent prediction; its input history is equal.
                horizon=first if first is not None else min(len(captured),len(reference_logits[i]))
                errors=[float((a[i:i+1]-b).abs().max()) for a,b in zip(captured[:horizon],reference_logits[i][:horizon])]
                quality.append(dict(question=i,equal=torch.equal(actual,ref),first_divergent_token=first,
                    reference_ids=ref.tolist(),actual_ids=actual.tolist(),
                    same_history_raw_logit_max_abs=max(errors,default=None)))
        cache.reset();timings=[]
        for repeat in range(3):
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=perf_counter()
            answers,timed_diag=executor.try_generate(base,selected,do_sample=False,max_new_tokens=320,use_cache=True)
            torch.cuda.synchronize();timings.append(dict(warmup=repeat==0,wall_ms=(perf_counter()-start)*1000,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),diagnostics=timed_diag))
            cache.reset()
        cases.append(dict(count=count,maximum=maximum,adaptive=adaptive,workspace_budget_mb=6144,input_lengths=[x.length for x in selected],
            quality=quality,diagnostics=diag,timings=timings))
    budgets=[];old=old_executor()
    for budget in (0,128):
        for implementation in ('frozen_original','early_reject'):
            executor=BatchedExecutor(PrefixKVCache(budget*1024**2),6144*1024**2)
            if implementation=='frozen_original':executor.try_generate=types.MethodType(old,executor)
            calls=[];handle=base.register_forward_hook(lambda m,a,o:calls.append(list(o.logits.shape)))
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=perf_counter()
            try:answers,diag=executor.try_generate(base,xs,do_sample=False,max_new_tokens=320,use_cache=True)
            finally:handle.remove()
            torch.cuda.synchronize()
            budgets.append(dict(implementation=implementation,budget_mb=budget,wall_ms=(perf_counter()-start)*1000,
                base_forward_shapes=calls,diagnostics=diag,returned_batch=answers is not None,
                peak_gpu_bytes=torch.cuda.max_memory_allocated()))
            executor.cache.reset()
    # Same native Sampling stream across all five requests: batching must
    # reject before RNG reordering, and full fallback must preserve every ID.
    from worldstream.m3 import M3Runtime
    contract=dict(do_sample=True,temperature=.1,top_p=.75,num_beams=1,max_new_tokens=320,use_cache=True)
    seed(34000)
    expected=[lm.generate(inputs=x[None],images=vision,**contract).cpu() for x in ids]
    expected_rng=(torch.get_rng_state(),torch.cuda.get_rng_state())
    runtime=M3Runtime(dict(mode='exact',offline_experiment=True,batched_decode=True))
    seed(34000);torch.cuda.synchronize();start=perf_counter()
    actual,diag=runtime.generate(lm,ids,vision,context=dict(scene_id=snap['scene'],frame_id=path.stem,camera_ids=['all']),**contract)
    torch.cuda.synchronize();elapsed=(perf_counter()-start)*1000
    sampling=dict(contract=contract,wall_ms=elapsed,diagnostics=diag,
        complete_token_equality=[torch.equal(a.cpu(),b) for a,b in zip(actual,expected)],
        cpu_rng_equal=torch.equal(torch.get_rng_state(),expected_rng[0]),
        cuda_rng_equal=torch.equal(torch.cuda.get_rng_state(),expected_rng[1]))
    runtime.reset()
    assert versions==[(n,p._version,p.dtype) for n,p in base.named_parameters()]
    write(dest/'batch_quality_results.json',dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        independent_test=False,scope='Saved actual previously failed carrier; bounded native operator and batch-shape diagnosis',
        cases=cases,budget_precheck=budgets,sampling_fallback=sampling,production_auto_approved=False,
        note='Raw logits after first divergence excluded; adaptive raw logits unavailable because groups change row order'))
