"""Actual GPU diagnostic controls kept outside ordinary E2E statistics."""
import inspect,json,cProfile,pstats
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter,process_time
import torch
from .audit import write
from .v3 import ROOT,arguments
from .v2_probes import operator_scopes,frozen_question
from .experiment import seed


@torch.no_grad()
def run(driver):
    import hashlib
    source_bytes=Path(__file__).read_bytes()
    (ROOT/'executed_v3_probes.py').write_bytes(source_bytes)
    write(ROOT/'v3_probes_source.json',dict(path=__file__,sha256=hashlib.sha256(source_bytes).hexdigest(),late_imported_after_ordinary_runs=True))
    from .v3_cost_support import run as cost_support
    cost_support(driver)
    from .v2_probes import run as previous_probes
    from .v2_batch_probe import run as batch_probe
    from mmcv.parallel import collate,scatter
    from tools.worldstream_m3.refinement import freeze_sample_questions
    from .v2_state_supply import run as long_state_supply
    driver.args=arguments('main','single',120,8,4,[])
    previous_probes(driver,ROOT/'main',ROOT/'probes')
    thread_controls(driver)
    long_state_supply(driver,ROOT/'long',ROOT/'long_task_supply',checkpoints=(7,15),base_mode='certified_serial',post_perception=True)
    # One additional actual refined timeline; pair history then trace frame 2.
    specs=json.loads((ROOT/'main/questions.json').read_text());protocol=json.loads((ROOT/'main/protocol.json').read_text())
    source=ROOT/'probes';sink=SimpleNamespace(ids=[]);model,dataset=driver.model,driver.dataset
    native=inspect.getclosurevars(driver.native_generate).nonlocals['generate']
    model.lm_head.generate=native;driver.setup(model,'refined_overlap',sink,source,512,10,True)
    for ordinal,index in enumerate(protocol['scenes'][0]['indices'][:2]):
        seed(index);batch=scatter(collate([freeze_sample_questions(dataset[index],frozen_question(specs[str(index)]))],samples_per_gpu=1),[0])[0]
        model.save_path=str(source/f'refined_{ordinal}')+'/'
        seed(index+10000)
        if ordinal==0:model(return_loss=False,rescale=True,**batch);torch.cuda.synchronize();continue
        with operator_scopes(model),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
            model(return_loss=False,rescale=True,**batch);torch.cuda.synchronize()
        path=source/'refined_overlap_pipeline_trace.json';prof.export_chrome_trace(str(path));del prof
        manifests=json.loads((source/'profile_manifest.json').read_text());manifests.append(dict(path=str(path),mode='refined_overlap',worker=model.stream_runtime.last_record.m1.get('worker')))
        write(source/'profile_manifest.json',manifests)
    driver.model.stream_runtime.close()
    backbone=model.img_backbone
    write(ROOT/'eva_structure.json',dict(type=type(backbone).__name__,
        blocks=[dict(index=i,window_size=b.window_size,residual=b.use_residual_block) for i,b in enumerate(backbone.blocks)],
        patch=dict(kernel_size=list(backbone.patch_embed.proj.kernel_size),stride=list(backbone.patch_embed.proj.stride),padding=list(backbone.patch_embed.proj.padding))))
    # Independent actual-entry input capture for bounded admission profiling.
    from functools import wraps
    sink=SimpleNamespace(ids=[]);model.lm_head.generate=native
    driver.setup(model,'overlap',sink,source,512,10,False)
    original=model.simple_test;captured={}
    @wraps(original)
    def capture(*a,**kw):
        if not model.stream_runtime.active:
            captured.update(kw);captured['img']=kw['img'].view(*kw['img'].shape)
        return original(*a,**kw)
    model.simple_test=capture
    index=protocol['scenes'][0]['indices'][0];seed(index)
    batch=scatter(collate([freeze_sample_questions(dataset[index],frozen_question(specs[str(index)]))],samples_per_gpu=1),[0])[0]
    model.save_path=str(source/'admission_probe')+'/'
    seed(index+10000)
    try:model(return_loss=False,rescale=True,**batch);torch.cuda.synchronize()
    finally:model.simple_test=original
    runtime=model.stream_runtime;runtime.config=replace(runtime.config,cost_aware=True);cost_rows=[]
    prof=cProfile.Profile();prof.enable()
    for i in range(32):
        start=perf_counter();cpu=process_time();key=runtime.workload_key(model,captured)
        cost_rows.append(dict(iteration=i,wall_ms=(perf_counter()-start)*1000,cpu_ms=(process_time()-cpu)*1000))
    prof.disable();profile_stats=pstats.Stats(prof)
    functions=[dict(file=k[0],line=k[1],function=k[2],primitive_calls=v[0],calls=v[1],self_cpu_s=v[2],cumulative_cpu_s=v[3]) for k,v in profile_stats.stats.items()]
    matches=[];runtime._request_key=key
    for costs,quality in ((dict(records=[]),dict(records=[])),
            (json.loads((ROOT/'cost_certificate.json').read_text()),json.loads((ROOT/'quality_certificate.json').read_text()))):
        runtime.costs=costs;runtime.quality=quality
        for i in range(32):
            start=perf_counter();allowed,reason=runtime.optimization_admission(model,captured)
            matches.append(dict(iteration=i,certificate_rows=len(costs['records']),wall_ms=(perf_counter()-start)*1000,allowed=allowed,reason=reason))
    write(ROOT/'admission_overhead_profile.json',dict(scope='Separate cProfile and actual-entry input control; not ordinary E2E timings',key_construction=cost_rows,
        implementation_functions=sorted(functions,key=lambda r:r['cumulative_cpu_s'],reverse=True),lookup=matches,
        cache_bound_entries=64,no_certificate_fast_reject=True))
    runtime.close()
    # Repeat bounded old-failure diagnosis in the unchanged precision/LoRA contract.
    batch_probe(driver,ROOT/'batch_probes')


@torch.no_grad()
def thread_controls(driver):
    """Actual resource variations, without promoting these controls to scene CIs."""
    import cv2
    from mmcv.parallel import collate,scatter
    from tools.worldstream_m3.refinement import freeze_sample_questions
    from .v2_resources import ResidentSampler
    from .v2 import state_digest
    dest=ROOT/'thread_controls';dest.mkdir(exist_ok=False)
    model,dataset=driver.model,driver.dataset;native=inspect.getclosurevars(driver.native_generate).nonlocals['generate']
    sink=SimpleNamespace(ids=[])
    def generate(*a,**kw):
        out=native(*a,**kw);sink.ids.append(out.detach().cpu().tolist());return out
    model.lm_head.generate=generate
    protocol=json.loads((ROOT/'main/protocol.json').read_text());questions=json.loads((ROOT/'main/questions.json').read_text())
    indices=protocol['scenes'][0]['indices'][:2];references={};records=[]
    old_torch,old_cv=torch.get_num_threads(),cv2.getNumThreads()
    try:
        for cpu,opencv,workers in ((1,1,6),(1,1,1),(2,2,6)):
            torch.set_num_threads(cpu);cv2.setNumThreads(opencv)
            for mode in ('m1','overlap','refined_overlap'):
                driver.setup(model,mode,sink,dest,512,10,False)
                model.m1_manager.config=replace(model.m1_manager.config,temporal_workers=workers)
                for ordinal,index in enumerate(indices):
                    seed(index);sink.ids=[];model.save_path=str(dest/f'{cpu}_{opencv}_{workers}_{mode}_{ordinal}')+'/'
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=perf_counter()
                    with ResidentSampler() as sampler:
                        batch=scatter(collate([freeze_sample_questions(dataset[index],frozen_question(questions[str(index)]))],samples_per_gpu=1),[0])[0]
                        seed(index+10000);model(return_loss=False,rescale=True,**batch);torch.cuda.synchronize()
                    digest=state_digest(model.m1_manager)
                    if (cpu,opencv,workers,mode)==(1,1,6,'m1'):references[index]=(list(sink.ids),digest)
                    expected,state=references[index]
                    records.append(dict(torch_threads=cpu,opencv_threads=opencv,temporal_workers=workers,mode=mode,index=index,
                        wall_ms=(perf_counter()-start)*1000,complete_token_equality=sink.ids==expected,state_equal=digest==state,
                        gpu_peak_allocated=torch.cuda.max_memory_allocated(),gpu_peak_reserved=torch.cuda.max_memory_reserved(),**sampler.result()))
                model.stream_runtime.close()
    finally:torch.set_num_threads(old_torch);cv2.setNumThreads(old_cv)
    write(ROOT/'thread_resource_controls.json',dict(rows=records,scope='Actual two-frame resource controls, 20ms RSS sampling; not independent performance scenes',
        restored=dict(torch_threads=torch.get_num_threads(),opencv_threads=cv2.getNumThreads())))
