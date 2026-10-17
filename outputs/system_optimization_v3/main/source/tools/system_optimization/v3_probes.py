"""Actual GPU diagnostic controls kept outside ordinary E2E statistics."""
import inspect,json
from pathlib import Path
from types import SimpleNamespace
import torch
from .audit import write
from .v3 import ROOT,arguments
from .v2_probes import operator_scopes,frozen_question
from .experiment import seed


@torch.no_grad()
def run(driver):
    from .v2_probes import run as previous_probes
    from .v2_batch_probe import run as batch_probe
    from mmcv.parallel import collate,scatter
    from tools.worldstream_m3.refinement import freeze_sample_questions
    driver.args=arguments('main','single',120,8,4,[])
    previous_probes(driver,ROOT/'main',ROOT/'probes')
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
    # Repeat bounded old-failure diagnosis in the unchanged precision/LoRA contract.
    batch_probe(driver,ROOT/'batch_probes')
