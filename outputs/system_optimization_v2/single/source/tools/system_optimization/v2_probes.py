"""Separate real-weight profiler/logit controls; never ordinary E2E samples."""
from contextlib import ExitStack,nullcontext,contextmanager
from functools import wraps
import copy,hashlib,inspect,json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import torch
from .audit import write,stats
from .experiment import seed
from .profile_analysis import analyze
from worldstream.m3.native_decode import zero_mask_decode
from worldstream.m3.visual_state import base_model


@contextmanager
def operator_scopes(model):
    with ExitStack() as stack:
        for name,module in base_model(model.lm_head).named_modules():
            tail=name.split('.')[-1]
            wanted=name in ('lm_head','model.norm') or (name.startswith('model.layers.') and
                (tail.isdigit() or tail in ('self_attn','mlp','q_proj','k_proj','v_proj','o_proj',
                    'gate_proj','up_proj','down_proj','input_layernorm','post_attention_layernorm') or '.lora_' in name))
            if not wanted:continue
            owned='forward' in vars(module);original=module.forward
            @wraps(original)
            def forward(*a,_original=original,_name=name,**kw):
                with torch.autograd.profiler.record_function('decodeop/'+_name):return _original(*a,**kw)
            module.forward=forward
            if owned:stack.callback(setattr,module,'forward',original)
            else:stack.callback(delattr,module,'forward')
        yield


def frozen_question(spec):
    return dict(questions=spec['questions'],ids=[torch.tensor(x,dtype=torch.long) for x in spec['ids']])


@torch.no_grad()
def run(driver,source,dest):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    write(dest/'executed_source.json',dict(path=__file__,sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    source=Path(source);protocol=json.loads((source/'protocol.json').read_text());questions=json.loads((source/'questions.json').read_text())
    from mmcv.parallel import collate,scatter
    from tools.worldstream_m3.refinement import freeze_sample_questions
    model,dataset=driver.model,driver.dataset
    versions=[(n,p._version,p.dtype) for n,p in model.named_parameters()]
    native=inspect.getclosurevars(driver.native_generate).nonlocals['generate']
    snapshots=[];profiles=[]
    # Pair two actual consecutive frames, so the profiled precheck has history.
    for mode in ('m1','overlap','native_decode'):
        sink=SimpleNamespace(ids=[])
        driver.setup(model,mode,sink,dest,512,10,True)
        def generate(*a,**kw):
            sink.prompt=kw['inputs'].detach();sink.carriers=kw['images'].detach()
            out=native(*a,**kw);sink.ids.append(out.detach().cpu().tolist());return out
        model.lm_head.generate=generate
        indices=protocol['scenes'][0]['indices'][:2]
        for ordinal,index in enumerate(indices):
            seed(index)
            def prepare():return scatter(collate([freeze_sample_questions(dataset[index],frozen_question(questions[str(index)]))],samples_per_gpu=1),[0])[0]
            model.save_path=str(dest/f'{mode}_{ordinal}'/'language')+'/'
            if ordinal==0:
                batch=prepare();seed(index+10000)
                model(return_loss=False,rescale=True,**batch);torch.cuda.synchronize();continue
            with operator_scopes(model),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
                with torch.autograd.profiler.record_function('stream/input_preparation'):batch=prepare()
                seed(index+10000);model(return_loss=False,rescale=True,**batch);torch.cuda.synchronize()
            path=dest/f'{mode}_pipeline_trace.json';prof.export_chrome_trace(str(path))
            worker=model.stream_runtime.last_record.m1.get('worker')
            # Parse traces after releasing the model in the final reporting phase.
            profiles.append(dict(path=str(path),mode=mode,worker=worker))
        if driver.context is not None:driver.context.__exit__(None,None,None);driver.context=None
        model.stream_runtime.close()
    # Independent actual single-question inputs: one source frame per scene.
    for scene in protocol['scenes']:
        index=scene['indices'][0];sink=SimpleNamespace(ids=[])
        driver.setup(model,'original',sink,dest,512,10,False)
        def generate(*a,**kw):
            sink.prompt=kw['inputs'].detach();sink.carriers=kw['images'].detach()
            return native(*a,**kw)
        model.lm_head.generate=generate
        seed(index);batch=scatter(collate([freeze_sample_questions(dataset[index],frozen_question(questions[str(index)]))],samples_per_gpu=1),[0])[0]
        seed(index+10000);model.save_path=str(dest/f'snapshot_{index}'/'language')+'/'
        model(return_loss=False,rescale=True,**batch);torch.cuda.synchronize()
        snapshots.append(dict(index=index,scene=scene['scene'],split=scene['split'],prompt=sink.prompt,carriers=sink.carriers))
    write(dest/'profile_manifest.json',profiles)
    base=base_model(model.lm_head);rows=[];quality=[]
    contract=dict(do_sample=True,temperature=.1,top_p=.75,num_beams=1,max_new_tokens=320,use_cache=True)
    from tools.worldstream_m3.qa_analysis import planning_quality
    for snapshot in snapshots:
        references=None;reference_rng=None
        index=snapshot['index'];inputs=dict(inputs=snapshot['prompt'],images=snapshot['carriers'],**contract)
        def generate_with_logits():
            captured=[]
            handle=base.register_forward_hook(lambda module,args,out:captured.append(out.logits[:,-1].detach().cpu()))
            try:
                seed(index+10000);answer=native(**inputs);torch.cuda.synchronize()
                return answer.detach().cpu(),captured,torch.get_rng_state(),torch.cuda.get_rng_state()
            finally:handle.remove()
        expected,reference_logits,cpu_rng,gpu_rng=generate_with_logits()
        with zero_mask_decode(model.lm_head) as probe:
            actual,actual_logits,actual_cpu,actual_gpu=generate_with_logits()
        first=next((i for i,(a,b) in enumerate(zip(expected[0],actual[0])) if a!=b),None)
        horizon=first if first is not None else min(len(reference_logits),len(actual_logits))
        diffs=[float((a-b).abs().max()) for a,b in zip(reference_logits[:horizon],actual_logits[:horizon])]
        text=model.tokenizer.batch_decode(actual,skip_special_tokens=True)[0]
        quality.append(dict(index=index,scene=snapshot['scene'],split=snapshot['split'],
            complete_token_equality=torch.equal(expected,actual),first_different_token=first,
            native_ids=expected.tolist(),candidate_ids=actual.tolist(),same_history_raw_logit_max_abs=max(diffs,default=0.),
            cpu_rng_equal=torch.equal(cpu_rng,actual_cpu),cuda_rng_equal=torch.equal(gpu_rng,actual_gpu),
            planning=planning_quality(text,dataset.data_infos[index]),candidate_stats=dict(probe)))
        quality[-1]['native_planning']=planning_quality(model.tokenizer.batch_decode(expected,skip_special_tokens=True)[0],dataset.data_infos[index])
        for repeat in range(5):
            order=('native','zero_mask') if repeat%2 else ('zero_mask','native')
            for mode in order:
                context=zero_mask_decode(model.lm_head) if mode=='zero_mask' else nullcontext()
                seed(index+10000)
                with context as probe:
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=perf_counter()
                    answer=native(**inputs);torch.cuda.synchronize();elapsed=(perf_counter()-start)*1000
                    peak=torch.cuda.max_memory_allocated()
                rows.append(dict(index=index,scene=snapshot['scene'],split=snapshot['split'],repeat=repeat,warmup=repeat<2,
                    mode=mode,wall_ms=elapsed,gpu_peak_allocated=peak,tokens=answer.numel()-1,
                    complete_token_equality=torch.equal(answer.cpu(),expected)))
    write(dest/'decode_backend_results.json',dict(scope='Standalone native Sampling LM, separate from complete frame E2E',
        contract=contract,quality=quality,performance=rows,
        production_auto_approved=False,caveat='Mask checks and host synchronization are included; logits copying is outside timed runs'))
    assert versions==[(n,p._version,p.dtype) for n,p in model.named_parameters()],'Probe changed model weights'
    from .v2_lifecycle import run as lifecycle
    lifecycle(driver,source,dest/'overlap_lifecycle')
    print('PROBES COMPLETE',len(quality),'real prompts,',len(rows),'LM timings',flush=True)
