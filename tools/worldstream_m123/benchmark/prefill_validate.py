"""Real-model equivalence validation; native generation is a reference only.

Controlled suffix changes are labelled and never grant deployment approval.
"""
import argparse
import json
from pathlib import Path
import torch
import copy
import subprocess
import sys
import hashlib
from time import perf_counter
from dataclasses import asdict
from tools.worldstream_m123.benchmark.worker import effective_config,load_model,select_question
from worldstream.m3.visual_state import expand_input,base_model,deployment_key
from worldstream.m3.prefix_kv_cache import PrefixKVCache
from worldstream.m3.prefill_executor import PrefillExecutor
from .common import write,sha256


@torch.no_grad()
def wsdr_probe(model,call,out):
    """Two labelled carrier perturbations, never service quality or production reuse."""
    from worldstream.m3.reuse_planner import ReusePlanner
    from worldstream.m3.world_state import query_world_state
    from worldstream.m3.prefix_matcher import exact_lcp
    runtime=model.m3_runtime;entry=runtime.cache.entry
    if entry is None:
        write(out/'controlled_probes.json',dict(status='NOT_RUN',reason='No retained real prefill cache'))
        return
    layout=call['layout'];base=entry.expanded;qi=len(call['input_ids_list'])-1
    ids=call['input_ids_list'][qi];carriers=call['vision_carriers']
    generation={k:call[k] for k in ('do_sample','temperature','top_p','num_beams','max_new_tokens','use_cache')}
    generation.update(return_dict_in_generate=True,output_scores=True)
    cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state();records=[]
    try:
        for group,index in (('map',layout.object_tokens+layout.ego_tokens),('object',0)):
            altered=carriers.clone();altered[0,index,0]+=1.
            x=expand_input(model.lm_head,ids,altered,scope=base.scope,signature=base.signature,
                source=base.source,carrier_layout=layout,layout_signature=base.layout_signature)
            matched=exact_lcp(base,x)
            history=query_world_state(call['verified_state'],call['questions'][qi],call['context'])
            plan=ReusePlanner().plan(call['questions'][qi],x,layout,base,history,call['context'],call.get('m2_diagnostics'))
            torch.manual_seed(1729);torch.cuda.manual_seed_all(1729)
            torch.cuda.synchronize();tick=perf_counter()
            reference=model.lm_head.generate(inputs=ids[None],images=altered,**generation)
            torch.cuda.synchronize();reference_ms=(perf_counter()-tick)*1000
            expected_rng=torch.cuda.get_rng_state().clone()
            cache=copy.copy(runtime.cache);executor=PrefillExecutor(cache)
            torch.manual_seed(1729);torch.cuda.manual_seed_all(1729)
            torch.cuda.synchronize();tick=perf_counter()
            actual,diag=executor.generate(base_model(model.lm_head),x,reuse_length=matched,capture=False,profile=True,**generation)
            torch.cuda.synchronize();candidate_ms=(perf_counter()-tick)*1000
            tokens_equal=torch.equal(reference.sequences,actual.sequences)
            scores_close=len(reference.scores)==len(actual.scores) and all(torch.allclose(a,b,atol=1e-6,rtol=1e-5) for a,b in zip(reference.scores,actual.scores))
            binding=dict(strategy='wsdr_kv',strategy_version=plan.strategy_version,
                model_version=model.worldstream_audit_identity['checkpoint_sha256'],
                weights_sha256=model.worldstream_audit_identity['checkpoint_sha256'],
                execution_key=deployment_key(model.lm_head),task_type=plan.task_type,task_profile=plan.task_profile,
                layout_signature=plan.layout_signature,input_tokens=x.length,prefix_tokens=matched,
                reused_tokens=diag['reused_tokens'],dtype=str(x.embeddings.dtype),
                generation_key=hashlib.sha256(json.dumps(dict(defaults=base_model(model.lm_head).generation_config.to_dict(),
                    overrides=generation),sort_keys=True,default=repr).encode()).hexdigest())
            records.append(dict(case='controlled_'+group+'_carrier_delta',natural_frame=False,
                binding=binding,generation=generation,
                production_admitted=False,approved_prefix_tokens=0,matched_prefix_tokens=matched,
                offline_actual_reused_tokens=diag['reused_tokens'],effective_prefill_tokens=diag['effective_prefill_tokens'],
                input_tokens=x.length,question=call['questions'][qi],plan=plan.diagnostics(),
                tokens_equal=tokens_equal,rng_equal=torch.equal(expected_rng,torch.cuda.get_rng_state()),scores_close=scores_close,
                logits_max_abs=max((float((a-b).abs().nan_to_num().max()) for a,b in zip(reference.scores,actual.scores)),default=0.),
                native_wall_ms=reference_ms,offline_candidate_wall_ms=candidate_ms,
                diagnostics=diag,reference_tokens=reference.sequences.cpu().tolist(),actual_tokens=actual.sequences.cpu().tolist(),
                task_quality='NOT_ASSESSED: controlled carrier changes have no natural-frame GT'))
    finally:
        torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng)
    write(out/'controlled_probes.json',dict(status='MEASURED',records=records,
        production_approvals_added=0,scope='Direct existing PrefillExecutor numeric probes only, bypassing scheduler explicitly outside measured service. No candidate is production-admitted.'))
    write(out/'wsdr_candidates.json',dict(strategy='wsdr_kv',strategy_version=plan.strategy_version,
        production_ready=False,approved=[],records=[dict(binding=r['binding'],
            numerical=dict(tokens_equal=r['tokens_equal'],rng_equal=r['rng_equal'],scores_close=r['scores_close'],
                logits_max_abs=r['logits_max_abs']),
            measurements=[dict(sample_id=r['case'],baseline_ms=r['native_wall_ms'],optimized_ms=r['offline_candidate_wall_ms'])],
            independent_task_quality=False,quality_status='NOT_ASSESSED',
            evidence_path='controlled_probes.json',evidence_sha256=sha256(out/'controlled_probes.json')) for r in records]))


def compare_wsdr(args):
    """Three short jobs in the existing worker, with the same real frames/questions."""
    from .run import wait_gpu
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'experiments').mkdir();(out/'raw_logs').mkdir()
    wait_gpu(out)
    checkpoint_hash=sha256(args.checkpoint)
    write(out/'preflight.json',dict(checkpoint=dict(sha256=checkpoint_hash),
        environment=dict(python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),affinity=sorted(__import__('os').sched_getaffinity(0))),
        scope='Short M3 comparison; each worker validates checkpoint strictly and actual sequence/resources'))
    commands=[]
    for strategy in ('native_full','exact_prefix','wsdr_kv'):
        wait_gpu(out)
        command=['taskset','-c','0-3,6-31',sys.executable,'-u','-m','tools.worldstream_m123.benchmark.worker',
            '--output',str(out/'experiments'/strategy),'--run-root',str(out),'--condition','B3',
            '--workload','language','--limit',str(args.frames),'--warmup',str(args.warmup),
            '--questions-per-frame',str(args.questions_per_frame),'--checkpoint',args.checkpoint,
            '--m3-strategy',strategy]
        if strategy=='wsdr_kv':command.append('--m3-controlled-probe')
        commands.append(command);write(out/'invocations.json',commands)
        print('START',strategy,flush=True)
        with (out/'raw_logs'/(strategy+'.log')).open('w') as log:
            code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        if code:
            write(out/'status.json',dict(status='FAILED',strategy=strategy,returncode=code))
            raise RuntimeError('M3 comparison worker failed: '+strategy)
    write(out/'status.json',dict(status='COMPLETED',strategies=3,production_approvals_added=0))

@torch.no_grad()
def run(args):
    from mmdet3d.datasets import build_dataset
    from mmcv.parallel import collate,scatter
    import projects.mmdet3d_plugin
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    cfg=effective_config('B0','language',out)
    model=load_model(cfg,args.checkpoint,out)
    original=model.lm_head.generate;results=[]
    def verify(inputs=None,images=None,**generation):
        x=expand_input(model.lm_head,inputs,images,scope=('scene','frame'),signature=('same_model',),source=('scene','frame'))
        for sample in (False,True):
            kw=dict(generation,do_sample=sample,return_dict_in_generate=True,output_scores=True)
            if not sample:
                kw.pop('temperature',None);kw.pop('top_p',None)
            cache=PrefixKVCache(1024**3);executor=PrefillExecutor(cache)
            def check(label,exp,reuse,reference_inputs):
                torch.manual_seed(1729);torch.cuda.manual_seed_all(1729)
                reference=original(inputs=reference_inputs,images=images,**kw)
                rng=torch.cuda.get_rng_state().clone()
                torch.manual_seed(1729);torch.cuda.manual_seed_all(1729)
                actual,diag=executor.generate(base_model(model.lm_head),exp,reuse,**kw)
                row=dict(kind=label,controlled_suffix_change=label=='partial_prefix',sampling=sample,input_tokens=exp.length,reuse_length=diag['reused_tokens'],
                    tokens_equal=torch.equal(reference.sequences,actual.sequences),
                    rng_equal=torch.equal(rng,torch.cuda.get_rng_state()),
                    reference_tokens=reference.sequences.tolist(),actual_tokens=actual.sequences.tolist(),
                    logits_max_abs=max((float((a-b).abs().nan_to_num().max()) for a,b in zip(reference.scores,actual.scores)),default=0.),
                    scores_close=len(reference.scores)==len(actual.scores) and all(torch.allclose(a,b,atol=1e-6,rtol=1e-5) for a,b in zip(reference.scores,actual.scores)),
                    cache_bytes=cache.nbytes)
                results.append(row);print('CHECK',json.dumps(row),flush=True)
                (out/'results.json').write_text(json.dumps(results,indent=2))
                return actual
            answer=check('full_prefill',x,0,inputs)
            check('identical_prefix',x,cache.match(x),inputs)
            changed=inputs.clone();changed.reshape(-1)[-1]=int((changed.reshape(-1)[-1]+1)%100)
            y=expand_input(model.lm_head,changed,images,scope=x.scope,signature=x.signature,source=x.source)
            check('partial_prefix',y,cache.match(y),changed)
        return answer.sequences
    model.lm_head.generate=verify
    ds=build_dataset(cfg.data.test);ds.data_infos.sort(key=lambda i:(i['scene_token'],i['timestamp']))
    data=scatter(collate([ds[0]],samples_per_gpu=1),[0])[0];select_question(data,0,1)
    model(return_loss=False,rescale=True,**data)
    print('COMPLETE; no production approvals written',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--checkpoint',default='ckpts/iter_10548.pth')
    parser.add_argument('--wsdr-comparison',action='store_true')
    parser.add_argument('--frames',type=int,default=4)
    parser.add_argument('--questions-per-frame',type=int,default=3)
    parser.add_argument('--warmup',type=int,default=1)
    args=parser.parse_args()
    if args.frames<2 or args.questions_per_frame<2 or args.warmup<1:parser.error('Need frames>=2, questions>=2, warmup>=1')
    (compare_wsdr if args.wsdr_comparison else run)(args)
