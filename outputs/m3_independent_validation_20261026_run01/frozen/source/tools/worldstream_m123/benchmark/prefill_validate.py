"""Real-model equivalence validation; native generation is a reference only.

Controlled suffix changes are labelled and never grant deployment approval.
"""
import argparse
import json
from pathlib import Path
import torch
from tools.worldstream_m123.benchmark.worker import effective_config,load_model,select_question
from worldstream.m3.visual_state import expand_input,base_model
from worldstream.m3.prefix_kv_cache import PrefixKVCache
from worldstream.m3.prefill_executor import PrefillExecutor


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
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--calibrate-wsdr',action='store_true')
    mode.add_argument('--qualify-service',action='store_true')
    parser.add_argument('--policy')
    parser.add_argument('--native-reference')
    parser.add_argument('--service-run')
    parser.add_argument('--capture',help='Native worker output with captured real M3 inputs')
    parser.add_argument('--sequence-file',help='Scene-disjoint real sequence manifest')
    parser.add_argument('--quality-budget',type=float,default=.02)
    parser.add_argument('--block-size',type=int,choices=[16,32,64],default=32)
    parser.add_argument('--max-visual-tokens',type=int,default=None,help='One focused shorter-prefix trial; no token deletion')
    args=parser.parse_args()
    if not 0 <= args.quality_budget <= 1:parser.error('Invalid quality budget')
    if args.qualify_service:
        if not all((args.policy,args.native_reference,args.service_run,args.sequence_file)):
            parser.error('Service qualification needs policy, native-reference, service-run and sequence-file')
        from .quality_calibration import qualify_service
        qualify_service(args)
    elif args.calibrate_wsdr:
        if not args.capture or not args.sequence_file:parser.error('Calibration needs capture and sequence-file')
        from .quality_calibration import calibrate
        calibrate(args)
    else:run(args)
