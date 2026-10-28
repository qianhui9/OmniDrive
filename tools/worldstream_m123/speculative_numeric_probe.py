"""Same forced prefix, serial vs block real-model numeric witness; no timing claims."""
import argparse
from pathlib import Path
import json
import torch
from tools.worldstream_m123.benchmark.common import write


@torch.no_grad()
def probe(root,schedule=False):
    import projects.mmdet3d_plugin  # noqa
    from tools.worldstream_m123.benchmark.worker import effective_config,load_model
    from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
    from tools.worldstream_m2.global_residual_benchmark import wait_gpu
    from worldstream.m3.visual_state import expand_input,base_model,CarrierLayout
    failure=json.loads((root/'strict_gate_failure/decode_cuda_benchmark.json').read_text())
    rows=failure['trials'][-3:]
    native=next(r for r in rows if r['mode']=='native')
    task=native
    dest=root/('numeric_probe_schedule' if schedule else 'numeric_probe');dest.mkdir(exist_ok=False)
    with gpu_job_lock():
        wait_gpu()
        cfg=effective_config('B3','language',dest,'native_full')
        model=load_model(cfg,'ckpts/iter_10548.pth',dest).eval().requires_grad_(False)
        base=base_model(model.lm_head)
        cap=torch.load(root/'native_profile'/(task['frame_id']+'.pt'),weights_only=False)
        src=torch.load(root/'native_profile'/(task['source_frame']+'.pt'),weights_only=False)
        ids=(src if task['workload']=='fixed' else cap)['ids'][task['question_index']].cuda()
        ex=expand_input(base,ids,cap['carriers'].cuda(),carrier_layout=CarrierLayout(256,1,256))
        if schedule:
            from worldstream.m3.prefill_executor import PrefillExecutor
            from worldstream.m3.prefix_kv_cache import PrefixKVCache
            original=base.forward
            reference=[None]
            events=[]
            def observed(*args,**kwargs):
                fast=original(*args,**kwargs)
                if kwargs.get('past_key_values') is None:
                    reference[0]=fast.past_key_values
                    return fast
                length=kwargs['past_key_values'][0][0].shape[-2]
                # Mirror exactly the verifier's rejection crop before building
                # the independent SERIAL reference for the same input tokens.
                ref=tuple(tuple(t[...,:length,:] for t in layer) for layer in reference[0])
                for j in range(kwargs['input_ids'].shape[-1]):
                    single=original(input_ids=kwargs['input_ids'][:,j:j+1],past_key_values=ref,
                        attention_mask=kwargs['attention_mask'][:,:length+j+1],
                        position_ids=kwargs['position_ids'][:,j:j+1],use_cache=True,return_dict=True)
                    ref=single.past_key_values
                    a,b=single.logits[:,-1],fast.logits[:,j]
                    top=a.topk(2,-1).values
                    events.append(dict(predicted_output_index=length-ex.length+j+1,
                        forward_input_tokens=kwargs['input_ids'].shape[-1],
                        serial_argmax=int(a.argmax(-1)),block_argmax=int(b.argmax(-1)),
                        serial_top2_margin=float(top[0,0]-top[0,1]),
                        max_abs_logit_difference=float((a-b).abs().max())))
                reference[0]=ref
                return fast
            observed._worldstream_read_only_observer=True
            base.forward=observed
            draft=next(r for r in rows if r['mode']=='history8')['draft']
            actual,diag=PrefillExecutor(PrefixKVCache(0)).generate(base,ex,capture=False,
                draft_tokens=draft,draft_length=8,**dict(cap['generation'],do_sample=False))
            base.forward=original
            write(root/'gpu_schedule_numeric_probe.json',dict(status='COMPLETED',
                scope='Exact actual verifier block/rejection schedule; independent serial reference KV cropped to match accepted current-request prefix before every forward.',
                events=events,diagnostics=diag,native_tokens=native['tokens'],
                speculative_tokens=actual[0,1:].cpu().tolist(),
                generation_defaults=base.generation_config.to_dict(),
                argmax_disagreements=sum(e['serial_argmax']!=e['block_argmax'] for e in events)))
            return
        first=base(inputs_embeds=ex.embeddings,position_ids=ex.position_ids,
            attention_mask=ex.attention_mask,use_cache=True,return_dict=True)
        serial=blocked=first.past_key_values
        del first
        tokens=ex.token_ids.new_tensor(native['tokens'][:110]).reshape(1,-1)
        events=[]
        def forward(t,kv):
            n=kv[0][0].shape[-2];k=t.shape[-1]
            return base(input_ids=t,past_key_values=kv,
                attention_mask=torch.ones((1,n+k),dtype=ex.attention_mask.dtype,device=t.device),
                position_ids=torch.arange(n,n+k,device=t.device)[None],use_cache=True,return_dict=True)
        # Causal invariance: change only FUTURE draft IDs with exactly same block
        # shape; outputs before that suffix must remain bit-identical.
        a=tokens[:,:8].clone();b=a.clone();b[:,4:]=(b[:,4:]+333)%base.config.vocab_size
        oa=forward(a,serial);ob=forward(b,serial)
        causally_equal=torch.equal(oa.logits[:,:4],ob.logits[:,:4])
        future_diff=float((oa.logits[:,:4]-ob.logits[:,:4]).abs().max())
        del oa,ob
        offset=0
        while offset<tokens.shape[-1]:
            k=min(8 if offset==0 else 9,tokens.shape[-1]-offset)
            block=forward(tokens[:,offset:offset+k],blocked)
            blocked=block.past_key_values
            for j in range(k):
                single=forward(tokens[:,offset+j:offset+j+1],serial)
                serial=single.past_key_values
                reference=single.logits[:,-1];batched=block.logits[:,j]
                top=reference.topk(2,-1).values
                events.append(dict(conditioned_output_tokens=offset+j+1,
                    max_abs_logit_difference=float((reference-batched).abs().max()),
                    serial_argmax=int(reference.argmax(-1)),block_argmax=int(batched.argmax(-1)),
                    serial_top2_margin=float(top[0,0]-top[0,1])))
                del single
            del block
            offset+=k
        kv_difference=[dict(layer=i,key_max_abs=float((a[0]-b[0]).abs().max()),
            value_max_abs=float((a[1]-b[1]).abs().max())) for i,(a,b) in enumerate(zip(serial,blocked))]
        write(root/'gpu_numeric_probe.json',dict(status='COMPLETED',
            scope='Actual target, exactly the same forced token prefixes; no sampling and no historical KV. Serial native positions/mask versus 8/9-token current-context blocks.',
            scene_id=task['scene_id'],frame_id=task['frame_id'],question=task['question'],
            future_suffix_changed_prefix_logits_identical=causally_equal,
            future_suffix_prefix_logits_max_abs_difference=future_diff,
            events=events,kv_differences=kv_difference,
            dtype=str(base.get_input_embeddings().weight.dtype),
            generation_defaults=base.generation_config.to_dict(),
            argmax_disagreements=sum(e['serial_argmax']!=e['block_argmax'] for e in events)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('outputs/m3_world_guided_decode_run01'))
    parser.add_argument('--schedule',action='store_true')
    args=parser.parse_args();probe(args.root,args.schedule)
