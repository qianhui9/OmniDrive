"""Layer/operator evidence and full-sequence precision probes on saved real carriers."""
import argparse
from contextlib import AbstractContextManager
from functools import wraps
import inspect
import hashlib
import json
from pathlib import Path
from time import perf_counter
import torch
import torch.nn.functional as F
from worldstream.m3 import M3Runtime,expand_input
from worldstream.m3.visual_state import base_model
from worldstream.m3.prefix_matcher import exact_lcp
from worldstream.m3.numerics import precision_probe
from .experiment import write,Timer

ROOT=Path('outputs/module3/clean_multi_20261016_220113')


class LayerTrace(AbstractContextManager):
    """Record selected absolute rows at every layer, never use tracing for timing."""
    def __init__(self,model,rows,offset=0):
        self.model=model;self.rows=rows;self.offset=offset;self.data={};self.handles=[];self.layer=None
    def save(self,name,x,axis=1):
        if not isinstance(x,torch.Tensor):return
        ids=[i-self.offset for i in self.rows if 0<=i-self.offset<x.shape[axis]]
        if not ids:return
        index=torch.tensor(ids,device=x.device)
        self.data[name]=dict(shape=list(x.shape),dtype=str(x.dtype),rows=[i+self.offset for i in ids],
            axis=axis,value=x.detach().index_select(axis,index).cpu().clone())
    def __enter__(self):
        import transformers.models.llama.modeling_llama as llama
        self.llama=llama;self.matmul=torch.matmul;self.softmax=F.softmax;self.rope=llama.apply_rotary_pos_emb
        for name,module in self.model.named_modules():
            if name in ('model.norm','lm_head'):
                def final_hook(m,a,o,name=name):self.save(name,o)
                self.handles.append(module.register_forward_hook(final_hook))
            if name.startswith('model.layers.'):
                suffix=name.split('.',3)
                if len(suffix)==3:
                    number=int(suffix[2])
                    def pre(m,a,number=number):
                        self.layer=number;self.save(f'layer{number}.input',a[0])
                    self.handles.append(module.register_forward_pre_hook(pre))
                if (len(suffix)==3 or any(s in name for s in ('layernorm','self_attn','mlp'))) and 'rotary_emb' not in name:
                    def hook(m,a,o,name=name):
                        self.save(name,o[0] if isinstance(o,tuple) else o)
                    self.handles.append(module.register_forward_hook(hook))
        def matmul(a,b,*args,**kw):
            out=self.matmul(a,b,*args,**kw)
            if self.layer is not None and out.ndim==4:
                key=f'layer{self.layer}.attention_'+('score' if b.shape[-2]==self.model.config.hidden_size//self.model.config.num_attention_heads else 'value_output')
                self.save(key,out,2)
            return out
        def softmax(x,*args,**kw):
            out=self.softmax(x,*args,**kw)
            if self.layer is not None and x.ndim==4:self.save(f'layer{self.layer}.attention_softmax',out,2)
            return out
        def rope(q,k,*args,**kw):
            out=self.rope(q,k,*args,**kw)
            self.save(f'layer{self.layer}.rope_q',out[0],2);self.save(f'layer{self.layer}.rope_k',out[1],2)
            return out
        torch.matmul=matmul;F.softmax=softmax;llama.apply_rotary_pos_emb=rope
        return self
    def __exit__(self,*args):
        for h in self.handles:h.remove()
        torch.matmul=self.matmul;F.softmax=self.softmax;self.llama.apply_rotary_pos_emb=self.rope


def compare(a,b,allowed_rows):
    out=[]
    for name,x in a.items():
        if name not in b:continue
        y=b[name];rows=sorted(set(x['rows'])&set(y['rows'])&set(allowed_rows))
        if not rows:continue
        u=x['value'].index_select(x['axis'],torch.tensor([x['rows'].index(r) for r in rows]))
        v=y['value'].index_select(y['axis'],torch.tensor([y['rows'].index(r) for r in rows]))
        # Attention keys may include an extra, future masked suffix. Compare
        # common key columns; no attention to future keys is asserted equivalent.
        if u.shape!=v.shape:
            if u.ndim!=4:continue
            n=min(u.shape[-1],v.shape[-1]);u=u[...,:n];v=v[...,:n]
        if u.ndim==4 and ('attention_score' in name or 'attention_softmax' in name):
            stop=max(allowed_rows)+1;u=u[...,:stop];v=v[...,:stop]
        delta=(u.float()-v.float()).abs()
        out.append(dict(operator=name,source_shape=x['shape'],target_shape=y['shape'],rows=rows,
            max_abs=float(delta.max()),mean_abs=float(delta.mean()),equal=torch.equal(u,v)))
    return out


@torch.no_grad()
def diagnose(base,snap,policy,dest):
    ids=[snap['ids'][i].cuda() for i in snap['selected_indices'][:2]];vision=snap['carriers'].cuda()
    xs=[expand_input(base,x,vision) for x in ids];prefix=exact_lcp(*xs)
    selected=sorted(set([0,34,35,100,300,prefix-1]+list(range(prefix,xs[1].length))))
    traces=[];kv=[]
    for x in xs:
        with LayerTrace(base,selected) as trace:
            output=base(inputs_embeds=x.embeddings,position_ids=x.position_ids,attention_mask=x.attention_mask,use_cache=True)
        traces.append(trace.data);kv.append(output.past_key_values)
        del output
    with LayerTrace(base,selected,offset=prefix) as trace:
        output=base(inputs_embeds=xs[1].embeddings[:,prefix:],position_ids=xs[1].position_ids[:,prefix:],
            attention_mask=xs[1].attention_mask,past_key_values=tuple(tuple(t[...,:prefix,:] for t in layer) for layer in kv[0]),use_cache=True)
    prefix_errors=[dict(layer=i,key_max_abs=float((a[0][...,:prefix,:]-b[0][...,:prefix,:]).abs().max()),
        value_max_abs=float((a[1][...,:prefix,:]-b[1][...,:prefix,:]).abs().max())) for i,(a,b) in enumerate(zip(*kv))]
    result=dict(policy=policy,input_prefix_bitwise_equal=torch.equal(xs[0].embeddings[:,:prefix],xs[1].embeddings[:,:prefix]),
        prefix=prefix,source_tokens=xs[0].length,target_tokens=xs[1].length,
        prefix_layer_operators=compare(traces[0],traces[1],range(prefix)),
        suffix_layer_operators=compare(traces[1],trace.data,range(prefix,xs[1].length)),prefix_kv=prefix_errors,
        sampled_absolute_positions=selected,attention_backend=type(base.model.layers[0].self_attn).__module__+'.'+type(base.model.layers[0].self_attn).__qualname__)
    torch.save(dict(source=traces[0],target=traces[1],cached_suffix=trace.data),dest/f'{policy}_intermediates.pt')
    write(dest/f'{policy}_operators.json',result)
    return result


@torch.no_grad()
def run(args):
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False)
    manifest={}
    for path in list(Path('worldstream/m3').glob('*.py'))+[Path(__file__)]:
        relative=Path('tools/worldstream_m3/numerical_refinement.py') if path.is_absolute() else path
        target=dest/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
        manifest[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
    write(dest/'source_manifest.json',manifest)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    import cv2
    cv2.setNumThreads(1)
    from tools.worldstream_m2.rearchitecture.integration import load
    model,_=load();lm=model.lm_head
    lm.generate=inspect.getclosurevars(lm.generate).nonlocals['generate']
    base=base_model(lm);timer=Timer();timer.wrap(base,'llm',dynamic=True)
    write(dest/'environment.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
        attention_backend=type(base.model.layers[0].self_attn).__module__+'.'+type(base.model.layers[0].self_attn).__qualname__,
        settings=dict(tf32=False),args=vars(args)))
    failed=torch.load(ROOT/'carriers/e12ae352f31d4268a7f71c4d27aa0886.pt',map_location='cpu')
    versions=[(n,p._version) for n,p in base.named_parameters()]
    rows=[]
    policies=['native','deterministic','no_fp16_reduction','fp32_lora']
    references={}
    files=sorted((ROOT/'carriers').glob('*.pt'))
    # All original six scenes and 12 consecutive frames, not only the failing Q.
    for policy in policies:
        with precision_probe(base,policy):
            evidence=diagnose(base,failed,policy,dest)
            if policy=='native':
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
                    xs=[expand_input(lm,failed['ids'][i].cuda(),failed['carriers'].cuda()) for i in failed['selected_indices'][:2]]
                    for x in xs:base(inputs_embeds=x.embeddings,position_ids=x.position_ids,attention_mask=x.attention_mask,use_cache=True)
                prof.export_chrome_trace(str(dest/'gemm_shapes.json'))
            for path in files:
                snap=torch.load(path,map_location='cpu');token=path.stem
                ids=[snap['ids'][i].cuda() for i in snap['selected_indices']];vision=snap['carriers'].cuda()
                if token not in references:
                    # Reference is always the unmodified original model, including
                    # for a precision candidate which changes the full path.
                    if policy!='native':raise ValueError('Missing native reference')
                    references[token]=[lm.generate(inputs=x[None],images=vision,do_sample=False,max_new_tokens=320,
                        return_dict_in_generate=True,output_scores=True) for x in ids]
                    references[token]=[dict(ids=x.sequences.cpu(),scores=[s.cpu() for s in x.scores]) for x in references[token]]
                for mode in ['exact']:
                    runtime=M3Runtime(dict(mode=mode,offline_experiment=True))
                    timer.events.clear();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=perf_counter()
                    answers,diag=runtime.generate(lm,ids,vision,context=dict(scene_id=snap['scene'],frame_id=token,camera_ids=['all']),
                        do_sample=False,max_new_tokens=320,return_dict_in_generate=True,output_scores=True)
                    torch.cuda.synchronize();elapsed=(perf_counter()-tick)*1000;components=timer.collect()
                    qrecords=[]
                    for reference,actual in zip(references[token],answers):
                        actual_ids=actual.sequences.cpu();ref=reference['ids'];m=min(ref.shape[1],actual_ids.shape[1])
                        differences=[float((s-t.cpu()).abs().max()) for s,t in zip(reference['scores'],actual.scores)]
                        step=next((i for i in range(1,m) if ref[0,i]!=actual_ids[0,i]),None)
                        qrecords.append(dict(equal=torch.equal(ref,actual_ids),first_divergent_generated_token=step,
                            prefill_logit_max_abs=differences[0],decode_logit_max_abs=max(differences[1:],default=0),
                            step_logit_max_abs=differences,
                            same_history_decode_logit_max_abs=max(differences[1:step] if step else differences[1:],default=0),
                            limitation='Unfiltered decode max includes different histories after the first divergence',
                            reference_ids=ref.tolist(),actual_ids=actual_ids.tolist()))
                    row=dict(policy=policy,token=token,scene=snap['scene'],questions=[snap['questions'][i] for i in snap['selected_indices']],
                        comparisons=qrecords,diagnostics=diag,wall_ms=elapsed,components=components,peak_allocated=torch.cuda.max_memory_allocated())
                    rows.append(row)
                    with (dest/'sequences.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                    runtime.reset()
                print('NUMERICAL',policy,token,sum(q['equal'] for q in qrecords),flush=True)
        torch.cuda.empty_cache()
    assert versions==[(n,p._version) for n,p in base.named_parameters()]
    write(dest/'completion.json',dict(complete=True,rows=len(rows),weights_versions_unchanged=True,
        policies=policies,baseline='unmodified original generate, max_new_tokens 320',
        note='Post-divergence logit errors compare different generated histories and are not same-input numerical errors'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/module3_refinement/numerical_01');run(p.parse_args())
