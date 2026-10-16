"""Measure feasibility and A/B/C numerical attribution before adding reuse boundaries."""
import argparse
import json
from time import perf_counter
import torch
from tools.worldstream_m2.benchmark import kernel_profile
from worldstream.m2.profiler import measure
from worldstream.m2.window_executor import WindowExecutor
from worldstream.m2.packing import partition,unpartition
from .common import ROOT,backbone,write,error_location,stats,reference


class PackedDense:
    """Diagnostic B path: all windows updated through the exact deployed executor."""
    def __init__(self,model):
        self.model=model;self.stages={};self.ex=WindowExecutor()
    def __call__(self,images,restore_mlp=False,skip=0):
        from projects.mmdet3d_plugin.models.backbones.eva_vit import get_abs_pos
        x=self.model.patch_embed(images)
        if self.model.pos_embed is not None:x=x+get_abs_pos(self.model.pos_embed,self.model.pretrain_use_cls_token,x.shape[1:3])
        for i,b in enumerate(self.model.blocks):
            if i<skip:continue
            if i<2:
                if restore_mlp:
                    # Same original order: norm -> padded attention -> UNPAD -> residual/MLP.
                    normalized,pad=partition(b.norm1(x),b.window_size)
                    attention=unpartition(b.attn(normalized),b.window_size,pad,x.shape[1:3])
                    x=x+b.drop_path(attention)
                    x=x+b.drop_path(b.mlp(b.norm2(x)))
                else:
                    windows,_=partition(x,b.window_size)
                    ids=torch.arange(len(windows),device=x.device)
                    x,_=self.ex.execute(b,x,windows,ids)
            else:x=b(x)
            self.stages[i]=x.detach().clone()
        return [x.permute(0,3,1,2)]


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repeats',type=int,default=10)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(2801);torch.backends.cuda.matmul.allow_tf32=False
    model=backbone();images=torch.randn(6,3,640,640,device='cuda')
    write(ROOT/'environment.json',dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        threads=torch.get_num_threads(),checkpoint='ckpts/iter_10548.pth',seed=2801,
        tf32=False,fp32_weights=True,flash_internal_fp16=True))
    events={};records={};handles=[]
    for name,module in [('patch_embedding',model.patch_embed)]+[(f'block_{i}',b) for i,b in enumerate(model.blocks)]:
        def before(m,inp,name=name):
            a=torch.cuda.Event(enable_timing=True);a.record();events[name]=(a,perf_counter())
        def after(m,inp,out,name=name):
            b=torch.cuda.Event(enable_timing=True);b.record();a,t=events[name]
            records.setdefault(name,[]).append((a,b,(perf_counter()-t)*1000))
        handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
    for _ in range(3):model(images)
    records.clear();torch.cuda.reset_peak_memory_stats()
    for _ in range(args.repeats):model(images)
    torch.cuda.synchronize()
    components={n:dict(gpu_ms=stats([a.elapsed_time(b) for a,b,_ in v]),
                       cpu_enqueue_wall_ms=stats([c for _,_,c in v])) for n,v in records.items()}
    for h in handles:h.remove()
    full=measure(lambda:model(images),warmup=3,repeats=args.repeats)
    # Hooks capture a real current-input oracle. Skipping later blocks here is an
    # explicitly impossible perfect-cache upper bound, never runtime permission.
    layer={};hs=[b.register_forward_hook(lambda m,i,o,j=j:layer.update({j:o.detach().clone()})) for j,b in enumerate(model.blocks)]
    oracle=model(images)[0]
    for h in hs:h.remove()
    upper=[]
    for early in (2,3,6,12):
        def tail():
            x=layer[early-1]
            for b in model.blocks[early:]:x=b(x)
            return x.permute(0,3,1,2)
        timing=measure(tail,warmup=3,repeats=args.repeats)
        upper.append(dict(scope=f'skip_first_{early}',kind='idealized_upper_bound',timing=timing,
            saving_ms=full['wall_ms']-timing['wall_ms'],cache_or_validation_included=False))
    for n in (1,3,6):
        def subset():
            result=torch.empty_strided(oracle.shape,oracle.stride(),device=oracle.device,dtype=oracle.dtype)
            result[:n].copy_(oracle[:n])
            if n<6:result[n:].copy_(model(images[n:])[0])
            return result
        timing=measure(subset,warmup=3,repeats=args.repeats)
        upper.append(dict(scope=f'reuse_{n}_cameras',kind='idealized_upper_bound',timing=timing,
            saving_ms=full['wall_ms']-timing['wall_ms'],cache_or_validation_included=False,
            error=error_location(subset(),oracle)))
    profile=kernel_profile(lambda:model(images),ROOT/'traces/full_eva.json')
    write(ROOT/'feasibility_raw.json',dict(full=full,components=components,upper_bounds=upper,profile=profile,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        notes='Component CPU times are enqueue wall, GPU time is events. Upper bounds exclude certification.'))
    print('phase A completed',full['wall_ms'],flush=True)
    packed=PackedDense(model);old=reference();numerical=[]
    for seed in (2801,3509):
        torch.manual_seed(seed);base=torch.randn_like(images)
        for case in ('repeat','patch','window','multi_window'):
            current=base.clone()
            if case=='patch':current[0,:,:16,:16]+=.03
            elif case=='window':current[0,:,:256,:256]+=.03
            elif case=='multi_window':current[:2,:,:256,:256]+=.03
            stages={};hs=[b.register_forward_hook(lambda m,i,o,j=j:stages.update({j:o.detach().clone()})) for j,b in enumerate(model.blocks)]
            a=model(current)[0]
            for h in hs:h.remove()
            b=packed(current)[0];bs=dict(packed.stages)
            restored=packed(current,restore_mlp=True)[0];rs=dict(packed.stages)
            c=old.IncrementalVisionRuntime(model,dict(mode='exact'))
            ctx=dict(scene_id='numeric',frame_id='0',camera_ids=list(range(6)))
            c.forward(base,execution_context=ctx);cs={}
            c.stage_observer=lambda j,x:cs.update({j:x.detach().clone()})
            sparse,diag=c.forward(current,execution_context=dict(ctx,frame_id='1'))
            micro=torch.cat([model(x[None])[0] for x in current])
            numerical.append(dict(seed=seed,case=case,skipped=diag['actual_skipped_windows'],
                B_vs_A=error_location(b,a),C_vs_A=error_location(sparse[0],a),
                C_vs_B=error_location(sparse[0],b),restored_mlp_vs_A=error_location(restored,a),
                camera_microbatch_vs_A=error_location(micro,a),
                layers={str(j):dict(B_vs_A=error_location(bs[j],stages[j]),C_vs_A=error_location(cs[j],stages[j]),
                    C_vs_B=error_location(cs[j],bs[j]),restored_mlp_vs_A=error_location(rs[j],stages[j])) for j in stages}))
            write(ROOT/'numerical_raw.json',dict(cases=numerical))
            print('numerical',seed,case,'B/C/restored',numerical[-1]['B_vs_A']['max_abs'],numerical[-1]['C_vs_A']['max_abs'],numerical[-1]['restored_mlp_vs_A']['max_abs'],flush=True)
    write(ROOT/'phase_ab_status.json',dict(status='completed'))

if __name__=='__main__':main()
