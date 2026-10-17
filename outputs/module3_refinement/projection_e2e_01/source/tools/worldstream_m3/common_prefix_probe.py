"""Full native sequences versus an independently computed true shared prefix."""
import argparse
import inspect
import json
from pathlib import Path
from time import perf_counter
import torch
from worldstream.m3 import M3Runtime
from worldstream.m3.visual_state import base_model
from .experiment import Timer,write
from .refinement import snapshot_source


@torch.no_grad()
def run(args):
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False);snapshot_source(dest)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    import cv2
    cv2.setNumThreads(1)
    from tools.worldstream_m2.rearchitecture.integration import load
    model,_=load();model.lm_head.generate=inspect.getclosurevars(model.lm_head.generate).nonlocals['generate']
    timer=Timer();timer.wrap(base_model(model.lm_head),'llm',dynamic=True)
    root=Path('outputs/module3/clean_multi_20261016_220113/carriers');rows=[]
    versions=[(n,p._version) for n,p in model.named_parameters()]
    for path in sorted(root.glob('*.pt')):
        snap=torch.load(path,map_location='cpu');ids=[snap['ids'][i].cuda() for i in snap['selected_indices']]
        vision=snap['carriers'].cuda();kw=dict(do_sample=False,max_new_tokens=320,return_dict_in_generate=True,output_scores=True)
        reference=[model.lm_head.generate(inputs=x[None],images=vision,**kw) for x in ids]
        runtime=M3Runtime(dict(mode='exact',offline_experiment=True,common_prefix_seed=True,borrow_prefix=True))
        timer.events.clear();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=perf_counter()
        actual,diag=runtime.generate(model.lm_head,ids,vision,context=dict(scene_id=snap['scene'],frame_id=path.stem,camera_ids=['all']),**kw)
        torch.cuda.synchronize();wall=(perf_counter()-tick)*1000;components=timer.collect()
        comparisons=[]
        for qi,(a,b) in enumerate(zip(reference,actual)):
            n=min(a.sequences.shape[1],b.sequences.shape[1]);step=next((i for i in range(1,n) if a.sequences[0,i]!=b.sequences[0,i]),None)
            delta=[float((x-y).abs().max()) for x,y in zip(a.scores,b.scores)];valid=delta if step is None else delta[:step]
            comparisons.append(dict(equal=torch.equal(a.sequences,b.sequences),first_divergent_generated_token=step,
                prefill_logits_error=delta[0],same_history_decode_logits_error=max(valid[1:],default=0),
                reference_ids=a.sequences.cpu().tolist(),actual_ids=b.sequences.cpu().tolist()))
        row=dict(policy='independent_common_prefix',token=path.stem,scene=snap['scene'],comparisons=comparisons,
            diagnostics=diag,wall_ms=wall,components=components,peak_allocated=torch.cuda.max_memory_allocated())
        rows.append(row)
        with (dest/'sequences.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        runtime.reset();del reference,actual
        print('COMMON_PREFIX',path.stem,sum(r['equal'] for r in comparisons),flush=True)
    assert versions==[(n,p._version) for n,p in model.named_parameters()]
    write(dest/'completion.json',dict(complete=True,rows=len(rows),weights_versions_unchanged=True,
        frozen_target_pass=all(q['equal'] for r in rows for q in r['comparisons'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/module3_refinement/common_prefix_01');run(p.parse_args())
