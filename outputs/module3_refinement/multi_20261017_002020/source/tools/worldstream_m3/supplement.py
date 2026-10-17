"""Original quantity questions and a real shared-visual-prefix profiler trace.

Separate from the frozen six-scene performance table. No GT enters selection.
"""
import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from .experiment import Timer,ground_truth,write
from .llm_probe import quality
from worldstream.m3 import M3Runtime,CarrierLayout,expand_input
from worldstream.m3.visual_state import base_model
from worldstream.m3.prefix_kv_cache import PrefixKVCache
from worldstream.m3.prefill_executor import PrefillExecutor


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--candidates',default='outputs/module3/quantity_candidates.json')
    args=parser.parse_args();dest=Path(args.output)
    if dest.exists() and any(dest.iterdir()):raise ValueError('Fresh output directory required')
    dest.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    import cv2
    cv2.setNumThreads(1)
    specs=json.loads(Path(args.candidates).read_text())
    write(dest/'protocol.json',dict(candidates=specs,selection='Original question text only, no GT filtering',
        generation=dict(do_sample=False,max_new_tokens=320),performance_scope='Supplementary LLM-only; not added to primary E2E table'))
    (dest/'source.py').write_bytes(Path(__file__).read_bytes())
    from tools.worldstream_m2.rearchitecture.integration import load
    model,dataset=load()
    for transform in dataset.pipeline.transforms:
        if hasattr(transform,'load_type'):transform.load_type=['planning','short','conv']
    from mmcv.parallel import scatter,collate,DataContainer
    original_pts=model.simple_test_pts
    selected_info={};current_spec=[None];captured={}
    native=inspect.getclosurevars(model.lm_head.generate).nonlocals['generate']
    timer=Timer();timer.wrap(base_model(model.lm_head),'llm',dynamic=True)
    def pts_entry(metas,**data):
        meta=metas[0];spec=current_spec[0]
        if meta['sample_idx']!=spec['token']:raise ValueError('Dataset index/token mismatch')
        questions=list(meta['vlm_labels'].data)
        qi=questions.index(spec['question']);picked=[0,qi] if qi!=0 else [0]
        all_ids=list(data['input_ids'][0])
        data['input_ids'][0]=[all_ids[i] for i in picked]
        meta['vlm_labels']=DataContainer([questions[i] for i in picked],cpu_only=True)
        selected_info.clear();selected_info.update(questions=[questions[i] for i in picked],
            scene_id=str(meta['scene_token']),frame_id=spec['token'],timestamp=float(data['timestamp'].reshape(-1)[0]),
            camera_ids=[Path(f).parent.name for f in meta['filename']])
        return original_pts(metas,**data)
    model.simple_test_pts=pts_entry
    def record(*a,**kw):
        kw.update(do_sample=False);kw.pop('temperature',None);kw.pop('top_p',None)
        if 'carriers' not in captured:captured['carriers']=kw['images'].detach().clone()
        captured.setdefault('ids',[]).append(kw['inputs'].detach().clone())
        answer=native(*a,**kw)
        captured.setdefault('answers',[]).append(answer.detach().clone())
        return answer
    model.lm_head.generate=record
    rows=[];trace_done=False
    for spec in specs:
        current_spec[0]=spec;captured.clear();model.test_flag=False
        model.save_path=str(dest/'baseline_answers'/spec['token'])+'/'
        batch=scatter(collate([copy.deepcopy(dataset[spec['index']])],samples_per_gpu=1),[torch.cuda.current_device()])[0]
        torch.manual_seed(0);np.random.seed(0)
        timer.events.clear();torch.cuda.synchronize();tick=perf_counter()
        output=model(return_loss=False,rescale=True,**batch)
        torch.cuda.synchronize();full_wall=(perf_counter()-tick)*1000
        full_components=timer.collect()
        ids=captured['ids'];vision=captured['carriers'];questions=selected_info['questions']
        references=ground_truth(spec['token'])
        baseline_ids=[x.cpu().tolist() for x in captured['answers']]
        def save_row(mode,selector,ratio,answers,diagnostics,wall,components):
            texts=[model.tokenizer.batch_decode(x,skip_special_tokens=True)[0] for x in answers]
            value=dict(token=spec['token'],index=spec['index'],mode=mode,selector=selector,keep_ratio=ratio,
                questions=questions,answers=texts,token_ids=[x.cpu().tolist() for x in answers],
                token_ids_equal=[x.cpu().tolist()==y for x,y in zip(answers,baseline_ids)],
                ground_truth=[references.get(q) for q in questions],
                quality=[quality(text,references[q]) if q in references else None for text,q in zip(texts,questions)],
                diagnostics=diagnostics,wall_ms=wall,components=components)
            rows.append(value)
            with (dest/'results.jsonl').open('a') as handle:handle.write(json.dumps(value,ensure_ascii=False)+'\n')
        save_row('baseline','full',1.,captured['answers'],{},full_wall,full_components)
        for mode,selector,ratio in [('exact','full',1.),('compact','task',.5),('compact','topk',.5),('compact','random',.5),('combined','task',.5)]:
            runtime=M3Runtime(dict(mode=mode,selector=selector,keep_ratio=ratio,offline_experiment=True))
            torch.manual_seed(0);timer.events.clear();torch.cuda.synchronize();tick=perf_counter()
            answers,diagnostics=runtime.generate(model.lm_head,ids,vision,context=selected_info,
                questions=questions,layout=CarrierLayout(256,1,256),do_sample=False,max_new_tokens=320)
            torch.cuda.synchronize();wall=(perf_counter()-tick)*1000
            save_row(mode,selector,ratio,answers,diagnostics,wall,timer.collect())
            runtime.reset()
        if not trace_done:
            base=base_model(model.lm_head)
            first=expand_input(model.lm_head,ids[0],vision,scope=(selected_info['scene_id'],),signature=('same_loaded_model',),source=('planning',spec['token']))
            second=expand_input(model.lm_head,ids[1],vision,scope=(selected_info['scene_id'],),signature=('same_loaded_model',),source=('quantity',spec['token']))
            cache=PrefixKVCache(512*1024**2);executor=PrefillExecutor(cache)
            executor.generate(base,first,max_new_tokens=1,do_sample=False)
            matched=cache.match(second)
            if matched<=513:raise ValueError('Expected genuine shared visual prefix')
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
                _,diag=executor.generate(base,second,matched,max_new_tokens=2,do_sample=False)
                torch.cuda.synchronize()
            (dest/'traces').mkdir(exist_ok=True)
            prof.export_chrome_trace(str(dest/'traces/shared_visual_prefix.json'))
            write(dest/'shared_visual_prefix_proof.json',dict(diagnostics=diag,
                initial_prompt_tokens=first.length,current_prompt_tokens=second.length,actual_reused_tokens=matched,
                expected_suffix_tokens=second.length-matched,source_token=spec['token'],
                source_sha256=hashlib.sha256((dest/'source.py').read_bytes()).hexdigest()))
            cache.reset();trace_done=True
        print('QUANTITY_COMPLETE',spec['index'],spec['question'],flush=True)
    write(dest/'completion.json',dict(complete=True,frames=len(specs),rows=len(rows),trace=trace_done,
        raw_file_sha256=hashlib.sha256((dest/'results.jsonl').read_bytes()).hexdigest()))


if __name__=='__main__':main()
