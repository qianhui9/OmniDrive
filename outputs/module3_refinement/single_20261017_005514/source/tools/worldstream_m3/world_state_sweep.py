"""Longer real perception/M1 history feasibility, plus conditional original QA.

Perception-only frames establish coverage, NOT VLM latency or mock QA results.
Any QA uses real saved current carriers and separately frozen historical state.
"""
import argparse
import copy
from dataclasses import replace
import inspect
import json
import random
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from worldstream.m3 import M3Runtime,CarrierLayout
from worldstream.m3.world_state import query_world_state,insert_state_context
from .experiment import write,ground_truth,chosen_questions
from .refinement import scenes_for,snapshot_source
from .llm_probe import quality
from .offline_reference import resolve_reference


@torch.no_grad()
def run(args):
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=False);snapshot_source(dest)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    import cv2
    cv2.setNumThreads(1)
    from tools.worldstream_m2.rearchitecture.integration import load
    from worldstream.m1 import WorldStateManager,M1Config
    from mmcv.parallel import scatter,collate
    model,dataset=load();native=inspect.getclosurevars(model.lm_head.generate).nonlocals['generate']
    model.lm_head.generate=native
    from .refinement_final_probes import run as final_probes
    final_probes(model,dest/'final_probes')
    if args.primary:
        from .projection_probe import run as projection_probe
        projection_probe(model,dest/'projection_probe',args.primary)
    for transform in dataset.pipeline.transforms:
        if hasattr(transform,'load_type'):transform.load_type=['planning','short','conv']
    specs=scenes_for(dataset);groups={}
    for i,info in enumerate(dataset.data_infos):groups.setdefault(info['scene_token'],[]).append(i)
    picked=[specs[0],next(s for s in specs if 5489 in s['indices']),specs[-1]]
    for s in picked:
        indices=sorted(groups[s['scene']],key=lambda i:dataset.data_infos[i]['timestamp'])
        start=indices.index(s['indices'][0]);s['indices']=indices[start:start+args.frames]
        if len(s['indices'])<args.frames:raise ValueError('Not enough consecutive frames')
    write(dest/'protocol.json',dict(scenes=picked,frames_per_scene=args.frames,
        scope='Actual perception and M1 history coverage, no LLM service timing. Conditional QA is a separate saved-carrier replay.',
        state_source='Current precheck of earlier actual observations, frozen before current perception commit',
        ground_truth='Loaded only after query/coverage decisions; no GT accepted by M1 or retrieval'))
    versions=[(n,p._version) for n,p in model.named_parameters()]
    lm=model.lm_head;model.lm_head=None
    # The model explicitly supports LM-disabled perception inference. The same
    # original parameter objects remain alive and are restored for conditional QA.
    captured={};current={};contexts=[];rows=[]
    hooks=[]
    for key,head in [('object',model.pts_bbox_head),('map',model.map_head)]:
        def record(m,a,o,key=key):captured[key]=o[1].detach()
        hooks.append(head.register_forward_hook(record))
    original_pts=model.simple_test_pts
    def pts_entry(metas,**data):
        meta=metas[0];current.clear();current.update(token=meta['sample_idx'],scene_id=meta['scene_token'],
            frame_id=meta['sample_idx'],timestamp=float(data['timestamp'].reshape(-1)[0]),
            camera_ids=[Path(f).parent.name for f in meta['filename']],questions=list(meta['vlm_labels'].data),
            ids=list(data['input_ids'][0]))
        return original_pts(metas,**data)
    model.simple_test_pts=pts_entry
    try:
        for si,scene in enumerate(picked):
            model.test_flag=False
            model.m1_manager=WorldStateManager(replace(M1Config.load('configs/worldstream_m1_coverage.yaml'),log_path=str(dest/'m1_state.jsonl')))
            model.m2_runtime=None;model.m3_runtime=M3Runtime(dict(mode='compact',state_context='only',offline_experiment=True))
            for fi,index in enumerate(scene['indices']):
                captured.clear();model.save_path=str(dest/f'perception_only/{si}_{fi}')+'/'
                sample=scatter(collate([copy.deepcopy(dataset[index])],samples_per_gpu=1),[torch.cuda.current_device()])[0]
                tick=perf_counter();output=model(return_loss=False,rescale=True,**sample);torch.cuda.synchronize()
                state=model.m3_verified_state;ctx={k:v for k,v in current.items() if k not in ('questions','ids','token')}
                ctx.update(state_epoch=model.m1_manager.store.generation,state_execution_signature=model.m1_manager.store.execution_signature)
                inquiries=[];eligible=[]
                for qi,q in enumerate(current['questions']):
                    result=query_world_state(state,q,ctx);fits=False
                    if result['admitted']:
                        try:insert_state_context(current['ids'][qi],result['text'],model.tokenizer);fits=True
                        except ValueError:pass
                    if fits:eligible.append(qi)
                    inquiries.append(dict(question=q,admitted_by_task=result['admitted'],fits_context_budget=fits,reason=result['reason']))
                checks=model.m1_manager.last_result.diagnostics.get('anchors',{})
                reasons={}
                for check in checks.values():
                    for reason in check.get('reasons',[]):reasons[reason]=reasons.get(reason,0)+1
                row=dict(scene=scene['scene'],split=scene['split'],index=index,frame_offset=fi,token=current['token'],
                    verified_history_records=len(state.records) if state else 0,
                    state_diagnostics=state.diagnostics if state else None,inquiries=inquiries,
                    actual_history_anchor_count=len(checks),verification_rejection_reasons=reasons,
                    perception_and_m1_wall_ms=(perf_counter()-tick)*1000,
                    not_a_vlm_service_latency=True)
                rows.append(row)
                with (dest/'coverage.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                if eligible and len(contexts)<4:
                    chosen=[0]+[i for i in eligible if i!=0]
                    chosen+=[i for i in chosen_questions(current['questions'],5) if i not in chosen]
                    indices=sorted(chosen[:5])
                    record=dict(token=current['token'],context=ctx,verified_state=state,
                        carriers=torch.cat((captured['object'],captured['map']),1).cpu().clone(),
                        ids=[current['ids'][i].cpu().clone() for i in indices],questions=[current['questions'][i] for i in indices])
                    contexts.append(record);torch.save(record,dest/f'context_{len(contexts)-1}.pt')
                print('STATE_SWEEP',si,fi,index,row['verified_history_records'],len(eligible),flush=True)
    finally:
        for hook in hooks:hook.remove()
        model.lm_head=lm;model.m3_runtime=None;model.m1_manager=None;model.simple_test_pts=original_pts
    assert versions==[(n,p._version) for n,p in model.named_parameters()]
    qa=[]
    for record in contexts:
        ids=[x.cuda() for x in record['ids']];carriers=record['carriers'].cuda();reference=[]
        for mode in ('full','compact','state_only','state_full'):
            config=dict(mode='full' if mode=='full' else 'compact',offline_experiment=True)
            if mode in ('state_only','state_full'):config['state_context']='only' if mode=='state_only' else 'augment'
            runtime=M3Runtime(config)
            torch.manual_seed(0)
            outputs,diagnostics=runtime.generate(lm,ids,carriers,context=record['context'],questions=record['questions'],
                verified_state=record['verified_state'],tokenizer=model.tokenizer,layout=CarrierLayout(256,1,256),
                do_sample=False,max_new_tokens=320)
            values=[x.cpu().tolist() for x in outputs]
            if mode=='full':reference=values
            refs=ground_truth(record['token']);texts=[model.tokenizer.batch_decode(x,skip_special_tokens=True)[0] for x in outputs]
            references=[resolve_reference(record['token'],q,refs) for q in record['questions']]
            row=dict(mode=mode,token=record['token'],questions=record['questions'],answers=texts,ids=values,
                token_equal=[a==b for a,b in zip(reference,values)],diagnostics=diagnostics,
                gt=[r[0] for r in references],ground_truth_sources=[r[1] for r in references],
                quality=[quality(a,r[0]) if r[0] is not None else None for a,r in zip(texts,references)],
                scope='Controlled replay of actual current carriers and separately frozen historical evidence, not continuous E2E latency')
            qa.append(row)
            with (dest/'conditional_qa.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            runtime.reset()
    write(dest/'completion.json',dict(complete=True,frames=len(rows),scenes=len(picked),frames_per_scene=args.frames,
        verified_history_records=sum(r['verified_history_records'] for r in rows),
        task_queries_with_valid_context=sum(q['fits_context_budget'] for r in rows for q in r['inquiries']),
        controlled_context_frames=len(contexts),controlled_qa_rows=len(qa),weights_versions_unchanged=True,
        decision='Feasibility No-Go on this longer history workload' if not contexts else 'Controlled context quality requires independent validation'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/module3_refinement/state_sweep_01');p.add_argument('--frames',type=int,default=16)
    p.add_argument('--primary');run(p.parse_args())
