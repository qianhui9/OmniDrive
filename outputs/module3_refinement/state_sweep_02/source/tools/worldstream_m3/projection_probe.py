"""Genuine base Q/K/V FP32-prefill candidate on original and independent frames."""
from collections import Counter
import json
from pathlib import Path
from time import perf_counter
import torch
from worldstream.m3 import M3Runtime
from worldstream.m3.visual_state import base_model
from worldstream.m3.projection_precision import fp32_qkv_prefill
from .experiment import write,Timer
from .numerical_refinement import diagnose


@torch.no_grad()
def run(model,dest,primary):
    dest=Path(dest);dest.mkdir(exist_ok=False);base=base_model(model.lm_head)
    parameter_dtypes=dict(base_qkv=dict(Counter(str(m.weight.dtype) for n,m in base.named_modules()
        if n.startswith('model.layers.') and n.endswith(('q_proj.base_layer','k_proj.base_layer','v_proj.base_layer')))),
        lora=dict(Counter(str(m.weight.dtype) for n,m in base.named_modules()
        if isinstance(m,torch.nn.Linear) and ('.lora_A.' in n or '.lora_B.' in n))))
    write(dest/'parameter_dtypes.json',dict(actual=parameter_dtypes,
        fp32_lora_scope='Native loader already converts trainable LoRA parameters to FP32; that earlier policy is an arithmetic no-op sanity control, not a new precision improvement',
        new_policy='Only base Q/K/V multi-token prefill GEMMs use temporary FP32 operands; original weights and one-token decode arithmetic unchanged'))
    primary=Path(primary);protocol=json.loads((primary/'protocol.json').read_text())
    frame_rows=[json.loads(s) for s in (primary/'frames.jsonl').read_text().splitlines()]
    baseline_by_index={r['index']:r for r in frame_rows if r['mode']=='baseline'}
    baseline_by_token={r['token']:r for r in baseline_by_index.values()}
    records=[]
    for scene in protocol['scenes']:
        for index in scene['indices'][:2]:
            # Primary snapshots carry actual current perception, not history
            # relabeled as verified world state. No GT enters generation.
            frame=baseline_by_index[index]
            records.append((primary/'carriers'/f'{frame["token"]}.pt',scene['split']))
    failed=next(p for p,split in records if p.stem=='e12ae352f31d4268a7f71c4d27aa0886')
    snap=torch.load(failed,map_location='cpu')
    with fp32_qkv_prefill(base):diagnose(base,snap,'fp32_base_qkv',dest)
    original_forward=base.forward;timer=Timer();timer.wrap(base,'llm',dynamic=True)
    versions=[(n,p._version,p.dtype) for n,p in base.named_parameters()]
    try:
        for path,split in records:
            snap=torch.load(path,map_location='cpu');ids=[snap['ids'][i].cuda() for i in snap['selected_indices']]
            vision=snap['carriers'].cuda();context=dict(scene_id=snap['scene'],frame_id=path.stem,camera_ids=['original_native_camera_set'])
            score_kw=dict(do_sample=False,max_new_tokens=320,return_dict_in_generate=True,output_scores=True)
            timer.events.clear();torch.cuda.synchronize();tick=perf_counter()
            reference=[model.lm_head.generate(inputs=x[None],images=vision,**score_kw) for x in ids]
            torch.cuda.synchronize();reference_wall=(perf_counter()-tick)*1000;reference_components=timer.collect()
            baseline=baseline_by_token[path.stem]
            native_record=dict(policy='native_full_reference',token=path.stem,split=split,
                wall_ms=reference_wall,components=reference_components,
                comparisons=[dict(equal=a.sequences.cpu().tolist()==b,reference_ids=b,actual_ids=a.sequences.cpu().tolist())
                    for a,b in zip(reference,baseline['generated_ids'])],
                scope='Fresh original native execution, checked against primary original baseline, no answer reuse in generation')
            with (dest/'sequences.jsonl').open('a') as f:f.write(json.dumps(native_record)+'\n')
            full_policy=None
            for strategy in ('fp32_qkv_full','fp32_qkv_exact','fp32_qkv_batch'):
                runtime=M3Runtime(dict(mode='full' if strategy.endswith('full') else 'exact',offline_experiment=True,
                    borrow_prefix=True,batched_decode=strategy.endswith('batch')))
                generation=score_kw if not strategy.endswith('batch') else dict(do_sample=False,max_new_tokens=320)
                timer.events.clear();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=perf_counter()
                with fp32_qkv_prefill(base) as counts:
                    outputs,diag=runtime.generate(model.lm_head,ids,vision,context=context,**generation)
                torch.cuda.synchronize();wall=(perf_counter()-tick)*1000;components=timer.collect()
                if strategy.endswith('batch'):assert diag['batch']['admitted']
                comparisons=[]
                for qi,(a,b) in enumerate(zip(reference,outputs)):
                    actual=b.sequences if hasattr(b,'sequences') else b
                    n=min(a.sequences.shape[1],actual.shape[1]);step=next((i for i in range(1,n) if a.sequences[0,i]!=actual[0,i]),None)
                    delta=[float((x-y).abs().max()) for x,y in zip(a.scores,b.scores)] if hasattr(b,'scores') else None
                    valid=(delta if step is None else delta[:step]) if delta is not None else None
                    comparisons.append(dict(equal=torch.equal(a.sequences,actual),first_divergent_generated_token=step,
                        question=snap['questions'][snap['selected_indices'][qi]],
                        same_precision_full_token_equal=torch.equal(full_policy[qi].sequences,actual) if full_policy is not None else None,
                        prefill_logits_error=delta[0] if delta is not None else None,
                        same_history_decode_logits_error=max(valid[1:],default=0) if valid is not None else None,
                        score_scope='Same generated history before/at first divergence' if valid is not None else 'Batched structured score outputs deliberately unsupported; full token IDs compared',
                        reference_ids=a.sequences.cpu().tolist(),actual_ids=actual.cpu().tolist()))
                row=dict(policy=strategy,token=path.stem,split=split,scene=snap['scene'],comparisons=comparisons,
                    diagnostics=diag,projection_counts=counts,wall_ms=wall,components=components,
                    peak_allocated=torch.cuda.max_memory_allocated(),scope='Controlled LLM-only replay, memory includes retained reference scores; authoritative VLM memory/latency is the separate E2E protocol')
                assert counts['prefill_gemms']>0
                with (dest/'sequences.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                if strategy.endswith('full'):full_policy=outputs
                runtime.reset()
                print('BASE_FP32',strategy,path.stem,sum(q['equal'] for q in comparisons),flush=True)
            del reference,outputs,full_policy
    finally:base.forward=original_forward
    assert versions==[(n,p._version,p.dtype) for n,p in base.named_parameters()]
    write(dest/'completion.json',dict(complete=True,rows=len(records)*4,weights_and_dtypes_unchanged=True,
        calibrated_frames=sum(s=='calibration' for p,s in records),heldout_frames=sum(s=='heldout' for p,s in records)))
