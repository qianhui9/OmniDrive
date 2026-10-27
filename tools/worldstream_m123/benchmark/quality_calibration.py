"""Paired natural-input calibration for the single quality-bounded M3 path."""
from collections import Counter
from pathlib import Path
from time import perf_counter
import hashlib
import json
import math
import re
import numpy as np
import torch
from worldstream.m3 import M3Runtime
from worldstream.m3.config import M3Config
from worldstream.m3.reuse_planner import ReusePlanner, WEIGHTS, GROUPS
from worldstream.m3.scheduler import POLICY_VERSION
from worldstream.m3.visual_state import base_model, deployment_key
from .common import ground_truth, sha256, write
from .metrics import answer_quality, distribution


def numeric_values(text):
    words = dict(zero=0, one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9, ten=10)
    text = re.sub(r'\b('+'|'.join(words)+r')\b', lambda m:str(words[m[0]]), text.lower())
    return [float(x) for x in re.findall(r'(?<!\w)[+-]?\d+(?:\.\d+)?', text)]


def labels(text, kind):
    text=text.lower()
    patterns={
        'spatial': {'left':r'\bleft\b', 'right':r'\bright\b', 'front':r'\b(front|ahead)\b', 'rear':r'\b(behind|rear)\b'},
        'motion': {'stop':r'\b(stopped|stationary|parked)\b', 'moving':r'\b(moving|driving)\b',
                   'slow':r'\b(slow|slowing|decelerate)\b', 'fast':r'\b(accelerate|speed up)\b'},
        'safety': {'collision':r'\b(collision|collide|crash|hit)\b', 'unsafe':r'\b(unsafe|danger|dangerous|not safe)\b',
                   'stop':r'\b(stop|brake|braking)\b'}}
    return sorted(k for k,p in patterns[kind].items() if re.search(p,text))


def task_metrics(answer, reference):
    if reference is None:return {}
    x,y=numeric_values(answer),numeric_values(reference)
    result=dict(numeric_applicable=bool(y),numeric_exact=(len(x)==len(y) and all(
        math.isclose(a,b,rel_tol=0,abs_tol=1e-3) for a,b in zip(x,y))) if y else None,
        numeric_values=x,reference_numeric_values=y)
    unmatched=list(y);correct=0
    for value in x:
        match=next((i for i,v in enumerate(unmatched) if math.isclose(value,v,rel_tol=0,abs_tol=1e-3)),None)
        if match is not None:correct+=1;unmatched.pop(match)
    result['numeric_value_precision']=correct/len(x) if y and x else (0. if y else None)
    result['numeric_value_recall']=correct/len(y) if y else None
    result['numeric_aligned_mae']=float(np.mean(np.abs(np.array(x)-y))) if y and len(x)==len(y) else None
    for kind in ('spatial','motion','safety'):
        gold=labels(reference,kind)
        result[kind+'_label_match']=labels(answer,kind)==gold if gold else None
        pred=labels(answer,kind)
        result[kind+'_label_f1']=2*len(set(pred)&set(gold))/(len(pred)+len(gold)) if gold else None
    return result


def paired_quality(answer, baseline, reference):
    a=answer_quality(answer,reference);b=answer_quality(baseline,reference)
    if a is None or b is None:raise ValueError('Real calibration needs a labelled question')
    am,bm=task_metrics(answer,reference),task_metrics(baseline,reference)
    critical=sum(bm.get(k) is True and am.get(k) is False for k in (
        'numeric_exact','spatial_label_match','motion_label_match','safety_label_match'))
    critical+=sum(bm.get(k) is not None and am[k]<bm[k]-1e-12 for k in (
        'numeric_value_precision','numeric_value_recall','spatial_label_f1','motion_label_f1','safety_label_f1'))
    return dict(f1=a['token_f1'],baseline_f1=b['token_f1'],f1_loss=b['token_f1']-a['token_f1'],
        answer_equal=answer==baseline,output_token_f1=answer_quality(answer,baseline)['token_f1'],
        task_metrics=am,baseline_task_metrics=bm,critical_regressions=critical)


def quality_upper(rows):
    """Small-sample paired bound; empirical, not a semantic guarantee."""
    if not rows:return None
    values=np.array([r['quality']['f1_loss'] for r in rows],dtype=float)
    return max(0.,float(values.mean()+1.96*values.std(ddof=1)/math.sqrt(len(values)))) if len(values)>1 else max(0.,float(values[0]))


def aggregate(rows):
    return dict(samples=len(rows),quality_loss_upper=quality_upper(rows),
        mean_f1_loss=float(np.mean([r['quality']['f1_loss'] for r in rows])) if rows else None,
        critical_regressions=sum(r['quality']['critical_regressions'] for r in rows),
        runtime_ms=distribution(r['candidate_wall_ms'] for r in rows),
        baseline_ms=distribution(r['baseline_wall_ms'] for r in rows),
        reused_tokens=sum(r['diagnostics']['reused_tokens'] for r in rows))


def cross_frame_visual(row):
    """Require actual cross-frame visual reuse, not inherited same-frame KV."""
    d=row['diagnostics']
    source=d.get('cache_source_frame')
    target=row.get('frame_id')
    spans=d.get('visual_group_spans') or [(b['group'],b['start'],b['end']) for b in d.get('block_changes',[])]
    return bool(source and target and source!=target and spans
        and d['reused_tokens']>min(s[1] for s in spans))


def fit_policy(records, binding, sequence, budget=.02, block_size=32):
    """Select thresholds using calibration scenes ONLY, then validate once.

    Failed heldout rules are disabled, never replaced by a heldout-tuned threshold.
    Exact and approximate measurements share one policy and selection function.
    """
    cal_scenes={s for s,p in sequence['scene_splits'].items() if p=='calibration'}
    val_scenes={s for s,p in sequence['scene_splits'].items() if p=='validation'}
    if not cal_scenes or not val_scenes or cal_scenes & val_scenes:
        raise ValueError('Calibration and validation scenes must be disjoint')
    profiles={};curve=[];selection={}
    for profile in sorted({r['profile'] for r in records}):
        by_threshold={}
        for t in sorted({r['threshold'] for r in records}):
            rows=[r for r in records if r['profile']==profile and r['threshold']==t]
            cal=[r for r in rows if r.get('split','calibration' if r['scene_id'] in cal_scenes else 'validation')=='calibration']
            val=[r for r in rows if r.get('split','calibration' if r['scene_id'] in cal_scenes else 'validation')=='validation']
            c,v=aggregate(cal),aggregate(val)
            point=dict(profile=profile,threshold=t,calibration=c,validation=v)
            curve.append(point);by_threshold[t]=(rows,point)
        # Paired complete M3 wall cost, not candidate hit rate.
        eligible=[(t,p) for t,(_,p) in by_threshold.items() if p['calibration']['samples']
            and p['calibration']['quality_loss_upper']<=budget and not p['calibration']['critical_regressions']]
        if not eligible:continue
        chosen,_=min(eligible,key=lambda x:(x[1]['calibration']['runtime_ms']['mean'],x[0]))
        selection[profile]=chosen
        rows,point=by_threshold[chosen]
        rule={}
        for kind in ('exact','bounded_approx'):
            subset=[r for r in rows if r['diagnostics']['reuse_type']==kind and r['diagnostics']['reused_tokens']>0]
            cal=[r for r in subset if r.get('split','calibration' if r['scene_id'] in cal_scenes else 'validation')=='calibration']
            val=[r for r in subset if r.get('split','calibration' if r['scene_id'] in cal_scenes else 'validation')=='validation']
            loss=quality_upper(cal);heldout_loss=quality_upper(val)
            if not cal or not val:continue
            # Historical approximate fits could contain only inherited
            # same-frame KV (558 tokens) while the cross-frame cap was 163.
            # Keep all descendants in the quality bound, but require independent
            # cross-frame visual evidence for the approximate cost range.
            cost_cal=[r for r in cal if cross_frame_visual(r)] if kind=='bounded_approx' else cal
            cost_val=[r for r in val if cross_frame_visual(r)] if kind=='bounded_approx' else val
            if not cost_cal or not cost_val:continue
            if kind=='bounded_approx':
                input_range=[min(r['diagnostics']['input_tokens'] for r in cost_cal),max(r['diagnostics']['input_tokens'] for r in cost_cal)]
                reuse_range=[min(r['diagnostics']['reused_tokens'] for r in cost_cal),max(r['diagnostics']['reused_tokens'] for r in cost_cal)]
                cost_val=[r for r in cost_val if input_range[0]<=r['diagnostics']['input_tokens']<=input_range[1]
                    and reuse_range[0]<=r['diagnostics']['reused_tokens']<=reuse_range[1]]
                if not cost_val:continue
            # Costs fitted on calibration only; validation remains a check.
            lengths=np.array([r['diagnostics']['reused_tokens'] for r in cost_cal],dtype=float)
            savings=np.array([r['baseline_wall_ms']-r['candidate_wall_ms']+
                r.get('scheduler_overhead_ms',r['diagnostics']['planning_ms']) for r in cost_cal])
            if np.ptp(lengths)>=1:
                slope,intercept=np.polyfit(lengths,savings,1)
                residual=savings-(slope*lengths+intercept)
                fixed=max(0.,float(-intercept-np.percentile(residual,20)))
            else:
                slope=float(np.percentile(savings/lengths,20));fixed=0.
            n=[r['diagnostics']['input_tokens'] for r in cost_cal]
            evidence=dict(risk_limit=chosen,max_rms=2.,max_token_change=2.,
                max_visual_tokens=next((r.get('max_visual_tokens') for r in subset),None),
                quality_loss_upper=max(loss,heldout_loss),calibration_quality_loss_upper=loss,
                validation_quality_loss_upper=heldout_loss,
                critical_regressions=sum(r['quality']['critical_regressions'] for r in cal+val),
                calibration_samples=len(cal),validation_samples=len(val),
                cost=dict(net_saved_ms_per_token=float(slope),fixed_overhead_ms=fixed,
                    min_input_tokens=min(n),max_input_tokens=max(n),
                    min_reuse_tokens=min(r['diagnostics']['reused_tokens'] for r in cost_cal),
                    max_reuse_tokens=max(r['diagnostics']['reused_tokens'] for r in cost_cal),
                    samples=len(cost_cal),cross_frame_visual_samples=len(cost_cal) if kind=='bounded_approx' else None,
                    validation_cross_frame_visual_samples=len(cost_val) if kind=='bounded_approx' else None,
                    cost_scope='paired complete M3 wall incl. match, branch, cache update, prefill and affected decode',
                    reference_execution=cost_cal[0].get('cost_reference','legacy native LM head on identical M1/M2 carriers'),
                    overhead_scope='measured scheduler input preparation/signature plus planning' if all('scheduler_overhead_ms' in r for r in cost_cal)
                        else 'legacy planning-only evidence; online preparation/signature deduction remains conservative',
                    planning_ms=float(np.mean([r['diagnostics']['planning_ms'] for r in cost_cal])),
                    branch_ms=float(np.mean([r['diagnostics'].get('cache_branch_ms',0) or 0 for r in cost_cal])),
                    prefill_ms=float(np.mean([r['diagnostics']['prefill_ms'] for r in cost_cal])),
                    decode_ms=float(np.mean([r['diagnostics']['decode_ms'] for r in cost_cal])),
                    cache_update_enqueue_ms=float(np.mean([r['diagnostics'].get('cache_update_cpu_enqueue_ms',0) for r in cost_cal]))))
            rule['exact' if kind=='exact' else 'approx']=evidence
        profiles[profile]=rule
    return dict(version=POLICY_VERSION,binding=binding,block_size=block_size,quality_budget=budget,
        max_cache_age_frames=2,max_approx_depth=1,history_text=False,profiles=profiles,
        threshold_selection=selection,scene_splits=sequence['scene_splits'],
        calibration_samples=sum(r.get('split','calibration' if r['scene_id'] in cal_scenes else 'validation')=='calibration' for r in records),
        validation_samples=sum(r.get('split','calibration' if r['scene_id'] in cal_scenes else 'validation')=='validation' for r in records),
        quality_definition='paired GT Token F1 upper bound + no new correct-to-incorrect numeric/spatial/motion/safety parser errors',
        limitations=['Small-sample empirical bounds, not a mathematical guarantee',
            'Task labels are explicit lexical proxies, not human semantic approval',
            'Unseen profiles/input lengths stay on the same executor with reuse_length=0']),curve


def qualify_service(args):
    """Reject failed rules using actual closed-loop service observations.

    No new threshold is selected on validation data. Retesting after a rejection
    is a regression check, not another untouched independent validation set.
    """
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    policy=json.loads(Path(args.policy).read_text())
    sequence=json.loads(Path(args.sequence_file).read_text())
    def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
    baseline=rows(Path(args.native_reference)/'per_question_metrics.jsonl')
    actual=rows(Path(args.service_run)/'per_question_metrics.jsonl')
    if [(r['frame_id'],r['question']) for r in actual]!=[(r['frame_id'],r['question']) for r in baseline]:
        raise ValueError('Service and native inputs differ')
    evidence=[];groups={}
    for base,row in zip(baseline,actual):
        split=sequence.get('frame_splits',{}).get(row['frame_id'],sequence['scene_splits'][row['scene_id']])
        q=paired_quality(row['answer'],base['answer'],base['reference'])
        d=row['reuse_diagnostics']
        item=dict(frame_id=row['frame_id'],question=row['question'],split=split,profile=d['task_profile'],
            reuse_type=d['reuse_type'],quality=q)
        evidence.append(item)
        if split=='validation' and d['reuse_type']!='none':
            groups.setdefault((d['task_profile'],d['reuse_type']),[]).append(item)
    rejected=[]
    for (profile,kind),items in groups.items():
        rule=policy['profiles'][profile]['exact' if kind=='exact' else 'approx']
        critical=sum(i['quality']['critical_regressions'] for i in items)
        upper=quality_upper(items)
        rule['quality_loss_upper']=max(rule['quality_loss_upper'],upper)
        rule['critical_regressions']=max(rule['critical_regressions'],critical)
        rule['service_validation']=dict(samples=len(items),quality_loss_upper=upper,critical_regressions=critical)
        if critical or upper>policy['quality_budget']:rejected.append(dict(profile=profile,reuse_type=kind))
    with (out/'service_evidence.jsonl').open('w') as f:
        for item in evidence:f.write(json.dumps(item)+'\n')
    policy['service_qualification']=dict(evidence_path=str(out/'service_evidence.jsonl'),
        evidence_sha256=sha256(out/'service_evidence.jsonl'),rejected_rules=rejected,
        threshold_retuned=False,limitation='Post-rejection replay is regression evidence; further fresh-scene confirmation remains needed')
    write(out/'m3_reuse_policy.json',policy)
    write(out/'status.json',dict(status='COMPLETED',rejected_rules=rejected))


@torch.no_grad()
def calibrate(args):
    from .worker import effective_config,load_model
    from .observe import Observer
    from worldstream.m3.prefix_matcher import prefix_statistics
    from worldstream.m3.visual_state import expand_input
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    write(out/'status.json',dict(status='RUNNING'))
    write(out/'source_hashes.json',{str(p):sha256(p) for folder in ('worldstream/m3','tools/worldstream_m123/benchmark') for p in Path(folder).glob('*.py')})
    capture=Path(args.capture)
    sequence=json.loads(Path(args.sequence_file).read_text())
    cfg=effective_config('B3','language',out,'native_full')
    import projects.mmdet3d_plugin
    model=load_model(cfg,args.checkpoint,out)
    observer=Observer(model)
    from tools.worldstream_m123.latency_controls import load_capture
    frames=[load_capture(capture/'m3_inputs'/(t+'.pt')) for t in sequence['tokens']]
    native_rows=[json.loads(x) for x in (capture/'per_question_metrics.jsonl').read_text().splitlines()]
    native={(r['frame_id'],r['question']):r for r in native_rows}
    # Thresholds come from natural calibration-scene deltas only, in native dtype.
    risks=[];previous=None
    for frame in frames:
        if sequence['scene_splits'][frame['context']['scene_id']]!='calibration':continue
        carriers=frame['carriers'].cuda()
        x=expand_input(model.lm_head,frame['ids'][0].cuda(),carriers,carrier_layout=frame['layout'],
            scope=(frame['context']['scene_id'],),signature=('threshold_statistics',))
        if previous is not None and previous.scope==x.scope:
            _,_,blocks=prefix_statistics(previous,x,args.block_size)
            # Pool profiles using the conservative full_visual weight; trial
            # admission later applies the actual task and state weights.
            risks.extend(1.5*1.25*(b['rms']+.25*b['max_token']) for b in blocks)
        previous=x
    if not risks:raise ValueError('No natural calibration-frame carrier changes')
    thresholds=sorted(set([0.]+[float(np.quantile(risks,q)) for q in ((.65,) if args.max_visual_tokens else (.25,.65,.95))]))
    write(out/'thresholds.json',dict(values=thresholds,source='natural calibration scenes only',block_size=args.block_size))
    runtimes={t:M3Runtime(M3Config(block_size=args.block_size,quality_budget=args.quality_budget)) for t in thresholds}
    for t,runtime in runtimes.items():runtime.planner=ReusePlanner(runtime.scheduler,calibration_threshold=t,
        calibration_max_visual_tokens=args.max_visual_tokens)
    for runtime in runtimes.values():
        original=runtime.planner.plan
        def observed_plan(*a,_runtime=runtime,_original=original,**kw):
            _runtime._calibration_input_overhead_ms=kw.get('overhead_ms',0.)
            return _original(*a,**kw)
        runtime.planner.plan=observed_plan
    records=[];binding=None
    for ordinal,frame in enumerate(frames):
        scene=frame['context']['scene_id'];fid=frame['context']['frame_id']
        carriers=frame['carriers'].cuda();gt=ground_truth(fid)
        for qi,(ids,question,rng) in enumerate(zip(frame['ids'],frame['questions'],frame['rng'])):
            ids=ids.cuda();kw=frame['generation']
            torch.set_rng_state(rng[0]);torch.cuda.set_rng_state(rng[1]);observer.clear()
            torch.cuda.synchronize();tick=perf_counter()
            reference=model.lm_head.generate(inputs=ids[None],images=carriers,**kw)
            torch.cuda.synchronize();baseline_ms=(perf_counter()-tick)*1000
            if reference.cpu().tolist()!=native[(fid,question)]['token_ids']:
                raise RuntimeError('Captured natural input/RNG does not reproduce native tokens')
            baseline=model.tokenizer.batch_decode(reference,skip_special_tokens=True)[0]
            for threshold,runtime in runtimes.items():
                torch.set_rng_state(rng[0]);torch.cuda.set_rng_state(rng[1]);observer.clear()
                torch.cuda.synchronize();tick=perf_counter()
                output,diag=runtime.generate(model.lm_head,[ids],carriers,
                    context=dict(frame['context'],profile=True),questions=[question],layout=frame['layout'],
                    verified_state=frame['verified_state'],tokenizer=model.tokenizer,m2_diagnostics=frame['m2'],**kw)
                torch.cuda.synchronize();elapsed=(perf_counter()-tick)*1000
                answer=model.tokenizer.batch_decode(output[0],skip_special_tokens=True)[0]
                d=diag['requests'][0]
                if binding is None:
                    base=base_model(model.lm_head)
                    generation_key=hashlib.sha256(json.dumps(dict(defaults=base.generation_config.to_dict(),overrides=kw),sort_keys=True,default=repr).encode()).hexdigest()
                    binding=dict(model_version=frame['context']['weights_sha256'],weights_sha256=frame['context']['weights_sha256'],
                        execution_key=deployment_key(model.lm_head),generation_key=generation_key,
                        layout_signature=d['layout_signature'],dtype=str(base.get_input_embeddings().weight.dtype))
                record=dict(frame_ordinal=ordinal,scene_id=scene,frame_id=fid,question_index=qi,
                    split=sequence.get('frame_splits',{}).get(fid,sequence['scene_splits'][scene]),
                    max_visual_tokens=args.max_visual_tokens,profile=d['task_profile'],question=question,reference=gt.get(question),
                    threshold=threshold,baseline_wall_ms=baseline_ms,candidate_wall_ms=elapsed,
                    cost_reference='native LM head on identical M1/M2 carriers',
                    scheduler_overhead_ms=runtime._calibration_input_overhead_ms+d['planning_ms'],
                    baseline_answer=baseline,answer=answer,quality=paired_quality(answer,baseline,gt.get(question)),
                    diagnostics=d,token_ids=output[0].cpu().tolist())
                records.append(record)
                with (out/'paired_records.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        print('CALIBRATE',ordinal+1,'/',len(frames),fid,flush=True)
        write(out/'progress.json',dict(completed_frames=ordinal+1,total_frames=len(frames),records=len(records)))
    policy,curve=fit_policy(records,binding,sequence,args.quality_budget,args.block_size)
    policy.update(checkpoint=args.checkpoint,checkpoint_sha256=sha256(args.checkpoint),
        data_manifest_sha256=sha256(args.sequence_file),paired_records_sha256=sha256(out/'paired_records.jsonl'),
        paired_records_path=str(out/'paired_records.jsonl'),thresholds=thresholds)
    write(out/'m3_reuse_policy.json',policy);write(out/'pareto.json',curve)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    for profile in sorted({p['profile'] for p in curve}):
        points=[p for p in curve if p['profile']==profile and p['calibration']['samples']]
        ax.plot([p['calibration']['mean_f1_loss'] for p in points],
                [p['calibration']['runtime_ms']['mean'] for p in points],'o-',label=profile)
        for p in points:ax.annotate(f"{p['threshold']:.3g}",(p['calibration']['mean_f1_loss'],p['calibration']['runtime_ms']['mean']),fontsize=7)
    ax.axvline(args.quality_budget,color='red',linestyle='--',label='requested F1 budget')
    ax.set(xlabel='Paired GT Token F1 loss (absolute)',ylabel='Complete candidate M3 wall (ms)',title='Natural calibration scenes: quality / latency')
    ax.legend();fig.tight_layout();fig.savefig(out/'pareto.png',dpi=160);fig.savefig(out/'pareto.svg');plt.close(fig)
    write(out/'status.json',dict(status='COMPLETED',natural_frames=len(frames),paired_records=len(records),
        policy=str(out/'m3_reuse_policy.json'),no_legacy_approval_files_modified=True))
