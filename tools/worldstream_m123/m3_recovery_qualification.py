"""Fresh validation of an evidence-backed restriction, with no newly enabled rule."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from tools.worldstream_m123.benchmark.common import write
from tools.worldstream_m2.global_residual_benchmark import read,rows,execute,wait_gpu,evaluate
from tools.worldstream_m2.global_residual_analysis import compare_quality
from tools.worldstream_m123.m3_recovery_experiment import service,reference
from tools.worldstream_m123.m3_recovery_analysis import compare,reuse_counts,frame_metric
from tools.worldstream_m123.benchmark.quality_calibration import paired_quality,quality_upper
from tools.worldstream_m123.final_speed_report import table,dist,number
from tools.worldstream_m123.benchmark.metrics import distribution


def run_control(root):
    import projects.mmdet3d_plugin
    from tools.worldstream_m123.benchmark.worker import effective_config,load_model
    from tools.worldstream_m123.m3_recovery_control import paired
    out=root/'same_input_model';out.mkdir(exist_ok=False)
    cfg=effective_config('B3','language',out,'native_full');model=load_model(cfg,'ckpts/iter_10548.pth',out)
    paired(root,model,run_microbenchmark=False)


def analyze(root,install=False):
    from tools.worldstream_m2.production_acceptance import validate_rows
    zero=rows(root/'multi_free_r0_zero/per_question_metrics.jsonl');actual=rows(root/'multi_free_r0_reuse/per_question_metrics.jsonl')
    pairs=[]
    for a,b in zip(zero,actual):
        if (a['frame_id'],a['question'],a['reference'])!=(b['frame_id'],b['question'],b['reference']):raise ValueError('Questions/GT changed')
        pairs.append(dict(scene_id=b['scene_id'],frame_id=b['frame_id'],question=b['question'],baseline_answer=a['answer'],answer=b['answer'],reference=b['reference'],
            reuse_tokens=b['reused_prefix_length'],reuse_type=b['reuse_diagnostics']['reuse_type'],**paired_quality(b['answer'],a['answer'],b['reference'])))
    write(root/'natural_quality_pairs.json',pairs)
    differences=[p for p in pairs if not p['answer_equal']];write(root/'natural_answer_differences.json',differences)
    audits={};summaries={};native=rows(root/'multi_free_r0_native/per_frame_metrics.jsonl');inputs=[f['input_tensor_digest'] for f in native]
    perception=rows(root/'multi_free_r0_zero/per_frame_metrics.jsonl');digests=[f['output_digest'] for f in perception]
    refs=read(root/'multi_fixed_reference.json')
    for mode in ('free','fixed'):
        for kind in ('native','zero','reuse'):
            frames=[];questions=[]
            for repeat in (range(1) if mode=='free' else range(3)):
                name=f'multi_{mode}_r{repeat}_{kind}';p=root/name;ff=rows(p/'per_frame_metrics.jsonl');qq=rows(p/'per_question_metrics.jsonl')
                if read(p/'checkpoint_validation.json')['status']!='PASS' or read(p/'status.json')['integrity']!='PASS':raise ValueError('Checkpoint/ledger invalid')
                if [f['frame_id'] for f in ff]!=[f['frame_id'] for f in native] or [f['input_tensor_digest'] for f in ff]!=inputs:raise ValueError('Input mismatch')
                if kind!='native':
                    if read(p/'used_m3_policy.json')!=read(root/'frozen_m3_policy.json'):raise ValueError('Wrong policy executed')
                    if [f['output_digest'] for f in ff]!=digests:raise ValueError('M1/M2 perception changed')
                    a=validate_rows(ff,require_reuse=False)
                    if a['status']!='PASS':raise ValueError('M2 execution audit failed')
                    audits[name]=a
                    if kind=='zero' and any(q['reused_prefix_length'] for q in qq):raise ValueError('Zero control reused history')
                if mode=='fixed':
                    for f in ff:
                        rr=refs['frames'][f['frame_id']];steps=sum(len(q['tokens']) for q in rr)
                        if f['fixed_tokens']['generated_steps']!=steps or f['fixed_tokens']['actual_decode_forward_calls']!=steps-len(rr) or f['stages'].get('decode',{}).get('calls',0)!=steps-len(rr):raise ValueError('Actual fixed-step workload changed')
                    for q in qq:
                        ref=refs['frames'][q['frame_id']][q['question_index']]
                        if q['token_ids'][0][1:]!=ref['tokens'] or q['question']!=ref['question'] or q['quality'] is not None or q['reference'] is not None:raise ValueError('Forced tokens/quality isolation failed')
                frames+=ff;questions+=qq
            summaries[mode+'_'+kind]=dict(counts=reuse_counts(questions),**{k:distribution(frame_metric(f,k) for f in frames) for k in ('request_ms','model_ms','m3_ms','prefill_ms','preprocessing_ms','non_model_ms')},
                generated_tokens=sum(q['generated_tokens'] for q in questions),actual_decode_forwards=sum(f['stages'].get('decode',{}).get('calls',0) for f in frames),
                peak_allocated_bytes=max(f['peak_gpu_allocated_bytes'] for f in frames),throughput_fps=len(frames)*1000/sum(f['request_ms'] for f in frames))
    write(root/'execution_audits.json',audits)
    system=compare_quality(root,'multi_free_r0_native','multi_free_r0_reuse')
    comparisons={k:compare(root,'multi','reuse',v) for k,v in (('reuse_vs_zero','zero'),('reuse_vs_native','native'))}
    eligible=(not differences and sum(p['critical_regressions'] for p in pairs)==0 and quality_upper([dict(quality=p) for p in pairs])<=.02)
    result=dict(policy=read(root/'frozen_m3_policy.json'),plan=read(root/'experiment_plan.json'),summaries=summaries,comparisons=comparisons,system_quality=system,
        m3_quality=dict(requests=len(pairs),answers_equal=sum(p['answer_equal'] for p in pairs),mean_f1_loss=float(np.mean([p['f1_loss'] for p in pairs])),
            quality_loss_upper=quality_upper([dict(quality=p) for p in pairs]),critical_proxy_regressions=sum(p['critical_regressions'] for p in pairs),natural_answer_differences=len(differences)),
        semantic_review_clear=not differences,restriction_eligible=eligible,installed=Path('outputs/final_route/m3_reuse_policy.json').read_bytes()==(root/'frozen_m3_policy.json').read_bytes(),
        same_input_control=read(root/'same_input_control_summary.json'))
    if install and eligible:
        # Only a restriction is permitted; every old positive parameter stays.
        old=read(root.parent/'frozen_m3_policy.json');new=read(root/'frozen_m3_policy.json')
        for profile,rules in old['profiles'].items():
            for kind,evidence in rules.items():
                current=new['profiles'][profile][kind]
                if (profile,kind)==('map_relations','approx'):
                    if current['critical_regressions']<1:raise ValueError('Failed rule re-enabled')
                    for key,value in evidence.items():
                        if key!='critical_regressions' and current[key]!=value:raise ValueError('Positive rule parameter changed')
                elif current!=evidence:raise ValueError('Unrelated rule changed')
        if old['binding']!=new['binding']:raise ValueError('Binding changed')
        destination=Path('outputs/final_route/m3_reuse_policy.json')
        before=destination.read_bytes();allowed=[(root.parent/'frozen_m3_policy.json').read_bytes(),(root/'frozen_m3_policy.json').read_bytes()]
        if before not in allowed:raise ValueError('Unexpected concurrent policy change')
        destination.write_bytes((root/'frozen_m3_policy.json').read_bytes())
        result['installed']=True
        write(root/'deployment_decision.json',dict(status='RESTRICTED_POLICY_INSTALLED',policy=str(destination),policy_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            disabled_rule=['map_relations','approx'],reason='New unsupported road-rule claims in original-policy evaluation; preserve original automatic metrics and add separate visual-review rejection evidence. Fresh natural answers now match zero-history baseline.',
            new_rules_admitted=False,all_thresholds_and_cost_ranges_unchanged=True,full_visual_critical_regressions_retained=True,
            fresh_scenes=result['plan']['fresh_scenes'],system_quality_pass=system['within_quality_budgets'],single_runtime=True))
        write(root.parent/'deployment_decision.json',read(root/'deployment_decision.json'))
    write(root/'summary.json',result);write(root/'status.json',dict(status='COMPLETED' if eligible else 'NEEDS_SEMANTIC_REVIEW',installed=result['installed'],system_quality_pass=system['within_quality_budgets']))
    report(root,result)
    return result


def report(root,r):
    counts=r['summaries']['free_reuse']['counts'];c=r['comparisons']['reuse_vs_zero']
    out=['# 收紧 Map Approx 规则后的新独立场景验收',
        '上一阶段出现自动 F1/关键词代理遗漏的新增道路规则断言，因此仅增加失败证据并禁用 Map Relations Approx；没有重新拟合阈值、成本或开放新规则。该失败场景不再视为未见验证数据。本阶段冻结策略后另选4个未观察场景，每场景6帧、3个真实标注问题，按原生标注顺序执行所选问题。',
        f'新场景：`{r["plan"]["fresh_scenes"]}`。自然复用分解：`{counts}`。自然答案相同 `{r["m3_quality"]["answers_equal"]}/{r["m3_quality"]["requests"]}`；平均 F1 下降 `{r["m3_quality"]["mean_f1_loss"]}`；新增自动关键代理错误 `{r["m3_quality"]["critical_proxy_regressions"]}`；新增答案差异 `{r["m3_quality"]["natural_answer_differences"]}`。不再使用失败场景的 F1 上升宣称语义通过。',
        table(['条件（三轮）','Prefill CUDA mean/P50/P95 ms/帧','完整 M3 Wall','模型 Wall','请求 Wall','生成 Tokens','实际 Decode Forward'],[(k,dist(s['prefill_ms']),dist(s['m3_ms']),dist(s['model_ms']),dist(s['request_ms']),s['generated_tokens'],s['actual_decode_forwards']) for k in ('native','zero','reuse') for s in [r['summaries']['fixed_'+k]]]),
        table(['对照/阶段','加速比','平均节省 ms','场景 CI95 ms','所有场景/轮次均正收益'],[(comparison+'/'+metric,number(v['speedup']),number(v['saved_ms']['mean']),f'[{v["scene_bootstrap"]["lower"]:.3f}, {v["scene_bootstrap"]["upper"]:.3f}]',v['stable_net_speedup']) for comparison,values in r['comparisons'].items() for metric,v in values.items()]),
        '场景 Bootstrap 为4个独立场景，三轮重复先在场景内平均，不当作12个独立场景；10,000次、seed=20261027。请求改善不能在 M3/模型区间跨零时全部归因于 KV，预处理/模型之外时间单独保留。',
        table(['系统指标','Native','新策略 M123','绝对下降','相对下降'],[(k,number(v['native']),number(v['candidate']),number(v['loss']),number(v['relative_loss'])) for k,v in r['system_quality']['metrics'].items()]),
        table(['场景','mAP loss','NDS loss','Lane DET_l loss','Token F1 loss'],[(scene,*[number(v['metrics'][k]['loss']) for k in ('mAP','NDS','DET_l','Token_F1')]) for scene,v in r['system_quality']['per_scene'].items()]),
        f'四项 .10 预算通过：`{r["system_quality"]["within_quality_budgets"]}`。M3 内部 .02 不变。同初始实际 KV/同输入/相同强制 Token 的真实 Forward 与 logits 控制：`{r["same_input_control"]}`；详情在 same_input_controls.jsonl。强制答案不计质量。',
        f'正式安装状态：`{r["installed"]}`。仅 Map Approx 的既有 critical_regressions 因新增图像审核证据上调；旧 Full Visual 错误记录和 Object Exact 质量拒绝保留，其他规则参数逐项相同。仍只有一个 ReusePlanner/PrefillExecutor/PrefixKVCache，M1/M2默认不变。',
        '历史原策略5607 Tokens和本轮原策略1606 Tokens属于冻结候选复现，不应当作最终收紧策略的收益。最终收紧策略的实际复用与耗时以本文件为准。历史输出完整保留。']
    (root/'experiment_report.md').write_text('\n\n'.join(out)+'\n')


def run(root,stage):
    plan=read(root/'experiment_plan.json')
    if hashlib.sha256((root/'multi_sequence24.json').read_bytes()).hexdigest()!=plan['manifest_sha256'] or hashlib.sha256((root/'frozen_m3_policy.json').read_bytes()).hexdigest()!=plan['policy_sha256']:raise ValueError('Frozen qualification input changed')
    if stage=='control':run_control(root);return
    for kind in ('native','zero','reuse'):service(root,'multi',kind)
    reference(root,'multi')
    for repeat,order in enumerate(plan['orders']):
        for kind in order:service(root,'multi',kind,repeat,True)
    target=root/'multi_after_diagnostic'
    if not target.exists():
        wait_gpu();execute(root,'multi_after_diagnostic',['-m','tools.worldstream_m123.benchmark.worker','--output',str(target),'--run-root',str(root),'--condition','B3','--workload','language',
            '--sequence-file',str(root/'multi_sequence24.json'),'--limit','24','--questions-per-frame','3','--warmup','1','--seed','0','--m3-strategy','wsdr_kv','--m3-policy',str(root/'frozen_m3_policy.json'),'--trace-m3-decisions'])
    if not (root/'same_input_control_summary.json').exists():
        wait_gpu();execute(root,'same_input_controls',['-m','tools.worldstream_m123.m3_recovery_qualification','--root',str(root),'--stage','control'])
    for kind in ('native','reuse'):
        name='multi_free_r0_'+kind
        if not (root/(name+'_official/summary.json')).exists():evaluate(root,name)
    r=analyze(root,install=True)
    write(root.parent/'final_status.json',dict(status='COMPLETED' if r['installed'] else 'NEEDS_SEMANTIC_REVIEW',policy_installed=r['installed'],system_quality_pass=r['system_quality']['within_quality_budgets']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--stage',choices=['all','control'],default='all');a=p.parse_args()
    if a.stage=='control':run(Path(a.root),'control')
    else:
        from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
        with gpu_job_lock():run(Path(a.root),'all')
