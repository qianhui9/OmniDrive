"""Audit recorded artifacts and finish the delivery without rerunning inference."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import numpy as np
import torch
from .experiment import write,stats,typed_digest

MARKER='\n## 最终交付复核\n'


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def trace_summary(path):
    payload=json.loads(Path(path).read_text());events=payload['traceEvents']
    names=Counter(e.get('name','') for e in events)
    bmm=[e.get('args',{}).get('Input Dims') for e in events if e.get('name')=='aten::bmm']
    bmm=[shape for shape in bmm if shape and len(shape[0])==3]
    return dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        cuda_kernel_events=sum(e.get('cat')=='kernel' for e in events),
        regions={k:v for k,v in names.items() if k.startswith('m3_')},
        bmm_query_lengths=sorted({shape[0][-2] for shape in bmm}),bmm_calls=len(bmm))


def audit_frames(run,perception=False):
    rows=lines(run/'frames.jsonl');protocol=json.loads((run/'protocol.json').read_text())
    env=json.loads((run/'environment.json').read_text())['argv']
    expected={(rep,scene['scene'],index,mode)
        for rep in range(env['warmup']+env['repeats']) for scene in protocol['scenes']
        for index in scene['indices'][:env['frames']] for mode in protocol['modes']}
    actual=[(r['repeat'],r['scene'],r['index'],r['mode']) for r in rows]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('Incomplete or duplicate experimental cells')
    raw_verified=0
    for row in rows:
        if len(row['generated_ids'])!=len(row['questions']):raise ValueError('Question skipped or output count changed')
        if any(not 1<len(x[0])<=321 for x in row['generated_ids']):raise ValueError('Invalid output token count')
        if row['components']['prefill']['calls']!=len(row['questions']):raise ValueError('Missing actual prefill call')
        decode_tokens=sum(len(x[0])-2 for x in row['generated_ids'])
        if row['components']['decode']['calls']!=decode_tokens:raise ValueError('Decode call count mismatch')
        si=next(i for i,s in enumerate(protocol['scenes']) if s['scene']==row['scene'])
        fi=protocol['scenes'][si]['indices'].index(row['index'])
        answer_path=run/f'answers/r{row["repeat"]}_{si}_{row["mode"]}_{fi}'/row['token']
        if json.loads(answer_path.read_text())!=row['answers']:raise ValueError('Saved answer mismatch')
        if perception:
            path=run/f'perception/r{row["repeat"]}_{si}_{row["mode"]}_{fi}.pt'
            if typed_digest(torch.load(path,map_location='cpu'))!=row['perception_digest']:
                raise ValueError('Raw dtype/shape/content digest mismatch')
            raw_verified+=1
    return rows,dict(frames=len(rows),measured_frames=sum(not r['warmup'] for r in rows),
        unique_cells=True,answers_and_call_counts_verified=True,raw_perception_files_verified=raw_verified)


def paired_statistics(rows):
    rows=[r for r in rows if not r['warmup']]
    base={(r['repeat'],r['index']):r for r in rows if r['mode']=='baseline'}
    output={}
    for mode in sorted({r['mode'] for r in rows}-{'baseline'}):
        scenes=defaultdict(list)
        for row in rows:
            if row['mode']==mode:scenes[row['scene']].append((base[(row['repeat'],row['index'])],row))
        summary=[]
        for scene,pairs in scenes.items():
            summary.append(dict(scene=scene,cells=len(pairs),
                baseline_wall_mean_ms=float(np.mean([b['wall_ms'] for b,r in pairs])),
                method_wall_mean_ms=float(np.mean([r['wall_ms'] for b,r in pairs])),
                savings_mean_ms=float(np.mean([b['wall_ms']-r['wall_ms'] for b,r in pairs])),
                question_token_equal=sum(x==y for b,r in pairs for x,y in zip(b['generated_ids'],r['generated_ids'])),
                questions=sum(len(r['questions']) for b,r in pairs)))
        b=np.array([s['baseline_wall_mean_ms'] for s in summary]);m=np.array([s['method_wall_mean_ms'] for s in summary])
        # Scene is the resampling unit; repeated frames/replays are not independent seeds.
        rng=np.random.RandomState(73);draw=rng.randint(len(b),size=(4096,len(b)))
        reduction=1-m[draw].mean(1)/b[draw].mean(1)
        output[mode]=dict(scenes=summary,mean_paired_savings_ms=float((b-m).mean()),
            mean_paired_wall_reduction=1-float(m.mean()/b.mean()),
            descriptive_scene_bootstrap_reduction_ci95=np.percentile(reduction,[2.5,97.5]).tolist(),
            statistical_scope='Six observed scenes, one fixed RNG seed, two replays; scene-cluster bootstrap is descriptive, not a road-population guarantee')
    return output


def prefix_statistics(run):
    rows=json.loads((run/'prefix_ablation.json').read_text());result={}
    for mode in ('independent','text_prefix','same_frame_visual'):
        picked=[r for r in rows if r['mode']==mode and not r['warmup']]
        result[mode]=dict(samples=len(picked),wall_ms=stats([r['wall_ms'] for r in picked]),
            reused_tokens=sum(d['reused_tokens'] for r in picked for d in r['requests']),
            effective_prefill_tokens=sum(d['effective_prefill_tokens'] for r in picked for d in r['requests']))
    for mode in ('text_prefix','same_frame_visual'):
        differences=[];argmax_equal=0;questions=0
        for row in rows:
            if row['mode']!=mode or row['warmup']:continue
            baseline=torch.load(run/'logits'/f'{row["token"]}_independent_{row["repeat"]}.pt',map_location='cpu')
            actual=torch.load(run/'logits'/f'{row["token"]}_{mode}_{row["repeat"]}.pt',map_location='cpu')
            for a,b in zip(baseline,actual):
                differences.append(float((a-b).abs().max()));argmax_equal+=int(torch.equal(a.argmax(-1),b.argmax(-1)));questions+=1
        result[mode]['first_token_logits']=dict(questions=questions,argmax_equal=argmax_equal,
            max_abs=max(differences),max_abs_distribution=stats(differences),
            limitation='First-token agreement does not establish full generated-sequence equality')
    result['cross_frame']=dict(observations=[dict(token=r['token'],exact_lcp=r['exact_lcp'],
        current_request_wall_ms=r['requests'][0]['llm_wall_ms']) for r in rows if r['mode']=='cross_frame'],
        limitation='Single-token component probes; no cross-frame E2E speedup claim')
    return result


def finalize(root,supplement=None,test_log=None):
    root=Path(root);status=json.loads((root/'delivery_status.json').read_text())
    run=Path(status['source']);single=Path(status['single_question_source'])
    rows,main_audit=audit_frames(run);single_rows,single_audit=audit_frames(single,perception=True)
    manifest=json.loads((run/'source_manifest.json').read_text());source_checks={}
    for path,digest in manifest.items():
        if path.startswith('worldstream/m3/') or path.endswith('/detectors/petr3d.py'):
            source_checks[path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    if not source_checks or not all(source_checks.values()):raise ValueError('Inference source changed since GPU run')
    test_log=Path(test_log) if test_log else root/'logs'/('clean_tests_'+run.name.removeprefix('clean_multi_')+'.log')
    text=test_log.read_text();passed=int(re.findall(r'(\d+) passed',text)[-1])
    skipped=sum(map(int,re.findall(r'(\d+) skipped',text)))
    if re.search(r'\d+ (failed|errors?)',text):raise ValueError('Regression tests failed')
    paired=paired_statistics(rows);prefix=prefix_statistics(run)
    identical=trace_summary(run/'traces/identical_prefix.json')
    if identical['regions'].get('m3_probe_decode')!=2 or identical['bmm_query_lengths']!=[1]:
        raise ValueError('Identical-prefix trace does not show decode-only execution')
    (root/'traces').mkdir(exist_ok=True)
    shutil.copy2(run/'traces/identical_prefix.json',root/'traces/identical_prefix.json')
    supplemental=None;shared=None
    if supplement:
        supplement=Path(supplement);completion=json.loads((supplement/'completion.json').read_text())
        if not completion['complete']:raise ValueError('Incomplete quantity supplement')
        if hashlib.sha256((supplement/'results.jsonl').read_bytes()).hexdigest()!=completion['raw_file_sha256']:
            raise ValueError('Supplement raw data hash mismatch')
        supplemental=lines(supplement/'results.jsonl')
        protocol=json.loads((supplement/'protocol.json').read_text())
        expected={(s['token'],mode,selector) for s in protocol['candidates'] for mode,selector in
            [('baseline','full'),('exact','full'),('compact','task'),('compact','topk'),('compact','random'),('combined','task')]}
        actual=[(r['token'],r['mode'],r['selector']) for r in supplemental]
        if set(actual)!=expected or len(actual)!=len(expected):raise ValueError('Missing supplemental cells')
        shared=trace_summary(supplement/'traces/shared_visual_prefix.json')
        proof=json.loads((supplement/'shared_visual_prefix_proof.json').read_text())
        if shared['regions'].get('m3_suffix_prefill')!=1 or shared['regions'].get('m3_full_prefill',0)!=0:
            raise ValueError('Shared-prefix trace did not execute exactly one suffix prefill')
        if proof['diagnostics']['forward_prefill_lengths']!=[proof['expected_suffix_tokens']]:
            raise ValueError('Unexpected suffix computation length')
        if shared['bmm_query_lengths']!=sorted({1,proof['expected_suffix_tokens']}):
            raise ValueError('Attention trace query dimensions do not match the suffix')
        shared['proof']=proof
        shutil.copy2(supplement/'traces/shared_visual_prefix.json',root/'traces/shared_visual_prefix.json')
    memory={}
    for mode in sorted({r['mode'] for r in rows}):
        picked=[r for r in rows if r['mode']==mode and not r['warmup']]
        variation=[]
        for index in {r['index'] for r in picked}:
            peaks=[r['peak_allocated'] for r in picked if r['index']==index]
            variation.append(max(peaks)-min(peaks))
        memory[mode]=dict(min_allocated_gib=min(r['peak_allocated'] for r in picked)/2**30,
            max_allocated_gib=max(r['peak_allocated'] for r in picked)/2**30,
            max_same_frame_repeat_peak_delta_mib=max(variation)/2**20,
            actual_reused_cameras=sum(r['m2'].get('actual_reused_cameras',0) for r in picked))
    review=dict(main=main_audit,single=single_audit,inference_source_matches_gpu_snapshot=source_checks,
        regression=dict(passed=passed,skipped=skipped,log=str(test_log)),
        profiler=dict(identical=identical,shared_visual=shared),memory_stability=memory,
        all_frozen_quality_targets_pass=False,decision='NO_GO for online optimization; keep Auto full fallback')
    write(root/'delivery_audit.json',review);write(root/'paired_statistics.json',paired)
    write(root/'prefix_statistics.json',prefix)
    if supplemental is not None:
        write(root/'quantity_results.json',dict(source=str(supplement),rows=supplemental,
            scope='Supplementary original questions; no added primary E2E samples or quality certification',
            baseline_wall_scope='full VLM',candidate_wall_scope='LLM only; wall values cannot be compared directly'))
    append_report(root,paired,prefix,review,supplemental)
    status.update(status='finalized_gpu_protocol_quality_no_go',delivery_audit_complete=True,
        final_gpu_experiments_complete=True,quantity_supplement=str(supplement) if supplement else None,
        tests_passed=passed,tests_skipped=skipped,auto_approved=False,compact_approved=False,
        quality_gate='NO_GO',satisfies_frozen_quality=False)
    status.pop('current_blocker',None)
    write(root/'delivery_status.json',status)
    return review


def append_report(root,paired,prefix,review,supplemental):
    report=root/'experiment_report.md';text=report.read_text().split(MARKER)[0]
    quality=json.loads((root/'quality_results.json').read_text())['paired_e2e']
    exact=quality['exact']
    compact=json.loads((root/'compact_input_results.json').read_text())['summary']
    task=compact['compact_task_0.5']
    raw_task=[r for r in lines(Path(json.loads((root/'delivery_status.json').read_text())['source'])/'llm_probes.jsonl')
        if r['mode']=='compact' and r['selector']=='task']
    selected=sum(d['visual_tokens']<513 for r in raw_task for d in r['diagnostics']['requests'])
    diagnoses=json.loads((root/'exact_failure_diagnosis.json').read_text())
    diagnosis_text=[]
    for d in diagnoses:
        step=d['first_divergent_output_step'];v=d['divergence_logits']
        diagnosis_text.append(f'样本 {d["token"]} 的回答首次在输出索引 {step}（第 {step+1} 个生成 Token）分歧。'
            f'Prefill logits MaxAbs={d["prefill_logits"]["max_abs"]}；分歧位置 MaxAbs={v["max_abs"]}，'
            f'Baseline 前两候选 logits 为 {v["reference_top_values"][0][:2]}，缓存路径为 {v["cached_top_values"][0][:2]}，'
            f'首选 Token 从 {v["reference_top_ids"][0][0]} 变为 {v["cached_top_ids"][0][0]}。'
            f'相同输入前缀的第 0 层 K 已有 MaxAbs={d["prefix_kv_layer_differences"][0]["key_max_abs"]} 差异。')
    diagnosis_text.append('结果与不同 Prefill Shape 下 FP16 舍入变化一致；没有单独隔离具体 cuBLAS 算法，因此不将算法归因写成已证明事实。')
    conclusion='**交付结论：工程实现及本轮真实 GPU 实验已完成；优化未通过冻结的逐 Token 质量要求，正式 Auto 继续使用完整输入和原始生成路径。表中加速是离线实测，不能称为已通过质量约束的上线收益。**\n\n'
    if conclusion not in text:text=text.replace('## 协议与边界',conclusion+'## 协议与边界',1)
    out=[MARKER,'',f'完整主实验 {review["main"]["frames"]} 个帧执行记录、其中 {review["main"]["measured_frames"]} 个计时帧；单问题实验 {review["single"]["frames"]} 个记录。逐一核对原始回答文件、每问题 Prefill/Decode 调用次数，以及 {review["single"]["raw_perception_files_verified"]} 个原始 dtype/shape/内容感知与 Memory 文件。GPU 回归 {review["regression"]["passed"]} 项通过、跳过 {review["regression"]["skipped"]}；推理源码与本次 GPU 快照逐文件 SHA256 一致。','',
        '### 文本前缀与视觉前缀的独立消融','',
        '| 预填充探针 | 样本数 | 五问题累计 wall P50/P95 ms |','|---|---:|---:|']
    for mode in ('independent','text_prefix','same_frame_visual'):
        r=prefix[mode];out.append(f'| {mode} | {r["samples"]} | {r["wall_ms"]["p50"]:.2f}/{r["wall_ms"]["p95"]:.2f} |')
    out+=['','探针仅生成一个 Token，隔离 Prefill 并包括匹配、复制、缓存捕获及 logits GPU→CPU 转移；它不是完整服务延迟。35 个纯文本 Token 的复用在该测量中未获净收益；同帧长视觉前缀约三倍 Prefill 收益与主实验一致。跨帧六个观测均仅命中 35 个纯文本 Token，不推断为视觉复用。',
        f'同帧前缀探针首 Token argmax 一致 {prefix["same_frame_visual"]["first_token_logits"]["argmax_equal"]}/{prefix["same_frame_visual"]["first_token_logits"]["questions"]}；完整生成的逐 Token 一致为 {exact["token_ids_equal"]}/{exact["questions"]}。','',
        '### 数值分歧与质量准入','',
        ' '.join(diagnosis_text),
        f'GT Token F1 略升不构成更高驾驶质量的证明。任务选择 {task["questions"]} 问离线对照中只压缩 {selected} 问，整体 Carrier 减少 {100*task["carrier_reduction"]:.2f}%；不能将其误报为所有问题减少一半。数量、地图、安全和关系任务的质量必须分别看原始回答，Token F1 不替代语义或安全认证。','',
        '### 保序压缩对照','',
        '| 方法 | 问题数 | 实际视觉 Token 减少 | 输出 Token 一致 | 原始 GT Token F1 |',
        '|---|---:|---:|---:|---:|']
    for label in ('full_full_1.0','compact_task_0.5','compact_topk_0.75','compact_topk_0.5','compact_random_0.5','combined_task_0.5'):
        r=compact[label];out.append(f'| {label} | {r["questions"]} | {100*r["carrier_reduction"]:.2f}% | {r["token_ids_equal"]}/{r["questions"]} | {r["mean_gt_token_f1"]:.4f} |')
    out+=['',f'GT F1 仅有 {len(task["f1"])} 个带原始文本 GT 的问题；其他问题保持缺失，未按零分或满分填充。单问题部署所有方法实际复用 KV 均为 0，时延微小波动不能归因为跳算。','',
        '### 原始任务与轨迹 GT','',
        '| 方法 | 关系问题 Token 一致 | 地图问题 Token 一致 | 全局/安全问题 Token 一致 |',
        '|---|---:|---:|---:|']
    for mode in ('m12','compact','exact','combined','m123'):
        cells=[]
        for category in ('object_relation','map','global_safety'):
            r=quality[mode]['tasks'].get(category,dict(token_ids_equal=0,questions=0))
            cells.append(f'{r["token_ids_equal"]}/{r["questions"]}')
        out.append('| '+mode+' | '+' | '.join(cells)+' |')
    qa=json.loads((root/'qa_ground_truth_analysis.json').read_text())
    out+=['','轨迹 GT 直接按 sample token 读取原始验证集及有效 mask；不从生成文本反构 GT。','',
        '| 方法 | 有效轨迹回答数 | ADE m | FDE m |','|---|---:|---:|---:|']
    for r in qa['summary']:
        if not r['warmup'] and r['planning_samples']:
            out.append(f'| {r["mode"]} | {r["planning_parsed"]}/{r["planning_samples"]} | {r["planning_ade_m"]:.6f} | {r["planning_fde_m"]:.6f} |')
    out+=['','这是生成的六步轨迹误差，不是碰撞率或闭环驾驶评价；整体安全类分组中部分短动作问题没有文本 GT，不能据 Token 一致给出语义正确率。','',
        '### 统计与显存','',
        '| 方法 | 按场景配对平均 wall 节省 ms | 平均 wall 降幅 | 场景聚类 bootstrap 描述性 95% 区间 |',
        '|---|---:|---:|---:|---:|']
    for mode in ('m12','compact','exact','combined','m123'):
        r=paired[mode];lo,hi=r['descriptive_scene_bootstrap_reduction_ci95']
        out.append(f'| {mode} | {r["mean_paired_savings_ms"]:.2f} | {100*r["mean_paired_wall_reduction"]:.2f}% | {100*lo:.2f}%–{100*hi:.2f}% |')
    out+=['','区间以六个场景为重采样单元，两个重放不被当作不同随机种子；它只描述本次六场景，不外推到道路总体。配对均值与前表 P50 比例是不同统计量，不混用。']
    m=review['memory_stability']['exact'];out+=[f'修复后的 Exact 计时峰值 allocated 为 {m["min_allocated_gib"]:.3f}–{m["max_allocated_gib"]:.3f} GiB，同一帧两次重放峰值最大差 {m["max_same_frame_repeat_peak_delta_mib"]:.3f} MiB；本轮完整结束，没有复现 run03 的逐模式 KV 累积/OOM。真实移动场景 M2 复用 Camera 计数为 {review["memory_stability"]["m123"]["actual_reused_cameras"]}，不将 M3 的语言收益归因为视觉 Cache 命中。','',
        '### Profiler 与数量类补测','']
    t=review['profiler']['identical'];out+=[f'完全相同输入的真实 Profiler 只记录 {t["regions"].get("m3_probe_decode",0)} 次 Decode，所有 bmm query length 为 1，没有 Transformer Prefill；证明加载 Cache 后确实省略了 Prefill。']
    if supplemental is not None:
        t=review['profiler']['shared_visual'];p=t['proof']
        out+=[f'不同原始问题的视觉共享 trace 命中 {p["actual_reused_tokens"]} Token，实际只执行 {p["expected_suffix_tokens"]} Token 后缀；m3_suffix_prefill 记录为 {t["regions"]["m3_suffix_prefill"]}，m3_full_prefill 为 0。原 trace 在 traces/shared_visual_prefix.json，未把原始公共前缀重新送入 Transformer。',
            '补测 4 个实际 nuScenes 帧的原始数量类问题，GT 仅用于生成后的评价；不加入冻结的主实验时延表。','',
            '| 方法 | 数量问题 Token 一致 | 原始 GT Token F1 |','|---|---:|---:|']
        for mode,selector in (('exact','full'),('compact','task'),('compact','topk'),('compact','random'),('combined','task')):
            picked=[r for r in supplemental if r['mode']==mode and r['selector']==selector]
            equal=sum(r['token_ids_equal'][-1] for r in picked)
            f1=[r['quality'][-1]['token_f1'] for r in picked if r['quality'][-1] is not None]
            out.append(f'| {mode}/{selector} | {equal}/{len(picked)} | {float(np.mean(f1)):.4f} |')
        out+=['','数量 GT 及原始回答见 quantity_results.json；没有结构化明确数量的答案不强行转换为数值精度。该补测是质量覆盖，不比较 baseline 的 VLM wall 和候选的 LLM-only wall。']
    out+=['','### 决策与后续研究边界','',
        '1. 当前部署保持 Full 回退，cost_table.json 和 quality_certificate.json 的 approved 均为空；没有放宽冻结质量目标。',
        '2. 后续若研究数值稳定的前缀执行，需要独立验证后端/计算 Shape 与完整解码等价，不能沿用本轮失败的质量结论直接批准。',
        '3. 紧凑 Carrier 应先建立可验证的语义映射或单独适应性证据；当前无可靠 Anchor-to-query 映射，不能据 M1 ID 删 Token。',
        '4. 所有原始失败实验保留；本轮完整结果、补测和描述性统计分别归档，未覆写 M1/M2 交付。','']
    report.write_text(text+'\n'.join(out))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='outputs/module3')
    parser.add_argument('--supplement');parser.add_argument('--test-log');args=parser.parse_args()
    finalize(args.output,args.supplement,args.test_log)


if __name__=='__main__':main()
