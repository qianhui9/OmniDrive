"""Render measured evidence; all acceptance fields come from raw task scores."""
import argparse
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np
from tools.worldstream_m2.global_residual_benchmark import read, rows
from tools.worldstream_m2.global_residual_analysis import write
from tools.worldstream_m123.benchmark.metrics import distribution
from tools.worldstream_m123.latency_quality import summarize


def fmt(x,d=3):return '未定义' if x is None else f'{x:.{d}f}'
def latency(d):return ' / '.join(fmt(d[k]) for k in ('mean','p50','p95'))


def report(root):
    summarize(root)
    a=read(root/'summary.json');q=a['independent_quality'];n=a['aggregate']['native'];f=a['aggregate']['formal']
    m1=read(root/'m1_same_state_control.json');dec=read(root/'decode_control/summary.json');diag=read(root/'m3_rejection_diagnostics.json')
    historical=read(root/'historical_quality_reassessment/reassessment.json')
    curve=read(root/'calibration_curve.json');plan=read(root/'experiment_plan.json')
    pure=read(root/'pure_visual/results.json');pp=pure['natural']
    pure_stats={k:distribution(t for p in pp for t in p[v]['wall_samples_ms']) for k,v in [('native','native'),('formal','incremental')]}
    dev_preprocessing={name:float(np.mean([f['preprocessing_ms'] for f in rows(root/name/'per_frame_metrics.jsonl')])) for name in ('cal_native',a['selection']['name'])}
    executed=[r for r in diag['heldout_natural']['records'] if r['accepted_reuse_tokens']]
    gpu_test_summary=next(line.strip() for line in reversed((root/'tests_gpu_final.log').read_text().splitlines()) if ' passed' in line)
    kv_text=f"本轮自然独立任务发生 {len(executed)} 次合法 KV 复用、合计 {diag['heldout_natural']['actual_reused_tokens']} token。"
    if executed:
        e=executed[0]
        kv_text+=f" 首个案例：场景 {e['scene_id']}，当前帧 {e['frame_id']}，源帧 {e['cache_source_frame']}；输入 {e['input_tokens']} token，复用 {e['accepted_reuse_tokens']}，实际 suffix prefill {e['effective_prefill_tokens']}；age={e['cache_age_frames']}、depth={e['approx_reuse_depth']}、risk={e['estimated_quality_risk']:.6f}。输入落入既有风险及实测长度范围。"
    kv_text+=' 策略与阈值未改，复用情况随本轮自然输入变化，不能归因于调度算法修改。'
    write(root/'pure_visual_summary.json',dict(all_frames_including_cold=True,frames=len(pp),timing=pure_stats,
        saved_ms=pure_stats['native']['mean']-pure_stats['formal']['mean'],scope=pure['timing_scope']))
    lines=['# M1+M2+M3：0.10 任务预算与端到端时延实验','',
        f"四项独立质量预算：**{'通过' if a['quality_pass'] else '未通过'}**。完整请求加速比 **{a['service_speedup']:.4f}×**；"+
        ('三轮及逐场景获得净收益。' if a['stable_net_speedup'] else '**未证明稳定净加速**，不得把视觉收益或预算放宽解释为完整服务加速成功。'),'',
        f"开发集固定选择候选 AR{a['selection']['interval']} / drift={a['selection']['threshold']:.12f}。"+
        ('独立预算通过；实际正式参数以源码和 deployment_decision.json 的记录为准。复现工具只产生证据，不修改源码默认值。' if a['quality_pass'] else '**独立质量失败，候选不发布；保留原 AR2 / 0.7721027880907059 正式参数，不用独立集再次调参。以下 Formal 计时均指冻结的失败候选，不是已发布正式默认值的性能。**'),'',
        '## 质量标准及数据','',
        '四项默认预算由 0.05 改为 0.10，逐项检查 native−candidate≤0.10，包含等号；提升不能抵消另一项超标。这是绝对 10 个百分点，非相对 10%。M3 内部 WSDR-KV quality_budget=0.02 与策略文件字节保持不变。质量预算只用于离线选择和验收。','',
        '真实 checkpoint：`ckpts/iter_10548.pth`；六相机预处理、权重、必须刷新事件、ordinary_budget=3、max_reuse_frames=8、前两块窗口选择、依赖闭包及 HF 生成参数保持原值。','',
        '开发：2 个新场景×8 连续帧；独立：4 个新场景×6 连续帧。来自未参与 run01/run02/run03 的场景，按场景 ID 排序取固定数量连续帧，没有按模型输出筛选。每帧一个有真实标签的问题。这里是小规模子集任务评分，不能代替完整 nuScenes 验证集。场景/帧/时间/问题清单见 development_sequence.json、validation_sequence.json 及各运行的 per_frame_metrics.jsonl、per_question_metrics.jsonl。','',
        '开发场景：`'+'`, `'.join(plan['development_scenes'])+'`。','',
        '独立场景：`'+'`, `'.join(plan['validation_scenes'])+'`。','',
        '选择规则在完成开发输出前固定：'+plan['selection']+'。只比较同一算法的 2/3/4 三个间隔、已有漂移阈值 0.9120883047580719；没有使用额外漂移阈值。独立结果不参与选择。','',
        '## 历史结果只读重评','',
        'historical_quality_reassessment/ 内重评 run03 已有预测，quality_budget=0.10，new_inference_performed=false。原始证据 SHA256 和旧报告保留；这不是新的独立验证。旧完整服务仍为 0.9851×，性能结论没有改变。','',
        '| 指标 | 旧 Native | 旧 AR4 | 绝对下降 | 相对下降 | 新预算通过 |','|---|---:|---:|---:|---:|---|']
    h=historical['validation'] if 'validation' in historical else historical.get('independent_validation')
    if h is None:
        h=next(v for k,v in historical.items() if isinstance(v,dict) and 'metrics' in v and 'validation' in k)
    if 'quality' in h:h=h['quality']
    for k,v in h['metrics'].items():lines.append(f"| {k} | {fmt(v['native'],6)} | {fmt(v['candidate'],6)} | {fmt(v['loss'],6)} | {fmt(v.get('relative_loss',v['loss']/v['native'] if v['native'] else None),6)} | {v['loss']<=.10} |")
    lines += ['', '## 开发曲线','', '| 间隔 | 完整请求 mean ms | 视觉 span mean ms | 后缀复用帧 | mAP/NDS/DET_l/F1 绝对下降 | 四项通过 |','|---:|---:|---:|---:|---|---|']
    for p in curve['candidates']:
        lines.append(f"| {p['interval']} | {fmt(p['timing']['request_ms']['mean'])} | {fmt(p['timing']['visual_cuda_span_ms']['mean'])} | {p['timing']['global_suffix_reuse_frames']} | "+' / '.join(fmt(p['quality']['metrics'][k]['loss'],6) for k in ('mAP','NDS','DET_l','Token_F1'))+f" | {p['quality']['within_quality_budgets']} |")
    lines += ['', '[开发质量—时延曲线](quality_latency_curve.svg)。calibration_curve.json 保留每个开发场景的质量和完整服务时延。'+f"开发阶段预处理存在较大波动（Native {dev_preprocessing['cal_native']:.1f} ms，选中候选 {dev_preprocessing[a['selection']['name']]:.1f} ms）；"+'因此开发加速比不能全归因于模型，正式结论使用下面的三轮交替实测，不用独立数据反向重选。','',
        '## 新独立精度','', '| 指标 | Native | 正式候选 | 绝对下降 | 相对下降 | 通过 |','|---|---:|---:|---:|---:|---|']
    for k,v in q['metrics'].items():lines.append(f"| {k} | {fmt(v['native'],6)} | {fmt(v['candidate'],6)} | {fmt(v['loss'],6)} | {fmt(v['relative_loss'],6)} | {q['metric_budget_pass'][k]} |")
    lines += ['', '相对下降是 (Native−候选)/Native；负值表示改善。以下逐场景指标只作透明诊断，不新增质量硬门槛。','',
        '| 独立场景 | mAP native→formal | NDS native→formal | DET_l native→formal | F1 native→formal | 各项绝对下降 |','|---|---|---|---|---|---|']
    for s,v in q['per_scene'].items():
        metrics=v['metrics'];lines.append('| '+s+' | '+' | '.join(fmt(metrics[k]['native'],4)+'→'+fmt(metrics[k]['candidate'],4) for k in ('mAP','NDS','DET_l','Token_F1'))+' | '+' / '.join(fmt(metrics[k]['loss'],6) for k in ('mAP','NDS','DET_l','Token_F1'))+' |')
    pairs=read(root/'r0_formal_quality_pairs.json');counts=Counter()
    for p in pairs:
        b,c=p['baseline_task_metrics'],p['task_metrics']
        for kind,keys in {'numeric':['numeric_exact','numeric_value_precision','numeric_value_recall'],
            'spatial':['spatial_label_match','spatial_label_f1'],'motion':['motion_label_match','motion_label_f1'],'safety':['safety_label_match','safety_label_f1']}.items():
            if any(b.get(k) is not None and c.get(k) is not None and c[k]<b[k]-1e-12 for k in keys):counts[kind]+=1
    write(root/'critical_question_summary.json',dict(regressed_questions_by_type=dict(counts),events=q['critical_proxy_regressions_vs_native'],scope='Lexical proxies; overlapping categories; complete examples in r0_formal_quality_pairs.json'))
    lines += ['', f"数字/空间/运动/安全词法代理出现退化的问题数：{dict(counts)}；退化事件数 {q['critical_proxy_regressions_vs_native']}。逐问原生答案、正式答案、GT、数字值和各类匹配保留在 r0_formal_quality_pairs.json；代理指标不等于人工语义或安全判断。",'',
        '## 三轮交替完整服务','',
        '顺序：第 0/2 轮 Native→Formal，第 1 轮 Formal→Native。每轮相同 24 帧、4 个独立场景，warmup 1 后计时。三轮是重复测量，不是 12 个独立场景。所有正常负收益请求保留。','',
        '| 条件 | 请求 mean/P50/P95 ms | 吞吐 fps | 生成 token 总数 | 复用 KV token | 峰值 allocated/reserved GiB |','|---|---|---:|---:|---:|---|']
    for kind,d in [('Native',n),('Formal',f)]:lines.append(f"| {kind} | {latency(d['request_ms'])} | {fmt(d['throughput_fps'],5)} | {d['generated_tokens']} | {d['kv_reused_tokens']} | {fmt(d['peak_allocated_bytes']/2**30)} / {fmt(d['peak_reserved_bytes']/2**30)} |")
    lines += ['', '| 条件 | 预处理 mean/P50/P95 ms | transfer/collate mean/P50/P95 ms | 模型 mean/P50/P95 ms |','|---|---|---|---|']
    for kind,d in [('Native',n),('Formal',f)]:lines.append(f"| {kind} | {latency(d['preprocessing_ms'])} | {latency(d['transfer_and_collate_ms'])} | {latency(d['model_ms'])} |")
    lines += ['', '三轮重复输出一致性（任务摘要/回答）：'+str({k:(v['repeats_output_equal'],v['repeats_answers_equal']) for k,v in a['aggregate'].items()})+'。独立精度对第 0 轮评分，未把重复帧当作新增独立精度样本。']
    lines += ['', '| 同一独立请求分项 CUDA span | Native mean/P50/P95 ms | 候选 mean/P50/P95 ms |','|---|---|---|']
    for label,ns,fs in [('EVA/M2 wrapper','eva_native','m2_total'),('Detection','detection','detection'),('Map','map','map'),('Prefill','prefill','prefill'),('Decode','decode','decode')]:
        lines.append(f"| {label} | {latency(n['stages'][ns])} | {latency(f['stages'][fs])} |")
    lines += ['', '| 轮次 | Native mean ms | Formal mean ms | 净节省 ms | 加速比 |','|---:|---:|---:|---:|---:|']
    for i,p in enumerate(a['rounds']):lines.append(f"| {i} | {fmt(p['native']['request_ms']['mean'])} | {fmt(p['formal']['request_ms']['mean'])} | {fmt(p['net_saved_ms'])} | {fmt(p['speedup'],5)} |")
    ci=a['paired_round_saved_ms_ci95'];lines += ['',f"E2E_net_saved_ms={fmt(ci['mean'])}，重复轮均值 t(df=2) 95% 区间 [{fmt(ci['lower'])}, {fmt(ci['upper'])}] ms。{a['normal_negative_samples']}/72 个正常配对请求为负收益；详见 paired_requests.json。",'',
        '| 场景 | 配对净节省 mean/P50/P95 ms |','|---|---|']
    for s,d in a['per_scene_paired_saved_ms'].items():lines.append(f'| {s} | {latency(d)} |')
    lines += ['', '## 性能归因','',
        '以下 Host wall 与 CUDA event span 可能嵌套、包含排队和同步等待，不能逐项相加。真正净节省仅来自上面的配对完整请求 wall。Cache publication 主机等待不能全部解释为复制。','',
        f"同 Snapshot 的视觉控制包含首帧及所有正常负收益帧，8 帧×3 次交替：Native {latency(pure_stats['native'])} ms；WSGRR-AR {latency(pure_stats['formal'])} ms；M2_visual_saved_ms={fmt(pure_stats['native']['mean']-pure_stats['formal']['mean'])}。该边界含当前 M2 管理，排除 M1 precheck/read 与下游任务。",'',
        '| 冻结候选服务分项 | mean/P50/P95 ms |','|---|---|']
    for key in ('visual_cuda_span_ms','m1_wall_ms','snapshot_wall_ms','m2_wall_ms','cache_publication_host_ms','m3_wall_ms','prefill_cuda_ms','decode_cuda_ms'):
        d=distribution(r['formal'][key]['mean'] for r in a['rounds'])
        allf=[r for i in range(3) for r in rows(root/f'r{i}_formal/per_frame_metrics.jsonl')]
        if key=='m1_wall_ms':d=distribution(r['m1_commit_timings_ms']['m1_total_including_adapters_ms'] for r in allf)
        elif key=='snapshot_wall_ms':d=distribution(r['m2']['M1_M2_state_read_ms'] for r in allf)
        elif key=='m2_wall_ms':d=distribution(r['m2']['M2_total_ms'] for r in allf)
        elif key=='cache_publication_host_ms':d=distribution(r['m2']['M2_cache_publication_ms'] for r in allf)
        elif key=='m3_wall_ms':d=distribution(r['m3']['runtime_host_elapsed_ms'] for r in allf)
        elif key in ('visual_cuda_span_ms','prefill_cuda_ms','decode_cuda_ms'):
            stage={'visual_cuda_span_ms':'m2_total','prefill_cuda_ms':'prefill','decode_cuda_ms':'decode'}[key];d=f['stages'][stage]
        lines.append(f'| {key} | {latency(d)} |')
    for stage,d in f['stages'].items():
        if stage in ('detection','map') or 'head' in stage or stage.startswith('m2_') and not stage.startswith(('m2_window_block_','m2_suffix_block_')):lines.append(f'| CUDA {stage} | {latency(d)} |')
    allf=[r for i in range(3) for r in rows(root/f'r{i}_formal/per_frame_metrics.jsonl')]
    attribution=dict(M2_visual_saved_ms=pure_stats['native']['mean']-pure_stats['formal']['mean'],
        M2_service_visual_span_saved_ms=n['stages']['eva_native']['mean']-f['stages']['m2_total']['mean'],
        M1_state_added_ms=float(np.mean([r['m1_commit_timings_ms']['m1_total_including_adapters_ms'] for r in allf])),
        M2_management_added_ms=float(np.mean([r['m2']['M2_validation_ms']+r['m2']['M2_region_scoring_ms'] for r in allf])),
        M3_prefill_saved_ms=n['stages']['prefill']['mean']-f['stages']['prefill']['mean'],
        M3_decode_per_token_ms=dec['summary']['after']['per_token_cuda_span_ms']['mean'],E2E_net_saved_ms=ci['mean'],
        scopes=dict(M2_visual_saved_ms='Same-snapshot complete visual boundary including M2 management; separate 8-frame control',
            M2_service_visual_span_saved_ms='Same heldout 24-frame three-pair native EVA versus M2 wrapper CUDA spans, includes host gaps and management; not a pure kernel saving',
            M1_state_added_ms='Formal CPU state/adapters; native has no M1',M2_management_added_ms='Observed validation + region-scoring host spans only, lower-bound proxy, excludes execution enqueue and publication GPU waits',
            M3_prefill_saved_ms='Native minus formal CUDA spans under free generation; not controlled prefill algorithm speedup',
            M3_decode_per_token_ms='Fixed actual token sequence control',E2E_net_saved_ms='Paired full request wall; never sum other fields'))
    write(root/'latency_attribution.json',attribution)
    lines += ['', '明确归因字段见 latency_attribution.json：`'+json.dumps({k:v for k,v in attribution.items() if k!='scopes'})+'`。其中 M2_management_added_ms 仅指已观测的 validation+region-scoring Host 段，是管理成本的下界代理，不包含执行入队等未独立分离的管理开销；不能与视觉净收益再次相加。']
    skipped=sum(p['formal']['global_suffix_skipped_blocks'] for p in a['rounds']);reuse=sum(p['formal']['global_suffix_reuse_frames'] for p in a['rounds'])
    lines += ['', f"实际 suffix hooks 验证：三轮 {reuse}/72 帧跳过后续 22 个 EVA Block，共 {skipped} 次调用；窗口复用、更新集合及原子发布审计见 formal_execution_audit.json，逐帧计数见原始 JSONL。",'',
        '## M1 同状态因果控制','',
        '4 个真实帧×3 次交替回放，输入状态、感知输出和 M2 回执完全一致。关联结果、锚点和导出 Snapshot（仅去除计时字段）全部一致。另将优化前/后整条真实请求在同样 AR4 参数下比较：4 帧的任务输出摘要、自由生成答案及 token IDs 全部相同，见 real_request_equivalence.json；Profiler 时延不参与加速结论。摘要没有跨事务缓存，每次完整绑定全部可变元数据、数组内容和 Tensor 身份/版本；异常和事务一致性检查仍保留。','',
        '| M1 CPU 同状态控制 | 优化前 mean/P50/P95 ms | 优化后 mean/P50/P95 ms |','|---|---|---|']
    for key in ('precheck_ms','snapshot_ms','commit_ms','total_ms'):lines.append(f"| {key} | {latency(m1['summary']['before'][key])} | {latency(m1['summary']['after'][key])} |")
    old=m1['summary']['before']['total_ms']['mean'];new=m1['summary']['after']['total_ms']['mean']
    lines += ['', f"M1_state_added_ms 是正式服务相对无 M1 的新增维护开销，不等于优化节省。可归因的 M1 自身优化节省为 {fmt(old-new)} ms（{100*(old-new)/old:.2f}%）。",'',
        '优化：来源摘要一次完整树遍历，移除重复 geometry/certificate/descriptor 展开；同一请求共享灰度图；patch 支持中的 float 归一化只做一次。未删除世界状态事件。fork/freeze/完整历史验证在 Profile 中占比较小，保留现有语义。M2 窗口集合输入的完整 Snapshot 对照一致。','',
        '## M3 KV 触发、拒绝诊断与 Decode','',
        kv_text,'',
        f"4 个诊断开发帧中 {diag['candidate_decisions']} 次产生候选、共 {diag['candidate_count']} 个候选；决定原因 {diag['decision_reasons']}。逐候选第一拒绝原因 {diag['first_rejection_counts']}。实际复用 {diag['actual_reused_tokens']} token，实际 prefill 长度 {diag['effective_prefill_lengths']}。",'',
        '三轮自然独立请求的候选请求数/候选总数：'+str(diag['heldout_natural']['candidate_requests'])+'/'+str(diag['heldout_natural']['candidate_count'])+'；决定原因 '+str(diag['heldout_natural']['decision_reasons'])+'。全部候选长度、精确/结构连续长度、实际 prefill、来源及 frame/root/age/depth 保留在 m3_rejection_diagnostics.json 的 heldout_natural 中。','',
        '首次请求正常建缓存。后续缓存来源和策略绑定有效，精确前缀较短；近似候选受既有视觉 token 上限、连续因果前缀和校准范围约束。既有 full_visual 策略包含 critical regression，且未提供 approx 证据；4 帧诊断中 map_relations 的输入长度为 571，超出已测范围。object→ego→map 中前部 object 变化，不能跳过它而拼接后面的 map KV。未放宽内部 0.02 策略、没有强制产生非零复用。完整候选、视觉 RMS/最大变化、来源与 root/age/depth 见 m3_rejection_diagnostics.json。自然服务的决定原因及 prefix 长度在各逐问记录中。','',
        f"固定 Decode：相同实际视觉/语言输入、同一真实 token 序列、同一完整 prefill 初始 KV、相同 FP16、{dec['steps']} 步、5 次交替重复；每一步 logits bitwise equal，初始 KV 的版本未变。该控制只有 teacher forcing，没有改变正式自由生成逻辑。",'',
        '| 固定 token 控制 | 原实现 mean/P50/P95 ms | 优化后 mean/P50/P95 ms |','|---|---|---|',
        '| 单 token CUDA span | '+latency(dec['summary']['before']['per_token_cuda_span_ms'])+' | '+latency(dec['summary']['after']['per_token_cuda_span_ms'])+' |',
        '| 固定步数总 wall | '+latency(dec['summary']['before']['total_wall_ms'])+' | '+latency(dec['summary']['after']['total_wall_ms'])+' |','',
        'Decode 改动只消除每步对完整 attention mask 的 cumsum/masked_fill，改用请求内 GPU prompt count 与追加 token 数计算同一 position_id。保持 HF EOS、采样、停止条件及生成上限。对照均值差极小且 P95 未改善，**不能宣称可靠的 Decode 加速**。正常自由生成 token 数有差异，所以自由生成 Decode 总时延差不当作单 token 算法加速。','',
        '真实 GPU trace 中主要是 FP16 矩阵乘；KV concat、attention、归一化和逐 token kernel 启动也有开销。类别 kernel 累计 ms：`'+json.dumps(diag['decode_kernel_profile']['category_kernel_ms'])+'`。内核累计与 Host wall 不可相加；采样/同步及全请求 trace 在 before_profile/、after_profile/，固定序列 trace 在 decode_control/。剩余主要开销是逐 token 大模型 forward，而 M1 真实图像验证/关联也继续占用 CPU。','',
        '## 单主线、测试及复现','',
        '`M1 Precheck → M2 WSGRR → DependencyPlanner → WindowExecutor → Global-State Update → StateCache Commit → Detection/Map → M1 Commit → M3 PrefillExecutor → Decode`。正式路径仍只有这一条；本次新增工具均在 tools/benchmark，未新增备用模型、在线 M2 CostModel 门控或多执行器。','',
        'GPU 回归：'+gpu_test_summary+'。日志 tests_gpu_final.log，覆盖 tests/worldstream_m1、m2、m3、m123；包括 source tamper、只读 Snapshot、事务、真实 CUDA 缓存及 0.10 包含等号的预算判断。新测试另覆盖 Tensor 替换/原地修改及 padding 下逐步位置和 logits 一致性。','',
        '复现入口：`taskset -c 0-3,6-31 bash '+str(root)+'/reproduce.sh <新的输出目录>`。所有 Python 使用 /home/qh/miniconda3/envs/OmniDrive/bin/python；CPU 4/5 不使用。GPU 任务串行锁覆盖 checkpoint 加载，GPU 被其他任务占用时等待，未终止其他用户进程。commands.json 和独立诊断日志保留真实失败与退出码；曾并发加载两个本任务模型，基线进程以 -9 结束（疑似内存压力，未读取系统 OOM 证据），未生成有效测量，已保留 *_load_failure 并重跑。','',
        '## 仪表变更记录','', '| 仪表变更 | 用途 | 对正式三轮时延的影响 |','|---|---|---|',
        '| 可选 cProfile + CPU/CUDA Profiler | before/after 第 1 帧诊断 | 正式三轮不启用，诊断时延排除 |',
        '| M1 状态与真实 M3 输入/token 捕获 | 同状态及固定序列控制 | 正式三轮不启用；诊断序列计时外落盘 |',
        '| 原有 Observer suffix hooks / event spans | 验证真实执行及分项 | Native/Formal 统一，原始请求 wall 为最终口径 |',
        '| concrete function Profile 汇总 | 避免同名包装函数嵌套重复计数 | 离线读取 cpu.prof，不影响运行 |',
        '| 进程级 GPU 实验锁 | 避免空闲检查后并发加载 | 只在实验进程启动/等待，不在请求边界内 |','',
        '原始 Tensor 捕获、大型 trace、预测文件保留在服务器本目录。raw_artifacts_sha256.json 记录原始文件校验；Git 排除大型 Tensor/trace 和详细状态转储，原始逐帧、逐问指标、评分和源码快照保留。']
    (root/'experiment_report.md').write_text('\n'.join(lines)+'\n')
    print('REPORT',root/'experiment_report.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);a=p.parse_args();report(Path(a.root))
