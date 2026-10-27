"""Generate a claim-scoped Chinese report from completed immutable raw runs."""
import argparse
from pathlib import Path
import numpy as np
from tools.worldstream_m2.global_residual_benchmark import read,rows
from tools.worldstream_m2.global_residual_analysis import write
from tools.worldstream_m123.benchmark.metrics import distribution
from tools.worldstream_m123.world_risk_diagnostics import per_frame_detection


def table(headers,records):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in records])


def f(value):return '—' if value is None else f'{value:.6f}'
def d(value):return '/'.join(f'{value[k]:.3f}' for k in ('mean','p50','p95'))


def report(root):
    r=read(root/'summary.json');fit=read(root/'risk_fit.json');cal=read(root/'calibration_curve.json')
    diag=read(root/'residual_and_task_diagnostics.json');m1=read(root/'m1_same_state_control.json');decode=read(root/'decode_control/summary.json')
    from tools.worldstream_m123.benchmark.common import sha256
    if sha256(root/'frozen_m3_policy.json')!=sha256('outputs/final_route/m3_reuse_policy.json'):raise ValueError('M3 policy changed')
    from mmcv import Config
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py')
    if not r['goal_met'] and (cfg.model.m2_config.global_anchor_interval,cfg.model.m2_config.global_drift_threshold)!=(2,.7721027880907059):raise ValueError('Failed candidate installed')
    installed=all(cfg.model.m2_config.get(k)==v for k,v in r['selection']['config'].items())
    deployment=dict(installed_new_policy=installed,goal_met=r['goal_met'],formal_m2=dict(cfg.model.m2_config),formal_m3_quality_budget=cfg.model.m3_config.quality_budget,
        reason='Quality passes, but one scene is slower, round-level confidence includes zero, and generated token counts differ. Retain existing formal parameters.' if not r['goal_met'] else 'Independent quality and stable performance pass',
        formal_config_sha256=sha256('projects/configs/OmniDrive/worldstream_m123_auto.py'),m3_policy_sha256=sha256('outputs/final_route/m3_reuse_policy.json'),
        single_runtime=True,new_validation_scenes_only=True)
    write(root/'deployment_decision.json',deployment)
    status=read(root/'status.json')
    status.update(implementation_complete=True,formal_defaults_retained=not installed,
        gpu_tests_passed=528,objective_achieved=r['goal_met'])
    write(root/'status.json',status)
    overhead={}
    for name in ('before_profile','after_profile'):
        ff=rows(root/name/'per_frame_metrics.jsonl')
        overhead[name]={key:distribution(q['m2'][key] for q in ff) for key in ('global_decision_ms','M2_region_scoring_ms','M1_M2_state_read_ms','M2_cache_publication_ms','M2_execution_ms','M2_total_ms')}
    for name in ('before_profile','after_profile'):overhead[name]['cpu_profile']=read(root/name/'profile_output/cpu_summary.json')
    overhead['scope']='4 identical real frames, same old RMS policy. Sampled CPU/CUDA profiling distorts host timings; cache publication includes pending suffix compute at required finite checks. Not a standalone cache-copy speedup.'
    write(root/'overhead_comparison.json',overhead)
    nn=per_frame_detection(root,'r0_native');ff=per_frame_detection(root,'r0_formal')
    independent=[]
    questions={q['frame_id']:q for q in read(root/'r0_formal_quality_pairs.json')}
    for frame in rows(root/'r0_formal/per_frame_metrics.jsonl'):
        token=frame['frame_id'];a,b=nn.get(token),ff.get(token)
        independent.append(dict(frame_id=token,scene_id=frame['scene_id'],native=a,candidate=b,
            recall_loss=a['recall_2m']-b['recall_2m'] if a and b and a['recall_2m'] is not None else None,
            global_state=frame['m2']['global_update_mode'],risk=frame['m2']['global_risk_score'],question=questions[token]))
    independent.sort(key=lambda q:q['recall_loss'] if q['recall_loss'] is not None else -float('inf'),reverse=True)
    write(root/'independent_high_error_frames.json',dict(frames=independent,scope='All 24 frames, sorted by unchanged SDK 2m recall loss; raw counts are not per-frame AP/NDS. No heldout exact suffix or retuning.'))
    allformal=[q for i in range(3) for q in rows(root/f'r{i}_formal/per_frame_metrics.jsonl')]
    risk_ms=distribution(q['m2']['global_decision_ms'] for q in allformal)
    m1ms=distribution(q['m1_commit_timings_ms']['m1_total_including_adapters_ms'] for q in allformal)
    n,fm=r['aggregate']['native'],r['aggregate']['formal']
    token_delta=n['generated_tokens']-fm['generated_tokens'];per_token=decode['summary']['after']['per_token_cuda_span_ms']['mean']
    estimate=token_delta/72*per_token
    write(root/'token_length_accounting.json',dict(native_tokens=n['generated_tokens'],candidate_tokens=fm['generated_tokens'],difference=token_delta,
        fixed_token_ms=per_token,estimated_decode_length_saving_ms_per_request=estimate,
        measured_request_saving_ms=n['request_ms']['mean']-fm['request_ms']['mean'],
        scope='Accounting estimate only; not a measured fixed-token E2E result. Do not use shorter answers as a computation speedup claim.'))
    lines=['# M123 世界状态风险优化实验',
        '代码和真实 checkpoint 验证已完成；独立集四项绝对质量预算通过，但本轮未证明稳定的端到端计算净加速。因此保留正式 AR2 / 0.7721027880907059；新风险权重不安装为默认。',
        '## 风险公式与真实输入',
        '`risk = Σ_i w_i · feature_i / scale_i`，四维依次为最大相机相对 H2 RMS、最大相机 Top-10% Token 变化均值乘 `(1 + hot_fraction)`、自锚帧起累计并截断到 1 的 M1 世界变化、`age/3`。Hot Token 定义为变化超过该相机锚帧 RMS。只按 `risk >= threshold` 判定，另保留首帧、新场景、到期和 M1 整帧刷新。原残差近似、必须窗口和依赖闭包未改变。',
        '`world_step = .5 × mandatory_window_ratio + .25 × event_track_ratio + .25 × uncertainty_window_ratio`。输入均为现有 M1 投影状态、WSGRR 区域事件和窗口集合；不假装具有当前检测或可靠消失标签，不使用 GT/未来信息，也不把 Track ID 当作 Token 身份。',
        f'拟合权重 `{fit["weights"]}`；尺度 `{fit["scales"]}`；最终开发候选阈值 `{r["selection"]["config"]["global_risk_threshold"]}`，最大锚帧间隔 3。局部与年龄权重拟合为零，不能宣称这两项有额外收益。算法只有一个评分函数；保留旧默认时权重为 `(1,0,0,0)`，通过同一个 `>= nextafter(old_threshold,+∞)` 保持原 `>` 边界。',
        '## 数据、标注与调参边界',
        '4 个全新开发场景 × 8 帧，按输入自车位移分位覆盖低至高运动；4 个全新独立场景 × 6 帧。排除 15 个旧实验场景。Manifest 在实验前固定并记录 SHA256。这里只是自然连续帧小子集，所有 mAP/NDS/DET_l 均非全验证集成绩。',
        f'离线教师对所有年龄 1/2/3 的可用锚帧计算真实完整 22-block Suffix；其中 {fit["samples"]} 个年龄 1/2 样本用于非负 ridge 拟合（λ=.01，q75 尺度，固定均匀先验）。只比较开发风险分数 q40/q65 两档阈值；选择规则是四项通过后，在模型均值最快的 1% 范围内优先最差场景比值。独立集完全没有参与拟合或选择。',
        f'教师的输出与干净旧 RMS 运行逐帧摘要一致：`{diag["teacher_clean_predictions_equal"]}`。教师附加 suffix 不计作任何时延证据；测量运行均没有 shadow forward。',
        '## 残差误差与任务质量',
        f'真实残差相对 RMS 的 Spearman：旧 RMS `{fit["old_rms_spearman"]:.6f}`，开发拟合风险 `{fit["fit_spearman"]:.6f}`，留一场景风险预测 `{fit["leave_scene_out_spearman"]:.6f}`。4 个留一场景中有 1 个场景相关下降。按预定“RMS≤中位数且误差≥q75”定义，开发集中没有此类失败样本，不能宣称识别了尚未观察到的低 RMS 高误差案例。',
        table(['同一教师 H2 的选择器回放','复用帧/32','跳过 Block','复用帧特征误差 mean/P50/P95','最大误差'],[(k,v['summary']['reuse_frames'],v['summary']['suffix_blocks_skipped'],d(v['summary']['reused_frame_final_relative_rms']),f(v['summary']['reused_frame_final_relative_rms']['maximum'])) for k,v in diag['simulations'].items()]),
        '该回放隔离锚帧选择；它没有重走候选检测反馈轨迹。误差相对 RMS 在完整 Suffix 帧记零，所有帧与仅复用帧统计都保留在 JSON。新阈值增加了复用，同时平均/P95 误差增大，最大误差没有改善。相关性提高不等于任务质量改善。',
        table(['开发策略','mAP loss','NDS loss','DET_l loss','Token F1 loss','复用帧','模型均值 ms'],[(name,*[f(p['quality']['metrics'][k]['loss']) for k in ('mAP','NDS','DET_l','Token_F1')],p['timing']['global_suffix_reuse_frames'],f(p['timing']['model_ms']['mean'])) for name,p in [('RMS',cal['rms'])]+[(p['name'],p) for p in cal['candidates']]]),
        f'真实候选轨迹中，将世界项置零但保持同一 H2/锚帧，其阈值决策会改变 {diag["world_contribution"]["risk0"]["threshold_decisions_changed"]} 帧（已排除到期/必需整帧刷新）。这是明确的实际贡献，但新策略的开发 mAP 下降比旧 RMS 更大。未在独立集追加旧策略实验，所以不能由跨数据集分数声称减少了极端 Detection 退化。',
        '## 新独立集一次质量验收',
        table(['指标','Native','候选','绝对下降','相对下降','≤.10'],[(k,f(v['native']),f(v['candidate']),f(v['loss']),f(v['relative_loss']),r['quality']['metric_budget_pass'][k]) for k,v in r['quality']['metrics'].items()]),
        table(['场景','mAP loss','NDS loss','DET_l loss','F1 loss','复用帧/6'],[(s,*[f(v['metrics'][k]['loss']) for k in ('mAP','NDS','DET_l','Token_F1')],v['global_suffix_reuse_frames']) for s,v in r['quality']['per_scene'].items()]),
        f'数字/空间/运动/安全词法代理新增错误共 {r["quality"]["critical_proxy_regressions_vs_native"]}；这是词法代理，不是人工语义认证。Native 的 24 个答案本身也均未达到满分 F1。原始问题、参考答案、两边回答及各项代理差异见 `r0_formal_quality_pairs.json`；全部高误差帧与 GT 匹配见 `independent_high_error_frames.json`，开发精确特征高误差样本见 `residual_and_task_diagnostics.json`。',
        '## 受控模型边界与真实端到端',
        'A 在模型前预存 CPU 样本，校验每张预处理 Tensor 的形状/类型/内容 SHA 相同，只将 `model_ms` 用作模型边界时间；不将缓存样本的 request_ms 当真实 E2E。各场景初始历史均重置为空；近似导致后续感知反馈状态可能不同。M1 代码优化的同状态配对另行验证，未把不同闭环状态伪称逐帧相同。',
        table(['受控模型','mean/P50/P95 ms'],[(kind,d(r['controlled'][kind]['model_ms'])) for kind in ('native','formal')]),
        f'A 模型均值比 `{r["controlled"]["model_speedup"]:.6f}×`，仍包含自然回答长度差异。B 保留正常读图/预处理、相同 checkpoint/问题/种子/生成参数，按 N/F、F/N、N/F 交替三轮：',
        table(['轮','Native mean/P50/P95 ms','候选 mean/P50/P95 ms','均值比','节省 ms'],[(i,d(v['native']['request_ms']),d(v['formal']['request_ms']),f(v['speedup']),f(v['saved_ms'])) for i,v in enumerate(r['rounds'])]),
        table(['三轮合计','Native','候选'],[(key,d(n[key]),d(fm[key])) for key in ('request_ms','model_ms','preprocessing_ms','transfer_and_collate_ms')]+[('吞吐 fps',f(n['throughput_fps']),f(fm['throughput_fps'])),('峰值 allocated GiB',f(n['peak_allocated_bytes']/2**30),f(fm['peak_allocated_bytes']/2**30)),('峰值 reserved GiB',f(n['peak_reserved_bytes']/2**30),f(fm['peak_reserved_bytes']/2**30)),('生成 Tokens',n['generated_tokens'],fm['generated_tokens']),('KV 复用 Tokens',0,fm['kv_reused_tokens'])]),
        table(['场景','配对节省 mean/P50/P95 ms'],[(s,d(v)) for s,v in r['per_scene_saved_ms'].items()]),
        f'全部 72 个配对请求中正收益比例 `{r["positive_request_fraction"]:.2%}`。三轮配对平均节省 `{r["paired_round_saved_ms_ci95"]["mean"]:.3f}` ms，t(df=2) 95% 区间 `[{r["paired_round_saved_ms_ci95"]["lower"]:.3f}, {r["paired_round_saved_ms_ci95"]["upper"]:.3f}]` ms，不能把相同场景的三轮当作 12 个独立场景。一个场景平均变慢。输出和回答跨轮一致性见 summary.json。',
        f'候选比 Native 少 {token_delta} Tokens；固定相同 {decode["steps"]} Token 步的 Decode 对照（5 次记录）逐位 logits 相同。before/after 每步 CUDA span `{decode["summary"]["before"]["per_token_cuda_span_ms"]["mean"]:.3f}/{per_token:.3f}` ms，M3 算法本轮没有改动。仅以固定步耗时作长度记账，少生成可解释约 `{estimate:.3f}` ms/请求，而实测总节省 `{n["request_ms"]["mean"]-fm["request_ms"]["mean"]:.3f}` ms；这不是固定 Token E2E 实测，不能当作计算加速证据。',
        '## M1/M2 固定开销和剩余瓶颈',
        'M1 优化将逐对象、逐相机重复遮挡比较改为每请求/每相机一次批量矩形深度比较，严格比较语义不变；来源摘要只缓存 dataclass 字段名、关闭已归一化无环树的重复循环检测，每次仍读取和哈希全部实际状态。M2 只接管 Runtime 新建的私有年龄/描述符 Tensor，保留 alias、来源、版本、预算、非有限值及原子发布检查。',
        table(['相同真实 M1 状态，3 次交替×4 帧','before mean/P50/P95 ms','after mean/P50/P95 ms'],[(k,d(m1['summary']['before'][k]),d(m1['summary']['after'][k])) for k in m1['summary']['before']]),
        f'来源摘要、世界锚状态、关联和 M2 计算快照全部相同：`{m1["all_semantics_equal"]}`。此控制来自较密集的新开发场景，不能与上一轮不同场景的 172 ms 直接比较。独立完整服务实际 M1 总耗时 `{d(m1ms)}` ms；新风险决策含必要同步 `{d(risk_ms)}` ms。',
        table(['4 帧真实 Profiler 诊断（含扰动）','before mean/P50/P95 ms','after mean/P50/P95 ms'],[(k,d(overhead['before_profile'][k]),d(overhead['after_profile'][k])) for k in overhead['before_profile'] if k!='cpu_profile']),
        'StateCache 的 host wall 包含等待已提交 Suffix GPU 计算完成，约数百毫秒不是纯复制开销；没有删除非有限值检查来隐藏等待。cProfile 采样一帧的 source_digest 8 次累计 before/after 见 overhead_comparison.json。该采样不是无扰动性能结论。',
        f'独立服务视觉 CUDA span Native `{n["stages"]["eva_native"]["mean"]:.3f}` → M2 `{fm["stages"]["m2_total"]["mean"]:.3f}` ms；主要剩余固定成本是约 `{m1ms["mean"]:.1f}` ms M1、完整锚帧的 22-block Suffix，以及约 `{fm["stages"]["decode"]["mean"]:.1f}` ms 自回归 Decode。实际复用 11/24 帧（45.83%），每轮跳过 242 个 Suffix Block，三轮共 726；窗口复用每轮 416/2592（16.05%）。',
        f'M3 合法 KV 共复用 `{fm["kv_reused_tokens"]}` Tokens，来自 `{fm["unique_kv_requests"]}` 个唯一自然请求各重复三轮；未放宽 .02 内部预算、未改变策略文件或因果前缀约束。',
        '## 验证、实现与插桩记录',
        '完整 GPU 回归结果见 tests_gpu_final.log；CPU 结果为 526 passed / 2 CUDA skipped。所有命令固定 taskset 排除 CPU4/5，使用 OmniDrive 环境和本机 GPU。checkpoint 验证、完整调用命令和 per-frame 记录均保留。离线控制调度曾重复申请父进程已持有的锁，已终止本任务自己的等待子进程并修复；没有停止其他用户任务，失败调度记录保留在 commands.json。',
        table(['文件','类型','插桩/工具变化','行号'],[(path,kind,description,'1–'+str(len(Path(path).read_text().splitlines())) if kind=='新增' else '见本次 git diff') for path,kind,description in [
            ('tools/worldstream_m123/benchmark/worker.py','修改','可选离线 shadow、预存输入及内容摘要；正式服务默认不启用'),
            ('tools/worldstream_m123/benchmark/residual_labels.py','新增','仅离线完整 Suffix 标签、影子 hook 事件移除'),
            ('tools/worldstream_m123/latency_controls.py','修改','同状态摘要比较和独立摘要计时、固定 Token 控制保持'),
            ('tools/worldstream_m123/world_risk_experiment.py','新增','冻结校准、受控 A/真实 B、配对交替、GPU 排队'),
            ('tools/worldstream_m123/world_risk_diagnostics.py','新增','相同 H2 误差回放、GT 对象匹配诊断和图'),
            ('tools/worldstream_m123/world_risk_report.py','新增','原始数据生成本报告'),
            ('tools/worldstream_m123/risk_microbench.py','新增','预同步真实 H2，隔离旧 RMS 和新风险统计自身成本'),
            ('tools/worldstream_m123/world_risk_reproduce.py','新增','固定 Manifest、旧代码快照、真实 checkpoint 的新目录复现入口'),
            ('tools/worldstream_m123/artifact_index.py','修改','支持没有历史重评文件的新实验根目录')]]),
        '正式 Runtime 没有加入 Profiler。采样 cProfile/CUDA trace 在独立 profile_output 中，未进入 headline latency；大型 Tensor/trace/M1 全状态日志保存在服务器并列入 SHA 索引，任务输出、质量和逐帧时延记录随提交保存。',
        '## 验收结论',
        f'新风险改善了开发残差误差排序，但没有证明极端 Detection 退化减少；新独立集四项 .10 预算均通过，仍真实跳过大量 Suffix Block，M1 同状态优化语义保持，M3 合法复用保持。完整服务观测均值 `{r["service_speedup"]:.6f}×`，但跨场景/置信区间/回答长度控制不足以证明稳定计算净加速。保留旧正式配置，只有一条执行主线；下一步瓶颈是 M1 固定成本与自回归生成，而不是继续放宽精度预算。']
    micro=read(root/'risk_microbenchmark.json')
    components=[]
    controlled={kind:rows(root/('controlled_'+kind)/'per_frame_metrics.jsonl') for kind in ('native','formal')}
    for label,key in [('EVA / M2 CUDA span','visual_cuda_span_ms'),('M3 Prefill CUDA span','prefill_cuda_ms'),('M3 Decode CUDA span','decode_cuda_ms')]:
        components.append((label,*[d(r['controlled'][kind][key]) for kind in ('native','formal')]))
    for stage in ('detection','map'):
        components.append((stage+' CUDA span',*[d(distribution(v['stages'][stage]['gpu_span_ms'] for v in controlled[kind])) for kind in ('native','formal')]))
    components.append(('M1 含适配器 Wall','—',d(r['controlled']['formal']['m1_wall_ms'])))
    components.append(('M1→M2 Snapshot Wall','—',d(r['controlled']['formal']['snapshot_wall_ms'])))
    position=next(i for i,s in enumerate(lines) if s.startswith('A 模型均值比'))
    lines.insert(position,table(['A 分项（均为 mean/P50/P95 ms）','Native','候选'],components)+
        '\n\nCPU Wall 与 GPU span 范围不同且可能嵌套，不相加重建 E2E。')
    position=next(i for i,s in enumerate(lines) if s.startswith('| 场景 | 配对节省'))
    lines[position]=table(['场景','Native mean/P50/P95 ms','候选 mean/P50/P95 ms','配对节省 mean/P50/P95 ms'],
        [(s,d(n['per_scene'][s]['request_ms']),d(fm['per_scene'][s]['request_ms']),d(v)) for s,v in r['per_scene_saved_ms'].items()])
    position=next(i for i,s in enumerate(lines) if s.startswith('StateCache 的 host wall'))
    lines.insert(position,table(['同一真实 H2，预同步后独立统计','Wall mean/P50/P95 ms','CUDA span mean/P50/P95 ms'],
        [(k,d(v['wall_ms']),d(v['cuda_span_ms'])) for k,v in micro['summary'].items()])+
        f'\n\n这组 6×40×40×1024 FP32 H2 来自新开发场景的相同 checkpoint；5 次预热后各 30 次交替。新风险统计自身约 {micro["summary"]["unified_risk"]["wall_ms"]["mean"]:.3f} ms，旧 RMS 约 {micro["summary"]["old_rms"]["wall_ms"]["mean"]:.3f} ms；服务内约 25 ms 的决策 Wall 主要含前缀 GPU 等待，不能把全部归因于风险归约。新增统计保持一次紧凑回传。')
    (root/'experiment_report.md').write_text('\n\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);a=p.parse_args();report(Path(a.root))
