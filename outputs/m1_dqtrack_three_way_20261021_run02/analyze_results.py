"""Post-commit report and independent trace audit; never replays any method."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parent

def load(name):return json.loads((ROOT/name).read_text())
def save(name,value):(ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

summary=load('summary.json');verification=load('verification.json');failures=load('failure_cases.json')
traces=[json.loads(s) for s in (ROOT/'per_frame_trace.jsonl').read_text().splitlines()]
methods=('A','B','C');extra={};audits={}
for method in methods:
    tt=[t for t in traces if t['method']==method];counts=collections.Counter();lastscene=None
    ambiguity_reasons=collections.Counter();component_reasons=collections.Counter();veto_cases=[]
    for t in tt:
        if lastscene!=t['scene_id']:previous={};last={};ordinal=0;lastscene=t['scene_id']
        current=t['scored_identity_by_anchor'];labels={int(k):v for k,v in t['prediction_gt_labels'].items()}
        eligible=set(previous.values())&set(current.values());counts['eligible']+=len(eligible)
        if ordinal:counts['candidates']+=len(t['selected_prediction_indices'])
        correct_ids=set()
        for e in t['edges']:
            old=previous.get(e['anchor_id']);new=labels.get(e['prediction_index'])
            status='correct' if old is not None and old==new else 'wrong' if old is not None and new is not None else 'unknown'
            assert status==e['status']
            counts['accepted']+=1;counts[status]+=1;counts['scored']+=int(status!='unknown')
            if status=='correct':correct_ids.add(new)
        counts['continuity_breaks']+=len(eligible-correct_ids)
        for aid,instance in current.items():
            if instance in last:
                counts['identity_switches']+=int(last[instance][0]!=aid)
                counts['fragments']+=int(last[instance][1]<ordinal-1)
            last[instance]=(aid,ordinal)
        bd=t['dqtrack']
        if bd:
            counts['tracks']+=bd['available_tracks'];counts['aligned']+=bd['aligned']
            counts['ambiguous']+=bd['ambiguous'];counts['rejected']+=len(bd['rejected'])
            counts['unaligned_tracks']+=bd['unaligned_tracks'];counts['alignment_possible']+=len(t['bridge_gt_labels']['possible_instances'])
            for j,h in bd['alignments'].items():
                old=t['bridge_gt_labels']['gt_by_tracking_id'].get(h['tracking_id'])
                new=labels.get(t['selected_prediction_indices'][int(j)])
                known=old is not None and new is not None
                counts['alignment_scored']+=int(known);counts['alignment_unknown']+=int(not known)
                counts['alignment_correct']+=int(known and old==new)
            ambiguity_reasons.update(r['reason'] for r in bd['unaligned_records'])
            component_reasons.update(c['reason'] for c in bd['policy']['components'])
            for i,j in bd['policy']['vetoes']:
                index=t['selected_prediction_indices'][j]
                baseline=next(x for x in traces if x['method']=='A' and x['frame_id']==t['frame_id'])
                edge=next(e for e in baseline['edges'] if e['prediction_index']==index)
                veto_cases.append(dict(scene_id=t['scene_id'],frame_id=t['frame_id'],prediction_index=index,
                    baseline_evaluation=edge,current_external_alignment=bd['alignments'][str(j)],
                    interpretation='oracle unknown; rejection is not evidence of corrected error'))
        previous=current;ordinal+=1
    actual=summary['methods'][method]['metrics']
    for k in ('accepted','candidates','correct','wrong','unknown','scored','eligible','identity_switches','fragments',
              'continuity_breaks','tracks','aligned','alignment_correct','alignment_scored','alignment_unknown',
              'alignment_possible','ambiguous','rejected','unaligned_tracks'):
        assert counts[k]==actual[k],(method,k,counts[k],actual[k])
    for t in tt:
        assert len(t['pairs'])==len(t['edges'])
        assert all(t['selected_prediction_indices'].index(e['prediction_index']) in range(len(t['committed_anchor_ids'])) for e in t['edges'])
    # Candidate denominators were never defined for old class/motion strata.
    # Remove derived placeholders instead of publishing negative rejection counts.
    for group in summary['methods'][method]['strata'].values():
        group.pop('not_accepted',None);group.pop('candidates',None)
    extra[method]=dict(unaligned_reasons=dict(ambiguity_reasons),component_reasons=dict(component_reasons),
        certified_anchor_observations=sum(len(t['certified']) for t in tt),veto_cases=veto_cases)
    failures[method]['policy_vetoes']=veto_cases
    for e in failures[method]['wrong']:
        t=next(t for t in tt if t['frame_id']==e['frame_id'])
        j=t['selected_prediction_indices'].index(e['prediction_index'])
        e['current_bridge_alignment']=(t['dqtrack'] or {}).get('alignments',{}).get(str(j))
        e['bridge_explanation']='No valid current bridge alignment; B/C preserve A behavior' if method!='A' else 'No external identity used'
    audits[method]=dict(recomputed_counts=dict(counts),matches_summary=True)

scenes=list(summary['scenes']);rng=np.random.RandomState(20261021);bootstrap={}
for method in ('B','C'):
    bootstrap[method]={}
    for metric in ('correct','wrong','identity_switches','continuity_breaks','accepted'):
        delta=np.array([summary['scenes'][s][method]['metrics'][metric]-summary['scenes'][s]['A']['metrics'][metric] for s in scenes])
        draws=delta[rng.randint(0,4,(2000,4))].sum(axis=1)
        bootstrap[method][metric]=dict(observed_sum=int(delta.sum()),scene_resample_sum_interval=np.percentile(draws,[2.5,97.5]).tolist(),
            n_scenes=4,draws=2000,independent_validation=False)
summary['decision']='B、C 均无可靠收益：保留默认 M1，DQTrack 不纳入默认决策路径'
summary['decision_scope']='Only this exposed four-scene development replay; insufficient evidence of external/general risk'
summary['post_commit_analysis']=extra;summary['paired_scene_bootstrap']=bootstrap
summary['report_processing']=dict(strata_undefined_denominators='Removed candidates=0 and derived not_accepted placeholders; no scored counts, ratios, denominators or method outputs changed',
    script='analyze_results.py',source_sha256=digest(__file__))
save('summary.json',summary);save('failure_cases.json',failures)
verification['independent_trace_audit']=audits
verification['all_groups_zero_physical_certifications']=all(v['certified_anchor_observations']==0 for v in extra.values())
verification['B_C_assignments_identical']=all(a['pairs']==b['pairs'] and a['committed_anchor_ids']==b['committed_anchor_ids'] for a,b in zip([t for t in traces if t['method']=='B'],[t for t in traces if t['method']=='C']))
verification['prior_preflight_failure']=dict(directory='outputs/m1_dqtrack_three_way_20261021_run01',method_frames_executed=0,
    cause='preflight source tag assertion too strict; corrected to accept original historical_full_model_detection_replay without rewriting inputs')
verification['report_processing']=summary['report_processing']
save('verification.json',verification)
shutil.copyfile('/tmp/m1_dqtrack_three_way_replay_20261021_run02.log',ROOT/'replay.log')

m=summary['methods'];A=m['A']['metrics']
def ratio(method,name):
    d=m[method]['metrics']['ratio_details'][name]
    return f"{d['numerator']}/{d['denominator']} = {d['value']*100:.4f}%" if d['value'] is not None else '不适用（0/0）'

lines=['# OmniDrive M1 / DQTrack 三组真实缓存回放报告','',
'结论：**B、C 均无可靠收益，保留默认 M1，不将 DQTrack 纳入默认决策路径。** 三组正确延续、错误延续、身份切换和连续性断裂相同。B/C 只拒绝一条 oracle 不可评分的关联，不能据此宣称消除了错误。C 的 686 个竞争分量均因外部映射不完整或不唯一而回退 B，实际新增消歧边为 0。','',
'## 修改文件','',
'新增 `tools/worldstream_m1/dqtrack_three_way_policy.py`（仅离线 B/C 策略）和 `dqtrack_three_way_eval.py`（冻结输入与三组独立回放）。`worldstream/m1/state_manager.py` 只加入默认等价的 `_associate()` 委托；`tools/worldstream_m1/dqtrack_policy_eval.py` 复用原 scorer，增加可注入的管理器与可重算 trace，不改变 strict_labels/rates 定义。新增 `tests/worldstream_m1/test_dqtrack_three_way.py`；新增三组使用文档，并在两份历史 DQTrack 文档顶部标明退役策略的当前状态。没有删除文件，没有修改 association.py、桥接算法、M1Config 默认值、物理验证或 M2/M3。','',
'## 数据、配置与执行边界','',
'真实复用四个已暴露开发场景、161 帧的原 OmniDrive 候选与 DQTrack 导出，共执行 483 个方法帧。所有组独立初始化管理器，采用相同帧顺序与候选，场景间不传递状态；按场景轮换 A/B/C 执行顺序。A 是本次从原缓存重新运行的唯一基线，没有用历史汇总代替。没有模型推理、训练、权重修改、依赖升级、其他跟踪器、官方完整验证或参数搜索。CPU 使用 0–3、6–31，解释器为 OmniDrive 环境。','',
'使用 `M1Config()` 当前真实默认值，三组配置 SHA256 完全一致：`'+load('config_manifest.json')['methods']['A']['sha256']+'`。`dqtrack_mode=off`、`association_track_prior=False`、`temporal_visibility=False`；本次未切换成历史 balanced 或 world_identity YAML。默认生产逻辑、Hint 拒绝、物理认证、M2/M3 未改变。A 不调用桥接或读取轨迹结果；预检阶段独立校验外部文件不属于 A 的关联输入。','',
'A：当前普通 WorldStateManager。B：原合法分配中，成熟外部身份明确冲突时拒绝未认证延续；缺失身份保留原行为，不主动消歧。C：B 加完整竞争分量一对一外部身份映射，所有选边必须原本合法，否则回退 B。共同研究 profile 固定为 score≥0.5、连续3次已提交对齐、年龄≤1秒，不将 score 当概率或距离权重，不根据结果调整。','',
'外部身份保存 run/scene/ID 命名空间与帧、epoch、执行签名、来源、状态版本；仅在事务私有 fork 上发布。普通管理器新增的 `_associate` 等价转发原 `associate`；离线子类才覆盖该方法。外部 ID 不签发物理证据、不增加认证观察、不提升可靠速度或 M2 特征资格。483 个方法帧中物理认证锚点观测均为 0。','',
'原始协议 SHA、候选文件/选择文件 provenance SHA 均校验通过，161 帧场景与时间/样本映射完整，原 DQTrack run/checkpoint 声明一致，历史逐帧清单完整；源码与图像、标注、候选、轨迹文件在运行前冻结哈希，运行后核验未变。来源声明和哈希证明本地文件一致性，不能独立证明生产者权重声明或检查点训练场景归属。候选保留原 `historical_full_model_detection_replay` 标记。','',
'run01 的来源类型预检假设过严，未执行任何方法帧；失败日志保留。修正校验后完整实验仅在 run02 各执行一次，没有重复方法或调参。运行命令、Git commit、工作区 diff、实际配置、全量输入 SHA、源文件快照见 manifests。','',
'## 同口径总体结果','',
'GT 只在每帧 commit 后使用：类别一致、尺寸比≤2、传感器支持、中心≤1米且两侧唯一。unknown 排除 scored，不能算作正确，也不能把拒绝算作错误已消除。effective recall 分母是相邻帧共同可评分的实例；coverage 分母是除各场景首帧外的已选择候选。以下 IDS/Fragments 是部分 oracle 诊断，**不是官方 nuScenes IDS/FRAG、AMOTA**。Fragments 沿用原定义：同一可评分 GT 观测之间的帧缺口计数，不能用其声称预测轨迹碎片改善。','',
'| 指标 | A 默认 M1 | B 保守桥接 | C 身份消歧 |','|---|---:|---:|---:|']
for name,label in [('correct','正确延续'),('wrong','可评分错误延续'),('unknown','不可评分延续'),('identity_switches','身份切换'),('fragments','原定义 Fragments'),('continuity_breaks','连续性断裂')]:lines.append('| '+label+' | '+' | '.join(str(m[x]['metrics'][name]) for x in methods)+' |')
for name,label in [('wrong_association_rate','可评分错误率'),('effective_recall','有效召回率'),('association_coverage','关联覆盖率'),('rejection_rate','拒绝率')]:lines.append('| '+label+' | '+' | '.join(ratio(x,name) for x in methods)+' |')
lines+=['',
'相对 A，B/C 的正确延续、可评分错误、IDS、Fragments、连续性断裂均变化 0；accepted 从 3302 降至 3301，绝对 -1，相对 -0.0303%；覆盖率下降 0.0215 个百分点，相对 -0.0303%。unknown 从2520降至2519。所有绝对与相对变化见 comparison.csv，基线为0或未知时相对变化留空。','',
'历史记录仅供背景：原 804正确/3错误、82.97% 召回、69.39%覆盖、258 IDS、165断裂来自旧实现/配置；本次 A 的正确数779、IDS282和断裂190不同。虽然 969/4646 分母一致，本任务没有将旧实现重新运行，也不把当前默认配置与旧复杂配置的差异归因于 DQTrack。真正受控结论仅来自本次 A/B/C。','',
'## 逐场景、类别与运动分层','',
'| Scene | 帧数 | A/B/C 正确 | A/B/C 错误 | A/B/C IDS | A/B/C 断裂 | A/B/C accepted |','|---|---:|---|---|---|---|---|']
for scene,rr in summary['scenes'].items():
    n=sum(t['scene_id']==scene and t['method']=='A' for t in traces)
    lines.append('| '+scene+' | '+str(n)+' | '+' | '.join('/'.join(str(rr[x]['metrics'][k]) for x in methods) for k in ('correct','wrong','identity_switches','continuity_breaks','accepted'))+' |')
lines+=['','完整分层在 summary.json 的 methods.*.strata 与逐场景原始 JSON。原分层没有定义候选分母，未伪造 class/motion coverage。主要类别如下（A/B/C 正确、可评分错误和 eligible 相同）：','',
'| 类别 | 正确延续 | 错误延续 | 有效召回（分子/分母） |','|---|---:|---:|---|']
for key,v in sorted(m['A']['strata'].items()):
    if key.startswith('class:'):
        assert all(m[x]['strata'].get(key,{}).get('correct',0)==v.get('correct',0) for x in methods)
        num=v.get('correct',0);den=v.get('eligible',0);fraction=f'{num}/{den}'
        lines.append(f"| {key[6:]} | {num} | {v.get('wrong',0)} | {fraction} |")
lines+=['',
'快速运动按 GT 世界中心差分速度≥8m/s 分组，三组均为 **0/115** 正确延续，没有改善。几何/年龄误差预算继续生效，外部 ID 未扩大门限。本次默认 association_track_prior=False，没有启用 dormant 记录或独立恢复算法；未把离开可评分集合后的未知重现算作恢复成功。本次任务没有单独运行遮挡恢复实验。','',
'## 桥接可用性与失败原因','',
'B/C 桥接完全相同：2270 个有效轨迹观测，对齐982次，覆盖982/2270=43.2599%；可评分对齐584/584=100%，另398次未知，不能声称982次均正确；可评分对齐召回584/639=91.3928%。原桥接 ambiguous 计数414，非法拒绝0，未对齐1288次。未对齐原因另按实际记录统计：620次存在竞争合法候选，668次没有合法类别/距离/尺寸边；414是既有桥接被提议后拒绝的计数，不能误称所有竞争轨迹的数量。','',
'C 共检查686个竞争分量，全因映射不完整或不唯一回退 B，没有完整分量获准。桥接本身要求跟踪器与检测双向无竞争，叠加成熟历史与源绑定要求后，可用的完整分量不足。这是没有实际消歧证据，不等于证明消歧算法在独立数据上一定无效。','',
'三条可评分错误在 A/B/C 均保留，目标预测在这些帧都缺少有效当前桥接，策略按要求退回普通 M1：','',
'| 帧 | Scene | 旧 GT → 当前 GT |','|---|---|---|']
for e in failures['A']['wrong']:lines.append(f"| {e['frame_id']} | {e['scene_id']} | {e['previous_instance']} → {e['current_instance']} |")
veto=extra['B']['veto_cases'][0]
lines+=['',
'第一场景的前两条错误在相邻帧发生于同一旧锚点，关联 GT 来回切换；第三条出现在另一场景。未用 GT 修正任何在线身份。B/C 唯一冲突拒绝发生在帧 `'+veto['frame_id']+'`，原预测索引 '+str(veto['prediction_index'])+'，A 的原关联标签为 unknown；不能判断该拒绝是消错还是损失真实延续。完整源/目标 ID、预测索引、当前外部提示、282个切换事件、190个断裂实例及所有拒绝预测均可由 failure_cases.json / per_frame_trace.jsonl 定位。','',
'## 真实耗时','',
'| 项目 | A | B | C |','|---|---:|---:|---:|']
for key,label in [('p50_ms','M1 P50（ms）'),('p95_ms','M1 P95（ms）'),('throughput_frames_per_s','M1 回放吞吐（帧/s）')]:lines.append('| '+label+' | '+' | '.join(f"{m[x]['m1'][key]:.3f}" for x in methods)+' |')
for key,label in [('p50_ms','桥接 P50（ms）'),('p95_ms','桥接 P95（ms）')]:lines.append('| '+label+' | '+' | '.join(f"{m[x]['bridge'][key]:.3f}" for x in methods)+' |')
lines+=['',
'计时覆盖 precheck、候选深拷贝和完整 commit，包含桥接与其首次文件解析；不含图像加载、GT 评分、离线报告和外部 DQTrack 前向。A 未调用桥接，桥接耗时为0，单独桥接吞吐无意义。单轮四场景存在缓存、顺序、调度噪声，C 的实测耗时比 B 小不能说明消歧更快；B/C 均较 A 增加开销，没有相应质量收益。这些值不能作为端到端 DQTrack+OmniDrive/VLM 服务时延或 GPU 加速结论。','',
'按场景配对bootstrap仅作描述性补充（4场景、2000次重采样）：正确/错误/IDS/断裂差异区间均为[0,0]；不构成统计等价性或独立风险保证。接受数量差异区间见 summary.json。没有合格独立场景，检查点训练场景归属未验证。','',
'## 验证、产物与采用决定','',
'轻量策略测试15项通过；全量 `python -m pytest tests/worldstream_m1 -q` 为 **613 passed，19 warnings，106.44秒**；最终桥接/策略聚焦复核42项通过。warnings 为既有库弃用提示。测试覆盖冲突、缺失、重复、过期、场景/来源/版本/epoch不一致、完整与不完整竞争分量、Hint拒绝、默认等价、失败原子回滚和源篡改。没有重新运行任何模型或历史基线。','',
'逐帧独立审计重新从当前/前帧 GT 标签及原始配对计算正确/错误/unknown、eligible、候选、IDS、Fragments、断裂及桥接可评分计数，与汇总完全一致；三组选择候选一致、一对一约束满足、A无桥接，B/C最终分配完全相同。源码与输入哈希的运行后核验通过，见 verification.json。分层原始报告没有候选分母，后处理仅移除了无定义的 candidates=0 / not_accepted占位项，没有改任何可评分计数或评价分母，分析脚本与SHA已保存。','',
'主要交付：experiment_report.md、summary.json、comparison.csv、per_scene.csv、per_frame_trace.jsonl、failure_cases.json、config_manifest.json、source_manifest.json、verification.json；另含12个场景/方法原始结果、测试与回放日志、源码快照、patch、可重算的 analyze_results.py。没有覆盖历史结果。','',
'**工程执行完成；DQTrack 默认采用结论为 No-Go。** B 不减少可评分错误或切换，C 没有观测到额外消歧收益；维持默认 off。四个已暴露开发场景不足以确认更广泛的正确性或无效性，本次不发布在线资格。']
(ROOT/'experiment_report.md').write_text('\n'.join(lines)+'\n')
print('Independent audit passed; report generated')
