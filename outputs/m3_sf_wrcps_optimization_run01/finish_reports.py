"""Report the observed adaptation-data failure, without fabricating training."""
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys
from tools.worldstream_m123.benchmark.common import sha256,write
from tools.worldstream_m123.latency_controls import load_capture
from worldstream.m3.stable_world import StableAnchorManager

root=Path(__file__).resolve().parent
def read(name):return json.loads((root/name).read_text())
def rows(name):return [json.loads(s) for s in (root/name).read_text().splitlines() if s.strip()]
def md(name,value):(root/name).write_text(value.strip()+'\n')
manifest=read('training_manifest.json');managers={};counts=defaultdict(Counter);scenes=defaultdict(Counter);events=[]
for sample in manifest['samples']:
    capture=load_capture(sample['capture']);s=capture['stable_snapshot']
    anchor,hit=managers.setdefault(s.scene_id,StableAnchorManager()).update(s,capture['stable_context'])
    split=manifest['scene_splits'][s.scene_id]
    for c in (counts[split],scenes[s.scene_id]):
        c['frames']+=1;c['questions']+=len(sample['pairs'])
        c['core_valid_frames']+=int(anchor.stable_state_valid);c['anchor_hits']+=int(hit)
    events.append(dict(scene_id=s.scene_id,frame_id=s.frame_id,split=split,
        core_valid=anchor.stable_state_valid,anchor_hit=hit,reason=anchor.update_reason,
        facts=anchor.facts))
assert any(not counts[s]['anchor_hits'] for s in ('train','validation')), 'This reporter is only for failed data admission'
assert not (root/'adapter_training/adapter.pt').exists(), 'Do not overwrite trained experiment conclusions'
admission=dict(training_started=False,optimizer_updates=0,status='REJECTED_REAL_DEVELOPMENT_CORE_COVERAGE',
    reason='Both train and validation need real frames and cross-frame stable anchors',
    split_counts={k:dict(v) for k,v in counts.items()},per_scene={k:dict(v) for k,v in scenes.items()},
    timeline=events,manifest_sha256=sha256(root/'training_manifest.json'),
    rule_unchanged=True,acceptance_logs_opened=False,checkpoint=None)
write(root/'training_admission.json',admission)
write(root/'composer_training_config.json',dict(status='PLANNED_NOT_STARTED',primary_K=64,rank=128,epochs=3,
    learning_rate=1e-4,seed=20261028,base_model_frozen=True,gradient_checkpointing=True,
    loss='Answer KL + critical-query-weighted task CE + 0.5 directional KL when reference/Teacher agree',
    selection='Core-valid development validation loss',optional_K32_run=False,
    optimizer_updates=0,training_admission=str(root/'training_admission.json')))
coverage=read('semantic_core_coverage.json');cv=coverage['observed_coverage']['qualification_validation/val']
ct=coverage['observed_coverage']['qualification_validation/train'];allc=coverage['observed_coverage']['all']
structure=read('structural_diagnostic.json');valid=[r['sf_exact_native_slots'] for r in structure['frames'] if r['core_valid']]
no_core=read('no_core_equivalence/summary.json')
captures=['qualification_train_capture','qualification_val_capture','adaptation_capture']
request_counts={};records=[]
for name in captures:
    rr=rows(name+'/per_question_metrics.jsonl');request_counts[name]=len(rr)
    records.extend(dict(r,experiment_role=name,comparison_method='native_layout_B3_ZeroKV_candidate_capture',
        sf_composer_executed=False,formal_performance_evidence=False) for r in rr)
for r in rows('no_core_equivalence/per_request.jsonl'):
    records.append(dict(r,experiment_role='old_failure_no_core_paired_regression',
        comparison_method='B0_native_vs_SF_native_identity_same_carrier_RNG',
        sf_composer_executed=False,formal_performance_evidence=False))
with (root/'per_request_metrics.jsonl').open('w') as f:
    for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
write(root/'request_scopes.json',dict(capture_requests=request_counts,paired_no_core_diagnostic_requests=24,
    per_request_rows=len(records),no_core_model_generations=48,
    note='A paired diagnostic row contains two generated answers. Native-layout B3 captures are not formal A/B/C efficiency trials.'))
reuse=dict(formal_sf_exact_campaign_executed=False,qualification_anchor_hits=19,
    independent_qualification_VAL=dict(hits=cv['anchor_hits'],transitions=cv['transitions']),
    adaptation_real_anchor_hits=sum(v['anchor_hits'] for v in counts.values()),
    matcher_exact_crossframe_visual_requests=0,scheduler_approved_crossframe_visual_requests=0,
    executor_crossframe_visual_reuse_requests=0,actual_crossframe_exact_visual_KV_tokens=0,
    prefill_forward_tokens_actually_skipped=0,
    prefill_CUDA_improvement_ms=None,M3_wall_improvement_ms=None,E2E_speedup=None,
    reason='Training data admission failed before trained representation quality. No ExactKV campaign was admitted.',
    scope='Zeros describe executed work only; no claim about a hypothetical trained SF adapter. Unit-test reuse is excluded.')
write(root/'crossframe_reuse_summary.json',reuse)
decision=dict(recommended=False,replace_formal_configuration=False,
    stage_one_independent_qualification_pass=True,target_domain_nonzero_correct_core=True,
    development_training_data_pass=False,adapter_training_started=False,optimizer_updates=0,
    no_core_native_equivalence_pass=True,core_valid_trained_quality_pass=None,overall_four_metric_quality_pass=None,
    exact_KV_pass=None,E2E_pass=None,final_acceptance_logs_opened=False,
    stop_stage='real_adaptation_data_admission',reason=admission['reason'],
    qualified_semantic_types=['car'],newly_admitted_semantic_types=[],semantic_expansion_pass=False,
    formal_configuration='projects/configs/OmniDrive/worldstream_m123_auto.py',
    quality_budgets=dict(mAP=.1,NDS=.1,DET_l=.1,Token_F1=.1),
    quality_measured=dict(mAP=None,NDS=None,DET_l=None,Token_F1=None),
    limitation='Qualification target-domain hits concentrate in one of three scenes; coverage is not a guarantee on new logs.')
write(root/'deployment_decision.json',decision)
bus=[r for r in rows('no_core_equivalence/per_request.jsonl') if r['frame_id']=='5193a061a8c24adb9663daa3a3c8e2cc' and 'bus stop' in r['question']][0]
write(root/'spatial_failure_regression.json',dict(frame_id=bus['frame_id'],question=bus['question'],
    reference='Yes, there is a bus stop to the right-front, suggesting the vehicle is on a main route that accommodates public transport.',
    old_composer_answer='Yes, there is a bus stop to the left, which suggests that we are in an urban area with public transportation facilities.',
    native_answer=bus['native_answer'],sf_no_core_answer=bus['sf_answer'],checks=bus['checks'],
    conclusion='Old no-core left/right regression removed by strict native identity; no conclusion about trained core-valid compression.',
    scope='Previously exposed diagnostic scene, not independent acceptance; reference/Teacher are not physical certification.'))
md('composer_training_report.md',f'''
# Composer 训练与数据准入

训练命令实际调用了现有 wrcps_train.py --spatial --stable-tokens 64 --rank 128 --epochs 3，但 validate_manifest 在加载基础模型、构建优化器之前拒绝。不是训练完成，也不是训练OOM。实际优化器更新0，没有SF checkpoint、最佳epoch或训练loss；旧K128权重不作为新结构权重使用。原始traceback见 adapter_training.log、adaptation_driver.log，返回码见 commands.json。

真实开发采集10个新TRAIN Log、80帧、240个原始QA，划分6训练Log/48帧、4验证Log/32帧。训练集有效核心帧 {counts['train']['core_valid_frames']}/48、跨帧核心延续 {counts['train']['anchor_hits']}/42；开发验证有效核心帧 {counts['validation']['core_valid_frames']}/32、跨帧延续 {counts['validation']['anchor_hits']}/28。逐场景和生命周期见 training_admission.json。大量真实无核心帧不能复制或伪装成有效核心序列。

独立资格VAL的5次正确命中满足最初适用性门槛，但不足以保证另选开发Log中也有持续核心。本轮在固定开发采集后观察到这一失败，没有转用资格或最终验收Log补训练，也没有按输出搜索新场景、放宽car规则或增加epoch。

实现已支持原始Answer KL、Task CE与固定0.5方向词KL。方向增强只在参考方向词位置且Teacher argmax同意参考时启用，冲突单列；基础权重冻结、梯度检查点沿用。无核心为原生identity，不做虚假更新；checkpoint选择只看有效核心开发验证loss。配置见 composer_training_config.json。以上为实现和合同测试，不是本轮实际学习结果。

K32只验证接口/布局合同，没有开发质量证据触发额外训练。峰值训练显存未测量（null），不能用旧训练显存代填。
''')
md('spatial_quality_report.md',f'''
# 空间保真证据与边界

最重要基础合同通过：同一真实M1/M2载体、同一问题/原采样RNG，8帧24问的Native与SF无核心输入实际扩展视觉embedding、position IDs、attention mask、token IDs、visual mask、prefill最后位置logits和完整自然生成tokens全部严格一致（24/24）。同一tensor存储身份也保留。温度0.1、top_p0.75、max_new_tokens320未改。证据见 no_core_equivalence/summary.json 和逐请求记录。

旧5193…公交站问题由旧Composer的left恢复为与Native相同的right，参考为right-front，详见 spatial_failure_regression.json。这修复了旧无核心压缩造成的方向翻转；只是一例已暴露回归。没有重新验证所有图像物理事实，也没有将Teacher或自然语言参考当作物理证书。

真实旧开发载体的13帧结构诊断中含5个有效核心帧。旧K128自由attention重构仅Ego一个原生槽位逐元素保留，新未训练K64保留{min(valid)}–{max(valid)}个；8个无核心帧完整保留513个且同一存储。K64按object/map特征余弦依赖做有限二部匹配，每组只移除32槽，未参与目标合并的槽位保持原值及原顺序；Ego不变。相邻索引不被当作物理邻接。余弦高并不能证明物理空间保真。详见 structural_diagnostic.json。

有效核心下的新结构会增加64个稳定槽，移除64个原生槽并修改少量合并目标，因此仍可能损失方向、运动和数量关系。本轮训练数据门失败，没有已训练新Composer的同载体核心质量结果；不能因无核心严格等价就宣称WRCPS质量通过。K64相对K128的结构保留改进不等同于任务精度最优，也没有证据触发K32开发比较。
''')
md('quality_report.md','''
# 质量准入

无核心真实模型等价24/24通过；上轮公交站left/right回归已在相同载体及RNG下消除。新资格目标域非零正确核心已观察到，但80帧独立开发采集中，训练集48帧只有1个孤立有效核心帧、0次延续；开发验证32帧有6个有效核心帧、5次延续。训练集不满足持续核心要求，因此训练准入拒绝。

因此没有新的已训练SF Adapter，Core-Valid ZeroKV质量未建立；最终6个保留Log没有打开。本轮mAP、NDS、DET_l、Token F1及其下降全部未测量（null），四项绝对下降预算仍为0.10。上轮K128的四指标不属于本轮结果，未复制充当新测量。

关键方向、运动、数量、车道/规则、灯态、新增/消失对象与空间依赖均没有完整独立SF验收证据。不能认定“没有新增错误”，只能报告无核心identity对照没有新增模型输出差异。旧例修复与有核心训练表示的泛化质量严格分开。

保留可执行的core-quality诊断和最终质量工具，前者检查整个开发验证序列，后者需绑定已训练Adapter的开发关键语义审阅通过文件。禁止绕过这些门槛进行ExactKV或部署宣称。
''')
timings={name:read(name+'/metrics.json') for name in captures}
write(root/'capture_timing_context.json',dict(scope='One natural native-layout capture per dataset, 3 questions/frame; not paired SF performance or CI',
    runs={n:{k:d.get(k) for k in ('frames','questions','request_ms','model_ms','m3_ms','gpu_peak_memory_bytes')} for n,d in timings.items()}))
md('performance_report.md','''
# 性能结果与停止阶段

没有准入正式A/B/C服务测试、固定tokens/相同步数控制、三轮交替或场景bootstrap。因此ΔT_KV、Prefill CUDA改善、M3 Wall改善、E2E加速比、配对95%CI、FPS/QPS改善均为null。不能从单次数据采集计时或CPU资格回放推断加速。

真实执行的192个六相机连续帧/576个自然问题均为原生布局B3-ZeroKV载体/候选采集；另有旧故障载体8帧24问的Native/SF无核心成对诊断。前者禁止历史复用，后者是同载体表示合同。真实跨帧视觉KV请求0、tokens0、跳过prefill tokens0；这不代表已训练SF方案的性能测量。

原始逐请求prefill/decode CUDA事件、cache成本、完整M3 wall等保留在各capture的per_question_metrics.jsonl，整帧Request/Model统计在metrics.json。采集上下文摘录见 capture_timing_context.json，明确不构成正式A/B/C对比。GPU内存为现有模型采集内存，训练和正式SF效率内存均未测量。

四个计数层次保持分开：19个资格回放Anchor命中（其中独立验证9）只是CPU状态延续；它们没有进入SF PrefixMatcher、Scheduler准许或PrefillExecutor跳过计算。开发训练集没有真实核心延续（开发验证有5次），训练先被拒绝，因而不能把资格命中乘K当作实际复用量。

保持原正式配置。主要瓶颈已从旧的无核心表示损失，进一步定位到核心跨新Log的覆盖不足、单car语义信息量小、以及有效核心空间质量尚无可训练且独立验证的数据证据。
''')
answers=[
('无核心是否严格一致？','是。真实8帧24问，实际embedding/position/mask/prefill logits/完整生成tokens全部严格一致；同tensor原生513槽，基础权重与生成参数不变。'),
('旧左右翻转是否修复？','旧无核心公交站案例由left恢复为与Native相同的right，参考right-front；仅此已暴露回归与无核心合同通过，不声称有效核心压缩的空间问题全部解决。'),
('真实有效核心下仍有什么差异？','结构上64稳定槽替代64原生槽，有限目标被合并。尚无训练后的真实核心质量结论，方向/运动/数量风险未排除。'),
('K64比K128更好平衡精度和复用吗？',f'结构诊断保留{min(valid)}–{max(valid)}个原生槽，旧K128只有Ego1个；稳定容量从128降64。任务精度与净收益未验证，不能判为最优。'),
('是否需要K32？','没有实际依据触发。仅实现并测试声明的K32布局，不进行搜索或用保留集选择K。'),
('覆盖更多场景/类型了吗？','新独立VAL的1/3场景出现正确持续核心；类型仍只有car，Map/Region及其他对象未获准。不同数据上的非零覆盖不能等同于总体覆盖改善。'),
('独立目标场景Anchor Hit Rate？',f'{cv["anchor_hits"]}/{cv["transitions"]} = {100*cv["anchor_hit_rate"]:.2f}%，3个VAL验证Log、24帧，命中集中一个Log；独立TRAIN {ct["anchor_hits"]}/{ct["transitions"]}。最终6保留Log未测。'),
('核心平均持续？',f'14个资格Log的有效Anchor平均{allc["mean_valid_anchor_frames"]:.1f}帧；独立VAL唯一有效Anchor持续6帧；开发训练集0次延续；开发验证有效Anchor持续6帧、5次延续。各场景见semantic_core_coverage.json。'),
('事实有实际语言价值吗？','仅承诺car观察类别与M1关联身份连续性，Stable编码不含opaque ID/动态位置。没有证明对驾驶语言的增量价值，不能将car事实视为道路/交通语义。'),
('新Composer实际精度和关键语义？','无核心严格等价通过；实际训练因数据门拒绝，更新0，无SF checkpoint；有核心及整体四指标未测，不能授予质量通过。'),
('实际跨帧Exact Visual KV Tokens？','0。没有正式SF Exact campaign；资格Anchor命中和单元测试均不计入真实服务复用。'),
('Prefill CUDA与M3 Wall改善？','未测量，null；采集的原始计时不是A/B/C配对效率证据。'),
('真实端到端加速？','没有证明；未执行三轮交替、同token控制或场景CI，保留正式配置。'),
('最主要瓶颈？','目标域命中稀疏且集中，新开发训练6个Log仅有1个孤立有效帧、0次核心延续（验证4个Log有5次延续），导致真实训练准入失败；可承诺语义仍贫乏，有核心表示质量及端到端净收益仍待建立。')]
md('result_answers.md','# 本轮14项回答\n\n'+'\n\n'.join(f'{i}. **{q}**\n\n{a}' for i,(q,a) in enumerate(answers,1)))
print(json.dumps(admission['split_counts'],indent=2))
