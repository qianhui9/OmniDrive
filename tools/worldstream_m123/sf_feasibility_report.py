"""Evidence-linked No-Go delivery after data and actual CUDA control analysis."""
import json
from pathlib import Path
from tools.worldstream_m123.benchmark.common import write
from tools.worldstream_m123.sf_feasibility import PRIOR,rows


def report(root):
    def read(n):return json.loads((root/n).read_text())
    def md(n,t):(root/n).write_text(t.strip()+'\n')
    d=read('core_event_distribution.json');gate=read('feasibility_decision.json')
    if gate['training_go']:raise ValueError('Positive gate requires actual training/quality, not a No-Go report')
    bound=read('kv_speedup_upper_bound.json');micro=read('controlled_micro/summary.json')
    comp=read('controlled_micro/composer_cost.json');encoder=read('controlled_micro/stable_encoder_audit.json')
    semantic=read('controlled_micro/semantic_diagnostics.json');old=read('old_48_diagnosis.json')
    raw,enriched,val=d['raw_train'],d['enriched_train'],d['validation']
    w1,w3=bound['workloads']['1'],bound['workloads']['3'];cost=comp['summary']['warm']['wall_ms']['mean']
    manifest=read('training_dataset_manifest.json');pool=read('official_train_pool_audit.json')
    win=['|窗口|原6训练Log有效帧/延续|全部14训练Log有效帧/延续|开发验证有效帧/延续|',
         '|---|---:|---:|---:|']
    for n,r in d['windows'].items():
        cells=[str(n)]+[f"{r[k]['valid_frames']}/{r[k]['anchor_extensions']}" for k in ('original_train','all_train','validation')]
        win.append('|'+ '|'.join(cells)+'|')
    md('data_feasibility_report.md',f'''
# 真实训练数据可行性

结论：数据门 **{'GO' if gate['data_go'] else 'NO-GO'}**。全官方TRAIN元数据共{pool['frames']}帧、{pool['scenes']}场景、{pool['logs']}Log；有{pool['eligible_scenes']}个场景满足当前保护Log排除、真实连续32帧、六相机存在及原始前三问要求。没有机械排除全部历史研究TRAIN。历史源索引与Log身份见 training_historical_exposure.json；不称这些数据全历史未见。

真实审计576帧，原6训练Log加8个合法历史正样本Log、原4开发验证Log，各32帧；在线M1/M2按时间因果执行，零LLM Decode。当前资格/开发验证/最终保留Log禁止进入训练；最终6个验收Log未打开，资格SHA和门槛原样保持。audit_sequence.json在输出前冻结，GT在commit之后评分，不用于当前状态。

{chr(10).join(win)}

原48训练帧中47帧没有合格事实。原始快照car首个拒绝原因：{json.dumps(old['first_rejections_car'],ensure_ascii=False)}。这些是候选首失败计数，不能相加解释为独立帧因果比例。原6Log延长至32帧仍0次延续，说明窗口短不是这些Log的主要可恢复瓶颈；增加有真实正事件的合法Log才获得108次延续。

原始14训练候选Log：{raw['frames']}帧、{raw['valid_frames']}有效帧、{raw['anchor_extensions']}次延续、{raw['valid_sequences']}个有效Anchor（其中{raw['sustained_sequences']}个真正持续）、{raw['positive_logs']}个正样本Log，有效Anchor平均{raw['mean_valid_anchor_lifetime']:.3f}帧。训练事件富集后{enriched['frames']}帧，保留全部{enriched['valid_frames']}有效帧和{enriched['anchor_extensions']}次延续。开发验证维持完整{val['frames']}帧，{val['valid_frames']}有效帧、{val['anchor_extensions']}次延续，来自{val['positive_logs']}个正Log，平均有效Anchor{val['mean_valid_anchor_lifetime']:.3f}帧。原Anchor寿命上限8帧保持不变，horizon到期也计入刷新，不为提高命中延长上限。

训练富集只选真实有效事件前后各2帧、真实无核心边界和稀疏常态帧；不重复帧、不改时间戳/Track ID、不把未来信息写入快照。稀疏样本之间的缺帧不算连续转换。独立部分Oracle对训练/开发验证延续分别评分{raw['retained_fact_offline_checks']} / {val['retained_fact_offline_checks']}，未知独立计数，零观测错误不代表总体零风险。原始候选池本身按历史正证据选过，也不是自然部署覆盖率。

逐过滤器计数：{json.dumps(d['independent_filter_counts'],ensure_ascii=False)}。涉及检测分数、残差、上一帧关联缺失、当前支持和ID撤销。Core事件：{json.dumps(d['event_counts'],ensure_ascii=False)}。物理遮挡、漏检与关联ID变化在缺少可独立验证可见性标注时不能强行区分；报告保留这一不可识别性。每帧候选、合格事实、撤销信息及原始M1诊断在服务器raw审计，精简事件见 core_event_timeline.jsonl。

训练准入定义冻结在experiment_plan.json/profit_gate_definition.json：至少3正Log、6序列、24有效帧、12次已评分正确延续，以及独立开发验证正事件；错误0、未知≤5%不放松。现有wrcps_train.py --preflight-only已实际通过，产物在training_preflight/data_validation.json。数据不是本轮最终停止原因。
''')
    effect=[x for r in semantic for k,x in r['diagnostics'].items() if k!='untrained_car']
    max_delta=max(x['max_abs_last_prefill_logit_change'] for x in effect)
    md('semantic_value_analysis.md',f'''
# Stable Prefix的信息与语义价值

代码审计：WorldPrefixComposer.encode_stable只编码排序后的(kind,label)，opaque Anchor ID只控制缓存寿命。所有有效单事实car状态均对应相同字节 `[["object","car"]]`；无论训练后参数是什么，不同车辆ID和场景的内容编码仍相同。实测两个不同真实Anchor的当前编码torch.equal=True。条件在“有效单car核心”上，语义内容变化熵为0；这不等于声称所有64个query的embedding彼此相同，也不等于有效性/连续观察这一条件本身没有价值。

K64可以表现为一个由可靠car存在条件触发的学习soft prompt，但没有对象位置、速度、具体身份、车道或道路关系。原生视觉已经提供车辆视觉证据，尚未证明这份类别/稳定性条件对驾驶问答有增量价值。64个learned queries并不会凭空增加世界事实种类。对信息容量的结构性审计不依赖随机权重优劣。

有限诊断使用旧真实开发核心连续帧的3个原始自然问题，无历史KV。固定Dynamic449、Stable64布局、LLM和RNG，仅比较当前未训练car编码、Stable64全零反事实、bus类别反事实；Native完整视觉另列。最大prefill最后位置logit变化{max_delta:.6f}，说明改变Stable输入能影响模型。原始全部答案/参考、tokens、KL和关键标签辅助量见 controlled_micro/semantic_diagnostics.json。

这些是明确标记的反事实与未训练结构探针，不是合法bus在线事实、训练后的SF精度、自然场景改进或物理真值证据。若答案/F1发生变化，也不能由随机权重推断训练后有用或无用。Teacher/参考可能有偏差；不将Teacher单独当作物理真值。本轮没有已训练新Adapter，因此“已证明实用语言价值”答案仍为否。

实际案例：第一问灯色参考为green，Native为red，二者本身冲突；未训练car前缀产生重复数字。第三问oncoming traffic，参考与Native均为No，未训练car前缀为“There are several”，bus反事实为Yes并重复文本。这说明当前未经适配的Stable输入影响很大且没有可解释的实用语义证据，不能将变化当作正确car事实帮助。派生数字溢出记录Unknown，原始自然文本和tokens完整保存。收益No-Go独立来自硬件/覆盖估计，未用这些随机输出作为正式质量否决或已训练模型结论。

Map/Region维持前轮审计结论：lane_control_points潜变量缺少低成本独立道路类别/拓扑身份生产者，不新增地图认证或Tracking系统，不写入Stable Core。
''')
    lines=['|控制|模型Prefill CUDA stream ms|完整控制调用Wall ms|Planner Wall ms|','|---|---:|---:|---:|']
    for kind,r in micro['by_kind'].items():
        lines.append(f"|{kind}|{r['model_prefill_cuda_ms']['mean']:.4f}|{r['executor_wall_ms']['mean']:.4f}|{r['planner_wall_ms']['mean']:.4f}|")
    workloads=[]
    for label,w in [('每帧1问',w1),('每帧3问',w3)]:
        ci=w['timing_bootstrap_ci95'];workloads.append(f"{label}：Stable64机会{w['counters']['crossframe_stable64_exact_candidates']}/{w['counters']['requests']}，预期纯prefill节省{w['optimistic_prefill_saved_ms_per_request']:.4f}ms/request；总净值估计{w['estimated_net_ms_per_request']:.4f}ms、计时区间{ci['total_net_ms']}；扣除Composer/状态成本后的增量世界前缀净值{w['incremental_world_prefix_net_ms_per_request']:.4f}ms、区间{ci['marginal_world_net_ms']}。")
    md('performance_report.md',f'''
# CUDA潜在收益与正式效率边界

本轮完成真实原生LLM和原PrefillExecutor的有限CUDA控制：3个原始问题、2轮预热、9轮有效测量、5种交替控制，共{micro['measured_rows']}条有效计时。完整prefill与text/text+64/same-frame-prefix的embedding/position/mask严格一致；源来自真实核心延续相邻帧。控制性执行器实际suffix长度逐次断言，跳过量与forward长度严格相符。每次只生成1token/零decode-forward是micro定义，不是缩短自然服务来制造加速。另有5个Kineto GPU活动探针，独立于Wall统计。

{chr(10).join(lines)}

CUDA列是包围真实model forward的event stream区间，包含host提交空隙，不冒充纯kernel时间。GPU活动时长和事件数单列于 controlled_micro/gpu_activity_probes.json；CPU控制与完整Wall单列。Cache Branch、owned KV clone及cache更新在既有Executor的控制调用内计时，Native真实生成作为共同开销参照。

Stable64相比仅固定文本的模型CUDA边际节省均值 {micro['summary']['text_minus_text64_model_cuda_ms']['mean']:.4f}ms/hit；完整prefill减text+64为 {micro['summary']['full_minus_text64_model_cuda_ms']['mean']:.4f}ms/hit。K64 Composer真实结构warm Wall均值{cost:.4f}ms/core-valid帧；同帧3问只构造一次，已按请求摊分，cold/slot matching及GPU区间在composer_cost.json。成本使用未训练参数，只作结构/hardware测量，不作质量或正式SF加速证据。

{chr(10).join(workloads)}

覆盖依据是旧独立资格VAL的3Log/24帧真实生命周期，而非本轮富集训练率。回放按单entry、场景/layout重置、原问题顺序计算机会；同帧收益单列，不能解释成世界跨帧收益。真实现行Scheduler缺少新Adapter/Layout的质量成本绑定，实测准入0；乐观估计只假设未来对真正Exact候选校准准入，不假设Anchor命中100%。

期望总净值以Native Wall减去按机会加权的Executor Wall、Planner、warm Composer及资格选择/Anchor CPU回放成本。完整候选提取/protocol hashing、M3执行signature、SF展开、额外allocator成本未测而乐观置0；新建Anchor也用较低warm Composer成本，因此结果偏向方法。它不是已测完整M3/Request净值。bootstrap仅描述9轮micro计时不确定性，不是场景配对E2E置信区间。

独立资格原始输入长度569–588tokens，本次硬件控制为572/574/575tokens。成本转移仅限这一有限近邻长度假设，不是全局服务成本模型。首次控制脚本解包错误在预热阶段断言退出；第二次完成135条计时后因未训练反事实的非有限派生数字指标写JSON失败。两次失败日志保留，第二次完整计时也保留在earlier_timing_run_sensitivity，未隐藏；最终135条控制与完整开销来自成功重放。规则/问题/参数未因这些结果改变。非有限外部诊断标Unknown，性能计时仍严格要求finite。

收益门：**{'GO' if gate['speed_go'] else 'NO-GO'}**。按测量前冻结的规则，1问总净值和世界前缀增量净值的95%计时下界必须同时>0；多问同帧通用收益不能替代跨帧世界前缀收益。结论是当前单car/K64路线没有足够稳健的训练投资收益证据，不是数学证明一切硬件/布局都不可能加速。

正式A/B/C自然服务、固定完整参考tokens/相同decode步数、三轮交替及场景级配对CI均NOT_RUN。真实正式Exact Visual KV tokens=0，实际服务跳过prefill tokens=0，正式B→C Prefill/M3 Wall改善、A→C E2E speedup、FPS/QPS、自然Decode CUDA均null。micro控制跳过量绝不填入正式复用。
''')
    md('composer_training_report.md',f'''
# K64 Composer条件性训练

数据扩展已解除上轮训练阻塞：训练富集{enriched['frames']}帧/{enriched['valid_frames']}有效帧/{enriched['anchor_extensions']}次正确延续，开发验证{val['frames']}帧/{val['valid_frames']}有效帧/{val['anchor_extensions']}次正确延续；真实旧训练入口预检通过。

本轮训练状态 **NOT_RUN**：收益门未通过，按用户要求在高成本训练前停止。没有优化器创建/更新，没有新Adapter、最佳epoch、训练/验证loss或真实训练峰值显存（全部null/0）。这与上轮“数据不足拒绝训练”是不同停止原因，不能混淆。

已提供可直接交给现有wrcps_train.py的training_dataset_manifest.json，保留原始前三问及答案、真实载体SHA、当前因果快照、官方TRAIN校验、Log分离和无核心状态。预定K64/rank128/3epoch，原有Answer KL+Task CE+方向KL及梯度检查点不改。只有未来在同样严格质量和收益证据下改变Go判断，才有理由实际训练；本轮没有运行K32/K128分支或修改阈值。
''')
    md('core_valid_quality_report.md','''
# Core-Valid质量

NOT_RUN：收益前置门未通过，没有训练后的SF Adapter。未训练编码反事实仅说明Stable输入能影响原生LLM，不授予空间/运动/数量或类别质量通过。无法回答已训练K64相对Native是否满足有效核心质量预算。

无核心Native identity、旧Bus Stop修复及612项CUDA合同属于前轮已验证工程成果，本轮原样保留；它们不代替有效核心子集的独立质量。真实捕获中的所有Core来自当前/过去观测，离线GT仅评分；没有用GT生产当前Core。
''')
    md('quality_report.md','''
# 正式质量

NOT_RUN：没有新的已训练SF Adapter，收益门先No-Go，最终保留验收数据未打开。mAP、NDS、DET_l、Token F1、Core-Valid/No-Core/整体新质量及其下降均为null。四项绝对下降预算0.10和关键语义要求保持不变。没有把旧实验指标或随机Composer的反事实F1当成本轮正式质量。

方向、前后、数量、运动、车道、信号及新增/消失对象均没有已训练SF的验收证据，不宣称“无新增严重错误”。反事实原始答案全部独立保留，Teacher不是物理真值。
''')
    write(root/'crossframe_reuse_summary.json',dict(formal_SF_service_executed=False,
        actual_crossframe_visual_KV_tokens=0,actual_skipped_service_prefill_tokens=0,
        actual_service_reuse_requests=0,micro_control_rows=micro['measured_rows'],
        micro_control_token_skips_excluded_from_service=True,
        independent_lifecycle_opportunities={k:v['counters'] for k,v in bound['workloads'].items()},
        deployed_new_SF_scheduler_admissions=0,formal_B_minus_C_prefill_CUDA_ms=None,
        formal_B_minus_C_M3_wall_ms=None,formal_A_div_C_E2E_speedup=None,
        reason='Data admission passes; potential net gain gate does not pass before training.'))
    write(root/'deployment_decision.json',dict(recommended=False,replace_formal_configuration=False,
        data_go=gate['data_go'],speed_go=gate['speed_go'],training_started=False,optimizer_updates=0,
        trained_SF_adapter=None,core_valid_quality=None,overall_quality=None,ExactKV=None,E2E=None,
        official_metrics=dict(mAP=None,NDS=None,DET_l=None,Token_F1=None),
        acceptance_logs_opened=False,stop_stage='controlled_computation_benefit_feasibility',
        reason=gate['reason'],formal_configuration='projects/configs/OmniDrive/worldstream_m123_auto.py'))
    answers=[
        ('为什么过去48训练帧只有1个有效Core？',f"47帧无合格事实；car首失败为{old['first_rejections_car']}。分数/连续观测及ID当前支持限制占主，残差是更后级过滤；不能将失观测全部解释为物理遮挡。"),
        ('扩大窗口提高多少？','原6训练Log：8/16/32帧窗口有效帧1/2/3，延续始终0。全部14训练Log有效帧45/93/195、延续28/53/108；开发验证6/13/22有效帧、5/9/11延续。'),
        ('更多合法TRAIN Log有足够正样本吗？',f"是，数据门GO。14训练Log中8个正Log，{raw['valid_sequences']}个有效Anchor、{raw['sustained_sequences']}条真正持续序列，108次已评分正确延续，0错误/未知；保护资格/开发验证/验收Log不进训练。历史暴露已披露。"),
        ('富集前后规模？',f"训练原始候选448帧、195有效/108延续；富集296帧、195有效/108延续。有效帧占比{195/448:.2%}→{195/296:.2%}，不能当作部署覆盖率。验证完整128帧、22有效/11延续。"),
        ('car Prefix有语言语义价值吗？','所有有效单car事实的内容编码相同，ID不进embedding；有效性条件可能有价值，但尚无训练后增量语言效用证据。三问有限反事实证明能影响logits/回答，不证明影响有用。'),
        ('K64潜在信息损失？','以64稳定槽替换64原生视觉槽，object/map各移除32并修改部分合并目标。Ego/未混合槽保留，但空间/运动/数量依赖仍可能丢失；原生latent索引不是物理邻接。'),
        ('真实覆盖下prefill节省？',f"旧独立VAL Anchor5/21转换；按请求和layout回放，1问Stable64机会{w1['counters']['crossframe_stable64_exact_candidates']}/{w1['counters']['requests']}，模型预期prefill节省{w1['optimistic_prefill_saved_ms_per_request']:.4f}ms/request。CUDA上限与服务结果分列。"),
        ('扣除额外成本有净收益吗？',f"单问乐观总净值{w1['estimated_net_ms_per_request']:.4f}ms/request，世界前缀增量{w1['incremental_world_prefix_net_ms_per_request']:.4f}ms/request；相应95%计时区间{w1['timing_bootstrap_ci95']}。收益门No-Go。未测额外成本置0，已偏向方法。"),
        ('完成训练了吗？','否。现有训练预检已通过，但收益门未通过，因此优化器更新0，新checkpoint/loss/训练显存null；不是再次数据拒绝训练。'),
        ('有效核心关键质量达标吗？','尚无已训练SF证据，NOT_RUN。未训练的反事实、无核心identity及旧回归均不能代替验收。'),
        ('新独立场景命中率和寿命？',f"本轮独立于训练Log的开发验证11/124转换（{11/124:.2%}），有效Anchor均值2帧，2/4 Log有延续；这些是既有开发验证Log扩窗，非完全未见最终VAL。旧独立资格VAL5/21、单Anchor6帧用于收益机会估计。最终保留场景未测。"),
        ('真实Exact Visual KV tokens多少？','正式服务0；micro控制执行确实按原Executor跳过了合法prefix，但所有控制tokens从正式服务统计中排除。'),
        ('B→C真实Prefill/M3 Wall改善？','NOT_RUN/null。只有控制性全prefill/text/text+64及同帧prefix计时，不是已训练B/C因果服务对照。'),
        ('A→C可重复E2E净加速？','未证明。未执行完整自然tokens、相同decode步数、三轮正式交替和场景配对95%CI。多问同帧控制收益属于通用同帧复用，不能冒充跨帧世界收益。'),
        ('当前路线是否值得继续？','数据阻塞已解决，工程实现保留；在当前单car内容、K64预算及真实稀疏覆盖下，收益前置门不足，建议暂停高成本训练和部署替换。该判断限于当前结构/硬件/覆盖证据，不是宣称所有世界状态KV方法不可能有效。')]
    md('result_answers.md','# 15项问题回答\n\n'+'\n\n'.join(f'{i}. **{q}**\n\n{a}' for i,(q,a) in enumerate(answers,1)))
    write(root/'request_scopes.json',dict(causal_state_frames=576,state_audit_LLM_decode_requests=0,
        controlled_micro_measured_rows=micro['measured_rows'],diagnostic_original_questions=len(semantic),
        semantic_natural_generations=len(semantic)*4,formal_service_requests=0,
        warning='Control max_new_tokens1/zeroDecode and untrained semantic counterfactuals are NOT formal service or quality requests.'))
    print(json.dumps(dict(data_go=gate['data_go'],speed_go=gate['speed_go'],reports_written=True),indent=2))


def main():
    import argparse,os
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);a=p.parse_args()
    if {4,5}&set(os.sched_getaffinity(0)):p.error('Use taskset -c 0-3,6-31')
    report(Path(a.root))


if __name__=='__main__':main()
