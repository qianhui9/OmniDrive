"""Reduce completed M1.2.1 evidence, without changing any scoring/acceptance."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from .effective_acceptance import sha,source_manifest


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def run(output):
    out=Path(output);root=out/'main';spec=read(root/'frozen_protocol.json');records=[];micro={};funnel={};evidence={}
    for cohort,variants in [('fixed',list(spec['variants'])),('expanded',spec['expanded_variants'])]:
        micro[cohort]={};funnel[cohort]={}
        for variant in variants:
            path=root/cohort/variant;s=read(path/'summary.json');e=root/(cohort+'_evaluation')/variant
            motion=read(e/'correspondence'/'summary.json');identity=read(e/'identity'/'summary.json');f=read(e/'funnel'/'summary.json')
            assert s['reproduced'] and s['source_unchanged']
            assert read(e/'state_ledger.json')['all_pass'] and read(e/'depth_ledger.json')['all_pass']
            micro[cohort][variant]=dict(operational=s['micro'],macro=s['macro'],latency=s['latency_micro'],
                correspondence=motion['micro'],correspondence_macro=motion['macro'],identity=identity['micro'],identity_macro=identity['macro'],
                qualification=identity['log_qualification'])
            funnel[cohort][variant]=dict(micro=f['micro'],macro=f['macro'],scenes=f['scenes'])
            for x in s['scenes']:
                sid=x['scene_id'];c=x['counts'];m=next(t for t in motion['scenes'] if t['scene_id']==sid);i=next(t for t in identity['scenes'] if t['scene_id']==sid)
                supported=m['observation_supported']['counts'];ic=i['counts']
                records.append(dict(cohort=cohort,variant=variant,scene_id=sid,log_token=x['log_token'],frames=c['frames'],
                    projection_candidates=c['historical_projection_candidates'],supported_anchors=c['observation_supported_anchor_events'],coverage=x['observation_anchor_coverage'],
                    identity_candidates=ic['candidates'],identity_total=ic['total_proposals'],identity_correct=ic['candidate_correct'],identity_wrong=ic['candidate_wrong'],identity_unknown=ic['candidate_unscorable'],
                    observed_points=supported['total'],correct_points=supported['correct'],wrong_points=supported['wrong'],unscorable_points=supported['unscorable'],
                    certified=c['certified_anchor_events'],e2e_p50_ms=x['latency']['end_to_end_wall_ms']['p50'],e2e_p95_ms=x['latency']['end_to_end_wall_ms']['p95']))
            evidence[str(path)]=dict(protocol_sha256=sha(path/'protocol.json'),summary_sha256=sha(path/'summary.json'),
                executed_scenes=s['executed_scene_ids'],unexecuted_scenes=s['unexecuted_scene_ids'],full_dataset_validated=False)
    write(out/'results_micro_macro.json',micro);write(out/'recovery_funnels.json',funnel);write(out/'executed_unexecuted.json',evidence)
    with (out/'scene_comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    tests=ET.parse(out/'regression_gpu.xml');suite=next(tests.iter('testsuite'))
    assert int(suite.attrib['failures'])==int(suite.attrib['errors'])==0
    test_groups={}
    for t in tests.iter('testcase'):
        group=t.attrib['classname'].split('.')[1];d=test_groups.setdefault(group,Counter(total=0,passed=0,failed=0,skipped=0));d['total']+=1
        d['failed' if t.find('failure') is not None or t.find('error') is not None else 'skipped' if t.find('skipped') is not None else 'passed']+=1
    write(out/'test_summary.json',dict(final_suite=suite.attrib,groups=test_groups,
        new_tests=39,initial_sandbox=dict(passed=839,skipped=1,reason='CUDA unavailable; subsequently rerun with real GPU'),
        historical_command_errors=['First regression invocation used nonexistent tests/worldstream_cross_module; collected 0 tests; corrected to worldstream_m123'],
        synthetic_tests_are_not_qualification=True))
    gpu=read(out/'gpu'/'final_verification.json');qual=read(out/'qualification_inventory.json')
    unchanged=spec['source_sha256']==source_manifest()
    assert unchanged and gpu['sources_unchanged'] and gpu['all_scene_native_parity']
    baseline=read(root/'fixed'/'baseline'/'summary.json');old=read('outputs/m12_ego_motion_20261020/run03/baseline/summary.json')
    legacy=read(root/'fixed'/'old_b'/'summary.json');old_b=read('outputs/m12_ego_motion_20261020/run03/b/summary.json')
    comparisons=dict(baseline_count_parity=baseline['micro']['counts']==old['micro']['counts'],
        old_B_count_parity=legacy['micro']['counts']==old_b['micro']['counts'])
    write(out/'legacy_count_parity.json',comparisons)
    gates=dict(Gate_A_Operational=dict(status='PASS',scope='8 independent RGB/cache replay scenes and 2 full-model validation scenes; not complete datasets'),
        Gate_A_Functional=dict(status='FAIL',reason='Multi-scene improvement with preserved independently scorable correspondence quality has not been established'),
        Gate_B=dict(status='BLOCKED',formal_certified=0,qualification_file=None,risk_limit=.05,confidence=.95,
            independent_validation_logs=qual['validation_log_units'],best_case_upper_95=qual['best_case_zero_error_upper_95'],actual_risk_upper_95=None),
        Gate_C=dict(status='PASS',scope='Atomic source/version/depth ledgers, reset/rollback tests; no neural Tensor reuse authorization'),
        acceptance='部分达标：实现和运行验收已完成；核心功能收益未达标，独立资格阻塞，不进入 M2',
        source_unchanged=unchanged,m2_enabled=False,m3_enabled=False,execution_mode='observe_only')
    write(out/'formal_gates.json',gates)
    lines=['# M1.2.1 历史观测恢复与运动补偿实验','',gates['acceptance'],'',
        '本轮所有统计来自本目录的新运行。历史 M1.2 只作为冻结场景和旧计数回归参照。原有实验、失败记录与 M2/M3 代码未覆盖。',
        '',f"最终回归：{suite.attrib['tests']} 项，失败 {suite.attrib['failures']}、错误 {suite.attrib['errors']}、跳过 {suite.attrib['skipped']}。各测试结果见 regression_gpu.xml；新增 39 项见 new_tests_final.xml。",'',
        '固定六场景：每场景独立 manager、store、灰度图像/深度/身份历史、计时与输出；每组12帧、2次重复。使用约0.5秒 keyframe，不是高频视频。先完成固定组回归与离线评估，再执行预声明的两个字典序扩展场景。',
        '', '场景 | Baseline | 旧 B | A | B | C | C+实测裁剪对齐', '---|---:|---:|---:|---:|---:|---:']
    for sid in spec['fixed_scene_ids']:
        cells=[]
        for v in spec['variants']:
            r=next(x for x in records if x['cohort']=='fixed' and x['variant']==v and x['scene_id']==sid)
            cells.append(f"{r['supported_anchors']}/{r['projection_candidates']}")
        lines.append(sid+' | '+' | '.join(cells))
    lines+=['','上表为有效历史 Anchor / 历史投影候选；原始比率分母保持不变，新增实测深度点同样参与局部支持分母，未删除难样本。', '',
        '扩展场景 | Baseline | C | C+实测裁剪对齐','---|---:|---:|---:']
    for sid in spec['expanded_scene_ids']:
        cells=[]
        for v in spec['expanded_variants']:
            r=next(x for x in records if x['cohort']=='expanded' and x['variant']==v and x['scene_id']==sid)
            cells.append(f"{r['supported_anchors']}/{r['projection_candidates']}")
        lines.append(sid+' | '+' | '.join(cells))
    lines+=['','逐场景对应评分、身份正确/错误/不可评分、时延及正式认证数量见 scene_comparison.csv；所有 Micro/Macro、场景均值/中位数/分位数及差异保存在 results_micro_macro.json 和各组 summary.json。未执行场景逐一列于 executed_unexecuted.json，未声称全数据集验证。','',
        '组别 | Anchor 支持率 Micro | Macro 均值 | 对应正确/错误/不可评分（排除后） | 身份候选正确/错误/不可评分 | E2E P50/P95 ms','---|---:|---:|---:|---:|---:']
    for cohort,variants in [('fixed',list(spec['variants'])),('expanded',spec['expanded_variants'])]:
        for v in variants:
            x=micro[cohort][v];c=x['operational']['counts'];m=x['correspondence']['observation_supported']['counts'];i=x['identity']['counts'];lat=x['latency']['end_to_end_wall_ms']
            avg=x['macro']['observation_anchor_coverage']['mean']
            lines.append(f"{cohort}/{v} | {c['observation_supported_anchor_events']}/{c['historical_projection_candidates']} | {avg:.6f} | {m['correct']}/{m['wrong']}/{m['unscorable']} | {i['candidate_correct']}/{i['candidate_wrong']}/{i['candidate_unscorable']} | {lat['p50']:.1f}/{lat['p95']:.1f}")
    lines+=['','延迟为实际运行观测值，CPU 多组实验和 GPU 验证有并行时段，墙钟差值受资源竞争影响；不据此宣称加速。m1_cpu_process_ms 含各线程 CPU 开销；完整逐帧 Precheck/Commit/CPU/E2E 及 P50/P95 保存在各组 summary.json。', '',
        '核心改动和调用链：真实 RGB/位姿/标定输入 → 原有来源与历史验证 → 保留 GFT 原始点并追加此前真实跟踪深度点 → raw/rotation/depth 统一完整验证后选择 → 原有静态几何/遮挡/Anchor 支持门 → 可选的实测仿射裁剪描述子 → 提交当前模型感知 → 独立资格门 → 原子发布状态及下一帧深度历史。当前帧生成的深度只供下一帧使用。', '',
        '此前已有世界坐标投影、相对相机极线验证及 RGB 三角化；新增的是持久实测像素关联、双视图轴向扰动深度区间、多候选完整必要门后消歧、暂定运动搜索和离线 Oracle。没有用全局单应解释不同深度。实测仿射仅在空间分布、留出点误差、纹理、描述子和原有独立可见性门全部通过时提供裁剪观察支持，不认证前景归属或静止性。', '',
        '深度记录绑定 scene/frame/log/epoch/execution/commit version 和相机图像/位姿/K/映射摘要。原子提交失败不发布新深度，场景切换清空深度和身份历史。每帧每相机漏斗和拒绝原因见 main/*_evaluation/*/funnel；逐 Anchor 相机原始检查和图像保存在各组 scenes。', '',
        '搜索运动读取同场景前三次真实提交中心和版本链，不读取检测速度/GT；方向变化、关联歧义、时间异常或不确定性过大保持 Unknown。仅候选边和 LK 初始化读取假设；identity_tracking_policy 强制 shadow，普通关联、可信速度、正式运动历史和资格门不因搜索成功改变。world_motion.py 严格条件保留。', '',
        '深度/候选/运动漏斗详见 recovery_funnels.json。旧 B 的 nearest-match 计数沿用旧诊断命名，不代表新增的一对一歧义验证；新 B/C 才执行一对一检查。恢复数的新定义为 raw 完整必要门未通过而其他候选完整通过，不能直接与旧 LK-only recovered 计数比较。', '',
        'Oracle 在所有在线帧提交后独立读取源 LiDAR，限定1px近邻、前表面、深度范围/间断、既有4px预测残差及完整局部门。共享完全相同源像素人口比较 raw/RGB/独立稀疏深度初始化，缺参考保留 Unknown。没有状态提交或证书调用；main/oracle_strict 保存最终逐点记录、每场景、Micro/Macro。它是稀疏条件静态诊断，不能保证稠密深度上界，也不反推 Anchor counterfactual 支持（Not Run）。早期 main/oracle 未加4px预测残差的诊断结果保留，仅作审计存档，不纳入最终验收；复核补齐后在新目录重跑，39项相关测试重新通过。', '',
        '原 motion_scoring.py 的目标近邻1px、误差2px规则未修改。source_anchored_scoring.py 另行按源像素选独立参考，只在 Oracle 的 additional_source_anchored_measured 中单列，不替换旧评分。稀疏 LiDAR、相机异步和刚体标注近似导致大量 Unknown；不同组可评分子集不同，少量点准确率不具普遍性，不能当作身份准确率或资格风险置信上界。', '',
        '原生模型对照：两个冻结验证场景各12帧，M1关闭/开启的检测、地图、Planning/语言输出与模型计算摘要一致；见 gpu/final_verification.json、各场景 results.json / language_comparison.json。无 M2/M3 复用、网络跳算或神经 Tensor/KV 复用。', '',
        f"Gate B：验证集仅 {qual['validation_log_units']} 个独立日志，零错误最佳单侧95%上界仍为 {qual['best_case_zero_error_upper_95']:.6f}，大于0.05。无当前生产者资格文件、无正式接受人口，实际风险及上界为 null。59 个独立零错误验证单位的要求未改变；增加同日志场景/帧数不能替代。", '',
        'Gate_A_Operational PASS（已执行范围）；Gate_A_Functional FAIL（尚未建立多场景覆盖及质量收益）；Gate_B BLOCKED；Gate_C PASS（状态来源/版本/失效接口）。不进入 M2。', '',
        '复现：taskset -c 0-3,6-31 bash outputs/m121_history_recovery_20261020/reproduce.sh outputs/m121_reproduce_NEW。必须使用新目录；全部命令使用 OmniDrive 解释器并避开 CPU4/5。GPU命令仅在空闲时运行，不停止其他任务。', '',
        '初次回归命令误写跨模块目录，收集0项并失败，保留 regression_initial.log/xml；修正目录后839通过、CUDA一项跳过，之后在真实 GPU 环境完整重跑。探索 smoke 为3帧，只作工具调试，不加入固定组或资格统计。', '',
        '剩余最小修复：为零覆盖场景逐项解决深度/遮挡/描述子和空间支持的必要证据缺口；对大不确定性运动假设收集真实可验证轨迹及独立不确定性校准，不能直接放宽像素预算；获取更多独立日志与更密集可评分对应，再验证覆盖增益与错误关联变化。']
    lines+=['','固定组深度及搜索完整漏斗（每个比率的原始分子/分母见 recovery_funnels.json）：','',
        '组别 | 追加实测点 | 深度匹配/查询 | depth 尝试/完整支持/选中 | raw 完整门失败后的恢复 | 运动假设/尝试 | 动态图像尝试/选中',
        '---|---:|---:|---:|---:|---:|---:']
    for v in spec['variants']:
        c=funnel['fixed'][v]['micro']['counts']
        def n(k):return c.get(k,0)
        depth='Not Run' if v in ('baseline','a') else f"{n('depth_association_one_to_one')}/{n('depth_association_query_points')}"
        stages=(f"{n('candidate_depth_attempted')}/{n('candidate_depth_local_supported')}/{n('candidate_depth_selected')}" if v in ('b','c','aligned')
                else 'Not Available（旧接口无分组）' if v=='old_b' else 'Not Run')
        recovered=str(n('seed_recovered')) if v in ('a','b','c','aligned') else 'Not Available（旧定义）' if v=='old_b' else 'Not Run'
        search=f"{n('dynamic_search_generated')}/{n('dynamic_search_attempts')}" if v in ('c','aligned') else 'Not Run'
        image=f"{n('dynamic_image_attempted')}/{n('dynamic_image_selected')}（有候选但全拒）" if v in ('c','aligned') else 'Not Run'
        lines.append(f"{v} | {n('retained_points')} | {depth} | {stages} | {recovered} | {search} | {image}")
    lines+=['','阶段计数用斜杠连接时不代表比率。新 B/C 深度匹配12281/153663的分母只包括存在历史深度的查询相机人口；按全部444391个源点计为12281/444391（约2.76%）。候选未执行、旧接口未分组以及有候选却全部拒绝已分别标记；零分母比率仍为 null。']
    lines+=['','Oracle（固定六场景同一源像素人口；数字为完整局部验证后，非身份或 Anchor 准确率）：','',
        '方案 | 完整局部验证 | 原评分正确/错误/不可评分 | 源锚定另评分正确/错误/不可评分 | 共享稀疏源深度分母',
        '---|---:|---:|---:|---:']
    oracle=read(root/'oracle_strict'/'summary.json')
    for arm,x in oracle['micro'].items():
        a=x['measured']['counts'];b=x['additional_source_anchored_measured']['counts']
        lines.append(f"{arm} | {a['total']} | {a['correct']}/{a['wrong']}/{a['unscorable']} | {b['correct']}/{b['wrong']}/{b['unscorable']} | {x['common_reference_source_denominator']}")
    lines+=['','Oracle 全部源像素数相同；上表完整支持后人口不同，公平的共享参考子集计数另保留 common_sparse_reference；不能把其比率差异解释为普遍准确性或统计显著收益。', '',
        '修改文件及目的：','',
        '| 文件 | 目的 |','|---|---|',
        '| worldstream/m1/config.py | 新功能显式 opt-in；搜索只允许 temporal shadow；旧配置默认行为保留 |',
        '| worldstream/m1/ego_motion.py | 唯一实际像素深度关联、轴向扰动区间、raw/rotation/depth/search 实测候选 |',
        '| worldstream/m1/correspondence_recovery.py（新） | 保留所有原始角点，追加真实深度轨迹；统一完整必要门选择、留出实测裁剪对齐 |',
        '| worldstream/m1/search_motion.py（新） | 已提交暂定中心的短时搜索假设；不写可信速度或证书 |',
        '| worldstream/m1/temporal_visibility.py | 接入真实深度轨迹与多候选诊断，继续原有后置遮挡和支持门 |',
        '| worldstream/m1/identity_temporal.py | 在有限身份视图中消费搜索初始化和深度轨迹，保留竞争与周期检查 |',
        '| worldstream/m1/observation_validator.py | 可选实测裁剪对齐；同一描述子阈值和独立可见性门 |',
        '| worldstream/m1/state_manager.py | 只扩展影子身份搜索候选，真实历史原子维护及诊断导出 |',
        '| configs/worldstream_m121_baseline.yaml、worldstream_m121_old_b.yaml、worldstream_m121_a.yaml、worldstream_m121_b.yaml、worldstream_m121_c.yaml、worldstream_m121_aligned.yaml | 六组冻结设置，门限一致，无独立资格文件 |',
        '| tests/worldstream_m1/test_history_recovery.py（新） | 39项真实测量/合成几何/来源/歧义/事务/搜索隔离/Oracle数据流测试 |',
        '| tools/worldstream_m1/history_recovery_experiment.py（新） | 冻结同M1.2场景及预声明扩展，调用既有实验回放 |',
        '| tools/worldstream_m1/history_recovery_metrics.py（新） | 独立场景深度、候选、运动漏斗与 Micro/Macro |',
        '| tools/worldstream_m1/depth_oracle_diagnosis.py（新） | 提交之后的纯离线稀疏深度初始化对照，不调用 manager/commit/certificate |',
        '| tools/worldstream_m1/source_anchored_scoring.py（新） | 单列源锚定稀疏评分，不替换既有规则 |',
        '| tools/worldstream_m1/postprocess_history_recovery.py（新） | 已完成提交的既有独立评分、状态和深度账本验证 |',
        '| tools/worldstream_m1/history_recovery_report.py（新） | 从完成的原始证据汇总交付，不改变评分/接受 |',
        '| tools/worldstream_m1/history_recovery_cases.py（新） | 只画真实LK有效测量的恢复/拒绝案例，缺案例明确标记 |',
        '| tools/worldstream_m1/search_motion_revalidate.py（新） | 无GT重建历史中心/来源/运动及像素不确定性，验证零激活原因 |',
        '| outputs/m121_history_recovery_20261020/ | 新实验及复现脚本、代码摘要、逐项测试和失败记录；不覆盖历史结果 |']
    lines+=['','进一步诊断见 [diagnostic_findings.md](diagnostic_findings.md)。相对 Baseline，额外对齐组逐事件为新增3个、丢失2个，净增1个；不能用净数量掩盖旧支持损失。完整得失来源见 support_event_changes.json。三个动态对象为主的零覆盖场景首先缺运动/身份资格；另外两个缺历史裁剪/描述子/空间证据，不能归因于同一个光流瓶颈。', '',
        '固定组时延开销（P50/P95 ms；并行条件下的描述统计，非受控加速结论）：','',
        '组别 | Precheck | Commit | M1 CPU进程时间 | E2E','---|---:|---:|---:|---:']
    for v in spec['variants']:
        t=micro['fixed'][v]['latency'];cells=[f"{t[k]['p50']:.1f}/{t[k]['p95']:.1f}" for k in ('precheck_wall_ms','commit_wall_ms','m1_cpu_process_ms','end_to_end_wall_ms')]
        lines.append(v+' | '+' | '.join(cells))
    lines+=['','新增机制有真实CPU成本：多候选LK及完整独立门、历史深度不确定性传播、来源审计都继续执行。没有以覆盖增加推导加速。', '',
        '原生完整模型逐场景（M1开启；同场景关闭/开启各12帧）：','',
        '场景 | Anchor支持/候选 | 身份候选/提案 | 正式认证 | 原生输出/语言字节一致','---|---:|---:|---:|---:']
    gs=read(out/'gpu'/'summary.json')
    for x in gs['scenes']:
        c=x['counts'];lines.append(f"{x['scene_id']} | {c['observation_supported_anchor_events']}/{c['historical_projection_candidates']} | {c['identity_candidate_proposals']}/{c['identity_proposals']} | {c['certified_anchor_events']} | true")
    lines+=['','完整回归区分：旧M1用例496项 + 本轮新增39项 = M1共535项；M2旧回归127项、M3旧回归131项、跨模块旧回归55项。全部848项实际通过；这些测试不充当独立风险资格样本。', '',
        '未运行：高频视频验证、全训练/验证数据集模型推理、具有合格独立日志数的生产者认证、Oracle反事实Anchor提交实验。M2/M3复用和GPU跳算不属于本轮实现。']
    (out/'experiment_report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args();run(a.output)


if __name__=='__main__':main()
