"""Publish measured results and explicit No-Go/Not Run outcomes without qualification fabrication."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from .quality_coverage_audit import ROOT, sha
from .quality_coverage_experiments import methods, source_manifest
from .io_utils import write_json


def freeze():
    target=ROOT/'frozen_validation_protocol.json'
    if target.exists():raise RuntimeError('Already frozen; never reselect from heldout/final')
    selection=json.loads((ROOT/'data_selection.json').read_text())
    calibration=ROOT/'calibration_summary.json'
    result=json.loads(calibration.read_text()) if calibration.exists() else {}
    chosen_identity=None;chosen_region=None
    if result.get('status')=='Run':
        identity=result.get('identity',{}).get('temporal_identity',{})
        risk=identity.get('log_risk',{})
        if (identity.get('independently_correct_proposals',0)>0 and risk.get('incomplete_accepted_logs')==0
                and risk.get('complete_log_event_upper95') is not None and risk['complete_log_event_upper95']<=.05):
            chosen_identity='temporal_identity'
        region=result.get('region',{}).get('surface_local_plane',{})
        baseline=result.get('region',{}).get('conservative',{})
        if (region.get('risk_gates_passed') and region.get('independently_correct_stable_support',0)>0
                and region['effective_stable_area']>baseline['effective_stable_area']):chosen_region='surface_local_plane'
    decision=dict(identity_method=chosen_identity,region_method=chosen_region,qualified_physical_producer=None,
        calibration_status=result.get('status','Not Run'),no_final_reselection=True,
        reason='No unexposed accessible acquisition logs' if not selection['splits']['calibration'] else 'Frozen Calibration hard constraints')
    write_json(ROOT/'algorithm_selection.json',decision)
    write_json(target,dict(selection=decision,source_sha256=source_manifest(),methods=methods(),
        experiment_protocol_sha256=sha(ROOT/'experiment_protocol.json'),data_selection_sha256=sha(ROOT/'data_selection.json'),
        frozen_before_heldout_final=True,latency_in_selection=False))
    for stage in ('heldout','final_test'):
        if not selection['splits'][stage]:
            write_json(ROOT/(stage+'_summary.json'),dict(status='Not Run',reason='No new independent log available',
                selected_identity_method=chosen_identity,selected_region_method=chosen_region,
                qualified_certifications=0,qualified_FAR=None,effective_stable_area_ratio=None,
                old_exposed_results_reused_as_fresh=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');args=p.parse_args()
    if args.freeze:freeze();return
    frozen=json.loads((ROOT/'frozen_validation_protocol.json').read_text())
    if source_manifest()!=frozen['source_sha256']:raise RuntimeError('Source changed after freeze')
    development=json.loads((ROOT/'development_summary.json').read_text())
    identities=development['identity'];regions=development['region']
    data=json.loads((ROOT/'data_selection.json').read_text())
    regression={}
    for name in ('regression','regression_gpu'):
        xml=ROOT/(name+'.xml')
        if xml.exists():
            tree=ET.parse(xml);suites=tree.findall('.//testsuite')
            regression[name]={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    write_json(ROOT/'regression_results.json',regression)
    gpu=ROOT/'gpu_parity/environment.json';comparison=ROOT/'gpu_parity/comparison.json'
    parity=dict(status='Not Run',reason='No successful current-code CUDA parity result available')
    if gpu.exists():
        env=json.loads(gpu.read_text())
        parity=dict(status='Passed' if env.get('observe_only_equal') else env.get('status','Not Run'),environment=env)
        if comparison.exists():parity['comparisons']=json.loads(comparison.read_text())
    write_json(ROOT/'original_model_parity.json',parity)
    qualification=dict(status='No-Go',qualified_certifications=0,qualified_FAR=None,qualified_upper95=None,
        qualification_artifact=None,new_independent_logs=data['available_unexposed_logs'],
        reason='No independent Calibration/confirmation units; Development candidates do not establish qualification',
        required_zero_error_complete_accepted_logs=59)
    write_json(ROOT/'identity_qualification.json',qualification)
    complete=list((ROOT/'traces/development').glob('*/complete.json'));parities=[json.loads((p.parent/'baseline_parity.json').read_text()) for p in complete]
    engineering=bool(regression and all(not v['failures'] and not v['errors'] for v in regression.values())
        and all(p.get('different',0)==0 for p in parities) and len(complete)==len(data['splits']['development']))
    old=identities['selective_identity'];new=identities['temporal_identity']
    feasibility=(new['independently_correct_proposals']>old['independently_correct_proposals']
        and new['wrong_proposals']==0)
    gates=dict(engineering=dict(status='Go' if engineering else 'No-Go',regression=regression,
                replay_completed_scenes=len(complete),baseline_parity=parities,gpu_parity=parity['status']),
        identity_feasibility=dict(status='Development positive; independent confirmation Not Run' if feasibility else 'No-Go',
            correct_delta=new['independently_correct_proposals']-old['independently_correct_proposals'],
            wrong_delta=new['wrong_proposals']-old['wrong_proposals'],independent_confirmation=False),
        physical_certification=qualification,
        region_quality=dict(status='No-Go',reason='No fresh Calibration/Heldout/Final support; Development metrics cannot grant formal validation',
                            development_risk_gates=regions['surface_local_plane']['risk_gates_passed']),
        scientific_feasibility=dict(status='Limited observation feasibility; no formal qualification',
            limitations=['Repeated texture and co-moving/background crop tracks do not identify a physical surface or object uniquely',
                'Local plane support assumes a continuous visible surface and is conditional on RGB matching; wrong coherent matches remain possible',
                'Sparse annotation-owned LiDAR does not label unreturned/unannotated pixels',
                'No fresh acquisition logs remain; repeated scenes/frames/objects cannot reduce independent uncertainty'],
            minimum_new_observation='New disjoint annotated driving logs with synchronized dense instance-owned depth/correspondences; calibrated stereo/depth is an auxiliary ablation, not an installed dependency'))
    write_json(ROOT/'formal_go_nogo.json',gates)
    # Add new method funnels without changing the frozen pre-change audit.
    funnel=json.loads((ROOT/'surface_acceptance_funnel.json').read_text());aggregates=defaultdict(lambda:defaultdict(int))
    for path in (ROOT/'traces/development').glob('*/pixels.jsonl'):
        for row in map(json.loads,path.read_text().splitlines()):
            if row['method']!='surface_local_plane':continue
            for gate,counts in row['camera_diagnostics'].get('acceptance_funnel',{}).items():
                for k,v in counts.items():aggregates[gate][k]+=v
    write_json(ROOT/'surface_acceptance_funnel_new_method.json',dict(gates=dict(aggregates),
        baseline_funnel_sha256=sha(ROOT/'surface_acceptance_funnel.json')))
    core_diff=subprocess.check_output(['git','diff','--stat'],text=True)
    manifest=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sources=source_manifest(),
        python=sys.version,executable=sys.executable,platform=platform.platform(),affinity=sorted(os.sched_getaffinity(0)),
        protocol_sha256=sha(ROOT/'experiment_protocol.json'),frozen_validation_sha256=sha(ROOT/'frozen_validation_protocol.json'),
        development_scenes=len(complete),development_frames=development['region']['conservative']['frames'],
        new_independent_quality_data=False,original_model_changed=False,checkpoint_changed=False,M2_M3_changed=False,
        observations='Real RGB/pose and archived original GPU detector outputs; LiDAR/annotation used only in offline evaluation',
        diff_stat=core_diff)
    write_json(ROOT/'execution_manifest.json',manifest)
    lines=['# M1 质量与可信覆盖专项实验报告','',
        '本轮完成可关闭的新身份和局部表面方法、真实 Development 回放及机制消融。正式物理认证与区域验收仍为 **No-Go**：当前可访问数据中没有剩余未曝光采集 Log，新 Calibration/Heldout/Final 均为 **Not Run**，方法选择保持 null。',
        '', '## 身份方法与结果','',
        '新路径从实际历史图像直接测量裁剪内双向 LK、光度支持和局部仿射残差，使用真实来源时间和互斥图像支持解决合法竞争；短时缺帧从有限已提交图像中直接重测。唯一两帧证据足够时不强制三帧；竞争时需要更早实际图像的闭合支持。它只提出 object_identity_continuity，不认证静止、中心、速度、Box像素或特征复用。所有正式接受仍经过真实资格验证，未创建资格文件。','',
        '|方法|原几何分母|候选|可评分正确|错误|不可评分|唯一实例|支持Log|正式认证|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,v in identities.items():
        lines.append(f"|{name}|{v['geometric_candidates']}|{v['identity_proposals']}|{v['independently_correct_proposals']}|{v['wrong_proposals']}|{v['unscorable_proposals']}|{v['unique_physical_identity_recovery']}|{v['correct_support_logs']}|{v['qualified_certifications']}|")
    lines+=['',f"相对 Selective：可评分正确链接变化 {new['independently_correct_proposals']-old['independently_correct_proposals']:+d}；候选覆盖 {old['proposal_coverage']:.6%} → {new['proposal_coverage']:.6%}；可评分正确覆盖 {old['scorable_correct_proposal_coverage']:.6%} → {new['scorable_correct_proposal_coverage']:.6%}。正式覆盖均为0、qualified_FAR=null。",
        '', '完整接受 Log 的事件单侧95% CP上界、不完整可评分 Log、已知错误 Log 和最坏情况上界见 identity_risk_coverage.json。仅把完全可评分 Log 用于资格；未评分候选不算正确。分母沿用原 Hungarian 人口，新增恢复边单列。',
        '', '## 局部表面方法与结果','',
        '新方法在小型局部单元中拟合逆深度平面。平面诱导跨帧投影，在观测充分的关键对应凸包内部逐像素核对预测源坐标与真实 Dense Flow；使用未参与拟合的关键对应、局部边界/折叠/FB和光度纹理共同拒绝不连续表面。相邻帧可观测性不足时可用更长真实时间基线的对应提供平面，并检查三时刻闭合。没有凸包外扩张、ROI填充或语义静态真值。该模型提供待独立验证的几何依据，不能独自证明像素归属。','',
        '|方法|Effective Stable|可评分接受|错误|Sparse FAR|事件召回|关键点召回|接受Scene/Log|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    def pct(v):return 'null' if v is None else f'{v:.6%}'
    for name,v in regions.items():
        lines.append(f"|{name}|{pct(v['effective_stable_area'])}|{v['accepted_scorable_points']}|{v['false_stable_points']}|{pct(v['sparse_reference_far'])}|{pct(v['critical_change_recall'])}|{pct(v['critical_point_recall'])}|{v['accepted_scene_support']}/{v['accepted_log_support']}|")
    area=regions['surface_local_plane']['effective_stable_area'];base=regions['conservative']['effective_stable_area']
    lines+=['',f'相对 Conservative 的全输入有效面积变化为 {(area-base)*100:+.6f} 个百分点。该变化只有满足质量约束且有独立支持时才属于可信收益；本轮没有独立确认，不能宣称正式稳定覆盖优化成功。所有有效面积均由最终 refresh_required_mask 计算，启动/周期强刷包含在分母。',
        '', '逐 Scene/相机/帧及事件变化类型、ROI刷新比、稀疏 CP95 和按采集 Log 聚类的不确定性见 region_quality_by_*.json 与 region_risk_coverage.json。Bootstrap 中无可评分接受的重复样本保留为未定义，0/0 FAR 为 null。无完整实例表面GT的像素不默认正确。',
        '', '## 根因与失败边界','',
        '修改前审计、互斥/多标签拒绝和逐门槛面积见 root_cause_audit.md、identity_rejection_diagnostics.json 与 surface_acceptance_funnel.json。原 Local R12 的169个错误点、RGB/Flow/Pose/LiDAR引用及哈希保持；46个低视差与123个 Unknown Root Cause 不重新主观归因。新错误保存在各场景 false_stable.jsonl 与 visualizations，身份错误保存在 identity_false_acceptance_cases.json。',
        '', '两射线/平面/三帧自洽仍可能来自重复纹理或错误可见表面。更好的局部模型能减少过度拒绝，但无法凭缺失的独立表面标签认证完整图像；身份裁剪也可能包含一致背景，必须依赖独立风险验证。新增辅助模型/传感器未假定存在。',
        '', '## 独立验证与五项结论','',
        f"曝光审计覆盖 {len(data['previously_exposed_scenes'])} Scene/{len(data['previously_exposed_logs'])} Log；剩余可访问新Log={data['available_unexposed_logs']}。模型训练人口曝光显式记录，不称模型独立泛化测试。Calibration没有可用新人口，algorithm_selection=null；Heldout/Final Not Run，无后验重选。",
        '',f"1. Engineering：{gates['engineering']['status']}；回归 {regression}；真实CUDA感知对照 {parity['status']}，详情 original_model_parity.json。",
        f"2. Identity Feasibility：{gates['identity_feasibility']['status']}。完整正确、错误、未知和唯一实例/Log见表。",
        '3. Physical Certification：No-Go，独立完全可评分接受Log不足；正式认证0，正式FAR=null。',
        '4. Region Quality：No-Go，没有新独立Calibration/确认支持；开发面积或零错误均不能替代风险资格。',
        '5. Scientific Feasibility：保留候选可行性结论及不可辨识条件，不能解释为仅参数未调好。',
        '', '下一轮最小需要新增互不重叠采集Log，提供完整独立身份及实例拥有的同步深度/源像素对应参考。标定双目/深度可作为显式辅助观测消融，但需先验证输入可用性；不会自动改变原模型。',
        '', '## 实现、复现与交付','',
        '改动仅限 M1、研究配置、测试与 tools/worldstream_m1。默认保守路径保持；Precheck不读取当前检测/Oracle，Commit事务后才发布。来源裁剪、相机、坐标空间、有限时间、状态版本/epoch、执行签名、帧和协议/资格摘要均保留。原模型权重、输入预处理、M2/M3算法未修改。',
        '', '完整命令见 commands.sh；源码哈希、数据和协议指纹见 execution_manifest.json 与 frozen_validation_protocol.json。图见 figures/；所有未运行项显式 Not Run。']
    (ROOT/'experiment_report.md').write_text('\n'.join(lines)+'\n')
    (ROOT/'implementation_changes.md').write_text('\n'.join(lines[lines.index('## 身份方法与结果'):lines.index('## 根因与失败边界')])+'\n\n'+core_diff)


if __name__=='__main__': main()
