"""Supplement the final report with population accounting and immutable checks."""
from collections import Counter,defaultdict
from pathlib import Path
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET

root=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text())
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


summary=read(root/'development_summary.json');identity=summary['identity'];region=summary['region']
scenes=list((root/'traces/development').glob('*/complete.json'))
assert len(scenes)==20
frozen=read(root/'frozen_validation_protocol.json')
assert all(read(p)['sources']==frozen['source_sha256'] for p in scenes), 'Run source changed'
assert all(read(p.parent/'baseline_parity.json')['different']==0 for p in scenes)
for name,digest in read(root/'diagnostic_freeze.json').items():assert sha(root/name)==digest, 'Pre-change audit modified: '+name
for name,digest in read(root/'surface_error_taxonomy.json')['preservation'].items():assert sha(Path(name))==digest

ids=[];population=[];pixels=[];errors=[]
for scene in scenes:
    folder=scene.parent
    ids.extend(map(json.loads,(folder/'identity.jsonl').read_text().splitlines()))
    population.extend(map(json.loads,(folder/'population.jsonl').read_text().splitlines()))
    pixels.extend(map(json.loads,(folder/'pixels.jsonl').read_text().splitlines()))
    p=folder/'false_stable.jsonl'
    if p.exists():errors.extend(map(json.loads,p.read_text().splitlines()))
new=[r for r in ids if r['method']=='temporal_identity'];selected=[r for r in new if r['candidate']]
structural=Counter(x for r in selected for x in r.get('qualification_reasons',[]) if x!='independent_producer_qualification_missing')
assert not structural, structural


def breakdown(rows):
    return dict(candidate_links=len(rows),scenes=len({r['scene'] for r in rows}),logs=len({r['log_token'] for r in rows}),
        unique_online_anchors=len({(r['scene'],r['anchor_id']) for r in rows}),
        correct=sum(r['correct'] is True for r in rows),wrong=sum(r['correct'] is False for r in rows),unscorable=sum(r['correct'] is None for r in rows))


reasons=sorted({x for r in new for x in r['rejection_reasons']})
view_reasons=sorted({v['reason'] for r in new for v in (r.get('measurements') or {}).get('views',[])})
write(root/'identity_new_rejection_diagnostics.json',dict(population=breakdown(new),
    multilabel={reason:breakdown([r for r in new if reason in r['rejection_reasons']]) for reason in reasons},
    primary={reason:breakdown([r for r in new if (r['rejection_reasons'] or ['proposal'])[0]==reason]) for reason in [*reasons,'proposal']},
    view_multilabel={reason:breakdown([r for r in new if any(v['reason']==reason for v in (r.get('measurements') or {}).get('views',[]))]) for reason in view_reasons},
    candidate_only_qualification_rejections=dict(Counter(x for r in selected for x in r.get('qualification_reasons',[]))),
    candidate_structural_rejections=dict(structural),
    counting_unit='An object candidate source/current observation pair. Unique online anchors and physical instances are reported separately.'))

write(root/'independent_population_accounting.json',dict(original_selected_detection_observations=len(population),
    statuses=dict(Counter(r['status'] for r in population)),
    unmatched_geometric_target_observations=sum(not r['matched_geometric'] for r in population),
    original_geometric_candidate_links=identity['selective_identity']['geometric_candidates'],
    new_evaluated_legal_edges=identity['temporal_identity']['evaluated_edges'],
    candidate_acceptance_paths=dict(Counter(r['acceptance_path'] for r in selected)),
    new_recovered_gap_proposals=breakdown([r for r in selected if r['recovered_short_gap']]),
    new_resolved_competition_proposals=breakdown([r for r in selected if r['resolved_competition']]),
    baseline_and_new_unique_reference_instances={k:v['unique_physical_identity_recovery'] for k,v in identity.items()},
    note='All unscorable predictions and unmatched targets remain in the inventory. Expanded legal edges never replace the original coverage denominator.'))

original=[r for r in ids if r['method']=='selective_identity' and r['candidate']]
def key(row):return row['scene'],row['source_frame'],row['frame'],row['source_instance'],row['current_instance']
before={key(r):r for r in original if r['correct'] is True};after={key(r):r for r in selected if r['correct'] is True}
recovery=read(root/'identity_recovery_results.json');recovery.update(
    correct_link_set_comparison=dict(retained=len(before.keys()&after.keys()),new=len(after.keys()-before.keys()),
        lost=len(before.keys()-after.keys()),net=len(after)-len(before)),
    acceptance_paths=dict(Counter(r['acceptance_path'] for r in selected)))
write(root/'identity_recovery_results.json',recovery)

ablation=read(root/'surface_method_ablation.json')
write(root/'surface_method_ablation.json',dict(results=ablation,mechanisms=dict(
    surface_local_plane='Full local-unit plane/hull verification with optional longer baseline and boundary rejection',
    surface_no_history='Same local units and thresholds, older image path removed',
    surface_visibility='No local unit interpolation: original strict per-pixel three-view reference',
    surface_no_boundary_guard='Additional local depth/fold/source-collision boundary guard removed; dynamic swept masks retained',
    surface_observable_ablation='Original individual two-view observability reference'),
    note='No parameter grid; ablations are Development only and cannot select a qualified method.'))

# Keep the pre-change taxonomy byte-for-byte frozen. New errors have verified
# sparse reference failure, but a low-parallax flag is not by itself causation.
write(root/'surface_error_taxonomy_new_method.json',dict(
    sparse_false_points=region['surface_local_plane']['false_stable_points'],
    primary_physical_cause='Unknown_Root_Cause unless independently established',
    diagnostic_flags_by_method={m:dict(Counter(r.get('root_cause','Unknown_Root_Cause') for r in errors if r['method']==m)) for m in sorted({r['method'] for r in errors})},
    raw='traces/development/*/false_stable.jsonl',
    note='The existing diagnostic root_cause field is retained as a measured observability flag. It does not prove ownership, background tracking, or a specific physical mechanism.'))

expected=['root_cause_audit.md','experiment_protocol.json','implementation_changes.md','identity_rejection_diagnostics.json',
    'surface_acceptance_funnel.json','data_selection.json','independent_reference_audit.json','identity_proposal_results.json',
    'identity_risk_coverage.json','identity_recovery_results.json','identity_false_acceptance_cases.json','identity_qualification.json',
    'region_quality_by_scene.json','region_quality_by_camera.json','region_quality_by_frame.json','region_risk_coverage.json',
    'surface_error_taxonomy.json','surface_method_ablation.json','algorithm_selection.json','frozen_validation_protocol.json',
    'heldout_summary.json','final_test_summary.json','formal_go_nogo.json','regression_results.json','original_model_parity.json',
    'execution_manifest.json','commands.sh','experiment_report.md']
assert all((root/p).is_file() for p in expected)
write(root/'artifact_validation.json',dict(required_artifacts=len(expected),all_present=True,
    all_20_scene_source_manifests_match=True,prechange_audit_unchanged=True,old_error_evidence_unchanged=True,
    baseline_compared_camera_rows=sum(read(p.parent/'baseline_parity.json')['compared'] for p in scenes),
    baseline_different_camera_rows=0,new_identity_candidate_structural_rejections=dict(structural),
    fresh_validation_status={stage:read(root/(stage+'_summary.json'))['status'] for stage in ['calibration','heldout','final_test']}))

# A complete patch includes new source files, which ordinary git diff omits.
patch=subprocess.check_output(['git','diff','--binary','HEAD'])
new_files=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],text=True).splitlines()
for name in new_files:
    if name.startswith(('worldstream/m1/','tools/worldstream_m1/','tests/worldstream_m1/')) or name=='configs/worldstream_m1_quality_coverage.yaml':
        result=subprocess.run(['git','diff','--no-index','--binary','--','/dev/null',name],stdout=subprocess.PIPE)
        assert result.returncode in (0,1);patch+=result.stdout
(root/'source_changes.patch').write_bytes(patch)

new=identity['temporal_identity'];old=identity['selective_identity'];plane=region['surface_local_plane'];base=region['conservative']
answers=[
    '有限实际图像历史和测得的局部运动替代单一相邻静态射线限制；竞争用正向排除测量与真实三时刻闭合解析，无法区分仍Unknown。原Selective两帧路径作为单列备用生产器保留，未把保留链接算成全部新增收益。',
    '身份路径只提出同实体候选。区域路径使用 $\\rho(u)=n^T K^{-1}u$ 的局部逆深度平面、空间留出对应及凸包内部逐像素核对；不进行凸包外填充。自洽模型仍须独立风险参考。',
    '针对静态射线错误排除运动、来源只保留上一帧、共同视图不足即整体拒绝和逐像素三帧门槛进行改进。原0.5m外仅1条旧条件可观测链接，说明单独放宽位移阈值不能解决问题。',
    f"候选覆盖 {old['proposal_coverage']:.6%} → {new['proposal_coverage']:.6%}；可评分正确覆盖 {old['scorable_correct_proposal_coverage']:.6%} → {new['scorable_correct_proposal_coverage']:.6%}。正确链接 {old['independently_correct_proposals']} → {new['independently_correct_proposals']}，错误 {new['wrong_proposals']}、不可评分 {new['unscorable_proposals']}、唯一实例 {old['unique_physical_identity_recovery']} → {new['unique_physical_identity_recovery']}。正式覆盖仍为0。",
    f"新区域有效面积 {plane['effective_stable_area']:.6%}，Conservative {base['effective_stable_area']:.6%}，变化 {(plane['effective_stable_area']-base['effective_stable_area'])*100:+.6f} 个百分点。错误/接受 {plane['false_stable_points']}/{plane['accepted_scorable_points']}，FAR={plane['sparse_reference_far']:.6%}；事件/关键点召回 {plane['critical_change_recall']:.6%}/{plane['critical_point_recall']:.6%}。没有实现比Conservative更大的可信稳定收益。",
    '新Calibration、Heldout、Final均Not Run。全部68个可访问采集Log已被历史研究曝光，剩余新独立Log=0。方法选择保持null，未用旧Heldout/Final重新调参。',
    f"身份候选只有 {new['log_risk']['fully_scorable_accepted_logs']} 个完全可评分开发接受Log，事件CP95上界 {new['log_risk']['complete_log_event_upper95']:.6%}，并有不可评分接受人口；远未达到独立5%门槛。区域FAR超过5%，可评分接受仅 {plane['accepted_scene_support']} Scene/{plane['accepted_log_support']} Log，正式支持也不足。",
    '正式物理身份资格：没有，qualified_certifications=0、qualified_FAR=null。可信稳定区域正式收益：没有。身份的正向结果限于开发人口上的候选可行性。',
    '最终回归与CUDA对照见regression_results.json和original_model_parity.json。对象与地图混合提交、异步相机时间绑定已补测。原模型/权重/预处理、M2/M3算法均未修改，旧五方法7200组相机质量记录与归档相等；LLM执行数值对照不在perception-only测试范围。',
    '优先补充新的独立采集Log及完整身份、实例拥有的同步深度/像素源对应参考。重复纹理、共运动背景和沿极线错误对应可满足RGB自洽条件，缺少独立表面归属时无法仅靠更复杂拟合取得正式资格。至少59个完全可评分零错误独立接受Log只是5%上界的必要统计条件。'
]
with (root/'experiment_report.md').open('a') as handle:
    handle.write('\n## 十项验收问题的直接回答\n\n')
    for index,answer in enumerate(answers,1):handle.write(f'{index}. {answer}\n\n')
    handle.write('## 图与原始失败证据\n\n![身份候选覆盖](figures/identity_coverage.png)\n\n![区域风险与覆盖](figures/region_risk_coverage.png)\n\n![身份根因](figures/identity_root_causes.png)\n\n![修改前表面漏斗](figures/surface_funnel.png)\n\n')
    handle.write('修改前完整图像探针没有应用动态排除调度ROI，因此其前九级是未删减人口的诊断实验，最后两级是实际管理器输出，不应把两者差值单独归因于遮挡。当前方法实际调度后的逐级漏斗另见surface_acceptance_funnel_new_method.json。\n\n')
    handle.write('重叠拒绝原因按候选链接计数，不能相加或当成唯一物理实例。新拒绝分解、完整检测/未匹配人口分别见identity_new_rejection_diagnostics.json和independent_population_accounting.json；正确链接集合的保留/新增/丢失分别记录在identity_recovery_results.json。\n\n')
    handle.write('受控测试不计真实成功；Attempts中的中断、GPU地图元素失败、相机参考时钟修复前结果均不参与最终指标。GPU使用perception-only测试，Checkpoint SHA256与历史值一致。\n')
with (root/'implementation_changes.md').open('a') as handle:
    handle.write('\n\n额外接入修复：temporal身份路径仅处理object，地图仍走原关联；闭合时间使用绑定的相机观测时间，避免整帧参考时间与相机时间混用。历史图像/标定快照有独立摘要；真实测量包含运动矩阵、裁剪/描述子、轨迹、FB、光度残差、竞争排除点及闭合点，均纳入认证与协议绑定。\n')
print('All required artifacts and immutable evidence checks passed')
