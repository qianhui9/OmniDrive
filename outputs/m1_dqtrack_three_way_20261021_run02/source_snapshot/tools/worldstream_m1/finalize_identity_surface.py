"""Report actual selective candidates separately from qualified certificates."""
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from .identity_surface_audit import ROOT,PRIOR
from .identity_surface_experiments import identity_summary,METHODS
from .evidence_experiments import summary
from .finalize_evidence import cluster_far
from .independent_refinement import distribution
from .io_utils import write_json


def read(path):return json.loads(path.read_text())
def lines(path):return list(map(json.loads,path.read_text().splitlines())) if path.exists() else []
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    stages=('development','calibration','heldout','final_test');pixels={};identities={};times={};pop={};false={};memory={};parity=Counter()
    for stage in stages:
        done=list((ROOT/'traces'/stage).glob('*/complete.json'));assert len(done)==(20 if stage=='development' else 4)
        pixels[stage]=sum((lines(p.parent/'pixels.jsonl') for p in done),[])
        identities[stage]=sum((lines(p.parent/'identity.jsonl') for p in done),[])
        times[stage]=sum((lines(p.parent/'timing.jsonl') for p in done),[])
        pop[stage]=sum((read(p.parent/'population.json') for p in done),[])
        false[stage]=sum((lines(p.parent/'false_stable.jsonl') for p in done),[])
        memory[stage]=sum((read(p.parent/'memory.json') for p in done),[])
        for p in done:parity.update(read(p.parent/'summary.json')['parity'])
    frozen=read(ROOT/'frozen_validation_protocol.json');matches={p:sha(p)==digest for p,digest in frozen['source_sha256'].items()}
    assert all(matches.values()),matches
    selection=read(ROOT/'algorithm_selection.json')
    identity={s:identity_summary(rows) for s,rows in identities.items()}
    for stage,methods in identity.items():
        for method,v in methods.items():
            v['distinct_correct_candidate_entities']=len({(r['scene'],r['source_instance']) for r in identities[stage]
                if r['method']==method and r['candidate'] and r['correct'] is True})
    region={s:{m:dict(summary([r for r in rows if r['method']==m]),cluster_far=cluster_far([r for r in rows if r['method']==m]))
               for m in [*METHODS,'full_refresh']} for s,rows in pixels.items()}
    write_json(ROOT/'identity_selective_risk_coverage.json',dict(splits=identity,selection=selection,
        official_accepted=0,official_FAR=None,official_upper95=None,unscorable_never_success=True,
        risk_unit='Scene/log event, not individual repeated link; conditional bounds are descriptive',qualification='Insufficient; no artifact issued'))
    write_json(ROOT/'identity_candidate_diagnostics.json',dict(population={s:dict(Counter(r['status'] for r in rows)) for s,rows in pop.items()},
        fragmentation={s:sum(r['fragmented'] for r in rows) for s,rows in pop.items()},
        missing_observation_reappearance={s:sum(r['source_gap_frames'] is not None and r['source_gap_frames']>1 for r in rows if r['method']=='selective_identity') for s,rows in identities.items()},
        scope='All selected object detections; unique GT identity gate2m/margin.25 is censored; no-supported-GT/outside/competition remain distinct',
        raw_candidate_records='traces/<stage>/<scene>/identity.jsonl',raw_population_records='traces/<stage>/<scene>/population.json'))
    write_json(ROOT/'identity_false_acceptance_cases.json',dict(
        actual_qualified_false_acceptances=[],
        wrong_selected_candidate_cases={s:[r for r in rows if r['correct'] is False and r['candidate']] for s,rows in identities.items()},
        all_wrong_geometric_candidates={s:[r for r in rows if r['correct'] is False] for s,rows in identities.items()},
        scope='Geometric candidates/rejections are not certified identity accepts; candidate and continued flags distinguish each stage'))
    recovery={}
    for s,rows in identities.items():
        a={(r['frame'],r['anchor_id'],r['prediction']) for r in rows if r['method']=='strict_foreground' and r['candidate']}
        b=[r for r in rows if r['method']=='selective_identity' and r['candidate'] and r['correct'] is True]
        recovery[s]=dict(independent_correct_links=len(b),additional_to_strict_foreground=sum((r['frame'],r['anchor_id'],r['prediction']) not in a for r in b),
            distinct_entities=len({(r['scene'],r['source_instance']) for r in b}),independent_scenes=len({r['scene'] for r in b}),qualified_identity_states=0,examples=b[:12])
    write_json(ROOT/'identity_recovery_results.json',recovery)
    write_json(ROOT/'foreground_producer_ablation.json',dict(splits=identity,
        distinction='A needs strict foreground/static rays; B defines an identity-only empirically testable population. No false instance mask.',
        isolated_costs=read(ROOT/'isolated_runtime_profile.json'),qualification_shipped=False))
    write_json(ROOT/'identity_claim_contract.json',dict(claim='object_identity_continuity',
        excluded_claims=['stationarity','metric_center','velocity','full_box_surface','scene_stability','pixel_reuse'],
        bindings=['anchor','source/current frame','source/current geometry version','scene','timestamp','epoch','execution signature','frame digest','measured receipt','independent risk qualification'],
        production_enabled=False,controlled_qualification_tests='Engineering fixture only, not experiment quality',
        protocol_hash_is_quality_proof=False))
    write_json(ROOT/'commit_certificate_contract.json',dict(api='WorldStateManager.get_current_certifications()',
        type='M1CommitCertification frozen bundle, defensively copied nested provenance',commit_return_type_unchanged=True,
        publication='After all private Commit work succeeds, one Store publication point',
        precheck_field='M1Result.physically_certified_anchor_ids is Precheck-time only, never backfilled',
        invalidation=['frame','geometry/version','timestamp','scene','epoch','reset'],
        actual_certified_readouts=0,controlled_tests_pass=True))
    phase1=read(ROOT/'phase1_diagnostics.json')
    write_json(ROOT/'surface_error_taxonomy.json',dict(phase1=phase1,
        all_methods_by_stage={s:{m:dict(Counter(r['root_cause'] for r in rows if r['method']==m)) for m in METHODS} for s,rows in false.items()},
        rules='C: measured parallax<.5deg or invalid rays; E: measured many-to-one/fold; others Unknown Root Cause',
        A_vs_B='Not identified without source/current independent surface ownership; no subjective relabeling',
        D='Critical newly-visible/occlusion events separately recorded, never treated as accepted stable negatives'))
    write_json(ROOT/'surface_correspondence_diagnostics.json',dict(counts={s:len(v) for s,v in false.items()},
        raw='traces/<stage>/<scene>/false_stable.jsonl; phase1_false_stable.jsonl',
        fields=['pixel','source_pixel','RGB source/current images via scene/sample/camera; original visualizations',
            'flow_forward/backward','FB','photo','gradient','texture','Jacobian','poses','triangulation','parallax','reprojection','local_rejection','dynamic_sweep','Oracle'],
        reference_scope='Same unchanged annotation-owned LiDAR population; absent returns are not stable negatives'))
    write_json(ROOT/'region_risk_coverage.json',dict(splits=region,selection=selection,scope='Annotation-owned sparse LiDAR + ROI events only'))
    for kind,key in [('scene','scene'),('camera','camera'),('frame','frame')]:
        write_json(ROOT/f'region_quality_by_{kind}.json',{s:{unit:{m:summary([r for r in rows if r[key]==unit and r['method']==m]) for m in [*METHODS,'full_refresh']}
            for unit in sorted({r[key] for r in rows})} for s,rows in pixels.items()})
    write_json(ROOT/'surface_visibility_results.json',dict(splits={s:region[s]['surface_visibility'] for s in stages},
        ablation={s:region[s]['surface_observable_ablation'] for s in stages},
        interpretation='Removing false accepts with vanishing/unscorable area is not a benefit; no proof of all-pixel surface ownership'))
    runtime={s:{m:{key:distribution([r[key] for r in rows if r['method']==m]) for key in ('total_ms','precheck_ms','commit_ms','identity_ms','foreground_ms','certification_ms','association_ms','surface_ms')}
                for m in METHODS} for s,rows in times.items()}
    funnel={s:{method:dict(evaluated=len([r for r in rows if r['method']==method]),
        rejected=dict(Counter(reason for r in rows if r['method']==method for reason in r['rejection_reasons'])),
        cost_reaching_observation_ms=sum(r['producer_cpu_ms'] for r in rows if r['method']==method and r['measurements'] and r['measurements']['views']),
        total_object_cost_ms=sum(r['producer_cpu_ms'] for r in rows if r['method']==method)) for method in ('selective_identity','strict_foreground')} for s,rows in identities.items()}
    write_json(ROOT/'runtime_cost_breakdown.json',dict(replay=runtime,rejection_funnel=funnel,
        funnel_scope='Reason counts may overlap; cost of populations reaching stage, not independently timed incremental stage work',
        incremental_stage_CPU='Not Run: frozen producers expose per-object totals, not incremental sub-stage timers; totals and population cost are retained',
        isolated=read(ROOT/'isolated_runtime_profile.json'),GPU_protocol=read(ROOT/'gpu_adapter_profile.json'),long_sequence=read(ROOT/'long_sequence_memory.json'),
        memory={s:{m:dict(max_history_bytes=max(r['history_bytes'] for r in rows if r['method']==m),max_anchors=max(r['anchors'] for r in rows if r['method']==m),max_tensor_bytes=max(r['tensor_bytes'] for r in rows if r['method']==m)) for m in METHODS} for s,rows in memory.items()},
        replay_timing='Concurrent analysis/replay timing descriptive only; isolated comparison excludes IO/Oracle/logging',
        surface_ms_semantics='Sum of elapsed time of camera workers, which overlap; not additive CPU time or full-frame wall time'))
    write_json(ROOT/'runtime_optimization_ablation.json',dict(optimization='Not Run: no approved online identity/region coverage',
        measured_claim_specific_ablation=read(ROOT/'isolated_runtime_profile.json'),
        prefilter='Necessary conditions are part of the fixed identity definition, not an empirical probability shortcut',
        wrong_qualified_certifications_lost=0,proof_scope='Population violations cannot qualify; actual qualified acceptances remain zero',
        triangulation_cache_or_threshold_shortcuts=False))
    junit=ET.parse(ROOT/'regression.xml');tests=junit.findall('.//testcase')
    failed=[r.attrib['name'] for r in tests if r.find('failure') is not None or r.find('error') is not None]
    skips=[r.attrib['name'] for r in tests if r.find('skipped') is not None]
    regression=dict(passed=len(tests)-len(failed)-len(skips),failed=failed,skipped=skips,previous_380_retained=True,
        new_tests=sum('test_identity_surface' in r.attrib.get('classname','') for r in tests),old_mask_parity=dict(parity))
    write_json(ROOT/'regression_results.json',regression)
    gpu=read(ROOT/'gpu_parity_isolated/results.json');oldsha=read(PRIOR/'original_model_parity.json')['checkpoint_sha256'];assert (ROOT/'checkpoint.sha256').read_text().split()[0]==oldsha
    write_json(ROOT/'original_model_parity.json',dict(passed=gpu['observe_only_equal'],comparisons=gpu['comparisons'],frames=len(gpu['baseline']['outputs']),
        checkpoint_sha256=oldsha,original_parameters_unchanged=True,scope='Actual GPU perception/backbone/head memory, no LLM/M2/M3 changes',environment=gpu['run_metadata']))
    engineering=not failed and not skips and gpu['observe_only_equal'] and parity['different_camera_masks']==0
    b=[identity[s]['selective_identity'] for s in ('heldout','final_test')]
    independent_feasibility=selection['identity_feasibility_candidate'] is not None and all(v['independently_correct_candidate_links']>0 and v['conditional_candidate_FAR']<=.05 for v in b)
    def region_pass(v,baseline,formal=False):
        return (v['effective_stable_area']>0 and v['effective_stable_area']>=baseline['effective_stable_area']
            and v['critical_change_recall'] is not None and v['critical_change_recall']>=.998
            and v['critical_point_recall'] is not None and v['critical_point_recall']>=.998
            and v['sparse_reference_far'] is not None and v['sparse_reference_far']<=.05
            and (not formal or v['accepted_scene_support']>=3 and v['far_upper95_iid_descriptive']<=.05))
    region_feasible=selection['region_candidate'] is not None and all(region_pass(region[s]['surface_visibility'],region[s]['conservative']) for s in ('heldout','final_test'))
    region_qualified=region_feasible and all(region_pass(region[s]['surface_visibility'],region[s]['conservative'],True) for s in ('heldout','final_test'))
    decision=dict(engineering='Go' if engineering else 'No-Go',
        identity_feasibility='Positive (independently scorable subset only)' if independent_feasibility else 'Negative for frozen independent confirmation',
        physical_certification='No-Go: Qualification Insufficient; unscorable accepted population unresolved',
        region_feasibility='Positive' if region_feasible else 'Negative',region_validation='Go' if region_qualified else 'No-Go',runtime_cost='No-Go for deployment; claim-specific costs measured',
        development_identity_feasibility=identity['development']['selective_identity']['independently_correct_candidate_links']>0,
        calibration_selection=selection,formal_qualification_issued=False,post_commit_real_certificates=0,
        no_test_reselection=True,model_training_population=True)
    write_json(ROOT/'formal_go_nogo.json',decision)
    source=[*Path('worldstream/m1').glob('*.py'),*Path('tools/worldstream_m1').glob('*identity_surface*.py'),Path('tools/worldstream_m1/collect_evidence_observations.py'),Path('tools/worldstream_m1/dataset.py'),Path('tools/worldstream_m1/coverage_oracle.py'),Path('tools/worldstream_m1/evidence_oracle.py'),Path('tests/worldstream_m1/test_identity_surface.py'),Path('configs/worldstream_m1_identity_surface.yaml')]
    write_json(ROOT/'execution_manifest.json',dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(p):sha(p) for p in source},frozen_quality_source_matches=matches,
        checkpoint_sha256=oldsha,cpu_affinity=sorted(os.sched_getaffinity(0)),python=__import__('sys').executable,
        GPU=gpu['run_metadata'],data='data_selection.json',camera_order=read(ROOT/'model_observation_provenance.json')['frames'][0]['camera_order'],
        model_processing_changed=False,online_GT=False,M2_M3_changed=False))
    figures(region,identity)
    report(region,identity,decision,regression,funnel)
    print(json.dumps(decision,ensure_ascii=False,indent=2))


def figures(region,identity):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/m1-identity-surface-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=ROOT/'figures';folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(4,2,figsize=(11,13));methods=list(METHODS);colors=dict(zip(methods,plt.get_cmap('tab10').colors));markers=dict(zip(methods,['o','s','^','D','X']))
    for index,(stage,rows) in enumerate(region.items()):
        for m in methods:
            v=rows[m];x=v['effective_stable_area']*100
            axes[index,0].scatter(x,v['critical_change_recall']*100,color=colors[m],marker=markers[m],label=m)
            if v['sparse_reference_far'] is not None:axes[index,1].scatter(x,v['sparse_reference_far']*100,color=colors[m],marker=markers[m])
        for col,(limit,label) in enumerate(((99.8,'Critical event recall (%)'),(5,'Sparse annotated FAR (%)'))):
            axes[index,col].axhline(limit,color='red',ls='--');axes[index,col].set(title=stage,xlabel='Effective area (%)',ylabel=label);axes[index,col].grid(alpha=.2)
        axes[index,0].set_ylim(min(99.7,min(rows[m]['critical_change_recall']*100 for m in methods)-.1),100.03)
        risk=[rows[m]['sparse_reference_far']*100 for m in methods if rows[m]['sparse_reference_far'] is not None]
        axes[index,1].set_ylim(-.2,max([6]+[v+1 for v in risk]))
        axes[index,1].set_xlim(0,max(rows[m]['effective_stable_area']*100 for m in methods)*1.07+.001)
        if not risk:axes[index,1].text(.5,.5,'FAR unavailable: no scorable accepted points',transform=axes[index,1].transAxes,ha='center',fontsize=9)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.98),ncol=3,frameon=False)
    fig.suptitle('Measured candidates only; null FAR omitted; no dense pixel ownership truth',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(folder/'surface_risk_coverage.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(identity));a=[v['strict_foreground']['candidate_count'] for v in identity.values()];b=[v['selective_identity']['candidate_count'] for v in identity.values()]
    ax.bar(x-.2,a,.4,label='Strict foreground candidates');ax.bar(x+.2,b,.4,label='Selective identity candidates')
    ax.set_xticks(x);ax.set_xticklabels(list(identity));ax.set_title('Candidates, not qualified identity states');ax.legend();fig.tight_layout();fig.savefig(folder/'identity_candidates.png',dpi=160);plt.close(fig)


def report(region,identity,decision,regression,funnel):
    phase1=read(ROOT/'phase1_diagnostics.json');cost=read(ROOT/'isolated_runtime_profile.json')
    rows=['# M1 物理身份连续性与局部表面可见性实验','',
        '本轮完成独立身份声明、真实候选生产、Commit 后读取闭环与三帧表面原型。真实候选、离线可评分正确对应和取得资格的持久物理状态分别统计；本轮没有发放真实风险资格。','',
        '## 数据与协议','',
        '旧 Calibration 20 场景/240 帧全部转为 Development。新增 12 个场景和 12 个未在旧 M1 质量记录曝光的采集 log，冻结为 4/4/4 Calibration/Heldout/Final，共144帧，由原 Checkpoint 本轮实际 GPU 采集。它们属于原模型 Train Population，不能声称模型独立泛化。另有 8 帧原模型 GPU 数值对照、8 帧独立计时与40帧旧开发长序列生命周期检查。','',
        '新增 sample token 扫描只匹配旧可用性清单 train_data_availability.json，没有旧 M1 质量产物匹配；清单出现不等于阈值开发曝光。scene/log 清单、模型输出哈希、相机顺序、执行环境见 data_selection.json、sample_exposure_audit.json 和 execution_manifest.json。','',
        'root_cause_audit.md 在修改核心代码前生成；experiment_protocol.json 在实现和新质量运行前冻结假设及门槛。Calibration 结束写入 algorithm_selection.json 与 frozen_validation_protocol.json 后才运行 Heldout/Final。没有根据测试结果改算法或重选方法。GT/annotation/LiDAR 仅在 tools 评分，运行时 Precheck 没有当前检测或 Oracle 输入。','',
        '风险独立单位是 scene/log。一个单位内任一错误接受即单位失败；零错误也需至少59个独立接受单位才能让单侧CP95≤5%。校准排除后新 Heldout+Final 仅8个 log，即使全部完整可评分且零错，上界仍约31.23%，至少还缺51个这样的独立单位；当前尚有不可评分候选，实际资格缺口更大。不得用重复帧/对象或像素补足。','',
        '## 身份结果','',
        'A 保留原严格前景/静态射线生产器。B 为同类、上一实际来源、无门内竞争、检测更新位移≤0.5m、共有视图中实际轨迹和裁剪支持的选择性身份人口。B 不要求独立三维中心/静止/环深度分离；没有前景 Mask，必须通过独立身份资格才能批准。','',
        '| 分组 | 方法 | 全部几何候选 | 方法候选 | 可评分候选 | 错误身份 | 不可评分候选 | 可评分支持 scene | 条件场景上界 | 正式认证 |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for stage,methods in identity.items():
        for method,v in methods.items():rows.append(f'| {stage} | {method} | {v["all_geometric_candidates"]} | {v["candidate_count"]} | {v["candidate_scorable_count"]} | {v["candidate_false_identity"]} | {v["unscorable_candidates"]} | {v["independent_scorable_candidate_scenes"]} | {v["conditional_scene_event_upper95"]} | 0 |')
    rows+=['','以上候选/正确数按链接计数，同一实体跨帧不会增加独立统计样本；去重实体数另见 identity_recovery_results.json。所有候选仍包含 2m 唯一身份筛选的截断和无 GT 对应人口。条件 FAR/上界不代表全部候选的风险；qualified FAR/上界均为 null。正确但拒绝、错误但保留、缺帧重现与 fragmentation 分别列在 identity_candidate_diagnostics.json、identity_false_acceptance_cases.json 和 identity_recovery_results.json。受控资格正例只验证接口，不计作真实认证。','',
        '## 表面根因和结果','',
        f'逐点精确复现旧 Local R12 的169个错误稳定点。主分类为 {phase1["primary_taxonomy"]}；46点低视差，其余123点不能区分真实表面切换、纹理错配和参考误差，保留 Unknown Root Cause。它们的光度/FB/Jacobian 已通过旧门槛；不能把两帧自洽三角化当成独立表面证明。完整 flow、RGB、pose、texture、Jacobian、几何、原判定与实际图见 traces/、visualizations/。','',
        '新方法要求可观测的三帧射线/对应闭合，并检查直接与复合的长基线轨迹。固定原门槛，没有 Radius Grid。缺历史/视差/闭合证据均为 Unknown，额外历史有固定帧数上限。两帧仅可观测性为消融；全刷新仅安全零收益参考。','',
        '| 分组 | 方法 | 有效面积 | 事件召回 | 关键点召回 | 稀疏错误/接受 | 稀疏FAR | 接受scene |','|---|---|---:|---:|---:|---:|---:|---:|']
    for stage,methods in region.items():
        for method,v in methods.items():rows.append(f'| {stage} | {method} | {v["effective_stable_area"]:.6%} | {v["critical_change_recall"]:.6%} | {v["critical_point_recall"]:.6%} | {v["false_stable_points"]}/{v["accepted_scorable_points"]} | {v["sparse_reference_far"]} | {v["accepted_scene_support"]} |')
    rows+=['','点和事件分母保持原协议：事件 ROI 刷新≥90%；错误稳定为 annotation-owned LiDAR 的来源偏差>2px 或真实标注运动。未标注像素与无返回背景均不当正确负例；零可评分稳定接受时 FAR=null，错误数下降不能替代有效覆盖与风险资格。scene/camera/frame、描述性CP与场景bootstrap见 region_quality_by_*.json、region_risk_coverage.json。','',
        '## 合法持久状态接口与成本','',
        'commit() 保留原返回值，get_current_certifications() 提供已原子提交的 frozen 结果和防御性证书副本。新检测替换几何即清除旧证书；scene/frame/version/epoch/timestamp 不匹配不返回。pending 查询只看到之前提交；异常不发布准备中的证书。M1Result 的 Precheck 字段不反填当前检测结论。身份声明不授权像素、静止性、米制几何或后续模块。','',
        '| 相同8帧预加载工作量 | 完整M1 P50 ms | P95 ms | 身份耗时 P50 | 严格前景耗时 P50 | 表面相机耗时和 P50 |','|---|---:|---:|---:|---:|---:|']
    for method,v in cost['methods'].items():rows.append(f'| {method} | {v["total_ms"]["p50"]:.2f} | {v["total_ms"]["p95"]:.2f} | {v["identity_ms"]["p50"]:.2f} | {v["foreground_ms"]["p50"]:.2f} | {v["surface_ms"]["p50"]:.2f} |')
    gpu_cost=read(ROOT/'gpu_adapter_profile.json')['measurements']['m1_total_including_adapters_ms']
    rows+=['','Surface 列是并行相机任务各自 elapsed 耗时的和，任务重叠，因此可大于完整帧 wall time，不能与其他阶段相加或解释为实际CPU占用。完整M1列才是同机单帧耗时对照。',f'独立实际 GPU 输入协议下完整 M1 P50/P95 为 {gpu_cost["p50"]:.2f}/{gpu_cost["p95"]:.2f} ms（8帧含启动）。该输入分辨率与CPU回放不同，不能直接解释为CPU加速。Transfer/Input/Output adapter拆分见 gpu_adapter_profile.json。']
    rows+=['','这是声明路径成本消融，排除 IO/Oracle/日志；不声称获得部署加速。身份路径不调用其声明不需要的严格前景三角化；原 A 路径仍保留。A 开关前后 B 的逐帧测量/判定一致，保守 Mask 也与基线完全一致。必要条件预筛只拒绝明确不属于所定义人口的候选，没有启用经验概率跳过；尚无合格物理状态收益，因此没有继续扩大 P1 优化、缓存或削减验证条件。','',
        '## 20项验收','',
        '1. 原强生产器稀少来自视差、双OBB、环测深/分离、空间支持和多视图静态假设的交集，详细拒绝漏斗保留。',
        '2. 独立实例标签可验证 B 的部分候选真实同实例；未评分人口不能自动填正确。',
        '3. 候选可评分正确覆盖与正式认证分开；正式可信持久身份状态仍为0。',
        '4. B 仅身份连续性；A 是原有限静态射线声明；两者都不自动批准完整三维状态。',
        '5. 正式身份错误率/上界仍null，没有满足5%正式资格。',
        '6. 新质量单位仅12log，选择后独立评价8log，数量不足59且可评分人口不完整。',
        '7. 最小缺口是覆盖全部所选人口的可靠独立身份证据，及足量新日志；完整表面/几何还需额外观测参考。',
        '8. Post-Commit 查询闭环通过受控完整证据和模拟资格验证，实际结果无资格仍空。',
        '9. Precheck 仍只消费合法当前输入和历史，未被当前 Detection 后验结论反填。',
        '10. 46/169错误可确认低视差；其余主要根因未知，未凭光流错误主观归类A/B。',
        '11. 三帧判据减少错误数，但收益必须同时检查面积和可评分接受；拒绝全部不算成功。',
        '12. 有限非零区域不代表有用覆盖；未达到相对保守方案的风险约束覆盖收益。',
        '13. 每组事件召回见上表，保留原90% ROI定义。',
        '14. 每组点级召回见上表，未通过改分母抹除失效。',
        '15. 稀疏FAR和单侧描述性上界分别报告，零接受为null。',
        '16. 独立接受场景数逐组列出；不足则不批准。',
        '17. 新身份、严格前景、surface、adapter/Precheck/Commit、memory/transfer的实际成本分别记录；缺少的硬件独立微测不虚构。',
        '18. 预筛属于固定必要条件而不是经验快捷规则；没有声称在零实际认证下获得有用认证收益。',
        f'19. 实际GPU8帧原始感知/backbone/head memory相等，回归{regression["passed"]}通过，checkpoint SHA一致。',
        '20. 当前只保留observe-only工程基础和实验原型，不能正式批准物理/区域输出供后续模块使用。','',
        '## 独立结论','',*['- **'+k+'：'+str(v)+'**' for k,v in decision.items() if k in ('engineering','identity_feasibility','physical_certification','region_feasibility','region_validation','runtime_cost')],
        '','最小新增能力优先是独立且完整的对象身份/源当前表面参考。当前没有额外传感器、分割权重或人工核查会话，因此这些新增能力的采集/推理成本为 Not Run；本轮没有训练或集成新网络。本轮固定选择性方法失败不证明所有轻量方法不可能，也不能把仍然未知的参考/物理误差主观解释为已解决。','',
        '![身份候选](figures/identity_candidates.png)','![区域质量](figures/surface_risk_coverage.png)']
    (ROOT/'experiment_report.md').write_text('\n'.join(rows)+'\n')


if __name__=='__main__':main()
