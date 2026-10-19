"""Derive auditable state-interface deliverables from fresh replay and JUnit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .io_utils import write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",default="outputs/module1_state_interface_rearchitecture")
    args=parser.parse_args()
    root=Path(args.root);replay=root/"replay"
    complete=json.loads((replay/"completion.json").read_text())
    method=json.loads((replay/"method_comparison.json").read_text())
    rows=[json.loads(line) for line in (replay/"pixels.jsonl").read_text().splitlines()]
    timings=json.loads((replay/"timings.json").read_text())
    sources=json.loads((replay/"source_audit.json").read_text())
    leases=json.loads((replay/"leases.json").read_text())
    funnel=json.loads((replay/"funnel.json").read_text())
    identity=json.loads((replay/"identity.json").read_text())
    xml=ET.parse(root/"regression_final.xml")
    cases=list(xml.iter("testcase"))
    failed=[t.attrib for t in cases if t.find("failure") is not None or t.find("error") is not None]
    skipped=[dict(t.attrib,reason=t.find("skipped").get("message")) for t in cases if t.find("skipped") is not None]
    tests=dict(total=len(cases),passed=len(cases)-len(failed)-len(skipped),
        failed=failed,skipped=skipped,execution="CPU; CUDA unavailable",source="regression_final.xml")
    def write(name,value):write_json(root/name,value)
    population="exploratory_reused_population"
    missing=dict(status="Not Run",reason="CUDA unavailable inside sandbox; two outside-sandbox read-only GPU queries were not executed because automatic approval review timed out",
        E2E_wall_ms=None,E2E_GPU_ms=None,perception=None,temporal_memory=None,LLM=None)
    engineering=[]
    for test in cases:
        for prop in test.findall("./properties/property"):
            if prop.get("name")=="engineering_observations":
                engineering.extend(json.loads(prop.get("value")))
    pins=complete["source_sha256"]
    pin_match={p:Path(p).exists() and hashlib.sha256(Path(p).read_bytes()).hexdigest()==value for p,value in pins.items()}
    audit=dict(population=population,status="passed" if all(pin_match.values()) else "source_mismatch",
        replay_source_hash_matches=pin_match,records=len(sources),
        computation_available=sum(s["computation_available"] for s in sources),
        observation_only_or_evicted=sum(not s["computation_available"] for s in sources),
        current_frame_computation_leaks=sum(s["computation_frame_id"]==s["frame"] for s in sources),
        metadata_only_replay=True,M2_cache_in_replay=False,
        source="replay/source_audit.json",engineering_source_tests=[t.attrib["name"] for t in cases if
            "source" in t.attrib["name"] and t.find("failure") is None])
    write("source_provenance_audit.json",audit)
    write("validity_lease_results.json",dict(population=population,by_method={name:dict(
        valid=sum(v["source_valid"] for v in leases if v["method"]==name),
        revoked=dict(Counter(v["revoked_reason"] for v in leases if v["method"]==name and not v["source_valid"])))
        for name in method if name.startswith("source_linked")},source="replay/leases.json",
        invalid_and_expired_testcases=[t.attrib for t in cases if "test_lease" in t.attrib["name"]],
        interpretation="Existing hard maximum source age applies in all arms; optional lease export adds no separate lifecycle."))
    write("m1_method_comparison.json",dict(population=population,results=method,
        measurement="Fresh six-camera CPU replay through original test image preprocessing; historical original-model detections enter Commit",
        lease_and_refresh_ablations="Contract exports share mandatory safety gates and candidate decisions; equal coverage does not establish an algorithmic benefit."))
    write("m1_candidate_quality.json",dict(population=population,methods={k:dict(
        coverage=v["effective_stable_area"],accepted_scorable=v["accepted_scorable_points"],
        correct=v["accepted_scorable_points"]-v["false_stable_points"],false=v["false_stable_points"],
        candidate_error_rate=v["sparse_reference_far"],scorable_reference_points=v["scorable_lidar_points"],
        unscorable_candidate_pixels=sum(r["effective_pixels"] for r in rows if r["method"]==k),
        dense_correct_area=None,dense_false_area=None,dense_unscorable_area=None) for k,v in method.items()},
        sparse_reference_scope="Annotated, depth-visible LiDAR points used only after online Precheck. No dense truth. Zero accepted sparse reference points do not imply zero physical errors.",
        GT_used_online=False,source="replay/pixels.jsonl"))
    refresh={}
    for name in method:
        selected=[r for r in rows if r["method"]==name]
        strict_events=[e for r in selected for e in r.get("strict_refresh_score",{}).get("events",[])]
        failures=[dict(frame=r["frame"],camera=r["camera"],event=e) for r in selected
            for e in r.get("events",[]) if e["invalidation_coverage"]<.9]
        refresh[name]=dict(candidate_diagnostic_events=method[name]["critical_events"],
            candidate_diagnostic_event_recall=method[name]["critical_change_recall"],
            candidate_diagnostic_point_recall=method[name]["critical_point_recall"],
            candidate_complement_min_ROI_refresh=method[name]["minimum_critical_ROI_refresh"],
            candidate_complement_failures=failures,
            strict_refresh_min_ROI=min((e["invalidation_coverage"] for e in strict_events),default=None),
            strict_refresh_event_recall=sum(e["recalled"] for e in strict_events)/len(strict_events) if strict_events else None,
            dirty_area=sum(r["changed_pixels"] for r in selected)/sum(r["pixels"] for r in selected),
            unknown_area=sum(r["unknown_pixels"] for r in selected)/sum(r["pixels"] for r in selected),
            candidate_area=method[name]["effective_stable_area"],
            strict_observation_area=method[name]["strict_observation_area"])
    write("refresh_contract_results.json",dict(population=population,partition_assertions="passed every replay camera/method",
        methods=refresh,warning="Candidate complement is a diagnostic of hypothetical candidate-based refresh, never the deployed refresh permission. Actual strict refresh is separately scored."))
    write("m1_m2_admission_funnel.json",dict(population=population,methods={name:dict(
        candidate_windows=sum(x["candidate_windows"] for x in funnel if x["method"]==name),
        exact_RGB_window_opportunities=sum(x["exact_RGB_window_opportunities"] for x in funnel if x["method"]==name),
        verified_exact_windows=None,actual_skipped_windows=None,M2_validated_candidate_population_ratio=None)
        for name in method if name.startswith("source_linked")},
        scope="CPU M1 opportunity screening only; RGB window equality is not M2 neural admission. CUDA M2 execution Not Run."))
    write("m2_dependency_correctness.json",dict(status="engineering_tests_passed" if not failed else "failed",
        tests=[t.attrib["name"] for t in cases if any(s in t.attrib["name"] for s in
            ("candidates_never_bypass_exact","refresh_gates","camera_bundle","cache_publication","dependency","closure"))],
        engineering_observations=engineering,natural_original_EVA_CUDA=missing,
        evidence_scope="Actual small EVAViT CPU engineering fixture, synthetic inputs; not nuScenes acceleration evidence."))
    write("m2_actual_reuse_results.json",dict(natural_nuScenes=missing,
        controlled_engineering_fixture=dict(status="passed" if engineering and not failed else "incomplete",
            observations=engineering,scope="CPU unit fixtures only; no task-quality or acceleration claim")))
    overhead={}
    for name in method:
        selected=[t for t in timings if t["method"]==name]
        def stat(key):
            values=[t[key] for t in selected]
            return dict(p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95)))
        overhead[name]=dict(precheck_ms=stat("M1_precheck_ms"),commit_ms=stat("M1_commit_ms"),
            state_maintenance_ms=stat("M1_wall_ms"),input_adapter_ms=stat("input_adapter_ms"),
            validation_requested=sum(t["validation_requested"] for t in selected),
            requests=len(selected),M2_validation_ms=None,M2_execution_ms=None,
            coordinate="CPU preprocessing; no GPU transfers",service="persistent_state")
    write("runtime_overhead.json",dict(population=population,CPU_replay=overhead,
        CPU_rss_high_water_kib=complete["rss_high_water_kib"],CUDA_memory=None,
        scope="CPU replay host wall; historical detections loaded before measurement. Not complete model/E2E latency."))
    write("e2e_comparison.json",dict(status="Not Run",arms={k:missing for k in
        ("original","strict_exact","source_linked_exact","source_linked_on_demand","full_maintain")},
        entry="tools/worldstream_m2/state_interface_integration.py",reason=missing["reason"]))
    comparable=[r for r in identity if r["method"] in ("dense_observation","source_linked_refresh","source_linked_on_demand")]
    by_frame={}
    for row in comparable:by_frame.setdefault(row["frame"],{})[row["method"]]=row
    same=all(v["dense_observation"]["anchor_ids"]==v["source_linked_refresh"]["anchor_ids"]==
        v["source_linked_on_demand"]["anchor_ids"] for v in by_frame.values())
    write("identity_certification_regression.json",dict(population=population,
        shadow_ID_sequences_equal=same,frames=len(by_frame),
        qualified_certifications=0,qualified_FAR=None,
        emitted_certified_ids=sum(r["certified"] for r in identity),
        physical_certifier_changed=False,M3_changed=False,
        limitations="This compares committed IDs and certificate isolation, not all 13,314 historical identity proposal edges."))
    write("original_model_parity.json",dict(CUDA_full_original_model=missing,
        CPU_actual_entry_with_test_doubles=dict(status="passed" if not failed else "failed",
            test="test_original_entry_source_mode_output_parity",frames=4,
            scope="Executed Petr3D methods with deterministic lightweight heads/LLM doubles; not real-model quality"),
        engineering_EVA_prefix=engineering))
    write("regression_results.json",tests)
    formal=dict(decision="No-Go",qualification=dict(Calibration="Not Run",Heldout="Not Run",
        Final_Test="Not Run",Qualified_Certification=0,Qualified_FAR=None),
        population=population,gates=dict(G1="CPU source and transaction tests passed; CUDA lineage unverified",
            G2="Coverage measured; candidate physical error rate not established (no accepted scorable points)",
            G3="Partition and strict refresh pass; hypothetical candidate complement has critical-ROI failures",
            G4="CPU exact-admission and fail-closed tests passed",
            G5="CPU EVA closure passed; full original-model CUDA/quality Not Run",
            G6="No E2E or natural compute-saving result; CPU maintenance measured",
            G7="CPU regression passed; CUDA full model Not Run",
            G8="No independent log qualification; No-Go"),
        runtime_speedup_claim=False)
    write("formal_go_nogo.json",formal)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures=root/"figures";figures.mkdir(exist_ok=True)
    names=list(method)
    fig,ax=plt.subplots(figsize=(11,4))
    ax.bar(range(len(names)),[method[n]["effective_stable_area"]*100 for n in names])
    ax.set_xticks(range(len(names)),[n.replace("_","\n") for n in names],fontsize=8)
    ax.set_ylabel("Observed coverage (%)");ax.set_title("Fresh nuScenes CPU replay: candidates versus strict Stable")
    fig.tight_layout();fig.savefig(figures/"coverage.png",dpi=160);plt.close(fig)
    sample=np.load(replay/"spatial_masks.npz")
    fig,axes=plt.subplots(1,4,figsize=(12,3))
    for ax,key in zip(axes,("image","dirty","candidate","unknown")):
        ax.imshow(sample[key],cmap=None if key=="image" else "gray");ax.set_title(key);ax.axis("off")
    fig.tight_layout();fig.savefig(figures/"spatial_distribution.png",dpi=160);plt.close(fig)
    count=sum(v["candidate_windows"] for v in funnel if v["method"]=="source_linked_refresh")
    exact=sum(v["exact_RGB_window_opportunities"] for v in funnel if v["method"]=="source_linked_refresh")
    fig,ax=plt.subplots(figsize=(7,3))
    ax.bar(["M1 candidate windows","Exact RGB opportunities"],[count,exact])
    ax.text(.99,.95,"M2 Exact admission / actual compute reuse: Not Run",ha="right",va="top",transform=ax.transAxes)
    ax.set_title("nuScenes opportunity funnel; no CUDA execution");fig.tight_layout()
    fig.savefig(figures/"admission_funnel.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4))
    ax.bar(range(len(names)),[overhead[n]["state_maintenance_ms"]["p50"] for n in names])
    ax.set_xticks(range(len(names)),[n.replace("_","\n") for n in names],fontsize=8)
    ax.set_ylabel("CPU M1 precheck + commit P50 (ms)")
    ax.set_title("CPU maintenance only; M2 and E2E latency Not Run")
    fig.tight_layout();fig.savefig(figures/"runtime.png",dpi=160);plt.close(fig)
    table="\n".join(f"| {n} | {method[n]['effective_stable_area']*100:.6f} | "
        f"{method[n]['accepted_scorable_points']-method[n]['false_stable_points']}/{method[n]['false_stable_points']} | "
        f"{overhead[n]['state_maintenance_ms']['p50']:.2f}/{overhead[n]['state_maintenance_ms']['p95']:.2f} |" for n in names)
    (root/"experiment_report.md").write_text(f"""# M1 状态与 M1→M2 接口重构实测报告

已完成可选代码路径、CPU 回归与真实 nuScenes 回放；完整 CUDA 模型验收尚未完成。正式结论 **No-Go**，没有宣称端到端加速。

本次回放为 {complete['frames']} 连续帧、{complete['camera_observations']} 个相机观测、2 个已曝光场景；均标为 exploratory_reused_population。原 OmniDrive 的真实测试图像变换生成 640×640 preprocessed_image 输入；历史原模型检测仅在 Commit 使用。离线 LiDAR/Annotation 只在 Precheck 之后评分，未参与在线候选或刷新。该回放没有重新执行原始 Head、Memory 或 LLM。

| 方法 | 候选/旧 Stable 面积 % | 稀疏正确/错误 | CPU Precheck+Commit P50/P95 ms |
|---|---:|---:|---:|
{table}

候选误差率不能由零接受参考点推断为零。本人口没有候选命中可评分的标注内稀疏参考点，Candidate 正确/错误/不可评分稠密面积均无独立真值，保持 null。像素候选数单独保留，不能视作 Certified Stable。Lease 与 Refresh Contract 共享强制来源年龄检查；这些接口消融覆盖相同，不声称有额外算法收益。

轻量候选的补集曾出现关键 ROI 刷新不足。这是假设候选直接授权复用时的诊断失败，并非部署刷新掩码。每个失败事件、时间和相机在 refresh_contract_results.json 中保留。最终轻量路径的严格 refresh_required_mask/exact_invalidation_mask 保守全刷；候选单独进入 M2。Dirty 与 Candidate 不重叠，Unknown 只可通过完整独立 Exact 输入和依赖验证消除。

M2 CPU 工程测试实际执行小型 EVAViT，分别覆盖全候选、全 Unknown、相同/变化输入、Dirty、Full Refresh、已撤销事务、过期计算来源、全局依赖屏障及 OOM 原子发布。其实际跳窗和数值差异保留在 m2_dependency_correctness.json；这些是合成输入的单元测试证据，不能充当真实驾驶复用率或原模型任务质量证据。

三模块最终回归：{tests['passed']} 通过，{len(failed)} 失败，{len(skipped)} 跳过。跳过项目与原因见 regression_results.json。CPU 下实际 Petr3D 方法的四帧测试替身输出与时序状态相同；真实原模型 Detection/Map/Memory/LLM、完整 EVA 中间输出、CUDA 长序列、显存和 E2E P50/P95 为 Not Run，不能沿用旧实验通过结果。

沙箱内 CUDA 不可见；两次沙箱外只读 nvidia-smi 查询都因自动审批超时而未执行。该证据不能证明主机驱动损坏。完整五臂 CUDA 验证入口和自然/重复输入分离命令已提供；未完成 GPU 查询和本次完整质量对照前，不颁发任何新数值、任务或性能资格。

新状态保存 Observed Source 与 Computed Source 的区别。计算来源在后续验证帧和 Cache Publication 前进时保持不变。M1 只保存来源元数据；M2 拥有 Tensor。maintain 持续执行 Precheck/Commit/身份维护，拒绝机会时只推迟 Dense/Surface 验证；reset_on_reject 与 output_only 继续属于不同服务，不混报。

身份 ID 序列在 Dense、Source-Linked 和按需三臂比较一致性：{same}。物理证书生产器、身份算法、M3、原模型 Backbone/Head/LLM 和权重均未修改。尚未重放历史所有身份候选边，不将此项检查写成完整身份物理认证。

独立资格：Calibration/Heldout/Final Test = Not Run；Qualified Certification=0；Qualified FAR=null。G1–G8 逐项状态见 formal_go_nogo.json。候选更多与合法计算跳过之间仍有明确证据缺口；生产默认保留 Exact 检查、成本/质量门及 Full Forward 回退。

复现见 commands.sh；合同见 state_contract.md；实现审计见 implementation_changes.md。中间失败和修订前记录均保留在 attempts/ 和 logs/，最终统计只使用 replay/ 与 regression_final.xml。

![Coverage](figures/coverage.png)
![Spatial masks](figures/spatial_distribution.png)
![Admission funnel](figures/admission_funnel.png)
![CPU runtime](figures/runtime.png)
""")
    print(json.dumps(dict(tests=tests["passed"],failed=len(failed),skipped=len(skipped),source_match=all(pin_match.values()))))
    # Keep the current replay's measurements separate from the immutable older
    # CPU-stage snapshot when the CUDA report is regenerated.
    (root/'replay_stage_report.md').write_text((root/'experiment_report.md').read_text())
    from .state_interface_gpu_report import apply
    apply(root)


if __name__=="__main__":main()
