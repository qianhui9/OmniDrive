"""Merge only fresh CUDA evidence into the state-interface deliverables."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
from .io_utils import write_json


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def percentile(values):
    finite=[float(v) for v in values if v is not None and np.isfinite(v)]
    return dict(p50=float(np.percentile(finite,50)),p95=float(np.percentile(finite,95)),samples=len(finite)) if finite else None


def junit(path):
    if not path.exists():return dict(status="Not Run")
    cases=list(ET.parse(path).iter("testcase"))
    failures=[c.attrib for c in cases if c.find("failure") is not None or c.find("error") is not None]
    skipped=[dict(c.attrib,reason=c.find("skipped").get("message")) for c in cases if c.find("skipped") is not None]
    return dict(status="passed" if cases and not failures and not skipped else "incomplete",
        total=len(cases),passed=len(cases)-len(failures)-len(skipped),
        failures=failures,skipped=skipped,source=str(path))


def summarize(folder):
    runs=read(folder/"runs.json",[])
    quality=read(folder/"quality.json",[])
    state=read(folder/"status.json",dict(status="partial" if runs or quality else "Not Run"))
    result=dict(status=state.get("status"),source=str(folder),arms={},quality={})
    result['execution_boundaries']=dict(strict_exact='window',source_linked_exact='camera',
        source_linked_on_demand='camera',source_linked_zero_probe='camera (experimental zero-update qualification)',
        full_maintain='full recompute',original='full recompute')
    result['comparison_scope']='End-to-end configurations; differing cache boundaries do not isolate the effect of the M1 interface alone.'
    for mode in sorted({r["mode"] for r in runs}):
        groups=[r for r in runs if r["mode"]==mode and not r["warmup"]]
        frames=[f for r in groups for f in r["frames"]]
        checks=[q for q in quality if q["mode"]==mode]
        runtime=[f["runtime"] for f in frames]
        planning=[(f.get('current_model_outputs') or {}).get('planning') or {} for f in frames]
        ade=[p['ade_m'] for p in planning if p.get('ade_m') is not None]
        fde=[p['fde_m'] for p in planning if p.get('fde_m') is not None]
        result["arms"][mode]=dict(
            requests=len(frames),sequences=len(groups),
            service="current_output" if mode=="original" else "persistent_state",
            E2E_wall_ms=percentile([f["E2E_wall_ms"] for f in frames]),
            E2E_GPU_ms=percentile([f["E2E_GPU_ms"] for f in frames]),
            M1_time_ms=percentile([f["M1_time_ms"] for f in frames]),
            M2_validation_ms=percentile([d.get("M2_validation_ms") for d in runtime]),
            M2_execution_ms=percentile([d.get("M2_execution_ms") for d in runtime]),
            M2_host_wall_ms=percentile([d.get("M2_wall_ms") for d in runtime]),
            candidate_windows=sum(d.get("candidate_windows") or 0 for d in runtime),
            verified_exact_windows=sum(d.get("verified_exact_windows") or 0 for d in runtime),
            actual_skipped_windows=sum(d.get("actual_skipped_windows") or 0 for d in runtime),
            actual_reused_cameras=sum(d.get("actual_reused_cameras") or 0 for d in runtime),
            recomputed_windows=sum(d.get("recomputed_windows") or 0 for d in runtime),
            cache_hits=sum(d.get("cache_hits") or 0 for d in runtime),
            fallback_reasons=dict(Counter(d.get("fallback_reason") for d in runtime if d.get("fallback_reason"))),
            source_mismatch=sum(bool(d.get("source_mismatch")) for d in runtime),
            source_age=percentile([d.get("source_age") for d in runtime]),
            peak_gpu_allocated=max((r["peak_gpu_allocated"] for r in groups),default=None),
            peak_gpu_reserved=max((r["peak_gpu_reserved"] for r in groups),default=None),
            cpu_rss_high_water_kib=max((r["cpu_rss_high_water_kib"] for r in groups),default=None))
        result['arms'][mode]['planning_quality']=dict(parsed=sum(bool(p.get('parsed')) for p in planning),
            requests=len(planning),mean_ADE_m=float(np.mean(ade)) if ade else None,
            mean_FDE_m=float(np.mean(fde)) if fde else None)
        if checks:
            result["quality"][mode]=dict(checks=len(checks),
                includes_warmup_requests=True,
                frozen_target_pass=sum(q["frozen_targets_pass"] for q in checks),
                feature_max_abs=max(q["feature"]["max_abs"] for q in checks),
                memory_max_abs=max(q["memory"]["max_abs"] for q in checks),
                output_max_abs=max(q["output"]["max_abs"] for q in checks),
                text_equal=sum(q["text_equal"] for q in checks),
                planning_equal=sum(q["planning_equal"] for q in checks),
                executed_intermediate_max_abs=max([0.]+[
                    v["max_abs"] for q in checks for v in q.get("executed_intermediate_EVA",{}).values()]))
    result["all_quality_pass"]=(state.get('status')=='completed' and bool(quality)
        and all(q["frozen_targets_pass"] for q in quality))
    result["lineage"]=[
        dict(scene=r["scene"],mode=r["mode"],repeat=r["repeat"],frame=f["frame"],
            warmup=r['warmup'],
            sources=f["runtime"].get("output_camera_provenance",[]),
            cache_version=f["runtime"].get("cache_version"),
            backbone_commit=(f.get("m1_commit") or {}).get("backbone_computation_source"))
        for r in runs for f in r["frames"] if f["runtime"].get("actual_reused_cameras")]
    origins={};errors=[];units=available=0
    for run in runs:
        order={frame['frame']:i for i,frame in enumerate(run['frames'])}
        for i,frame in enumerate(run['frames']):
            bundle=frame.get('m1_computation_state') or {}
            for source in bundle.get('source_records',[]):
                units+=1
                if not source.get('computation_available'):continue
                available+=1
                origin=source.get('computation_frame_id')
                observed=source.get('source_frame_id')
                detail=dict(scene=run['scene'],mode=run['mode'],repeat=run['repeat'],
                    frame=frame['frame'],source=source)
                if origin not in order or order[origin]>=i or observed not in order or order[observed]>=i:
                    errors.append(dict(reason='non_committed_or_future_source',**detail))
                version=source.get('computation_version')
                namespace=source.get('computation_version_namespace')
                cache_version=source.get('source_cache_version')
                if version is None or not namespace:
                    errors.append(dict(reason='missing_producer_version',**detail))
                if namespace and namespace.startswith('m2_') and (cache_version is None or version is None or version>cache_version):
                    errors.append(dict(reason='producer_version_after_cache_publication',**detail))
                key=(run['scene'],run['mode'],run['repeat'],source.get('feature_id'),
                    source.get('source_camera_id'),origin,namespace)
                if key in origins and origins[key]!=version:
                    errors.append(dict(reason='immutable_producer_version_changed',**detail))
                origins[key]=version
    result['source_audit']=dict(status='passed' if units and available and not errors else
        'failed' if errors else 'Not Available',units=units,available_units=available,
        distinct_computation_sources=len(origins),errors=errors)
    pins=read(folder/'source_manifest.json',{})
    production={p:value for p,value in pins.items() if p.startswith('worldstream/') or
        p=='projects/mmdet3d_plugin/models/detectors/petr3d.py'}
    result['current_runtime_source_matches']={p:Path(p).exists() and
        hashlib.sha256(Path(p).read_bytes()).hexdigest()==value for p,value in production.items()}
    result['all_runtime_sources_match']=bool(production) and all(result['current_runtime_source_matches'].values())
    return result


def apply(root, gpu_dir=None, supplement_dir=None):
    root=Path(root)
    selection=read(root/'gpu_validation_current.json',{})
    gpu=Path(gpu_dir or selection.get('path') or root/"gpu_validation")
    supplemental = supplement_dir or selection.get('supplement_path')
    supplemental = Path(supplemental) if supplemental else None
    if not (gpu/"status.json").exists():
        access=read(root/'gpu_access_followup.json',{})
        report=root/'experiment_report.md'
        if access and report.exists():
            marker='> GPU 补测执行通道：'
            body=report.read_text()
            if not body.startswith(marker):
                report.write_text(marker+'用户确认主机 GPU 可用；当前执行沙箱无 /dev/nvidia*。'
                    '本轮沙箱外 nvitop 查询和补测启动共 '+str(len(access.get('outside_attempts',[])))+
                    ' 次均因自动审批超时未执行，实际 CUDA 结果仍为 Not Run。'
                    '完整补测入口见 gpu_followup_instructions.md。\n\n'+body)
        return
    write_json(root/'gpu_validation_current.json',dict(path=str(gpu),
        supplement_path=str(supplemental) if supplemental else None))
    status=read(gpu/"status.json")
    supplement_status = read(supplemental/'status.json', {}) if supplemental else {}
    regression=junit((supplemental if supplemental else gpu)/"regression_cuda.xml")
    parity=read(gpu/"model_parity/status.json",dict(status="Not Run"))
    natural=summarize(gpu/"natural")
    diagnostic_repeat=summarize(gpu/"repeat")
    repeat=summarize((supplemental if supplemental else gpu)/"repeat")
    write_json(root/"gpu_followup_results.json",dict(batch=status,regression=regression,
        supplement_batch=supplement_status,perception_parity=parity,natural=natural,
        controlled_repeat=repeat,controlled_unprofiled_diagnostic=diagnostic_repeat))
    # Preserve all CPU-stage documents once before replacing any current results.
    archive=root/"cpu_stage_delivery"
    if not archive.exists():
        archive.mkdir()
        for path in root.glob("*.json"):
            if path.name!="gpu_followup_results.json":shutil.copy2(path,archive/path.name)
        for name in ("experiment_report.md","commands.sh"):
            if (root/name).exists():shutil.copy2(root/name,archive/name)
    result=read(root/"regression_results.json",{})
    if regression['status']=='passed':
        if 'CPU_stage' not in result:
            result['CPU_stage']={k:v for k,v in result.items() if k!='CUDA'}
        result.update(total=regression['total'],passed=regression['passed'],
            failed=regression['failures'],skipped=regression['skipped'],
            execution='OmniDrive with CUDA visible; includes the actual CUDA EVA test',
            source=regression['source'])
    result["CUDA"]=regression
    write_json(root/"regression_results.json",result)
    write_json(root/"e2e_comparison.json",dict(natural=natural,controlled_repeat=repeat,
        controlled_unprofiled_diagnostic=diagnostic_repeat,supplement_batch=supplement_status,
        execution=status["status"],scope="Fresh GPU evidence only; repeated RGB requests are separate from natural driving."))
    parity_result=read(root/"original_model_parity.json",{})
    parity_result.update(CUDA_original_perception=parity,
        CUDA_full_original_model=dict(status=status['status'],natural=natural["quality"],controlled_repeat=repeat["quality"]))
    write_json(root/"original_model_parity.json",parity_result)
    reuse=read(root/"m2_actual_reuse_results.json",{})
    reuse.update(natural_nuScenes=natural,controlled_repeat=repeat)
    write_json(root/"m2_actual_reuse_results.json",reuse)
    dependency=read(root/"m2_dependency_correctness.json",{})
    dependency.update(CUDA_regression=regression,natural_original_EVA_CUDA=natural["quality"],
        controlled_repeat_EVA_CUDA=repeat["quality"])
    write_json(root/"m2_dependency_correctness.json",dependency)
    overhead=read(root/"runtime_overhead.json",{})
    overhead.update(CUDA_natural=natural["arms"],CUDA_controlled_repeat=repeat["arms"],
        CUDA_memory=dict(natural={k:dict(allocated=v['peak_gpu_allocated'],reserved=v['peak_gpu_reserved'])
            for k,v in natural['arms'].items()},controlled_repeat={k:dict(allocated=v['peak_gpu_allocated'],
                reserved=v['peak_gpu_reserved']) for k,v in repeat['arms'].items()}))
    write_json(root/"runtime_overhead.json",overhead)
    funnel=read(root/"m1_m2_admission_funnel.json",{})
    funnel["CUDA_natural"]=natural["arms"]
    funnel["CUDA_controlled_repeat"]=repeat["arms"]
    write_json(root/"m1_m2_admission_funnel.json",funnel)
    audit=read(root/"source_provenance_audit.json",{})
    audit["CUDA_reused_computation_lineage"]=natural["lineage"]+repeat["lineage"]
    audit['CUDA_source_audit']=dict(natural=natural['source_audit'],controlled_repeat=repeat['source_audit'])
    audit['CUDA_runtime_sources_match']=dict(natural=natural['all_runtime_sources_match'],
        controlled_repeat=repeat['all_runtime_sources_match'])
    write_json(root/"source_provenance_audit.json",audit)
    access=read(root/"cuda_access_attempts.json",{})
    access["current_followup"]=status
    write_json(root/"cuda_access_attempts.json",access)
    access_followup=read(root/'gpu_access_followup.json',{})
    access_followup.update(status='GPU_execution_completed' if status['status']=='completed' else status['status'],
        GPU_program_execution=status['status'],host_GPU=status.get('gpu'),
        current_execution=str(gpu),supplement_execution=str(supplemental) if supplemental else None,
        explanation='Early approval-service failures are historical; later approved host runs accessed CUDA successfully.')
    write_json(root/'gpu_access_followup.json',access_followup)
    gate=read(root/"formal_go_nogo.json",{})
    g=gate.setdefault("gates",{})
    g['G1']=dict(natural=natural['source_audit']['status'],controlled_repeat=repeat['source_audit']['status'])
    g['G4']=dict(contract='Candidates and Unknown require independent Exact inputs/dependencies; no candidate grants skip authority',
        CUDA_source_audit=dict(natural=natural['source_audit']['status'],controlled_repeat=repeat['source_audit']['status']),
        engineering_admission_regression=regression['status'])
    g["G5"]=dict(CUDA_regression=regression["status"],perception=parity.get("status"),
        natural_task_quality=natural["all_quality_pass"],repeat_task_quality=repeat["all_quality_pass"])
    g["G6"]=dict(natural_status=natural["status"],repeat_status=repeat["status"],
        natural_latency_measured=any(v["requests"] for v in natural["arms"].values()),
        natural_actual_reused_cameras=sum(v['actual_reused_cameras'] for v in natural['arms'].values()),
        natural_actual_skipped_windows=sum(v['actual_skipped_windows'] for v in natural['arms'].values()),
        speedup_claim=False)
    g["G7"]=dict(CUDA_regression=regression["status"],full_model_execution=status["status"])
    gate["decision"]="No-Go"
    gate["runtime_speedup_claim"]=False
    write_json(root/"formal_go_nogo.json",gate)
    base_path=root/'replay_stage_report.md'
    base=(base_path if base_path.exists() else archive/"experiment_report.md").read_text()
    table=[]
    for case,summary in (("自然连续帧",natural),("受控重复图像",repeat)):
        for name,values in summary["arms"].items():
            latency=values["E2E_wall_ms"]
            interval="Not Run" if latency is None else f"{latency['p50']:.2f}/{latency['p95']:.2f}"
            q=summary["quality"].get(name)
            quality_text="baseline" if name=="original" else (
                f"{q['frozen_target_pass']}/{q['checks']}" if q else "Not Run")
            table.append(f"| {case} | {name} | {values['requests']} | {interval} | "
                f"{values['actual_reused_cameras']} | {values['actual_skipped_windows']} | {quality_text} |")
    text=f"""# M1 状态与 M1→M2 接口：GPU 补充验证

本轮批处理状态：**{status['status']}**。CUDA 回归：{regression.get('passed',0)}/{regression.get('total',0)} 通过，
跳过 {len(regression.get('skipped',[]))}；原模型感知一致性：{parity.get('status')}。
完整模型自然序列：{natural['status']}；受控重复图像：{repeat['status']}。
补充零更新成本与完整任务验证：{supplement_status.get('status','未单独运行')}。
未执行项目继续按 Not Run 记录，不能以主机存在 GPU 或 CPU 测试通过替代 CUDA 证据。

| 人口 | 方法 | 实测请求 | E2E Wall P50/P95 ms | 复用相机 | 跳过 Window | 质量门 |
|---|---|---:|---:|---:|---:|---|
"""+("\n".join(table) if table else "| 本轮尚无已完成 GPU 请求 | — | 0 | Not Run | — | — | Not Run |")+"""

自然与受控重复输入分开统计；各序列重置，M1 各臂持续维护状态。质量门与既有阈值不变，
表中实测请求排除预热；质量核验包含预热，因此质量检查数可能大于计时请求数。
每个方法使用独立的新语言输出目录，并实际重新执行当前 Head、Temporal Memory 与 LLM。
原生 dtype Memory 快照、实际执行的 EVA 中间结果误差、计算来源、回退原因和显存峰值的确切目录见 gpu_followup_results.json 的 source 字段。
相机级复用与 Window 级复用分别计数；相机命中不能伪装成 Window 执行器的计数。
相机路径只测得 M2 总 host wall，未独立分离验证和执行时间，相关分项保留 null；
CUDA Event 区间包含主机调度间隙。计时包含诊断特征 CPU 拷贝，不冒充移除诊断后的部署时延。
每组序列先执行 original 以建立当前输出基准；补测按探测、Auto、按需、Full-Maintain 固定顺序执行，
并非完全平衡的时序消融。预热及测量轮数保留在各目录的 protocol.json；本表仅计测量请求。

original 仅提供当前模型输出；M1 各臂维护持久状态。strict_exact 保留旧 Window 边界，
新轻量两臂使用已有 Camera 边界及 Auto 成本/质量门；因此该比较衡量整体配置，
不能将差异全部归因于 M1 接口。完整模型没有强制启用未获下游资格的非零局部更新。

完整原始记录见 gpu_followup_results.json。CPU 阶段交付快照保留在 cpu_stage_delivery/。
独立人口资格不随 GPU 补测改变：Qualified Certification=0，Qualified FAR=null，正式结论保持 No-Go。
"""
    if supplemental:
        text+='''
旧成本表与当前签名不匹配时，受控重复输入也全部安全回退，原始记录仍保留为 controlled_unprofiled_diagnostic。
补测使用全新实验目录中的成本表：先实测完整相机输入完全相同时的特征复制误差及成本，
再通过 source_linked_zero_probe 实际执行完整 Head/Memory/LLM 质量验证，全部通过后才允许当前签名的 Auto 零更新复用。
该 probe 只用于 Development 验证，禁止部分相机更新，不授予非零局部更新或独立物理认证资格。
自然连续驾驶实测未发生合法计算跳过，不能由受控重复输入的命中率推断真实驾驶加速。
'''
    if status.get("error"):text+="\n本轮执行阻塞："+status["error"]+"\n"
    text+="\n## 先前 CPU 阶段记录（历史状态，不代表当前 GPU 结果）\n\n"+base
    (root/"experiment_report.md").write_text(text)
    if natural["arms"] or repeat["arms"]:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(11,4))
        labels=[];values=[]
        for case,summary in (("natural",natural),("repeat",repeat)):
            for name,arm in summary["arms"].items():
                if arm["E2E_wall_ms"]:
                    labels.append(case+"\n"+name);values.append(arm["E2E_wall_ms"]["p50"])
        ax.bar(range(len(values)),values)
        ax.set_xticks(range(len(values)),labels,rotation=30,ha="right",fontsize=8)
        ax.set_ylabel("Measured E2E Wall P50 (ms)")
        fig.tight_layout()
        fig.savefig(root/"figures/gpu_e2e.png",dpi=160)
        plt.close(fig)
        report=root/'experiment_report.md'
        report.write_text(report.read_text().replace('## 先前 CPU 阶段记录',
            '![GPU E2E latency](figures/gpu_e2e.png)\n\n## 先前 CPU 阶段记录',1))


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='outputs/module1_state_interface_rearchitecture')
    parser.add_argument('--gpu-dir')
    parser.add_argument('--supplement-dir')
    args=parser.parse_args()
    apply(args.root,args.gpu_dir,args.supplement_dir)
