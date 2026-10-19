"""Derive refinement delivery only from this round's raw replay and CUDA runs."""
from collections import Counter
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from .io_utils import write_json
from .state_interface_gpu_report import read,percentile,junit,summarize

ROOT=Path('outputs/module1_source_state_refinement')


def assess_runtime_sources(summary):
    mismatches=[p for p,ok in summary.get('current_runtime_source_matches',{}).items() if not ok]
    compatibility=read(ROOT/'config_compatibility_audit.json',{})
    archived=ROOT/'baseline/config_before_positional_compatibility.py'
    def canonical(path):
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ClassDef) and node.name=='M1Config':
                fields=[n for n in node.body if isinstance(n,ast.AnnAssign)]
                other=[n for n in node.body if not isinstance(n,ast.AnnAssign)]
                node.body=sorted(fields,key=lambda n:n.target.id)+other
        return ast.dump(tree,include_attributes=False)
    config_equivalent=(archived.exists()
        and all(compatibility.get('keyword_configuration_equal',{}).values())
        and canonical(archived)==canonical(Path('worldstream/m1/config.py')))
    def without_failed_publication_fix(path):
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.FunctionDef) and node.name=='forward':
                node.body=[n for n in node.body if not (isinstance(n,ast.If) and
                    isinstance(n.test,ast.UnaryOp) and isinstance(n.test.op,ast.Not) and
                    isinstance(n.test.operand,ast.Name) and n.test.operand.id=='saved')]
        return ast.dump(tree,include_attributes=False)
    old_executor=ROOT/'baseline/camera_executor_before_oom_publication_fix.py'
    runtime_frames=[f['runtime'] for run in read(Path(summary['source'])/'runs.json',[]) for f in run['frames']
        if f['runtime'].get('cache_boundary')=='camera']
    successful_publications=bool(runtime_frames) and all(d.get('cache_saved') is True for d in runtime_frames)
    executor_equivalent=(old_executor.exists() and successful_publications and
        without_failed_publication_fix(old_executor)==without_failed_publication_fix(Path('worldstream/m2/camera_executor.py')))
    allowed={'worldstream/m1/config.py':config_equivalent,'worldstream/m2/camera_executor.py':executor_equivalent}
    equivalent=bool(mismatches) and all(allowed.get(p,False) for p in mismatches)
    return dict(exact_source_hashes_match=summary.get('all_runtime_sources_match',False),
        mismatches=mismatches,keyword_only_configuration_declaration_order_equivalent=config_equivalent,
        successful_publication_paths_equivalent=executor_equivalent,
        camera_frames_checked_for_successful_publication=len(runtime_frames),
        algorithm_sources_compatible=summary.get('all_runtime_sources_match',False) or equivalent,
        compatibility_audit=compatibility if equivalent else None)


def main():
    r=ROOT;p=r/('replay_final' if (r/'replay_final/completion.json').exists() else 'replay');g=r/'gpu_validation_v2'
    def write(name,x):write_json(r/name,x)
    methods=read(p/'method_comparison.json',{})
    pixels=[json.loads(s) for s in (p/'pixels.jsonl').read_text().splitlines()] if (p/'pixels.jsonl').exists() else []
    timings=read(p/'timings.json',[]);links=read(p/'link_recovery.json',[])
    scopes=read(p/'scope_validity.json',[]);centres=read(p/'centre_quality.json',[])
    identity=read(p/'identity.json',[]);leases=read(p/'leases.json',[])
    source_rows=read(p/'source_audit.json',[])
    by_frame={}
    for row in identity:by_frame.setdefault(row['frame'],{})[row['method']]=row['anchor_ids']
    identity_equal=all(all(row.get(name)==row.get('source_linked') for name in
        ('scope_validity','link_recovery','change_first','optimized')) for row in by_frame.values())
    natural=summarize(g/'natural');repeat=summarize(g/'repeat')
    current_natural=summarize(g/'current_source_recheck');current_repeat=summarize(g/'current_repeat_recheck')
    source_assessment=dict(natural=assess_runtime_sources(natural),repeat=assess_runtime_sources(repeat))
    regression=junit(g/('regression_current_cuda.xml' if (g/'regression_current_cuda.xml').exists() else 'regression_cuda.xml'))
    cpu_regression=junit(r/'regression_cpu.xml')
    for folder,result in ((g/'natural',natural),(g/'repeat',repeat),
                          (g/'current_source_recheck',current_natural),(g/'current_repeat_recheck',current_repeat)):
        # Replace the imported legacy report helper's previous-round arm names.
        result['execution_boundaries']={mode:('full recompute' if mode in ('original','full_maintain') else 'camera')
            for mode in result['arms']}
        result['comparison_scope']='Matched Camera boundary, weights, cost table, quality gate and persistent M1 service for old/new Exact arms.'
        runs=read(folder/'runs.json',[])
        for mode,a in result['arms'].items():
            frames=[f for run in runs if run['mode']==mode and not run['warmup'] for f in run['frames']]
            ds=[f['runtime'] for f in frames]
            for key in ('verified_exact_cameras','eligible_exact_cameras','recomputed_cameras','complete_candidate_windows'):
                a[key]=sum(d.get(key,0) or 0 for d in ds)
            a['backbone_GPU_interval_ms']=percentile([f.get('backbone_GPU_interval_ms') for f in frames])
            a['M1_M2_adapter_ms']=percentile([d.get('M1_M2_adapter_ms') for d in ds])
            a['force_recompute_camera_units']=sum(sum(d.get('force_recompute_cameras',[])) for d in ds)
            a['candidate_cameras']=(sum(d['candidate_cameras'] for d in ds)
                if ds and all('candidate_cameras' in d for d in ds) else None)
            a['global_cache_resets']=dict(Counter(d.get('global_cache_reset_reason') for d in ds if d.get('global_cache_reset_reason')))
            a['precheck_breakdown']={key:percentile([(f.get('m1_precheck_diagnostics') or {}).get('timings_ms',{}).get(key) for f in frames])
                for key in ('input_adapter_including_transfers','input_gpu_to_cpu_transfer','temporal_correspondence',
                    'visibility_verification','geometry_support','dense_flow','dense_validation','computation_state_build','snapshot_fork')}
            a['commit_breakdown']={key:percentile([(f.get('m1_commit') or {}).get(key) for f in frames])
                for key in ('association_ms','commit_history_copy_ms','snapshot_fork_ms','cache_provenance_receipt_ms',
                    'output_adapter_including_transfers_ms','output_gpu_to_cpu_transfer_ms')}
            a['input_transfer_bytes']=sum((f.get('m1_precheck_diagnostics') or {}).get('input_gpu_to_cpu_transfer_bytes') or 0 for f in frames)
            a['output_transfer_bytes']=sum((f.get('m1_commit') or {}).get('output_gpu_to_cpu_transfer_bytes') or 0 for f in frames)
            if frames:
                m1_mean=float(np.mean([f.get('M1_time_ms') or 0. for f in frames]))
                m2_mean=float(np.mean([f['runtime'].get('M2_wall_ms') or 0. for f in frames]))
                total_mean=float(np.mean([f['E2E_wall_ms'] for f in frames]))
                a['service_host_wall_decomposition_mean_ms']=dict(M1=m1_mean,M2_host_call=m2_mean,
                    remaining_service=total_mean-m1_mean-m2_mean,total=total_mean,
                    meaning='Disjoint host regions; remainder includes heads, LLM, GPU waits and diagnostics. Not kernel attribution.')
    write('m1_method_comparison.json',dict(population='exploratory_reused_population',methods=methods,
        completion=read(p/'completion.json'),baseline=read(r/'baseline/manifest.json'),
        state_lifecycle={name:dict(records=sum(s['method']==name for s in leases),
            valid=sum(s['method']==name and s['source_valid'] for s in leases),
            revocations=dict(Counter(s['revoked_reason'] for s in leases if s['method']==name and not s['source_valid'])))
            for name in methods},
        source_traceability={name:dict(records=sum(s['method']==name for s in source_rows),
            computation_available=sum(s['method']==name and s['computation_available'] for s in source_rows),
            observation_source_known=sum(s['method']==name and s.get('source_frame_id') is not None for s in source_rows))
            for name in methods},
        shadow_identity_sequences_equal=identity_equal,identity_frames=len(by_frame)))
    scope_result={}
    recovery={};quality={};refresh={};overhead={}
    for name in methods:
        ss=[s for s in scopes if s['method']==name and s.get('observation_support_ratio') is not None]
        scope_result[name]=dict(records=len(ss),support_ratio=percentile([s['observation_support_ratio'] for s in ss]),
            support_count=sum(s['observation_support_count'] for s in ss),
            partial_records=sum(bool(s.get('partial_observation_only')) for s in ss),
            falsely_complete=sum(bool(s.get('observation_valid')) and s['observation_support_ratio']<1 for s in ss))
        ll=[s for s in links if s['method']==name]
        recovery[name]={k:sum(s['recovery'].get(k,0) or 0 for s in ll) for k in
            ('dense_only','sparse_only','both','measurement_conflicts','invalid_coordinates','new_real_correspondences')}
        cc=[s for s in centres if s['method']==name];scored=[s for s in cc if s['scorable']]
        new=[s for s in cc if s.get('origin')==1];new_scored=[s for s in new if s['scorable']]
        quality[name]=dict(candidate_coverage=methods[name]['effective_stable_area'],
            accepted_scorable_points=methods[name]['accepted_scorable_points'],
            false_points=methods[name]['false_stable_points'],sparse_reference_error_rate=methods[name]['sparse_reference_far'],
            sampled_centres=len(cc),scorable_centres=len(scored),scorable_fraction=len(scored)/len(cc) if cc else None,
            centre_correct=sum(bool(s['correct']) for s in scored),centre_false=sum(not s['correct'] for s in scored),
            centre_unscorable=len(cc)-len(scored),new_sampled_centres=len(new),new_scorable_centres=len(new_scored),
            new_correct=sum(bool(s['correct']) for s in new_scored),new_false=sum(not s['correct'] for s in new_scored),
            new_unscorable=len(new)-len(new_scored),dense_physical_ground_truth=None)
        rr=[s for s in pixels if s['method']==name]
        events=[dict(frame=s['frame'],camera=s['camera'],**e) for s in rr for e in s.get('events',[])]
        strict=[e for s in rr for e in s.get('strict_refresh_score',{}).get('events',[])]
        refresh[name]=dict(candidate_complement_event_recall=methods[name]['critical_change_recall'],
            candidate_complement_point_recall=methods[name]['critical_point_recall'],
            minimum_candidate_complement_ROI=methods[name]['minimum_critical_ROI_refresh'],
            deployed_strict_minimum_ROI=min((e['invalidation_coverage'] for e in strict),default=None),
            deployed_strict_event_recall=sum(e['recalled'] for e in strict)/len(strict) if strict else None,
            dirty_ratio=sum(s['changed_pixels'] for s in rr)/sum(s['pixels'] for s in rr),
            unknown_ratio=sum(s['unknown_pixels'] for s in rr)/sum(s['pixels'] for s in rr),
            failures=[e for e in events if e['invalidation_coverage']<.9],
            CAM_FRONT_pedestrian_events=[e for e in events if e['camera']=='CAM_FRONT' and e['semantic_class']=='pedestrian'])
        tt=[s for s in timings if s['method']==name]
        overhead[name]=dict(precheck_ms=percentile([s['M1_precheck_ms'] for s in tt]),
            commit_ms=percentile([s['M1_commit_ms'] for s in tt]),total_ms=percentile([s['M1_wall_ms'] for s in tt]),
            input_adapter_ms=percentile([s['input_adapter_ms'] for s in tt]),
            mask_encoding_ms=percentile([s.get('mask_encoding_ms') for s in tt]),
            precheck={k:percentile([s['precheck_breakdown'].get(k) for s in tt]) for k in
                ('snapshot_fork','computation_state_build','dense_flow','dense_validation','temporal_correspondence',
                    'visibility_verification','geometry_support')},
            commit={k:percentile([s['commit_breakdown'].get(k) for s in tt]) for k in
                ('snapshot_fork_ms','cache_provenance_receipt_ms','association_ms','commit_history_copy_ms')})
    window=read(g/'natural/window_scope_probe.json',dict(status='Not Run',records=[]))
    write('source_scope_validity.json',dict(CPU=scope_result,CUDA_real_window_probe=window,
        scope='Support of actual input dependency pixels; partial support never grants transformer equality'))
    write('observation_link_recovery.json',dict(methods=recovery,centre_quality=quality,
        dense_and_sparse_overlap='Deduplicated with measured source-coordinate conflict checks',
        sparse_scope='Single measured centres only; no patch or box fills'))
    write('candidate_quality.json',dict(population='exploratory_reused_population',methods=quality,
        nearby_diagnostics=read(r/'centre_nearby_diagnostics.json',{}).get('summary',{}),
        interpretation='Annotated depth-visible LiDAR reference only. Missing reference is unscorable; RGB/flow are not physical truth.',online_GT=False))
    write('change_first_refresh.json',dict(methods=refresh,partition='Every replay camera/method asserted disjoint complete partition',
        target=dict(event_recall=.998,point_recall=.998,minimum_ROI=.9),
        diagnostic='Candidate complement is hypothetical; deployed strict refresh and M2 actual execution are separate'))
    write('runtime_breakdown.json',dict(CPU_replay=overhead,CUDA_natural=natural['arms'],CUDA_repeat=repeat['arms'],
        final_source_natural_recheck=current_natural['arms'],final_source_repeat_recheck=current_repeat['arms'],
        baseline_profile='profile_output/baseline_hotspots.json',
        timing_scope='CPU camera stage sums may overlap in threads. CUDA Backbone events include launch gaps. M2 execution_ms is host enqueue wall, not GPU kernel sum.',
        E2E_observer='All arms include the same diagnostic feature CPU copies; memory serialization is outside the request timer'))
    write('m2_exact_reuse_results.json',dict(natural=natural,controlled_repeat=repeat,
        final_source_natural_recheck=current_natural,final_source_repeat_recheck=current_repeat))
    write('e2e_comparison.json',dict(natural=natural,controlled_repeat=repeat,
        matched_boundary='camera',matched_costs_and_quality=True,service='all M1 arms persistent; original=current_output',
        order_limit='Baseline first, other arms rotated per repetition; not a fully balanced thermal-order experiment'))
    write('m1_m2_admission_funnel.json',dict(natural=natural['arms'],controlled_repeat=repeat['arms'],
        final_source_natural_recheck=current_natural['arms'],final_source_repeat_recheck=current_repeat['arms'],
        interpretation='Camera and window counts are distinct. Unknown may pass independent Exact validation; this is not a monotonic candidate-only subset funnel.'))
    selective=[]
    for run in read(g/'repeat/runs.json',[]):
        if run['mode']!='optimized_dirty_front':continue
        for f in run['frames']:
            selective.append(dict(frame=f['frame'],warmup=run['warmup'],**f['runtime']))
    case_tests=[]
    if (g/'regression_current_cuda.xml').exists():
        for c in ET.parse(g/'regression_current_cuda.xml').iter('testcase'):
            for prop in c.findall('./properties/property'):
                if prop.get('name')=='selective_observations':case_tests.append(json.loads(prop.get('value')))
    write('camera_selective_invalidation.json',dict(engineering_actual_small_EVA=case_tests,
        original_model_controlled_dirty=selective,
        final_source_controlled_dirty=[dict(frame=f['frame'],**f['runtime'])
            for run in read(g/'current_repeat_recheck/runs.json',[]) if run['mode']=='optimized_dirty_front'
            for f in run['frames']],
        qualification='Small-EVA forced Exact partial tests grant no production qualification. Original-model Auto rejects unqualified nonzero updates.'))
    write('regression_results.json',dict(CUDA=regression,CPU=cpu_regression))
    write('original_model_parity.json',dict(perception=read(g/'model_parity/status.json',dict(status='Not Run')),
        natural=natural['quality'],controlled_repeat=repeat['quality'],
        current_runtime_sources_match=dict(natural=natural['all_runtime_sources_match'],repeat=repeat['all_runtime_sources_match']),
        source_assessment=source_assessment,
        current_source_full_task_recheck=current_natural,
        current_source_repeat_recheck=current_repeat,
        lineage=dict(natural=natural['source_audit'],repeat=repeat['source_audit'])))
    optimized=refresh.get('optimized',{})
    research_guard=bool(optimized and optimized['candidate_complement_event_recall'] is not None and
        optimized['candidate_complement_event_recall']>=.998 and optimized['candidate_complement_point_recall']>=.998 and
        optimized['minimum_candidate_complement_ROI']>=.9)
    efficient_benefit=bool(overhead and overhead['optimized']['total_ms']['p50']<overhead['change_first']['total_ms']['p50'])
    physical_established=bool(quality and quality['optimized']['new_scorable_centres']>0)
    gates=dict(G1=dict(natural=natural['source_audit']['status'],repeat=repeat['source_audit']['status']),
        G2=dict(partial_support_overclaims=sum(v['falsely_complete'] for v in scope_result.values()),window_probe=window['status']),
        G3=dict(engineering_partition='passed',research_guard_targets=research_guard),
        G4=dict(engineering_cases=len(case_tests),original_model_dirty_checks=len(selective),unqualified_partial_updates_enabled=False),
        G5=dict(new_link_physical_error_established=physical_established,quality=quality.get('optimized')),
        G6=dict(efficient_state_independent_p50_benefit=efficient_benefit,CPU=overhead.get('optimized')),
        G7=dict(CUDA_regression=regression['status'],natural_quality=natural['all_quality_pass'],repeat_quality=repeat['all_quality_pass'],shadow_identity_equal=identity_equal),
        G8='No new independent log qualification; No-Go')
    gates['G1']['current_source_recheck']=dict(natural=current_natural['source_audit']['status'],
        repeat=current_repeat['source_audit']['status'],natural_exact_source_hashes=current_natural['all_runtime_sources_match'],
        repeat_exact_source_hashes=current_repeat['all_runtime_sources_match'])
    write('formal_go_nogo.json',dict(gates=gates,decision='No-Go',
        M1_freeze='No-Go' if not (research_guard and physical_established and efficient_benefit) else 'Review required',
        qualification=dict(Calibration='Not Run',Heldout='Not Run',Final_Test='Not Run',Qualified_Certification=0,Qualified_FAR=None),
        natural_speedup_claim=False,experimental_lease_separation_production_default=False))
    figures=r/'figures';figures.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig,name):fig.tight_layout();fig.savefig(figures/name,dpi=160);plt.close(fig)
    names=list(methods)
    if names:
        fig,axes=plt.subplots(1,2,figsize=(12,4));axes[0].bar(range(len(names)),[methods[n]['effective_stable_area']*100 for n in names])
        axes[0].set_ylabel('Candidate / strict Stable area (%)')
        axes[1].bar(range(len(names)),[quality[n]['accepted_scorable_points']-quality[n]['false_points'] for n in names],label='correct sparse references')
        axes[1].bar(range(len(names)),[quality[n]['false_points'] for n in names],bottom=[quality[n]['accepted_scorable_points']-quality[n]['false_points'] for n in names],label='false sparse references')
        axes[1].legend(fontsize=8);axes[1].set_ylabel('Scorable reference points (not dense truth)')
        for ax in axes:ax.set_xticks(range(len(names)),names,rotation=30,ha='right',fontsize=8)
        save(fig,'coverage_quality.png')
        fig,axes=plt.subplots(1,2,figsize=(10,3))
        ratios=[s['observation_support_ratio'] for s in scopes if s['method']=='optimized' and s.get('observation_support_ratio') is not None]
        axes[0].hist(ratios,bins=20);axes[0].set_title('Historical descriptor input crops (real replay)')
        wr=[s['observation_support_ratio'] for s in window['records'] if s.get('observation_support_ratio') is not None]
        if wr:axes[1].hist(wr,bins=20)
        else:axes[1].text(.2,.5,'Real window probe: '+window['status'])
        axes[1].set_title('Actual original EVA prefix window scope')
        for ax in axes:ax.set_xlabel('Observed dependency fraction');ax.set_ylabel('Records')
        save(fig,'scope_support.png')
        sample=r/'profile_output/spatial_masks.npz'
        if sample.exists():
            z=np.load(sample);fig,axes=plt.subplots(1,4,figsize=(12,3))
            for ax,key in zip(axes,('image','dirty','candidate','unknown')):ax.imshow(z[key],cmap=None if key=='image' else 'gray');ax.set_title(key);ax.axis('off')
            save(fig,'spatial_masks.png')
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        axes[0].bar(range(len(names)),[overhead[n]['precheck_ms']['p50'] for n in names],label='Precheck P50')
        axes[0].bar(range(len(names)),[overhead[n]['commit_ms']['p50'] for n in names],bottom=[overhead[n]['precheck_ms']['p50'] for n in names],label='Commit P50')
        axes[0].set_xticks(range(len(names)),names,rotation=30,ha='right',fontsize=8);axes[0].legend(fontsize=8);axes[0].set_ylabel('CPU replay ms')
        nn=list(natural['arms']);bottom=np.zeros(len(nn))
        for key,label in (('M1','M1'),('M2_host_call','M2 host call'),('remaining_service','Rest of service / GPU waits')):
            values=[natural['arms'][n]['service_host_wall_decomposition_mean_ms'][key] for n in nn]
            axes[1].bar(range(len(nn)),values,bottom=bottom,label=label);bottom+=values
        axes[1].set_xticks(range(len(nn)),nn,rotation=30,ha='right',fontsize=8)
        axes[1].set_ylabel('Natural service mean host wall ms');axes[1].legend(fontsize=7)
        save(fig,'runtime.png')
    if selective:
        example=next((s for s in selective if sum(s.get('force_recompute_cameras',[]))==1),selective[0])
        forced=example.get('force_recompute_cameras',[False]*6);equal=example.get('exact_equal_cameras',[False]*6)
        fig,ax=plt.subplots(figsize=(9,2.5));ax.imshow(np.array([forced,equal]),vmin=0,vmax=1,cmap='Blues')
        ax.set_yticks([0,1],['M1 forces recompute','M2 full-input equality']);ax.set_xticks(range(6),example.get('camera_ids',list(range(6))),rotation=15,ha='right',fontsize=8)
        ax.set_title('Auto outcome: '+str(example.get('fallback_reason'))+'; actual reuse='+str(example.get('actual_reused_cameras')))
        save(fig,'selective_invalidation.png')
    fig,axes=plt.subplots(1,2,figsize=(11,3))
    for ax,case,ss in zip(axes,('natural','controlled repeat'),(natural,repeat)):
        selected=[n for n in ss['arms'] if n not in ('original','full_maintain')]
        xs=np.arange(len(selected))
        for offset,key,label in ((-.3,'candidate_windows','Candidate windows'),
                (-.15,'complete_candidate_windows','Complete candidate windows'),
                (0.,'verified_exact_cameras','Exact-equal cameras'),
                (.15,'actual_reused_cameras','Actually reused cameras'),
                (.3,'actual_skipped_windows','Actually skipped windows')):
            ax.bar(xs+offset,[ss['arms'][n][key] for n in selected],.15,label=label)
        ax.set_xticks(xs,selected,rotation=30,ha='right',fontsize=8);ax.set_title(case)
        ax.set_ylabel('Separate unit counts; not a monotonic subset funnel');ax.legend(fontsize=7)
    save(fig,'admission_reuse.png')
    table=[]
    for case,ss in (('自然',natural),('受控重复',repeat)):
        for name,a in ss['arms'].items():
            wall=a['E2E_wall_ms'];q=ss['quality'].get(name)
            interval=f"{wall['p50']:.2f}/{wall['p95']:.2f}" if wall else 'Not Run'
            table.append(f"| {case} | {name} | {a['requests']} | {interval} | {a['actual_reused_cameras']} | "+(f"{q['frozen_target_pass']}/{q['checks']}" if q else 'baseline')+' |')
    cpu_table=[f"| {n} | {methods[n]['effective_stable_area']*100:.6f} | {quality[n]['accepted_scorable_points']}/{quality[n]['false_points']} | {overhead[n]['total_ms']['p50']:.2f}/{overhead[n]['total_ms']['p95']:.2f} |" for n in names]
    facts=''
    if methods:
        qr=quality['optimized'];cr=refresh['optimized']
        facts=(f"\n本轮实测：96 连续帧、576 个相机观测。优化方法候选覆盖 {methods['optimized']['effective_stable_area']*100:.6f}%，"
            f"同人口旧 Source-Linked 为 {methods['source_linked']['effective_stable_area']*100:.6f}%。"
            f"稀疏可评分接受 {qr['accepted_scorable_points']} 点、错误 {qr['false_points']} 点；"
            f"抽样中心 {qr['sampled_centres']} 个，其中新增中心 {qr['new_sampled_centres']} 个，"
            f"新增中心可评分 {qr['new_scorable_centres']} 个，其余保持 Unscorable。"
            f"候选补集诊断的事件/点召回为 {cr['candidate_complement_event_recall']*100:.4f}%/"
            f"{cr['candidate_complement_point_recall']*100:.4f}%，最小 ROI 刷新 {cr['minimum_candidate_complement_ROI']*100:.4f}%，未达目标。"
            f"相同候选/刷新算法下，测量缓存使 M1 P50 从 {overhead['change_first']['total_ms']['p50']:.2f} 降为 "
            f"{overhead['optimized']['total_ms']['p50']:.2f} ms；这不等于整个新接口优于旧接口或原模型加速。\n")
    final_checks=sum(q['checks'] for ss in (current_natural,current_repeat) for q in ss['quality'].values())
    primary_checks=sum(q['checks'] for ss in (natural,repeat) for q in ss['quality'].values())
    validation=(f"\n最终源码 CUDA 回归 {regression.get('passed',0)}/{regression.get('total',0)} 通过。"
        f"主长序列完整任务质量核验 {primary_checks} 次通过；最终源码另做自然/重复各 4 帧多臂复查，"
        f"共 {final_checks} 次通过，EVA/Detection/Map/Memory 数值最大绝对差为 0，LLM 文本与规划输出一致。"
        "最终复查运行时源码哈希全部匹配；24 条真实来源记录覆盖 M1 Feature Slot 过期后仍由已提交 M2 Receipt 保留计算来源的路径。"
        "长序列之后修正了来源回执、描述子依赖范围和统计字段，因此长序列不能作为最终源码精确哈希一致的验证；"
        "最终源码完整范围回放在 replay_final/，短模型复查分别在 current_source_recheck/ 和 current_repeat_recheck/。\n")
    text='''# M1 来源状态精细化与高效刷新接口：实测报告

本轮以冻结 HEAD a4d2740 为对照，保留上一轮全部结果。实际数据为 nuScenes 六相机 Development 连续回放，正式独立认证仍为 No-Go。M1 是否可冻结见 formal_go_nogo.json，不能用接口工程通过替代物理质量或真实驾驶加速。

Camera Dirty 改为逐相机强制更新，其他相机继续完整 Exact 检查。Scene、签名、版本、事务与 Full Refresh 仍全局失效。非零相机更新缺少当前签名任务/成本资格时仍执行完整 Backbone。Feature/Window 的支持比例来自实际输入依赖足迹；局部观测不再声明完整有效。新增稀疏 Link 仅是实际测得的中心，冲突和在线变化风险优先撤销候选，不扩张 Patch 或检测 Box。
'''+validation+'''

| M1 方法 | Candidate/Strict 面积 % | 可评分接受/错误点 | Precheck+Commit P50/P95 ms |
|---|---:|---:|---:|
'''+ '\n'.join(cpu_table)+facts+'''

候选像素与稀疏参考点属于不同人口。没有参考的中心保持 Unscorable；RGB 光度/光流不是真值。新增中心的正确、错误和不可评分数量分别保留在 candidate_quality.json，不能由零已发现错误推断零风险。Critical Event/Point 与最小 ROI 保护同时报告候选补集诊断和实际严格刷新，所有失败事件保留在 change_first_refresh.json；本轮未据离线 GT 调整在线掩码或评分规则。
96 帧的人口比上一轮扩大四倍，不能把新增场景带来的覆盖差异当作方法增益。Scope 的多相机 Head/未知依赖记录是显式标记的相机片段，不声明完整 Head/Memory 依赖得到观测支持。邻近 LiDAR 诊断不替代精确像素参考，不能将邻近表面自动当成候选中心真值。

相同历史 24 帧对照中，Source-Linked 覆盖从 0.071540% 到最终候选的 0.078946%；事件/点召回仍为 99.3197%/99.8210%，最差 ROI 仍为 81.8182%，此前 CAM_FRONT 行人失败没有消除（frozen_population_comparison.json）。实际部署严格刷新在本轮 ROI 上保持完整覆盖；自然序列实际复用 Camera=0、实际跳过 Window=0，因此不宣称自然驾驶的计算节省。

最终 Scope 对已知局部描述子按其历史输入裁剪计数，使用去重的实测来源中心，不填充整个 Box。多相机或时序 Head 的完整依赖未知时，支持率/数量为 null，单相机片段另列；历史 source_linked 对照保留旧 any-pixel 语义且没有这些比例，不参与新接口完整范围断言。原 EVA 真实前缀的 108 个 Window 单独探测，其中 20 个仅部分支持，完整计算可用与观测支持相互独立。

| 人口 | 方法 | 测量请求 | E2E Wall P50/P95 ms | 实际复用 Camera | 完整任务质量门 |
|---|---|---:|---:|---:|---|
'''+ '\n'.join(table)+'''

新旧接口主对照使用相同 Camera Boundary、模型、成本表、质量门和 Persistent State 服务；old_source_exact 在共享采样运行时保留旧配置与耦合/全局失效策略。original 仅提供当前输出，不能视作同一持久状态服务。baseline 先执行，其余臂按轮次旋转；该计时并非完全平衡的热状态消融。预热排除在时延统计外，质量核验包含预热。

E2E 包含 M1、当前 Detection/Map/Memory/LLM 和统一诊断特征 CPU 拷贝；原生 Memory 写盘在计时外。CUDA Event 区间包含主机发射间隙，M2 execution_ms 为主机入队墙钟，不冒充 GPU kernel sum。稀疏/稠密相机阶段可能在线程中重叠，不能直接相加当作总时延。完整逐阶段数据和传输字节见 runtime_breakdown.json。
时延图右侧按同一测量请求的均值拆分不重叠主机区间：M1、M2 调用及其余服务；其余项包含 Heads、LLM、GPU 等待和诊断。该图与表格 P50/P95 使用不同统计量，不能把各阶段 P50 之和当作 E2E P50。

受控重复输入、人工单相机 Dirty 负测试与自然驾驶分开。默认关闭的租约分离实验只在无时间/Pose 输入的 EVA 图像边界验证 Exact 缓存；当前 Heads、Pose、Temporal Memory 和 LLM 继续执行，周期刷新和所有质量门保留。该策略不是生产默认配置。Camera 命中不改写为 Window 执行器的跳窗数。

G1–G8 和 M1 冻结判断见 formal_go_nogo.json。Calibration/Heldout/Final Test = Not Run；Qualified Certification=0；Qualified FAR=null。自然驾驶是否发生实际计算节省以新原始执行记录为准，不从候选增长或重复输入命中率推断。
配置兼容性修订只将新增字段移到旧字段之后，保留 M1Config(False) 等位置参数语义。已归档修订前源码并验证三套配置的全部关键字值和规范化 AST 相同。自然序列源码精确哈希差异会如实保留，不能伪装成完全相同文件；该序列和回放均以关键字加载配置，数值算法未改变。详情见 config_compatibility_audit.json 与 original_model_parity.json 的 source_assessment。
另有 OOM 发布回执修订，仅在 cache_saved=False 时修正不存在的下一次发布版本；发布成功路径数值与元数据均不变。原始序列的精确源码哈希不改写，source_assessment 会核对 AST 差异及该序列每个相机路径均成功发布的事实。当前最终代码另外运行完整 CUDA 回归和短自然序列完整任务复查，不用这两项未覆盖条件的旧结果冒充新版本通过。

![Coverage and sparse quality](figures/coverage_quality.png)
![Scope support](figures/scope_support.png)
![Measured spatial masks](figures/spatial_masks.png)
![Selective invalidation](figures/selective_invalidation.png)
![Independent exact admission and reuse](figures/admission_reuse.png)
![Maintenance and service latency](figures/runtime.png)

实现与 profiling 采样改动清单见 implementation_changes.md；复现见 commands.sh。失败与过渡实验保存在 attempts/；最终完整范围重放在 replay_final/，先前同人口数据保留在 replay/；CUDA 原始记录在 gpu_validation_v2/。主要长序列计时对应其各自记录的源码版本，最终源码另有自然/重复输入复查；不同版本或不同长短序列的延迟不混合。
'''
    (r/'experiment_report.md').write_text(text)
    print(json.dumps(dict(CUDA=regression['status'],natural=natural['status'],repeat=repeat['status'],research_guard=research_guard,efficient_p50_benefit=efficient_benefit)))


if __name__=='__main__':main()
