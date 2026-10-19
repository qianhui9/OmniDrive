"""Aggregate fresh observations; never turn controlled interventions into admission."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from tools.worldstream_m1.io_utils import write_json
from tools.worldstream_m1.state_interface_gpu_report import summarize,junit,percentile
from .state_aware_feasibility import ROOT
from .state_aware_lineage import window_lineage


def read(name):return json.loads((ROOT/name).read_text())


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_stats(frames):
    names=sorted({k for f in frames for k in f['pipeline']})
    return {name:dict(GPU_interval_ms=percentile([f['pipeline'].get(name,{}).get('gpu_ms',0) for f in frames]),
        host_enqueue_wall_ms=percentile([f['pipeline'].get(name,{}).get('cpu_enqueue_wall_ms',0) for f in frames])) for name in names}


def case_summary(case):
    folder=ROOT/'gpu_validation'/case;summary=summarize(folder)
    summary.pop('execution_boundaries');summary['comparison_scope']='Matched window prefix, same frozen persistent M1 and original complete VLM; original has no M1.'
    runs=json.loads((folder/'runs.json').read_text());quality=json.loads((folder/'quality.json').read_text())
    summary['window_lineage_audit']=window_lineage(runs)
    for mode,arm in summary['arms'].items():
        frames=[f for r in runs if r['mode']==mode and not r['warmup'] for f in r['frames']]
        q=[q for q in quality if q['mode']==mode]
        arm.update(pipeline=stage_stats(frames),cache_bytes=percentile([f['runtime'].get('cache_bytes') for f in frames]),
            conditional_reused_windows=sum(f['runtime'].get('conditional_reused_windows',0) for f in frames),
            actual_work_records=sum(len(f['actual_GPU_work']) for f in frames),
            prefix_attention_invocations=sum(w['component']=='attention' and w['block']<2 for f in frames for w in f['actual_GPU_work']),
            language_work_executed=all(f['pipeline'].get('LLM_generation',{}).get('calls',0)>0 and f['pipeline'].get('LLM_decoder',{}).get('calls',0)>0 for f in frames),
            prefix_MLP_input_tokens=sum(int(np.prod(w['shape'][:-1])) for f in frames for w in f['actual_GPU_work'] if w['component']=='MLP' and w['block']<2),
            all_tail_blocks_execute=all({(w['block'],w['component']) for w in f['actual_GPU_work']} >=
                {(i,c) for i in range(2,24) for c in ('attention','MLP')} for f in frames))
        if q:
            summary['quality'][mode].update(max_detection_center_m=max((v['detection'].get('max_center_difference_m') or 0 for v in q),default=0),
                max_detection_unmatched=max(v['detection'].get('unmatched',0) for v in q),
                max_map_coordinate_m=max((z['max_abs'] for v in q for z in v['map']['leaves'] if z['path']=='root[0][0]'),default=None),
                max_matched_map_score_error=max(v['map_matched']['max_matched_score_abs'] for v in q),
                max_matched_memory_point_m=max((b.get('max_matched_point_error_m',0) for v in q for b in v['memory_matched'].values()),default=None),
                max_matched_memory_embedding=max((b.get('leaves',{}).get('memory_embedding',{}).get('max_abs',0) for v in q for b in v['memory_matched'].values()),default=None),
                max_matched_memory_timestamp_error=max((b.get('leaves',{}).get('memory_timestamp',{}).get('max_abs',0) for v in q for b in v['memory_matched'].values()),default=None),
                memory_matching_shape_failures=sum(b.get('shape_mismatch',False) for v in q for b in v['memory_matched'].values()),
                parsed_planning=sum(v['planning'].get('parsed',False) for v in q),
                layers={k:dict(max_abs=max(v['layers'][k]['max_abs'] for v in q),
                    relative_l2_max=max(v['layers'][k]['relative_l2'] for v in q),
                    token_error_p95_max=max(v['layers'][k]['token_error_p95'] for v in q)) for k in q[0]['layers']})
    # Paired requests match scene/replay/ordinal; these are descriptive observed
    # deltas, not a claim of independently replicated generalization.
    indexed={(r['mode'],r['scene'],r['repeat'],f['ordinal']):f for r in runs if not r['warmup'] for f in r['frames']}
    paired={}
    for mode in summary['arms']:
        paired[mode]={}
        for baseline in ('original','full_maintain','conditional'):
            pairs=[(f,indexed[baseline,scene,rep,ordinal]) for (m,scene,rep,ordinal),f in indexed.items() if m==mode]
            deltas=[b['E2E_wall_ms']-a['E2E_wall_ms'] for a,b in pairs]
            paired[mode][baseline]=dict(saved_wall_ms=percentile(deltas),mean_saved_wall_ms=float(np.mean(deltas)) if deltas else None)
    summary['paired_observed_savings']=paired
    summary['units']='actual_skipped_windows is historical API: block-window computations (two prefix blocks); conditional_reused_windows counts distinct windows.'
    return summary,runs,quality


def main():
    assert read('gpu_validation/status.json')['status']=='completed'
    assert read('conditional_execution_profile.json')['status']=='completed'
    if not (ROOT/'prefix_only_profile.json').exists():
        from .source_state_refinement_cuda_batch import CPUS,PYTHON
        with (ROOT/'gpu_validation/prefix_only_profile.log').open('w') as log:
            p=subprocess.run(['taskset','-c',CPUS,PYTHON,'-u','-m','tools.worldstream_m2.state_aware_prefix_profile'],stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError('Original prefix-only GPU measurement failed')
    if not (ROOT/'token_error_details/completion.json').exists():
        # Timed VLM/profile work has finished. Preserve the first strict probe,
        # then collect final-source per-token details without resource overlap.
        from .source_state_refinement_cuda_batch import wait_idle,CPUS,PYTHON
        if (ROOT/'token_error_details').exists():raise RuntimeError('Incomplete detailed probe directory; preserve and inspect before rerunning')
        wait_idle(ROOT)
        shutil.copyfile(ROOT/'aligned_probe_raw.json',ROOT/'aligned_probe_initial.json')
        with (ROOT/'gpu_validation/aligned_detail_probe.log').open('w') as log:
            p=subprocess.run(['taskset','-c',CPUS,PYTHON,'-u','-m','tools.worldstream_m2.state_aware_probe'],stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError('Final-source detailed alignment probe failed')
    gap=ROOT/'temporal_gap_probe'
    if not (gap/'token_error_details/completion.json').exists():
        from .source_state_refinement_cuda_batch import wait_idle,CPUS,PYTHON
        if gap.exists():raise RuntimeError('Preserve and inspect partial temporal gap probe')
        wait_idle(ROOT)
        with (ROOT/'gpu_validation/temporal_gap_probe.log').open('w') as log:
            p=subprocess.run(['taskset','-c',CPUS,PYTHON,'-u','-m','tools.worldstream_m2.state_aware_probe',
                '--output',str(gap),'--frame-stride','2'],stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError('Different temporal interval probe failed')
    raw=read('aligned_probe_raw.json');rows=raw['rows'];assert raw['status']=='completed'
    assert read('token_error_details/completion.json')['archives']==len({r['frame'] for r in rows})
    census=[]
    for scene in sorted({r['scene'] for r in rows}):
        camera=next(r['camera'] for r in rows if r['scene']==scene)
        selected=sorted([r for r in rows if r['scene']==scene and r['camera']==camera],key=lambda r:r['timestamp'])
        yaw=np.unwrap([r['ego_yaw_rad'] for r in selected]);xyz=np.array([r['ego_translation'] for r in selected])
        census.append(dict(scene=scene,name=selected[0]['scene_name'],description=selected[0]['scene_description'],
            frames=len(selected),cameras=sorted({r['camera'] for r in rows if r['scene']==scene}),
            gaps_s=np.diff([r['timestamp'] for r in selected]),ego_yaw_range_degrees=float(np.ptp(yaw)*180/np.pi),
            traveled_distance_m=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()),
            camera_timestamp_skew_s=percentile([r['camera_timestamp']-r['timestamp'] for r in rows if r['scene']==scene]),
            low_texture_patch_gray_std=percentile([r['patch_gray_std']['p05'] for r in rows if r['scene']==scene]),
            first_frame_exact_repeated_RGB_patches=sum(r.get('repeated_RGB_patches',0) for r in rows if r['scene']==scene)))
    write_json(ROOT/'population_census.json',dict(population='exploratory_reused_population',scenes=census,
        semantics='Scene descriptions and measured ego yaw/texture are descriptive coverage evidence, not independent event/physical certification. No annotation identity or motion label enters admission.'))
    gap_rows=read('temporal_gap_probe/aligned_probe_raw.json')['rows']
    gap_summary=dict(source='temporal_gap_probe/aligned_probe_raw.json',frames=len({r['frame'] for r in gap_rows}),
        frame_stride=2,camera_observations=len(gap_rows),complete_source_windows=sum(r['windows'] for r in gap_rows),
        observed_tokens=sum(r['observed_tokens'] for r in gap_rows),complete_tokens=sum(r['complete_tokens'] for r in gap_rows),
        source_age_s=percentile([r['timestamp']-s['computation_timestamp'] for r in gap_rows for s in r['computation_sources'] if s.get('computation_timestamp') is not None]))
    probe=dict(population='exploratory_reused_population',frames=len({r['frame'] for r in rows}),camera_observations=len(rows),
        scenes=len({r['scene'] for r in rows}),observed_tokens=sum(r['observed_tokens'] for r in rows),
        complete_source_tokens=sum(r['complete_tokens'] for r in rows),complete_source_windows=sum(r['windows'] for r in rows),
        conflicting_tokens=sum(r['conflicting_tokens'] for r in rows),mapping_host_ms=percentile([r['mapping_host_ms'] for r in rows]),
        M1_M2_adapter_ms=percentile([r['wall_ms'] for r in raw['adapter_checks']]),
        probe_wall_ms=percentile([r.get('probe_wall_ms') for r in rows]),layers={},
        raw='aligned_probe_raw.json',different_interval_probe=gap_summary,population_census='population_census.json',source_identity='M1 measured observation association; no independent physical certification',
        gate='Whole window, all measured pixels, consistent source displacement, no dirty/unknown/conflicts, valid source lease and producer version',
        diagnostic_only='All features actually computed. Partial-token interpolation never authorizes prefix skip.')
    for name in ('patch','block_0','block_1','block_2','block_23'):
        values=[r['layers'][name] for r in rows if r['layers'].get(name,{}).get('tokens')]
        probe['layers'][name]=dict(camera_frames=len(values),tokens=sum(v['tokens'] for v in values),
            maximum_max_abs=max((v['max_abs'] for v in values),default=None),maximum_relative_l2=max((v['relative_l2'] for v in values),default=None),
            max_abs=percentile([v['max_abs'] for v in values]),relative_l2=percentile([v['relative_l2'] for v in values]),
            token_error_p50=percentile([v['token_error_p50'] for v in values]),token_error_p95=percentile([v['token_error_p95'] for v in values]),
            wrong_source_relative_l2=percentile([v['deliberate_wrong_source_control'].get('relative_l2') for v in values]),
            wrong_source_control='Roll previous features one token horizontally before measured alignment; diagnostic intervention only')
    legacy=ROOT/'legacy_probe_raw.json'
    if not legacy.exists():shutil.copyfile(ROOT/'source_aligned_feature_analysis.json',legacy)
    probe['legacy_probe']='legacy_probe_raw.json';write_json(ROOT/'source_aligned_feature_analysis.json',probe)
    cases={};allruns={};allquality={}
    for case in ('natural','repeat','local'):
        cases[case],allruns[case],allquality[case]=case_summary(case)
        assert cases[case]['status']=='completed'
    write_json(ROOT/'natural_sequence_results.json',dict(feature_opportunity=probe,full_VLM=cases['natural']))
    write_json(ROOT/'controlled_input_results.json',dict(repeat=cases['repeat'],local=cases['local'],
        intervention='Extra local arm uses known RGB permutation, deliberately bypasses M1 admission. Never a natural opportunity or automatic candidate.'))
    write_json(ROOT/'downstream_task_quality.json',dict(frozen_targets=json.loads(Path('outputs/module2_refinement/quality_targets.json').read_text()),
        populations={k:v['quality'] for k,v in cases.items()},
        actual_planning_metrics={k:{m:a['planning_quality'] for m,a in v['arms'].items()} for k,v in cases.items()},
        planning='Actual greedy LLM and annotation-referenced ADE/FDE; unparsed/missing-reference results remain null or failed, never zero-filled.',
        memory='Raw native-dtype NPZ per request plus timestamp/validity-grouped reference-point matching, permutations and per-leaf errors in quality.json.',
        certification='Exploratory reused population only. No independent conditional quality certification.'))
    feature=dict(source_aligned_probe=probe['layers'],full_model={k:{m:q['layers'] for m,q in v['quality'].items()} for k,v in cases.items()},
        raw_locations='gpu_validation/{natural,repeat,local}/quality.json retains each frame/camera/window MaxAbs/relative L2 and P50/P95 token errors.')
    write_json(ROOT/'feature_error_analysis.json',feature)
    refresh=[]
    for r in allruns['local']:
        for q in allquality['local']:
            if q['mode']==r['mode'] and q['ordinal'] in (5,6,7,10,11):
                refresh.append(dict(mode=r['mode'],ordinal=q['ordinal'],prefix_refreshed=q['ordinal'] in (6,11),
                    EVA_max_abs=q['feature']['max_abs'],detection=q['detection'],map=q['map_matched'],
                    memory_matched=q['memory_matched'],quality_pass=q['frozen_targets_pass']))
    write_json(ROOT/'window_dependency_and_refresh.json',dict(prefix=[0,1],first_global=2,unit='whole closed Window',
        partial_window_policy='Recompute entire window, including Q/K/V/softmax/MLP; no partial output substitution',
        positions='Current PE receives current ABS. Reused displaced post-attention outputs still contain historical ABS/RoPE and context; unresolved approximate semantics, default disabled.',
        refresh=refresh,downstream_reset=False,source_audits={k:v['source_audit'] for k,v in cases.items()},
        window_root_audits={k:v['window_lineage_audit'] for k,v in cases.items()},
        provenance='Producer frame/version/window index remain immutable; latest verification frame is separate. Approximate ancestry cannot become Exact.'))
    write_json(ROOT/'conditional_reuse_ablation.json',dict(cases=cases,
        exact_qualification='Existing Exact cost table and approved counts unchanged; experimental mode exact means forced research execution, not task qualification.',
        optimized='Existing SpatialWindowGather vs existing original gather only. No new scheduler or empirical score.',
        residual_threshold_ablation='Not Run: zero complete source-supported windows; relaxing patch residual cannot create missing observation support.',
        conditional_auto='Always full fallback: no qualified conditional cost/quality records exist. conditional_cost_table is reserved and grants no admission.'))
    upper=read('reuse_feasibility_upper_bound.json');profile=read('conditional_execution_profile.json')
    upper['oracle_scope_clarification']='Full EVA minus cached block-1 tail also removes PE/ABS, so this is a deliberately loose entire-frontend oracle. Conditional runtime still computes PE.'
    upper['separate_original_block0_block1_measurement']=read('prefix_only_profile.json')
    write_json(ROOT/'reuse_feasibility_upper_bound.json',upper)
    topology=read('eva_dependency_analysis.json');topology['final_GPU_position_mechanism']=read('eva_position_metadata.json')
    topology['source_locality']='Global EVA attention spans each camera token grid, not batch-camera indices. Detection/Map fuse six cameras and temporal memory.'
    write_json(ROOT/'eva_dependency_analysis.json',topology)
    m1=[f['M1_time_ms'] for r in allruns['natural'] if r['mode']=='full_maintain' and not r['warmup'] for f in r['frames']]
    costs=dict(impossible_oracle_frontend_saving_ms=upper['ideal_saving_wall_ms'],
        measured_block0_block1_wall_ms=upper['separate_original_block0_block1_measurement']['timing']['wall_ms'],PE_full_wall_ms=upper['patch']['wall_ms'],
        natural_allowed_whole_window_saving_ms=0 if probe['complete_source_windows']==0 else None,
        full_persistent_M1_ms=percentile(m1),source_mapping_host_ms_per_camera=probe['mapping_host_ms'],
        marginal_M2='Compare each arm to full_maintain, including cache/alignment/adapter and full M1 maintenance.',
        persistent_service='Same full M1 state contract in all non-original arms; all heads, temporal memory and greedy LLM remain live.',
        original_comparison='Original OmniDrive without M1; complete synchronized request wall.',
        observed_paired_savings={k:v['paired_observed_savings'] for k,v in cases.items()},
        qualified_conditional_records=[],conditional_costs='conditional_execution_profile.json',
        interpretation='Natural admissible saving is zero before positive validation/cache costs. Ideal 43ms-style all-prefix oracle cannot represent natural reuse. Single measured replay per scene; descriptive timing only.')
    write_json(ROOT/'cost_benefit_analysis.json',costs)
    write_json(ROOT/'end_to_end_comparison.json',dict(populations={k:v['arms'] for k,v in cases.items()},paired=costs['observed_paired_savings'],
        observer_effect='CUDA events and forward hooks in all arms. CPU feature/output/memory copies and numerical comparisons occur after E2E timer; feature probe itself stays inside. Per-stage event intervals overlap and cannot be summed.',
        timing='M2_execution_ms is host enqueue wall, not synchronized GPU duration; actual GPU and complete wall are separately retained. Heads include their temporal-memory maintenance.',
        limitations='Two natural scenes x8 frames, one repeat scene x8 frames with warm replay, one local scene x12 without warm replay. Probe covers 8 scenes x12. Exploratory, no independent certification.'))
    irregular=read('gpu_irregular_execution_profile.json');irregular['conditional_profile']='conditional_execution_profile.json'
    irregular['kernel_profiles']=profile['kernel_profiles'];irregular['qualified_conditional_records']=[]
    write_json(ROOT/'gpu_irregular_execution_profile.json',irregular)
    regression=junit(ROOT/'gpu_validation/regression_final_cuda.xml');write_json(ROOT/'regression_results.json',regression)
    baseline=read('baseline/manifest.json');protected={p:sha(p)==v for p,v in baseline['sha256'].items() if p.startswith(('worldstream/m1/','worldstream/m3/','configs/'))}
    original_paths=subprocess.check_output(['git','ls-files','projects/mmdet3d_plugin/models',
        'outputs/module2_refinement/break_even_results.json'],text=True).splitlines()
    for name in original_paths:
        protected[name]=Path(name).read_bytes()==subprocess.check_output(['git','show','HEAD:'+name])
    frozen=read('m1_frozen_contract_audit.json')
    frozen.update(final_protected_source_verified=all(protected.values()),
        source_audits={k:v['source_audit'] for k,v in cases.items()},baseline_HEAD=baseline['HEAD'])
    write_json(ROOT/'m1_frozen_contract_audit.json',frozen)
    exact_ok=regression['status']=='passed' and all(q['frozen_target_pass']==q['checks'] for v in cases.values() for m,q in v['quality'].items() if m in ('full_maintain','existing_exact','source_probe'))
    gate=dict(m1_interface_freeze='Go' if all(protected.values()) else 'No-Go',exact_m2_preserved='Yes' if exact_ok else 'No',
        conditional_reuse_feasible='No',conditional_reuse_quality_pass='No',conditional_reuse_profitable='No',production_conditional_reuse='Disabled',
        remaining_m2_bottleneck='No complete M1 source-supported natural windows; historical ABS/RoPE contextual outputs cannot be exactly relocated; validation/cache/M1 overhead exceeds zero natural allowed saving.',
        next_module_priority='转向 M3',reason=dict(natural_complete_windows=probe['complete_source_windows'],
            oracle_prefix_upper_ms=upper['ideal_saving_wall_ms'],unqualified_control=cases['local']['quality'].get('unqualified_layout_intervention'),
            conditional_quality='No M1-admitted displaced natural-window population. Safe full fallback does not certify approximate reuse.'),
        population='exploratory_reused_population',Qualified_Certification=0,Qualified_FAR=None,
        independent_qualification=dict(Calibration='Not Run',Heldout='Not Run',Final_Test='Not Run'),
        G1='Pass' if all(protected.values()) else 'Fail',G2='Measured No-Go',G3='Measured diagnostic errors, no equivalence claim',
        G4='Whole windows only; exact and conditional separated',G5='Actual skipped work and CUDA kernel traces archived',
        G6='Conditional unqualified/failed; existing frozen criteria unchanged',G7='No-Go natural profitability',G8=regression['status'])
    assert probe['complete_source_windows']==0,'Reevaluate gates if actual natural opportunities change'
    assert gap_summary['complete_source_windows']==0,'Reevaluate different-interval opportunity before No-Go'
    write_json(ROOT/'formal_go_nogo.json',gate)
    write_json(ROOT/'protected_source_integrity.json',protected)
    figures(probe,cases,allquality,irregular,upper,rows)
    report(gate,probe,cases,upper,regression)
    changes()
    required=['implementation_changes.md','m1_frozen_contract_audit.json','eva_dependency_analysis.json','reuse_feasibility_upper_bound.json',
        'source_aligned_feature_analysis.json','conditional_reuse_ablation.json','window_dependency_and_refresh.json','gpu_irregular_execution_profile.json',
        'cost_benefit_analysis.json','natural_sequence_results.json','controlled_input_results.json','feature_error_analysis.json','downstream_task_quality.json',
        'end_to_end_comparison.json','regression_results.json','formal_go_nogo.json','experiment_report.md','commands.sh']
    errors=[]
    if not exact_ok:errors.append('Exact/full/probe frozen quality or regression failed')
    if not all(protected.values()):errors.append('Protected source changed')
    for case,v in cases.items():
        if v['source_audit']['status']!='passed':errors.append(case+' source audit failed')
        if v['window_lineage_audit']['status']!='passed':errors.append(case+' window root audit failed')
        if not v['all_runtime_sources_match']:errors.append(case+' source pins changed')
        if not all(a['all_tail_blocks_execute'] for a in v['arms'].values()):errors.append(case+' missing tail work')
        if not all(a['language_work_executed'] for a in v['arms'].values()):errors.append(case+' missing actual language generation')
    probe_pins={p:h for p,h in raw['source_sha256'].items() if p.startswith('worldstream/')}
    gap_raw=read('temporal_gap_probe/aligned_probe_raw.json')
    gap_pins={p:h for p,h in gap_raw['source_sha256'].items() if p.startswith('worldstream/')}
    probe_match=bool(probe_pins) and all(sha(p)==h for p,h in probe_pins.items())
    gap_match=bool(gap_pins) and all(sha(p)==h for p,h in gap_pins.items())
    if not probe_match or not gap_match:errors.append('Source probe inference runtime hashes differ')
    errors.extend('Missing '+p for p in required if not (ROOT/p).exists())
    write_json(ROOT/'artifact_validation.json',dict(passed=not errors,errors=errors,required_files=len(required),
        source_probe_runtime_hashes_match=probe_match,temporal_gap_runtime_hashes_match=gap_match,
        source_audits={k:v['source_audit']['status'] for k,v in cases.items()},
        window_root_audits={k:v['window_lineage_audit']['status'] for k,v in cases.items()},
        full_model_requests=sum(len(r['frames']) for runs in allruns.values() for r in runs),
        quality_checks=sum(len(q) for q in allquality.values()),regression=regression,figures=len(list((ROOT/'figures').glob('*.png')))))
    print(json.dumps(read('artifact_validation.json'),indent=2))
    if errors:raise SystemExit(1)


def figures(probe,cases,quality,irregular,upper,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=ROOT/'figures';dest.mkdir(exist_ok=True)
    def finish(name):plt.tight_layout();plt.savefig(dest/(name+'.png'),dpi=170);plt.close()
    names=list(upper['component_GPU_intervals_ms']);plt.figure(figsize=(12,4))
    plt.bar(names,[np.median(upper['component_GPU_intervals_ms'][k]) for k in names]);plt.xticks(rotation=80);plt.ylabel('GPU interval ms');finish('eva_layer_costs')
    chosen=max(rows,key=lambda r:r['observed_tokens']);support=np.array(chosen['support_ratios']).reshape(40,40)
    plt.figure(figsize=(6,5));plt.imshow(support,vmin=0,vmax=1);plt.colorbar(label='Measured pixel support ratio')
    xy=np.array(chosen['source_token_coordinates']).reshape(40,40,2);yy,xx=np.mgrid[:40,:40]
    valid=(support>0)&~np.array(chosen['conflict_mask']).reshape(40,40)
    valid[1::2]=False;valid[:,1::2]=False
    plt.quiver(xx[valid],yy[valid],xy[...,0][valid]-xx[valid],xy[...,1][valid]-yy[valid],color='red',angles='xy',scale_units='xy',scale=1)
    for line in (15.5,31.5):plt.axhline(line,color='white',linewidth=.7);plt.axvline(line,color='white',linewidth=.7)
    plt.title(chosen['camera']+' / measured source displacement');finish('source_token_support')
    write_json(dest/'source_token_support_metadata.json',dict(frame=chosen['frame'],source_frame=chosen['source_frame'],camera=chosen['camera'],
        arrow='Current-token centre to proposed source-token coordinate, every other grid cell with actual support; diagnostic only',complete_windows=chosen['windows']))
    counts=[(r['updated'],r['timing']['wall_ms']) for r in irregular['baseline_records'] if r['strategy']=='original']
    plt.figure(figsize=(6,4))
    for strategy in ('original','spatial'):
        v=[r for r in irregular['baseline_records'] if r['strategy']==strategy];plt.plot([r['updated']/r['total'] for r in v],[r['timing']['wall_ms'] for r in v],'-o',label=strategy)
    plt.xlabel('Updated window fraction');plt.ylabel('Block executor wall ms');plt.legend();finish('window_update_gpu_latency')
    plt.figure(figsize=(7,4));plt.bar(['Observed tokens','Complete patches','Complete windows'],[probe['observed_tokens'],probe['complete_source_tokens'],probe['complete_source_windows']]);plt.ylabel('Count over 96 frames');finish('admission_funnel')
    plt.figure(figsize=(8,4))
    for m in ('conditional','conditional_optimized','unqualified_layout_intervention'):
        v=[r for r in quality['local'] if r['mode']==m];plt.plot([r['ordinal'] for r in v],[r['feature']['relative_l2'] for r in v],'-o',label=m)
    for x in (6,11):plt.axvline(x,color='gray',linestyle='--')
    plt.xlabel('Frame ordinal (prefix full refresh at 6/11)');plt.ylabel('Final EVA relative L2');plt.legend(fontsize=7);finish('feature_error_over_time')
    plt.figure(figsize=(10,4));modes=list(cases['natural']['arms']);plt.bar(modes,[cases['natural']['arms'][m]['E2E_wall_ms']['p50'] for m in modes]);plt.xticks(rotation=25,ha='right');plt.ylabel('Natural E2E wall P50 ms');finish('e2e_latency')
    plt.figure(figsize=(9,4))
    for case in cases:
        names=list(cases[case]['arms']);plt.plot(names,[cases[case]['arms'][m]['actual_skipped_windows'] for m in names],'-o',label=case)
    plt.xticks(rotation=25,ha='right');plt.ylabel('Skipped block-window computations');plt.legend();finish('actual_window_work')


def report(gate,probe,cases,upper,regression):
    lines=['# M2 状态感知增量视觉推理实验', '',
        '本轮完成冻结 M1、来源对齐探针、默认关闭的条件性前缀复用原型、真实 GPU 不规则执行分析和完整 OmniDrive 对照。当前边界的生产结论为 **No-Go，保持 Disabled**。','',
        f"自然探针覆盖 {probe['scenes']} 个已曝光场景、{probe['frames']} 帧、{probe['camera_observations']} 路相机观测。实际支持 Token {probe['observed_tokens']:,}，严格完整 Patch {probe['complete_source_tokens']:,}，完整可复用 Window **{probe['complete_source_windows']}**。这些均为 `exploratory_reused_population`，不构成独立物理或任务认证。",'',
        f"实际 EVA 为 24 Block，0/1 是 16×16 Window，Block 2 首次进入每路相机的全局 Attention。单独执行 Block 0+1 的真实 Wall 为 {upper['separate_original_block0_block1_measurement']['timing']['wall_ms']:.3f} ms。完整 EVA 减去缓存 Block 1 后的 Tail 得到偏宽 Oracle 上界 {upper['ideal_saving_wall_ms']:.3f} ms，其中连 PE/ABS 都免算；原型仍计算 PE，整层 PE 为 {upper['patch']['wall_ms']:.3f} ms。自然完整 Window 为 0，因此可获准的实际节省上界为 0，验证/对齐/缓存仍需成本。",'',
        '## 实现与边界','',
        '`source_alignment.py` 只使用 M1 真实来源点，统计实际像素支持、冲突、Dirty/Unknown、整 Patch 位移以及完整 Window 依赖。部分支持只进入诊断，不填充 Patch/Window。', '',
        '`conditional_prefix.py` 使用独立缓存和现有 WindowExecutor；被复用 Window 的前缀 Attention/MLP 确实不执行，其余 Window 完整计算。Block 2 及以后全部执行。当前 PE 加当前 ABS，但历史上下文内的 ABS/RoPE 无法解析消除，因此位移复用明确标记 approximate，禁止伪造 Transformer/任务认证。生产配置关闭；研究配置显式开启，Auto 对所有条件复用保持回退。', '',
        'GPU 优化只切换现有 original/spatial gather，保留索引顺序、Padding 和不可变历史来源；未另建调度器。成本记录含真实模型指纹/环境/Shape/Stride/Dtype、更新量和实测时间，所有条件记录未获任务资格，不安装进正式 Exact 成本表。','',
        '## 完整模型结果','',
        '| 输入 | 方法 | 计时请求 | E2E P50/P95 ms | 跳过 Block-Window | 冻结质量通过/比较 |','|---|---|---:|---:|---:|---:|']
    for case,v in cases.items():
        for mode,a in v['arms'].items():
            q=v['quality'].get(mode);qs=f"{q['frozen_target_pass']}/{q['checks']}" if q else '基准'
            lines.append(f"| {case} | {mode} | {a['requests']} | {a['E2E_wall_ms']['p50']:.2f}/{a['E2E_wall_ms']['p95']:.2f} | {a['actual_skipped_windows']} | {qs} |")
    lines += ['', '自然输入六方法在相同冻结 M1 持续维护契约下比较，Original 单独作为无 M1 的完整模型基准。B/C 是受控 RGB 输入，真实相机位姿、时间、Detection/Map/Memory/LLM 仍正常执行，不代表自然复用收益。', '',
        '额外 `unqualified_layout_intervention` 使用已知受控图像置换的来源映射，刻意绕过 M1 准入，只用于暴露历史位置上下文的误差传播；不得作为可用 M1 候选或自然性能证据。局部普通研究路径的同坐标未变 Window 也可能因不同 packed batch 浮点数值产生任务误差。', '',
        f"局部普通研究路径没有获准的 M1 位移复用 Window，其跳算依赖同坐标完整输入检查。其最大 Detection 中心差异为 {cases['local']['quality']['conditional']['max_detection_center_m']*100:.3f} cm，最大 Map 坐标差异为 {cases['local']['quality']['conditional']['max_map_coordinate_m']*100:.3f} cm，匹配后 Memory 参考点最大差异为 {cases['local']['quality']['conditional']['max_matched_memory_point_m']*100:.3f} cm。Map 超过冻结 1cm 门槛；这些结果不能当作 M1 来源对齐条件复用的合格人口。", '',
        f"刻意绕过准入的位移压力测试最大 Detection 中心差异为 {cases['local']['quality']['unqualified_layout_intervention']['max_detection_center_m']:.3f} m，只有 1/12 通过。视觉前缀刷新可将 EVA 误差恢复到零，但不能清除已经进入下游 Memory 的误差，详见逐帧刷新记录。", '',
        '所有质量比较沿用冻结的 Detection 中心 1cm、Map 1cm、文本/Planning 一致性协议，保留 raw/matched Memory 和实际解析 ADE/FDE；无法解析/无参考不填零。第 6/11 帧只刷新视觉前缀，未清除下游 Memory，恢复情况见 `window_dependency_and_refresh.json`。', '',
        '计时包含 M1 Precheck/Commit、来源验证、缓存及全部实际语言生成。输出/特征/Memory CPU 拷贝和质量计算在同步计时结束后；内部 Feature Probe 是方法工作量，计入计时。CUDA 事件与钩子具有观测开销，所有方法一致；分项间隔重叠不可直接相加。自然每场景一次预热和一次计时回放，局部一次无预热回放，因此时间为本轮描述性结果，不宣称独立重复性或泛化。', '',
        f"来源映射每路相机 CPU Wall P50/P95 为 {probe['mapping_host_ms']['p50']:.3f}/{probe['mapping_host_ms']['p95']:.3f} ms；M1→M2 Adapter 为 {probe['M1_M2_adapter_ms']['p50']:.3f}/{probe['M1_M2_adapter_ms']['p95']:.3f} ms。双倍帧间隔补充探针覆盖 {probe['different_interval_probe']['frames']} 帧、{probe['different_interval_probe']['camera_observations']} 路观测，完整来源 Window 仍为零。逐场景转弯幅度、纹理和相机时间信息见 `population_census.json`。", '',
        '自然两条研究路径都走 Full，因此其时延差不能归因于 packing 优化。GPU 优化对照来自实际执行的受控 Window microbenchmark；各头时延包含其内部 Memory 维护，不另行虚构独立 Memory 分项。', '',
        f"本轮真实 CUDA 回归：**{regression['passed']}/{regression['total']} 通过**。M1/M3/原模型代码不修改，正式 Exact 的依赖规划、执行器与成本表保持原样。详细来源/版本审计、实际 Attention/MLP 调用 Shape 和 CUDA kernel traces 均已归档。",'',
        '## 最终验收','', '| 项目 | 结论 |','|---|---|']
    for k in ('m1_interface_freeze','exact_m2_preserved','conditional_reuse_feasible','conditional_reuse_quality_pass','conditional_reuse_profitable','production_conditional_reuse','next_module_priority'):
        lines.append(f'| {k} | {gate[k]} |')
    lines += ['', '未具备完整来源支持的自然 Window；历史位置上下文无法安全视作当前位置等价表示；零自然跳算不能覆盖状态维护及验证开销。停止继续堆叠此边界的 Token/Window 复用算法，下一阶段转向完整 VLM 服务的任务条件化视觉计算/语义输入压缩边界。本轮不改动 M3。', '',
        '复现命令见 `commands.sh`；所有命令排除故障 CPU 4/5。保留旧实验，不覆盖先前结论。若需要清理分析钩子，可删除独立实验工具；生产原模型没有植入 profiling 代码。']
    (ROOT/'experiment_report.md').write_text('\n'.join(lines)+'\n')


def changes():
    paths=[Path('worldstream/m2/config.py'),Path('worldstream/m2/runtime.py'),Path('worldstream/m2/source_alignment.py'),Path('worldstream/m2/conditional_prefix.py'),
        *Path('tools/worldstream_m2').glob('state_aware*.py'),Path('tests/worldstream_m2/test_source_aligned_conditional.py'),
        Path('projects/configs/OmniDrive/worldstream_m2_state_aware.py'),Path('projects/configs/OmniDrive/worldstream_m2_conditional_research.py')]
    descriptions={'config.py':'默认关闭与简单硬门限配置','runtime.py':'隔离研究缓存与可关闭分支；Exact 默认行为保持',
        'source_alignment.py':'真实点支持/冲突和整 Window 对齐；诊断误差',
        'conditional_prefix.py':'实际 Window 跳算、源版本、当前位置与全局尾部执行',
        'state_aware_feasibility.py':'修改前实际 EVA GPU 上界/原始 packing 对照',
        'state_aware_probe.py':'96 帧真实来源对齐、分层误差和错误来源对照',
        'state_aware_integration.py':'原模型完整推理、CUDA事件、真实工作量、层/任务/Memory质量对照',
        'state_aware_execution_profile.py':'实际 prefix 跳算/缓存成本、CUDA kernel trace、权重指纹',
        'state_aware_cuda_batch.py':'空闲 GPU 等待与本轮回归/实验串行调度',
        'state_aware_finalize.py':'原始结果聚合、图表与严谨 No-Go/归档审计'}
    descriptions.update({'state_aware_prefix_profile.py':'单独补测原始 Block 0/1 GPU/Wall，排除 PE/ABS',
        'state_aware_lineage.py':'逐 Window 的真实根帧/版本/位置与实际计算量审计',
        'state_aware_finish_batch.py':'等待计时结束，执行最终源码回归和归档'})
    lines=['# Implementation / instrumentation changelog','',
        'M1 工程接口及配置、M3、原模型参数与网络结构、正式 Exact 成本表均不修改。新增生产配置默认关闭研究路径。','',
        '| File | Change type | Purpose / instrumentation | Lines |','|---|---|---|---|']
    for p in paths:
        modified=p.name in ('config.py','runtime.py')
        lines.append(f"| `{p}` | {'modified' if modified else 'created'} | {descriptions.get(p.name,'配置/测试，无原模型结构改动')} | 1–{len(p.read_text().splitlines())} |")
    lines += ['', '原始模型 Forward 的临时事件/钩子只在独立 runner 内安装，并在 finally 中移除。测试 fixture 不用于宣称自然驾驶质量。受控来源干预显式未通过 M1 准入。', '',
        '生产：`projects/configs/OmniDrive/worldstream_m2_state_aware.py`。研究：`worldstream_m2_conditional_research.py`。`conditional_cost_table` 为保留字段，不授予 Auto 准入；不存在合格 Conditional 成本人口。', '',
        'Raw artifacts 在本目录下，旧实验目录不写入。Baseline HEAD/source tar/hash 清单保留修改前状态。']
    (ROOT/'implementation_changes.md').write_text('\n'.join(lines)+'\n')
    write_json(ROOT/'delivery_source_manifest.json',{str(p):sha(p) for p in paths})


if __name__=='__main__':main()
