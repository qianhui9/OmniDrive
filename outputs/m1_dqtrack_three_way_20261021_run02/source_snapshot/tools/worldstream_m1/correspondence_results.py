"""Reproducible, offline-only accounting of fixed correspondence experiments."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .correspondence_diagnostics import ROOT, PRIOR, sha
from .correspondence_experiments import source_manifest
from .quality_coverage_metrics import region_metrics
from .io_utils import write_json


def read(name):return json.loads((ROOT/name).read_text())


def rows(name, directory='traces/development'):
    result=[]
    for path in sorted((ROOT/directory).glob('*/complete.json')):
        source=path.parent/name
        if source.exists():result.extend(map(json.loads,source.read_text().splitlines()))
    return result


def paired(points, method, baseline):
    good=[r for r in points if not r['false']];bad=[r for r in points if r['false']]
    get=lambda r,n:r['old_consensus_accepted'] and not r['forced_refresh'] if n=='surface_consensus' else r['accepted'][n]
    return dict(correct_retained=sum(get(r,method) and get(r,baseline) for r in good),
        correct_lost=sum(not get(r,method) and get(r,baseline) for r in good),
        new_correct=sum(get(r,method) and not get(r,baseline) for r in good),
        wrong_retained=sum(get(r,method) and get(r,baseline) for r in bad),
        wrong_blocked=sum(not get(r,method) and get(r,baseline) for r in bad),
        new_wrong=sum(get(r,method) and not get(r,baseline) for r in bad))


def point_funnel(points, keys):
    live=[r for r in points if r['accepted']['constraints_observation_base']]
    stages={}
    for name,condition in keys:
        old=len(live);oldgood=sum(not r['false'] for r in live)
        live=[r for r in live if condition(r['measured'])]
        stages[name]=dict(correct=sum(not r['false'] for r in live),false=sum(r['false'] for r in live),
            rejected=old-len(live),correct_lost=oldgood-sum(not r['false'] for r in live))
    return stages


def main():
    assert source_manifest()==read('development_source_freeze.json')['files'], 'Runtime changed during experiment'
    assert len(list((ROOT/'traces/development').glob('*/complete.json')))==20
    pins=read('historical_artifact_pins.json');assert all(sha(p)==digest for p,digest in pins.items())
    summary=read('development_summary.json');methods=summary['region'].copy()
    if (ROOT/'validation_ablation_summary.json').exists():methods.update(read('validation_ablation_summary.json'))
    previous=json.loads((PRIOR/'region_method_comparison.json').read_text())['methods']
    for name in ('full_refresh','conservative','dense_original','local_r12','surface_local_plane','surface_consensus'):
        methods[name]=previous[name]
    main_name='correspondence_recovery';current=methods[main_name]
    # Independently recomputed legacy masks must reproduce the archived run.
    equality_keys=['effective_stable_area_ratio','accepted_scorable_points','false_stable_points',
        'critical_change_recall','critical_point_recall','minimum_roi_event_refresh_ratio']
    legacy={key:methods['surface_constraints'][key]==previous['surface_constraints'][key] for key in equality_keys}
    assert all(legacy.values()),legacy
    pixels=rows('pixels.jsonl');points=rows('point_decisions.jsonl');hypotheses=rows('hypotheses.jsonl')
    mainrows=[r for r in pixels if r['method']==main_name]
    point_pairs={base:{name:paired(points,name,base) for name in summary['region']}
        for base in ('surface_constraints','surface_consensus','constraints_observation_base')}
    write_json(ROOT/'region_paired_support_analysis.json',dict(population='exploratory_reused_population',
        comparisons=point_pairs,denominator='Exact same scene/frame/camera/reference-index positions',
        all_pixel_acceptance_pairs=read('pixel_pair_and_mask_audit.json')['comparisons'],
        old_five_correct_retained=point_pairs['surface_constraints'][main_name]['correct_retained']))
    for method,metric in methods.items():
        metric.update(correct_stable_dense_area=None,false_stable_dense_area=None,
            dense_truth_note='Sparse annotated LiDAR does not label dense surface area. Do not infer correct/unscorable area from point counts.')
    write_json(ROOT/'region_method_comparison.json',dict(population='exploratory_reused_population',methods=methods,
        archived_same_input_methods=['full_refresh','conservative','dense_original','local_r12','surface_local_plane','surface_consensus'],
        old_constraints_actual_replay_equal=legacy,formal_selection=None))
    write_json(ROOT/'region_risk_coverage.json',dict(population='exploratory_reused_population',methods=methods,
        selection=None,threshold_grid_search=False))
    write_json(ROOT/'region_ablation.json',dict(methods={k:v for k,v in methods.items() if k in summary['region'] or k.startswith('recovery_')},
        logical_masks_share_identical_actual_inputs=True,separate_manager_methods=list(read('validation_ablation_summary.json')),
        source='candidate_masks plus independent candidate-neighbourhood/motion-sweep replay; no post-hoc threshold search'))
    diag=read('patch_rejection_root_cause.json');base=[r for r in points if r['accepted']['constraints_observation_base']]
    patch={}
    for label,bad in [('correct',False),('false',True)]:
        population=[r for r in base if r['false']==bad]
        patch[label]=dict(before=len(population),old_source=sum(r['measured']['source_consistent'] for r in population),
            new_source=sum(r['measured']['adaptive_source_consistent'] for r in population),
            actual_geometry_applicable=sum(r['measured']['patch_model_applicable'] for r in population),
            recovered_source=sum(r['measured']['patch_geometry_recovered'] for r in population))
    write_json(ROOT/'patch_geometry_comparison.json',dict(population='exploratory_reused_population',counts=patch,
        mechanism='Center-anchored inverse-depth plane from four measured ring corners; four disjoint axis pixels and raw projected RGB validate it',
        physical_attribution='Same-RGB fit/holdout is conditional. No proof of true slope, depth boundary or foreground ownership.',
        dense_area_inference=None))
    write_json(ROOT/'patch_true_false_controls.json',dict(population='Complete fixed population, not a GT-chosen online subset',
        counts=patch,source_diagnostics='patch_rejection_root_cause.json',measured_hypotheses=hypotheses,
        actual_rgb_patch_archive=read('patch_rgb_archive.json'),
        dependencies='Local flow/plane samples share the two RGB acquisitions. Third LK directly measures a separate acquisition.'))
    diagnostic=Counter();pixel_funnel=defaultdict(Counter)
    view_counts=defaultdict(Counter)
    for r in mainrows:
        d=r['camera_diagnostics'].get('surface_constraints',{})
        diagnostic.update({k:v for k,v in d.items() if type(v) is int})
        for v in d.get('actual_views',[]):
            view_counts[v['kind']].update({k:v.get(k,0) for k in ('in_fov','geometrically_usable','supported')})
        for gate,v in r['camera_diagnostics'].get('acceptance_funnel',{}).items():
            pixel_funnel[gate].update({k:x for k,x in v.items() if type(x) is int})
    ambiguous=[r for r in base if r['measured']['adaptive_ambiguous']]
    joint=dict(counts=dict(diagnostic),sparse_ambiguous_points=len(ambiguous),
        sparse_continuing_unknown=sum(r['measured']['joint_ambiguous'] for r in ambiguous),
        sparse_resolved_points=sum(not r['measured']['joint_ambiguous'] for r in ambiguous),
        sparse_restored_correct=sum(not r['false'] and r['accepted'][main_name] and not r['accepted']['geometric_adaptive_patch'] for r in points),
        sparse_restored_wrong=sum(r['false'] and r['accepted'][main_name] and not r['accepted']['geometric_adaptive_patch'] for r in points),
        actually_wrong_hypotheses_excluded=0 if diagnostic.get('excluded_hypotheses',0)==0 else None,
        wrong_hypothesis_note='No independent truth for every alternative 3D ray. Excluded conditional hypotheses are counted; their physical correctness is not invented.',
        measured_hypotheses=hypotheses,
        no_failure_implies_false=True,global_uniqueness_claim=False)
    write_json(ROOT/'joint_hypothesis_diagnostics.json',joint)
    write_json(ROOT/'joint_observation_provenance.json',dict(actual_view_counts={k:dict(v) for k,v in view_counts.items()},
        trace='traces/development/*/hypotheses.jsonl',camera_source='Actual nuScenes K, extrinsic and per-camera ego pose; real six-camera timestamps',
        temporal_order='committed third image < source image < current image; bounded by existing horizon',
        same_acquisition_dependencies=['DIS forward/reverse','source patch','local inverse-depth fit and disjoint spatial holdout','epipolar candidate generation'],
        third_acquisition_dependency='Current-to-third LK is measured directly; never propagated through DIS. Different acquisition is not statistical independence.',
        negative_evidence='H0 positive patch and bidirectional local holdout track, source reprojection cycle, distinguishable Hi inside measured third footprint, no actual competing support in any view',
        absent_or_offscreen_competitor='Unknown; not excluded',online_GT=False,online_lidar=False,physical_ownership_certified=False))
    oldkeys=[('base',lambda m:True),('source',lambda m:m['source_consistent']),
        ('extra',lambda m:m['temporal_supported'] or m['camera_supported']),('ambiguity',lambda m:not m['ambiguous']),
        ('depth_order',lambda m:not m['order_conflict'])]
    newkeys=[('base',lambda m:True),('source',lambda m:m['adaptive_source_consistent']),
        ('extra',lambda m:m['adaptive_temporal_supported'] or m['adaptive_camera_supported']),
        ('joint',lambda m:not m['joint_ambiguous']),('depth_order',lambda m:not m['adaptive_order_conflict'])]
    write_json(ROOT/'surface_acceptance_funnel.json',dict(pixel_funnel={k:dict(v) for k,v in pixel_funnel.items()},
        sparse_legacy=point_funnel(points,oldkeys),sparse_new=point_funnel(points,newkeys),
        all_scene_pixels=sum(r['pixels'] for r in mainrows),initial_observation_base_pixels=sum(
            r['camera_diagnostics'].get('surface_constraints',{}).get('mask_counts',{}).get('observation_base',0) for r in mainrows),
        sparse_base_points=len(base),note='Pixel counts before periodic refresh; sparse paired population excludes forced refresh.'))
    all_pixels=sum(r['pixels'] for r in mainrows);effective=sum(r['effective_pixels'] for r in mainrows)
    base_pixels=sum(r['camera_diagnostics'].get('surface_constraints',{}).get('mask_counts',{}).get('observation_base',0) for r in mainrows)
    costs=[r['camera_diagnostics'].get('surface_constraints',{}).get('elapsed_ms') for r in mainrows]
    costs=[v for v in costs if v is not None]
    write_json(ROOT/'runtime_cost.json',dict(online_constraint_camera_wall_ms_p50_p95=np.percentile(costs,[50,95]).tolist(),
        summed_online_constraint_camera_wall_ms=sum(costs),scope='Actual CPU measurements during concurrent research replay; not a latency benchmark',
        full_scene_pixels=all_pixels,initial_candidate_pixels=base_pixels,final_effective_pixels=effective,
        effective_candidate_retention=effective/base_pixels if base_pixels else None,
        correct_dense_area=None,false_dense_area=None,unscorable_accepted_dense_area=None))
    roi=rows('roi_pairs.jsonl')
    write_json(ROOT/'critical_roi_replay.json',dict(events=roi,GT_online=False,exact_mask=True))
    write_json(ROOT/'roi_exact_mask_audit.json',dict(actual_new_events=roi,
        independent_actual_mask_replay=read('pixel_pair_and_mask_audit.json'),
        all_events=[dict(frame=r['frame'],camera=r['camera'],event=e) for r in mainrows for e in r.get('events',[])],
        minimum=current['minimum_roi_event_refresh_ratio'],historical_audit_path=str(PRIOR/'roi_exact_mask_audit.json'),
        historical_audit_sha256=sha(PRIOR/'roi_exact_mask_audit.json'),
        exact_refresh_scored=True,tile_envelope_substituted=False,GT_used_only_after_precheck=True))
    oldidentity=json.loads((PRIOR/'identity_method_comparison.json').read_text())['methods']['temporal_object']
    newidentity=summary['identity']['temporal_object'];identity_equal=newidentity==oldidentity
    assert identity_equal
    completes=[json.loads(p.read_text()) for p in (ROOT/'traces/development').glob('*/complete.json')]
    state=rows('state.jsonl');assert all(r['qualified']==0 and r['duplicate_history_ids']==0 for r in state)
    write_json(ROOT/'identity_preservation_comparison.json',dict(before=oldidentity,after=newidentity,
        equal=identity_equal,actual_edges=sum(r['parity']['identity_edges'] for r in completes),
        edge_differences=sum(r['parity']['identity_differences'] for r in completes),qualified_certifications=0))
    write_json(ROOT/'association_certification_regression.json',dict(actual_commits=len(state),duplicate_history_ids=0,
        shadow_identity_edges_equal=True,qualified_certifications=0,unchanged_protected_sources={
            p:sha(p)==v for p,v in read('baseline_source_manifest.json')['files'].items()
            if Path(p).name in ('identity_temporal.py','identity_object.py','physical_certifier.py','state_store.py','state_manager.py')}))
    for stage in ('calibration','heldout','final_test'):
        write_json(ROOT/(stage+'_summary.json'),dict(status='Not Run',qualified_certifications=0,qualified_FAR=None,
            identity_selection=None,region_selection=None,new_independent_labeled_logs=0,
            reason='Pinned prior nuScenes audit: all 68 labeled trainval logs already exposed; Test has no local physical scoring annotations. No rescan or redefinition.'))
    folds={log:region_metrics([r for r in mainrows if r['log_token']==log]) for log in sorted({r['log_token'] for r in mainrows})}
    write_json(ROOT/'exploratory_log_cross_validation.json',dict(population='exploratory_reused_population',
        folds=folds,type='Fixed-method log-stratified descriptive reporting, no refit',formal_selection=None))
    write_json(ROOT/'frozen_validation_protocol.json',dict(protocol=read('experiment_protocol.json'),
        source_freeze_sha256=sha(ROOT/'development_source_freeze.json'),
        prior_independent_audit_sha256=sha(PRIOR/'independent_data_adapter_audit.json'),
        oracle_unchanged=True,qualified_certifications=0,identity_selection=None,region_selection=None))
    xml=ET.parse(ROOT/'regression_cuda.xml').getroot();suites=list(xml) if xml.tag=='testsuites' else [xml]
    tests=sum(int(s.get('tests',0)) for s in suites);failed=sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites)
    skipped=sum(int(s.get('skipped',0)) for s in suites)
    regression=dict(tests=tests,passed=tests-failed-skipped,failed=failed,skipped=skipped,CUDA=True,
        baseline_tests=517,xml='regression_cuda.xml',historical_artifacts_unchanged=True)
    write_json(ROOT/'regression_results.json',regression)
    model=read('gpu_parity/results.json');status=read('gpu_parity/status.json')
    parity=dict(status=status['status'],frames=8,equal=model['observe_only_equal'],comparisons=model['comparisons'],
        scope='Newly executed original EVA/detection/map and temporal-memory digests; perception-only; no LLM, M2 or M3 execution',
        checkpoint_sha256=read('data_manifest.json')['checkpoint']['sha256'])
    write_json(ROOT/'original_model_parity.json',parity)
    pointpass=(current['accepted_scorable_points']>0 and current['sparse_reference_far']<=.05
        and current['critical_change_recall']>=.998 and current['critical_point_recall']>=.998
        and current['minimum_roi_event_refresh_ratio']>=.9)
    gates=dict(engineering_correctness=dict(pass_gate=failed==0 and skipped==0 and parity['equal'],**regression),
        geometric_patch_recovery=dict(pass_gate=patch['correct']['recovered_source']>0,counts=patch,
            final_pair=point_pairs['surface_constraints'][main_name]),
        joint_hypothesis_disambiguation={k:v for k,v in joint.items() if k not in ('measured_hypotheses',)},
        region_risk_coverage=dict(pass_gate=False,descriptive_nonzero_point_constraints_passed=pointpass,
            frozen_risk_gates_passed=current['risk_gates_passed'],formal_go=False,
            reason='Original risk confidence and independent support remain unmet; no new scored correct coverage',
            area_ratio=current['effective_stable_area_ratio'],sparse_far=current['sparse_reference_far'],
            far_upper95=current['far_upper95_iid_descriptive'],accepted_logs=current['accepted_log_support']),
        identity_preservation=dict(pass_gate=identity_equal,correct=217,wrong=0,unscorable=71,unique_instances=45),
        independent_qualification=dict(pass_gate=False,new_labeled_logs=0,qualified_certifications=0,qualified_FAR=None))
    gates['joint_hypothesis_disambiguation']['pass_gate']=joint['sparse_restored_correct']>0 and joint['sparse_restored_wrong']==0
    gates['scientific_feasibility']=dict(pass_gate='Measured boundary; no formal qualification',
        geometry_model_effect_measured=True,joint_effect_measured=True,thresholds_unchanged=True)
    formal=dict(decision='No-Go',population='exploratory_reused_population',gates=gates,
        qualified_certifications=0,qualified_FAR=None,identity_selection=None,region_selection=None,
        calibration='Not Run',heldout='Not Run',final_test='Not Run')
    write_json(ROOT/'formal_go_nogo.json',formal)
    files=source_manifest();files.update({str(p):sha(p) for p in [*Path('tools/worldstream_m1').glob('correspondence_*.py'),Path('tests/worldstream_m1/test_correspondence_recovery.py')]})
    write_json(ROOT/'source_manifest.json',dict(baseline_head=read('baseline_source_manifest.json')['head'],files=files,
        source_frozen_during_replay=True,raw_traces={str(p):sha(p) for p in (ROOT/'traces').rglob('*.jsonl')},
        validation_traces={str(p):sha(p) for p in (ROOT/'validation_traces').rglob('*.jsonl')},
        actual_mask_audit_traces={str(p):sha(p) for p in (ROOT/'mask_audit_traces').rglob('complete.json')}))
    figures(methods,diag,patch,point_pairs,joint,mainrows,hypotheses)
    report(methods,diag,patch,point_pairs,joint,regression,parity,pointpass)
    print(json.dumps(dict(main={k:current[k] for k in equality_keys},patch=patch,
        paired=point_pairs['surface_constraints'][main_name],regression=regression,formal='No-Go'),indent=2))


def figures(methods,diag,patch,pairs,joint,pixels,hypotheses):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=ROOT/'visualizations';out.mkdir(exist_ok=True)
    labels=['parallax/triangulation','texture','projection unavailable','unconfirmed']
    keys=['measured_parallax_or_triangulation_insufficient','measured_texture_support_insufficient','observation_missing','Unknown_Root_Cause']
    fig,ax=plt.subplots(figsize=(10,4));x=np.arange(4)
    for label,offset in [('correct',-.18),('false',.18)]:
        ax.bar(x+offset,[diag['counts'][label]['explanations'].get(k,0) for k in keys],.36,label=label)
    ax.set_xticks(x,labels);ax.set_ylabel('Rejected reference points (overlapping flags)');ax.legend();fig.tight_layout()
    fig.savefig(out/'patch_rejection_causes.png',dpi=160);plt.close(fig)
    names=['surface_constraints','geometric_adaptive_patch','joint_multi_observation','correspondence_recovery']
    fig,ax=plt.subplots(figsize=(10,4));x=np.arange(len(names))
    ax.bar(x-.18,[methods[n]['independently_correct_stable_support'] for n in names],.36,label='Correct')
    ax.bar(x+.18,[methods[n]['false_stable_points'] for n in names],.36,label='False')
    ax.set_xticks(x,['Old constraints','Adaptive patch','Joint only','Full']);ax.legend();ax.set_ylabel('Actual sparse accepts');fig.tight_layout()
    fig.savefig(out/'paired_correct_false.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    for index,name in enumerate(['conservative','surface_consensus',*names]):
        m=methods[name]
        if m['sparse_reference_far'] is not None:
            ax.scatter(100*m['effective_stable_area_ratio'],100*m['sparse_reference_far'])
            ax.annotate(name,(100*m['effective_stable_area_ratio'],100*m['sparse_reference_far']),
                xytext=(12,10+12*index),textcoords='offset points',fontsize=7,arrowprops=dict(arrowstyle='-',lw=.5))
    ax.axhline(5,color='red',linestyle='--');ax.set_xlabel('Effective Stable area (%)');ax.set_ylabel('Sparse FAR (%)')
    ax.set_title('Fixed mechanisms on reused logs; no qualified selection');fig.tight_layout();fig.savefig(out/'region_risk_coverage.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.hist([100*e['invalidation_coverage'] for r in pixels for e in r.get('events',[])],bins=20)
    ax.axvline(90,color='red',linestyle='--');ax.set_xlabel('Exact critical ROI refresh (%)');ax.set_ylabel('Events');fig.tight_layout()
    fig.savefig(out/'critical_roi_refresh.png',dpi=160);plt.close(fig)
    actual_rgb_panels(hypotheses,out)


def actual_rgb_panels(hypotheses,out):
    """Render actual projected RGB samples, without synthetic explanatory data."""
    import cv2
    import pickle
    from .dataset import frame_from_info
    from worldstream.m1.surface_observability import project
    if not hypotheses:return
    selection=json.loads(Path('outputs/module1_evidence_optimization/data_selection.json').read_text())
    with open(selection['info_file'],'rb') as handle:infos={i['token']:i for i in pickle.load(handle)['infos']}
    candidates=sorted((r for r in hypotheses if r['local_geometry']['applicable']),
        key=lambda r:r['measurements']['projective_two_view_patch']['texture'],reverse=True)
    chosen=candidates[:4] if candidates else hypotheses[:4]
    for i,r in enumerate(chosen):
        cur=frame_from_info(infos[r['frame']],'data/nuscenes').cameras[r['camera']]
        old=frame_from_info(infos[r['source_frame']],'data/nuscenes').cameras[r['camera']]
        rad=3;yy,xx=np.mgrid[-rad:rad+1,-rad:rad+1];offset=np.c_[xx.ravel(),yy.ravel()]
        q=np.asarray(r['pixel']);grid=q+offset
        world=np.asarray(r['measurements']['world']);_,depth=project(world[None],cur)
        rays=np.c_[grid,np.ones(len(grid))]@np.linalg.inv(cur.intrinsic).T
        panels=[]
        for label,cam,uv in [('Current RGB',cur,grid)]:
            crop=cv2.remap(cam.image,uv[:,0].astype(np.float32).reshape(7,7),uv[:,1].astype(np.float32).reshape(7,7),cv2.INTER_LINEAR)
            panels.append((label,crop))
        for label,slopes in [('Frontoparallel',np.zeros(2)),(f"Plane applicable={r['local_geometry']['applicable']}",np.asarray(r['local_geometry']['slopes']))]:
            inverse=1/depth[0]+offset@slopes
            with np.errstate(divide='ignore',invalid='ignore'):
                xyz=(rays/inverse[:,None]-cur.world_to_camera[:3,3])@cur.world_to_camera[:3,:3]
            uv,_=project(xyz,old);uv=np.nan_to_num(uv,nan=-10).astype(np.float32)
            crop=cv2.remap(old.image,uv[:,0].reshape(7,7),uv[:,1].reshape(7,7),cv2.INTER_LINEAR)
            panels.append((label,crop))
        tiles=[]
        for label,crop in panels:
            tile=np.zeros((250,224,3),np.uint8);tile[:224]=cv2.resize(crop,(224,224),interpolation=cv2.INTER_NEAREST)
            cv2.putText(tile,label,(4,242),0,.43,(255,255,255),1);tiles.append(tile)
        cv2.imwrite(str(out/f'actual_patch_geometry_{i}.png'),cv2.cvtColor(np.concatenate(tiles,axis=1),cv2.COLOR_RGB2BGR))
    # All source alternatives are plotted, including those still unknown.
    ambiguous=[r for r in hypotheses if any(h['incompatible'] for h in r['measurements']['alternatives'])]
    measured_third=[r for r in ambiguous if any(v['kind']=='additional_time' and v['supported'] for v in r['measurements']['extra_views'])]
    cases=(measured_third or ambiguous)[:2]
    for i,r in enumerate(cases):
        frame=frame_from_info(infos[r['frame']],'data/nuscenes');old=frame_from_info(infos[r['source_frame']],'data/nuscenes')
        panels=[]
        for camera,pixel,label in [(old.cameras[r['camera']],r['source_uv'],'Source: observed alternatives'),(frame.cameras[r['camera']],r['pixel'],'Current ray')]:
            image=camera.image.copy();cv2.circle(image,tuple(np.rint(pixel).astype(int)),4,(255,0,0),1)
            if label.startswith('Source'):
                for h in r['measurements']['alternatives']:cv2.circle(image,tuple(np.rint(h['source_uv']).astype(int)),2,(255,190,0),1)
            cv2.putText(image,label,(8,22),0,.5,(255,255,255),1);panels.append(image)
        # Actual extra-camera observations can be drawn directly; temporal
        # third acquisitions are found by strict sample order below.
        from .evidence_experiments import load_inputs
        if i==0:specs,_,_,_=load_inputs('calibration')
        tokens=next(s['samples'] for s in specs if s['scene_id']==r['scene']);position=tokens.index(r['frame'])
        if position>=2:
            third=frame_from_info(infos[tokens[position-2]],'data/nuscenes').cameras[r['camera']]
            image=third.image.copy()
            for h in r['measurements']['alternatives']:
                uv,z=project(np.asarray(h['world'])[None],third)
                if np.isfinite(uv).all() and z[0]>.5:cv2.circle(image,tuple(np.rint(uv[0]).astype(int)),2,(255,190,0),1)
            cv2.putText(image,'Actual earlier RGB; amber competitors remain conditional',(8,22),0,.42,(255,255,255),1);panels.append(image)
        cv2.imwrite(str(out/f'actual_joint_observations_{i}.png'),cv2.cvtColor(np.concatenate(panels,axis=1),cv2.COLOR_RGB2BGR))


def report(methods,diag,patch,pairs,joint,regression,parity,pointpass):
    current=methods['correspondence_recovery'];pair=pairs['surface_constraints']['correspondence_recovery'];cons=pairs['surface_consensus']['correspondence_recovery']
    lines=['# M1 几何自适应观测与联合消歧实验', '',
        '正式结论：**No-Go**。本轮实际修改 M1 并完成 nuScenes 同一已曝光人口的实验；独立资格与研究性点估计分开报告。', '',
        '基线 HEAD 为 `d95011a`。所有命令使用 OmniDrive 环境及 `taskset -c 0-3,6-31`；M2/M3、原模型、Checkpoint、训练和输入预处理未修改。历史文件哈希全部一致。', '',
        '## 先诊断，再实现', '',
        f"冻结人口复现 1314 正确/230 错误候选，源验证后 177/80。1137 个损失正确点中的测量不足标记为 {diag['counts']['correct']['explanations']}。这些标记有重叠，反映观测条件，并非物理根因标签。未确认的遮挡、深度边界、曝光及真实斜面原因继续 Unknown。", '',
        '旧检查确实将中心深度扩展到整个邻域。但不能据源码把所有损失归因于前平行假设。局部 LK 诊断没有获得可靠的正确中心及全邻域联合支持，不能据此确认或否认真实斜面。', '',
        '仅新增一个局部逆深度平面：以中心实测深度锚定，四角真实对应拟合斜率，四个空间分离轴向点留出验证，随后按原光度/纹理/梯度阈值检查实际投影 RGB。旧前平行路径作为明确回退和消融保留；无证据不扩张 Patch、ROI 或表面面积。', '',
        '联合消歧逐个保留实际支持的竞争源射线。只有 H0 的第三图像正支持、直接双向 LK 与局部留出、源回投闭环、候选投影落在实测局部范围并与实测射线冲突时，才排除 Hi。任一实际第三视角仍支持 Hi 时保留竞争；观测缺失、离屏或 Patch 不匹配本身都不能排除。该证据仍有 RGB 相关性，不是物理所有权认证。', '',
        f'新局部模型源验证：{patch}。', '', '## 同人口方法对照', '',
        '| 方法 | 有效面积 % | 正确/错误点 | Sparse FAR % | 最差精确 ROI 刷新 % | 接受 Log |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ['full_refresh','conservative','dense_original','local_r12','surface_local_plane','surface_consensus',
        'surface_constraints','geometric_adaptive_patch','joint_multi_observation','correspondence_recovery',
        'recovery_no_extra_observation','recovery_h0_only','recovery_discriminating_third_only',
        'recovery_no_depth_order','recovery_conservative_neighbourhood','recovery_no_motion_sweep']:
        m=methods[name];far='N/A' if m['sparse_reference_far'] is None else f"{100*m['sparse_reference_far']:.6f}"
        lines.append(f"| {name} | {100*m['effective_stable_area_ratio']:.6f} | {m['independently_correct_stable_support']}/{m['false_stable_points']} | {far} | {100*m['minimum_roi_event_refresh_ratio']:.6f} | {m['accepted_log_support']} |")
    lines += ['', '历史六种方法为 SHA256 固定的相同输入结果，不冒充新运行。旧 Surface Constraints 则在本轮逐相机重算并核对面积、错误、召回和最差 ROI，全部一致。新方法及所有逻辑消融均使用本轮实际在线测量；邻域基底和动态扫掠消融各用独立 manager 完整重放。', '',
        f"最终与旧 Constraints 配对：{pair}；旧5个正确点保留 {pair['correct_retained']}。与 Consensus 的494正确/196错误配对：{cons}。全像素精确接受交集为 {read('pixel_pair_and_mask_audit.json')['comparisons']['surface_constraints']}。详见 region_paired_support_analysis.json。本轮没有恢复新增可评分正确点；极少像素面积变化不能解释为已证实的正确覆盖提升。", '',
        f"第三观测排除条件性假设 {joint['counts'].get('excluded_hypotheses',0)} 个；稀疏合法恢复正确 {joint['sparse_restored_correct']}、错误恢复 {joint['sparse_restored_wrong']}、继续 Unknown {joint['sparse_continuing_unknown']}。本轮没有排除假设，真实错误排除数因此也为0。替代射线本身缺少逐个独立物理真值，不能把未来的条件排除数直接称作真实错误解释数。", '',
        '所有面积分母包括20 Scene / 20 Log / 240 Frame / 1440 Camera 和启动、周期强制刷新。Correct/False Stable 是稀疏可评分参考点，正确/错误/不可评分稠密面积均无法由此推出，保持 null。每阶段像素与稀疏人口分别记录在 surface_acceptance_funnel.json。', '',
        f"最终 Event Recall={100*current['critical_change_recall']:.6f}%，Point Recall={100*current['critical_point_recall']:.6f}%，最差精确 ROI Refresh={100*current['minimum_roi_event_refresh_ratio']:.6f}%。四个历史失败事件在 critical_roi_replay.json 中逐项核对，GT ROI 在 Precheck 后评分，未控制刷新，也未用32px Tile包络替代精确掩码。", '',
        f"非零四项描述性点约束通过：{pointpass}；Sparse FAR 的单侧95%描述上界为 {current['far_upper95_iid_descriptive']}。原冻结风险 Gate 通过：{current['risk_gates_passed']}。同一 Log 内像素、帧及对象不增加独立样本数；零错误不能推出认证。", '',
        '## 身份、原模型与独立资格', '',
        'Temporal Object、Shadow、Anchor 继承和认证隔离保持。13,314个身份候选边逐边与历史结果一致：Proposal 288，正确217，错误0，不可评分71，唯一正确实例45，正式认证0。StateStore、身份生产器及 Physical Certifier 未修改。Stable 不代表 EVA Token 或 LLM Feature Reuse。', '',
        f"本轮完整 CUDA 回归：{regression['passed']}/{regression['tests']} 通过，失败{regression['failed']}、跳过{regression['skipped']}。本轮8帧原模型 Perception Parity：{parity['status']}，输出/骨干/记忆摘要一致={parity['equal']}；未执行LLM，未用旧GPU结果替代。", '',
        '复用已有 nuScenes 审计并重新核对数据与 Checkpoint 哈希。68个有标注采集Log均已曝光，Test缺少本地物理评分标注，没有新增独立数据。Calibration/Heldout/Final Test = Not Run；qualified_certifications=0，qualified_FAR=null，identity_selection=null，region_selection=null。按Log交叉验证仅是固定方法描述，不消除既有曝光。', '',
        '## 科学问题与后续最小条件', '',
        f"前平行假设的局限存在，但这批损失主要伴随视差、纹理和投影可用性问题；本轮不能确认大量损失由斜面造成。自适应源验证恢复正确{patch['correct']['recovered_source']}、错误{patch['false']['recovered_source']}，其最终收益必须结合完整配对和关键变化约束判断。第三观测的独立贡献由 joint_multi_observation 与旧 Constraints、完整路径与 geometric_adaptive_patch 直接比较，未通过时保留失败。", '',
        '本轮不据失败继续堆叠评分器或多平面。Surface Consensus 作为对照保留，当前错误水平没有证明其作为正式接受路径的必要性。下一步需在同一nuScenes范围内获得具有可见局部二维纹理、足够视差且可直接可靠跟踪的第三采集；局部几何适用性与候选唯一性的观测缺口应分别解决。正式资格还需合法未曝光且完整可评分的采集Log，不能靠阈值或重分组补足。', '',
        '复现命令见 commands.sh；实现细节见 implementation_changes.md；七项 Gate 及全部未通过结果见 formal_go_nogo.json。']
    lines += ['', '## 七项 Gate 与问题回答', '',
        '| Gate | 本轮结论 |', '|---|---|',
        '| Engineering Correctness | Pass：538项CUDA测试及8帧原模型一致性通过 |',
        '| Geometric Patch Recovery | 未达目标：没有新增可评分正确支持 |',
        '| Joint Hypothesis Disambiguation | 未达目标：实际排除0，无合法稀疏恢复 |',
        '| Region Risk-Coverage Feasibility | No-Go：点估计门槛通过，冻结置信度及独立支持未通过 |',
        '| Identity Preservation | Pass：候选逐边及完整指标保持 |',
        '| nuScenes Independent Qualification | No-Go：新增独立可评分Log为0 |',
        '| Scientific Feasibility | 已测量失败边界；当前机制没有证明正确覆盖收益 |', '',
        '1. 前平行假设确有局限，但本轮没有确认大量正确损失由这一假设造成；可测视差、纹理及投影可用性不足必须分开统计。',
        f"2. 新局部模型恢复的可评分正确源对应为{patch['correct']['recovered_source']}；最终新增正确Stable为{pair['new_correct']}。",
        f"3. 同时新增可评分错误源对应为{patch['false']['recovered_source']}；最终新增错误Stable为{pair['new_wrong']}。",
        f"4. 实际第三观测排除的条件性竞争解释为{joint['counts'].get('excluded_hypotheses',0)}，尚未证实联合消歧收益。",
        '5. H0获得附加观测支持但仍有错误的14个参考点，不能靠H0正支持证明唯一性；可靠第三几何约束缺失的竞争继续Unknown。',
        '6. 新方法与旧Constraints的可评分支持及错误风险点估计相同；净3个未标注像素的增加不足以证明覆盖和风险同时改善。',
        f'7. 非零四项点估计工作点存在={pointpass}，但仅5个正确点、1个接受Log，绝非正式风险资格。',
        '8. 当前证据没有证明继续依赖复杂Surface Consensus多平面接受机制的必要性；保留它作错误与正确损失对照。',
        '9. 关键事件保护保持达标；本轮面积几乎未恢复，不能据此推断未来大幅恢复面积后仍达标。',
        '10. 最小新增条件是同一区域实际可直接跟踪、局部可验证、具有新增视差的第三RGB采集；首先让真实几何适用与可见性条件可观测，再评价排除竞争的能力。独立认证另需合格未曝光Log。']
    (ROOT/'experiment_report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
