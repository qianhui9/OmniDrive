"""Reproducible offline analysis of the frozen nuScenes-only replay."""
from collections import Counter,defaultdict
from pathlib import Path
import json
import hashlib
import math
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from .io_utils import write_json
from .quality_coverage_metrics import identity_metrics,region_metrics,upper95
from .observability_experiments import source_manifest

ROOT=Path('outputs/module1_physical_observability_optimization')
PRIOR=Path('outputs/module1_quality_coverage_refinement')


def read(name):return json.loads((ROOT/name).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def rows(name):
    result=[]
    for p in sorted((ROOT/'traces/development').glob('*/complete.json')):
        q=p.parent/name
        if q.exists():result.extend(map(json.loads,q.read_text().splitlines()))
    return result


def point_counts(items):
    return dict(correct=sum(not r['false'] for r in items),false=sum(r['false'] for r in items),total=len(items))


def paired_analysis(points,diagnostic):
    bykey={(r['scene'],r['frame'],r['camera'],r['reference_index']):r for r in points}
    main='surface_constraints';comparisons={}
    for population in ('surface_consensus','surface_local_plane'):
        old=[r for r in diagnostic['records'] if population in r['populations']]
        retained=[];lost=[]
        for r in old:
            p=bykey.get((r['scene'],r['frame'],r['camera'],r['reference_index']))
            (retained if p is not None and p['accepted'][main] else lost).append(r)
        comparisons[population]=dict(old=point_counts(old),retained=point_counts(retained),rejected=point_counts(lost),
            new_accepts_outside_old=point_counts([r for r in points if r['accepted'][main] and (r['scene'],r['frame'],r['camera'],r['reference_index']) not in {(v['scene'],v['frame'],v['camera'],v['reference_index']) for v in old}]))
    # Marginal effect of measured ambiguity with the identical other checks.
    extra=[r for r in points if r['accepted']['constraints_no_ambiguity']]
    ambiguity_removed=[r for r in extra if not r['accepted'][main]]
    sequence=[];live=[r for r in points if r['accepted']['constraints_observation_base']]
    for key,label in [('source_consistent','actual_source_patch'),('extra','actual_additional_acquisition'),('ambiguous','observed_alternatives'),('order_conflict','conditional_depth_order')]:
        if key=='extra':keep=lambda r:r['measured']['temporal_supported'] or r['measured']['camera_supported']
        elif key in ('ambiguous','order_conflict'):keep=lambda r,k=key:not r['measured'][k]
        else:keep=lambda r,k=key:r['measured'][k]
        after=[r for r in live if keep(r)];removed=[r for r in live if not keep(r)]
        sequence.append(dict(gate=label,before=point_counts(live),after=point_counts(after),incrementally_rejected=point_counts(removed)));live=after
    assert point_counts(live)==point_counts([r for r in points if r['accepted'][main]])
    result=dict(population='exploratory_reused_population',comparisons=comparisons,
        ambiguity_marginal_rejection=point_counts(ambiguity_removed),sequential_scorable_funnel=sequence,
        root_cause='Physical errors confirmed offline; physical mechanism/root cause remains Unknown unless independently established',
        missing_measurement_note='A point outside the observation base is not tested by later gates; not treated as a measured contradiction')
    write_json(ROOT/'region_paired_error_correct_analysis.json',result)
    return result


def population_analysis(identities):
    original=set()
    for p in (PRIOR/'traces/development').glob('*/identity.jsonl'):
        for r in map(json.loads,p.read_text().splitlines()):
            if r['method']=='hungarian_geometry':original.add((r['scene'],r['frame'],r['anchor_id'],r['prediction']))
    assert len(original)==5798
    def key(r):return r['scene'],r['frame'],r['anchor_id'],r['prediction']
    assert original<={key(r) for r in identities}
    previous={}
    for spec in read('data_selection.json')['development']:
        previous.update({t:spec['samples'][i-1] if i else None for i,t in enumerate(spec['samples'])})
    definitions={
        'all_measured_legal_edges':'Every measured legal identity edge, including recovery/competitors; original denominator remains 5798',
        'original_geometry_edges':'Membership in original online Hungarian geometric edge population',
        'has_supported_crop':'At least one actual online view has producer status supported or legacy measured_identity_population_candidate reason',
        'multiple_supported_temporal_camera_crops':'At least two camera crops support their source/current observation; no foreground ownership or simultaneous triangulation claim',
        'measured_object_surface_and_background':'At least one online object-surface/ring view has producer status supported',
        'background_depth_separable':'At least one measured object/ring depth-gap array is finite, nonempty and all >= existing 1m foreground-depth gap',
        'unresolved_identity_competition':'Existing online unresolved_measured_identity_competition rejection',
        'resolved_identity_competition':'Existing online resolved_competition output',
        'short_gap_source':'Source frame differs from immediately previous committed frame; no GT occlusion claim',
        'adjacent_source':'Source is immediately previous committed frame',
        'source_age_at_most_half_second':'Source age <= 0.5s (descriptive nominal nuScenes keyframe bin, not a decision threshold)',
        'source_age_over_half_second':'0.5s < source age <= existing 1s lifetime',
        'predicted_displacement_at_most_05m':'Predicted center displacement <= prior selective producer 0.5m domain',
        'predicted_displacement_over_05m':'Predicted center displacement > 0.5m; measured temporal/object path domain'}
    groups={n:[] for n in definitions};classes=defaultdict(list);distances=defaultdict(list)
    for r in identities:
        views=(r.get('measurements') or {}).get('views',[])
        supported=[v for v in views if v.get('status')=='supported' or v.get('reason')=='measured_identity_population_candidate']
        separated=False
        for v in views:
            gap=v.get('background_gap_m')
            if gap is not None:
                a=np.asarray(gap,float)
                separated|=bool(a.size and np.isfinite(a).all() and (a>=1.).all())
        flags=dict(all_measured_legal_edges=True,original_geometry_edges=key(r) in original,
            has_supported_crop=bool(supported),multiple_supported_temporal_camera_crops=len(supported)>=2,
            measured_object_surface_and_background=any(v.get('observation_model')=='translation_compensated_box_surface_ring' for v in supported),
            background_depth_separable=separated,unresolved_identity_competition='unresolved_measured_identity_competition' in r['rejection_reasons'],
            resolved_identity_competition=r['resolved_competition'],short_gap_source=r['source_frame']!=previous[r['frame']],
            adjacent_source=r['source_frame']==previous[r['frame']],source_age_at_most_half_second=r['online_source_age_s']<=.5,
            source_age_over_half_second=.5<r['online_source_age_s']<=1.,
            predicted_displacement_at_most_05m=r['online_displacement_m']<=.5,predicted_displacement_over_05m=r['online_displacement_m']>.5)
        for n,yes in flags.items():
            if yes:groups[n].append(r)
        classes[r['online_class']].append(r)
        distances['under_20m' if r['online_distance_m']<20 else '20_to_40m' if r['online_distance_m']<40 else 'at_least_40m'].append(r)
    def stats(rr):
        result=identity_metrics(rr,5798)
        result.update(original_geometry_population_members=sum(key(r) in original for r in rr),
            original_geometry_population_fraction=sum(key(r) in original for r in rr)/5798,
            all_measured_edges_fraction=len(rr)/len(identities),
            within_population_proposal_coverage=sum(r['candidate'] for r in rr)/len(rr) if rr else None)
        return result
    result=dict(population='exploratory_reused_population',online_definitions=definitions,
        definitions_used_no_labels=True,nonexclusive_subsets=True,original_denominator=5798,expanded_edges=len(identities),
        populations={n:stats(rr) for n,rr in groups.items()},predicted_classes={n:stats(rr) for n,rr in classes.items()},
        predicted_ego_distance_bins={n:stats(rr) for n,rr in distances.items()},
        scope='Identity continuity only; crop/box geometry does not certify physical foreground ownership',
        limitation='No newly invented identity producer; conditional subsets preserve all unscorable accepts and log grouping')
    write_json(ROOT/'identity_population_analysis.json',result)
    write_json(ROOT/'identity_population_ablation.json',dict(population=result['population'],type='online-defined population restrictions, not new producer ablations',
        definitions=definitions,subsets=result['populations'],formal_population_selection=None,
        threshold_tuning=False,unscorable_accepts_retained=True))
    return result


def figures(methods,populations,pixels,diagnostic,paired):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=ROOT/'visualizations';out.mkdir(exist_ok=True)
    names=['conservative','surface_local_plane','surface_consensus','constraints_observation_base','constraints_two_view_only','constraints_no_ambiguity','surface_constraints']
    fig,ax=plt.subplots(figsize=(9,5))
    for n in names:
        v=methods[n]
        if v['sparse_reference_far'] is None:continue
        ax.scatter(v['effective_stable_area_ratio']*100,v['sparse_reference_far']*100)
        label=n+' (5 points, 1 exposed log)' if n=='surface_constraints' else n
        align='right' if n=='constraints_observation_base' else 'left'
        ax.annotate(label,(v['effective_stable_area_ratio']*100,v['sparse_reference_far']*100),xytext=(-4 if align=='right' else 4,4),ha=align,textcoords='offset points',fontsize=8)
    ax.axhline(5,color='red',ls='--');ax.set(xscale='log',xlabel='Effective observation Stable area (%)',ylabel='Annotated sparse FAR point estimate (%)',title='Exposed nuScenes Development; no independently qualified method')
    fig.tight_layout();fig.savefig(out/'region_risk_coverage.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for n in ('surface_consensus','conservative','surface_constraints'):
        values=sorted(methods[n]['roi_event_refresh_ratio']);axes[0].plot(np.arange(len(values)),values,label=n)
    axes[0].axhline(.9,color='red',ls='--');axes[0].set(xlabel='Sorted critical event ROI',ylabel='Exact refresh fraction');axes[0].legend(fontsize=7)
    p=paired['comparisons']['surface_consensus'];axes[1].bar(['Old correct','Retained correct','Old false','Retained false'],[p['old']['correct'],p['retained']['correct'],p['old']['false'],p['retained']['false']],color=['green','green','red','red']);axes[1].tick_params(axis='x',rotation=20);axes[1].set_ylabel('Offline scored correspondences')
    fig.tight_layout();fig.savefig(out/'roi_and_paired_support.png',dpi=180);plt.close(fig)
    names=['all_measured_legal_edges','has_supported_crop','multiple_supported_temporal_camera_crops','measured_object_surface_and_background','background_depth_separable','short_gap_source']
    fig,(ax,unique_ax)=plt.subplots(1,2,figsize=(13,5),gridspec_kw={'width_ratios':[3,1]});y=np.arange(len(names));good=np.array([populations['populations'][n]['independently_correct_proposals'] for n in names]);unknown=np.array([populations['populations'][n]['unscorable_proposals'] for n in names])
    ax.barh(y,good,color='seagreen',label='Scorable correct');ax.barh(y,unknown,left=good,color='gray',label='Unscorable');ax.set_yticks(y,labels=names,fontsize=8);ax.set_xlabel('Accepted proposals (nonexclusive online populations)');ax.legend()
    unique_ax.barh(y,[populations['populations'][n]['unique_physical_identity_recovery'] for n in names],color='steelblue');unique_ax.set_yticks(y,labels=['']*len(names));unique_ax.set_xlabel('Unique correct instances')
    fig.tight_layout();fig.savefig(out/'identity_population_recovery.png',dpi=180);plt.close(fig)
    manifest=json.loads((PRIOR/'data_manifest.json').read_text());frame_files={r['sample_token']:r['source_files'] for r in manifest['frames']}
    from .observability_diagnostics import panel
    main=[r for r in diagnostic['records'] if 'surface_consensus' in r['populations']]
    for label,rr in [('correct',[r for r in main if not r['false']][:4]),('false',[r for r in main if r['false']][:4])]:
        for i,r in enumerate(rr):
            cameras=[]
            from types import SimpleNamespace
            for t in (r['source_frame'],r['frame']):
                path=next(p for p in frame_files[t] if '/'+r['camera']+'/' in p)
                image=cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB);h,w=image.shape[:2];scale=min(1,640/w)
                cameras.append(SimpleNamespace(image=cv2.resize(image,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)))
            panel(*cameras,r,out/'true_false_controls'/f'{label}_{i}.png')


def main():
    freeze=read('development_source_freeze.json');assert freeze['files']==source_manifest(),'Source changed after frozen replay'
    diagnostic=read('surface_ambiguity_analysis.json');points=rows('point_decisions.jsonl');pixels=rows('pixels.jsonl');identities=rows('identity.jsonl')
    summary=read('development_summary.json');assert len(list((ROOT/'traces/development').glob('*/complete.json')))==20
    previous_region=json.loads((PRIOR/'region_method_comparison.json').read_text())['methods']
    previous_identity=json.loads((PRIOR/'identity_method_comparison.json').read_text())['methods']
    baseline_names=('full_refresh','conservative','dense_original','local_r12','surface_visibility','surface_local_plane','surface_consensus')
    methods={n:previous_region[n] for n in baseline_names};methods.update(summary['region'])
    base=[r for r in points if r['accepted']['constraints_observation_base']];good=[r for r in base if not r['false']]
    oracle=dict(role='offline_diagnostic_only',bound_scope='Known correct accepts within the measured observation-base candidate population',
        accepted_scorable_points=len(good),false_stable_points=0,source_candidate_points=len(base),
        rejected_correct_candidates_by_main=len(good)-summary['region']['surface_constraints']['independently_correct_stable_support'],
        effective_stable_area_ratio=None,dense_true_area_upper_bound=None,sparse_reference_far=0. if good else None,
        qualified_certifications=0,online_available=False,formal_selectable=False,
        limitation='Sparse labels cannot supply a dense surface visibility oracle; dense area/ROI scores are not invented')
    write_json(ROOT/'offline_oracle_upper_bound.json',oracle)
    methods['offline_sparse_candidate_oracle_upper_bound']=oracle
    provenance=dict(population='exploratory_reused_population',stage='Development',baseline_metrics_reuse='Archived exact same input population, source pinned; not newly independent tests',
        baseline_files={str(PRIOR/n):sha(PRIOR/n) for n in ('region_method_comparison.json','identity_method_comparison.json','data_manifest.json')},
        independent_reference='Unmodified Annotation-owned sparse LiDAR correspondence and event oracle; offline only',methods=methods)
    write_json(ROOT/'region_method_comparison.json',provenance)
    write_json(ROOT/'region_risk_coverage.json',dict(population=provenance['population'],methods=methods,formal_selection=None))
    write_json(ROOT/'region_ablation.json',dict(population=provenance['population'],methods=summary['region'],
        shared_actual_measurements=True,ablation_replay_check='independent_ablation_replay_check.json',
        fixed_predicate_ablations_not_threshold_grid=True))
    paired=paired_analysis(points,diagnostic)
    # Whole logs are held out for descriptive reporting only. No new fitting,
    # independent qualification or method selection is performed in these folds.
    mainrows=[r for r in pixels if r['method']=='surface_constraints'];logs=sorted({r['log_token'] for r in mainrows})
    folds={log:region_metrics([r for r in mainrows if r['log_token']==log]) for log in logs}
    write_json(ROOT/'exploratory_log_cross_validation.json',dict(population=provenance['population'],type='Leave-one-log reporting of a fixed method; no refit',
        held_log_folds=folds,new_independent_logs=0,formal_selection=None,zero_scorable_acceptance_folds=sum(v['accepted_scorable_points']==0 for v in folds.values()),
        interpretation='All held logs were exposed before this round; folds neither erase historical exposure nor multiply log units'))
    populations=population_analysis(identities)
    identity_names=('hungarian_geometry','selective_identity','temporal_identity','temporal_shadow','temporal_object','object_surface_only')
    identity_methods={n:previous_identity[n] for n in identity_names};assert identity_methods['temporal_object']==summary['identity']['temporal_object']
    write_json(ROOT/'identity_method_comparison.json',dict(population=provenance['population'],methods=identity_methods,
        new_actual_replay='temporal_object identical across all 13314 legal edges; no new identity producer',formal_selection=None))
    write_json(ROOT/'identity_risk_coverage.json',dict(population=provenance['population'],methods=identity_methods,qualified_FAR=None,formal_selection=None))
    funnels=defaultdict(Counter);maps=Counter();view=defaultdict(Counter);unavailable=Counter()
    for r in mainrows:
        d=r['camera_diagnostics']
        for gate,stats in d.get('acceptance_funnel',{}).items():funnels[gate].update({k:v for k,v in stats.items() if type(v) is int})
        c=d.get('surface_constraints',{});maps.update(c.get('mask_counts',{}));unavailable.update(c.get('unavailable_views',{}).values())
        for v in c.get('actual_views',[]):view[v['kind']].update({k:v.get(k,0) for k in ('in_fov','geometrically_usable','supported')})
    write_json(ROOT/'surface_acceptance_funnel.json',dict(pixel_funnel={k:dict(v) for k,v in funnels.items()},
        mask_totals=dict(maps),sparse_sequential_funnel=paired['sequential_scorable_funnel'],
        counting_scope='All camera pixels including startup/forced refresh; gates after dynamic exclusion; sparse scores after final refresh override'))
    write_json(ROOT/'surface_evidence_provenance.json',dict(dataset='nuScenes',actual_acquisition_view_counts={k:dict(v) for k,v in view.items()},
        unavailable_view_reasons=dict(unavailable),cross_camera_effective_stable_pixels=sum(r['effective_pixels'] for r in pixels if r['method']=='constraints_camera_only'),
        temporal_effective_stable_pixels=sum(r['effective_pixels'] for r in pixels if r['method']=='constraints_temporal_only'),
        simultaneous_cross_camera_scorable_support=summary['region']['constraints_camera_only']['accepted_scorable_points'],
        accepted_support_units={name:dict(reference_points=sum(r['accepted'][name] for r in points),
            distinct_pixel_locations=len({(r['frame'],r['camera'],tuple(r['pixel'])) for r in points if r['accepted'][name]}),
            distinct_frames=len({r['frame'] for r in points if r['accepted'][name]}),
            distinct_logs=len({r['log_token'] for r in points if r['accepted'][name]})) for name in summary['region']},
        camera_contract='Real affine-scaled nuScenes pinhole K, individual camera ego pose/extrinsic, coordinate-space equality, rigid finite matrices, reliable timestamps, <=10ms inter-camera skew and actual projected patch FOV; no additional distortion adapter',
        temporal_contract='Actual bounded committed source images; history earlier than source earlier than current, <=1s; >=0.5 degree measured parallax; no future observation',
        same_source_dependencies=['DIS forward/reverse','FB','patch texture/photo','ray triangulation','same-RGB LK re-tracking','fitted/withheld surface consensus keys'],
        additional_observation='Direct projective patch probe in a different actual acquisition; conditional evidence, not statistically independent truth',
        ambiguity='Finite +/-4px epipolar aperture at 0.5px increments; all actually supported incompatible >2px alternatives rejected, never residual-ranked',
        visibility='Observed depth-order collisions among supported hypotheses cause Unknown; missing/bad appearance alone is not occlusion proof',
        qualified_certifications=0,physical_ownership_certified=False,feature_reuse_certified=False,LiDAR_online=False,Annotation_online=False))
    audit=read('independent_data_adapter_audit.json');assert not audit['raw_unexposed_labeled_logs']
    for stage in ('calibration','heldout','final_test'):
        write_json(ROOT/(stage+'_summary.json'),dict(stage=stage,status='Not Run',population=None,
            reason='No remaining unexposed labeled nuScenes acquisition Log under frozen physical-quality protocol',
            samples=0,logs=0,qualified_certifications=0,qualified_FAR=None,method_selection=None))
    write_json(ROOT/'frozen_validation_protocol.json',dict(source_sha256=freeze['files'],dataset='nuScenes',
        data_selection_sha256=sha(ROOT/'data_selection.json'),experiment_protocol_sha256=sha(ROOT/'experiment_protocol.json'),
        selection_rule='No qualification or formal selection without fresh labeled complete acquisition logs; no reused scene/frame/object inflation',
        identity_selection=None,region_selection=None,calibration='Not Run',heldout='Not Run',final_test='Not Run'))
    xml=ET.parse(ROOT/'regression_cuda.xml').getroot();suites=list(xml) if xml.tag=='testsuites' else [xml]
    tests=sum(int(s.get('tests',0)) for s in suites);failed=sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites);skipped=sum(int(s.get('skipped',0)) for s in suites)
    pins=read('historical_artifact_pins.json');pin_files=pins.get('files',pins)
    # Historical pins may include a descriptive envelope; only digest mappings
    # are verified here. Every checked historical artifact remains untouched.
    checks={p:sha(p)==v for p,v in pin_files.items() if isinstance(v,str) and len(v)==64 and Path(p).is_file()}
    assert checks and all(checks.values()),checks
    regression=dict(tests=tests,passed=tests-failed-skipped,failed=failed,skipped=skipped,CUDA=True,
        command='taskset -c 0-3,6-31 env MPLCONFIGDIR=/tmp/m1_observability_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/qh/miniconda3/envs/OmniDrive/bin/python -m pytest tests/worldstream_m1 tests/worldstream_m2 tests/worldstream_m3 -q',
        xml='regression_cuda.xml',historical_artifact_checks=checks,baseline_tests=506,new_observation_contract_tests=11)
    write_json(ROOT/'regression_results.json',regression)
    model=read('gpu_parity/results.json');write_json(ROOT/'original_model_parity.json',dict(status=read('gpu_parity/status.json')['status'],
        frames=8,comparisons=model['comparisons'],equal=model['observe_only_equal'],
        original_checkpoint_sha256=read('data_manifest.json')['checkpoint']['sha256'],
        scope='Actual original EVA/detection/map outputs and temporal memory; perception-only, LLM not executed; M2/M3 disabled in this parity check',
        source='gpu_parity/results.json'))
    main=methods['surface_constraints'];identity=identity_methods['temporal_object']
    state=rows('state.jsonl');assert all(r['qualified']==0 and r['duplicate_history_ids']==0 for r in state)
    formal=dict(dataset='nuScenes + OmniDrive + M1',population='exploratory_reused_population',decision='No-Go',
        qualified_certifications=0,qualified_FAR=None,identity_selection=None,region_selection=None,
        calibration='Not Run',heldout='Not Run',final_test='Not Run',
        gates=dict(engineering=dict(pass_gate=not failed and not skipped and model['observe_only_equal'],tests=tests,actual_commits=len(state),duplicate_history_id_commits=0),
            surface_physical_observability=dict(pass_gate='partial',ambiguity_marginal=paired['ambiguity_marginal_rejection'],limitation='Different acquisition evidence rejects real errors, but costs many correct observations and does not prove physical ownership'),
            region_feasibility=dict(pass_gate=False,nonzero=True,four_descriptive_point_constraints_passed=main['sparse_reference_far']<=.05 and main['critical_change_recall']>=.998 and main['critical_point_recall']>=.998 and main['minimum_roi_event_refresh_ratio']>=.9,
                all_original_risk_gates_passed=main['risk_gates_passed'],sparse_iid_upper95=main['far_upper95_iid_descriptive'],accepted_scenes=main['accepted_scene_support'],accepted_logs=main['accepted_log_support'],
                effective_stable_area_ratio=main['effective_stable_area_ratio'],conservative_area_ratio=methods['conservative']['effective_stable_area_ratio']),
            identity_feasibility=dict(pass_gate=False,retained='temporal_object',correct_links=217,unique_instances=45,new_producer=False,reason='No measured evidence supporting a beneficial extra identity producer; population analysis does not create new certifications'),
            independent_quality=dict(pass_gate=False,new_labeled_logs=0,fully_scorable_accepted_identity_logs=identity['log_risk']['fully_scorable_accepted_logs'],identity_log_upper95=identity['log_risk']['complete_log_event_upper95'],minimum_zero_error_logs=59),
            scientific_feasibility=dict(pass_gate='boundary_identified',conclusion='Some measured alternatives can be rejected; present camera evidence leaves large unobserved/ambiguous populations. Independent data shortage is a separate limitation.')))
    write_json(ROOT/'formal_go_nogo.json',formal)
    source=source_manifest();source.update({str(p):sha(p) for p in [*Path('tools/worldstream_m1').glob('observability_*.py'),Path('tests/worldstream_m1/test_surface_observability.py')]})
    write_json(ROOT/'source_manifest.json',dict(baseline_head=read('baseline_source_manifest.json').get('head'),files=source,
        development_source_unchanged=freeze['files']==source_manifest(),new_runtime_policy='surface_constraints',original_model_and_oracle_unchanged=True,
        raw_trace_files={str(p):sha(p) for p in (ROOT/'traces').rglob('*.jsonl')}))
    figures(methods,populations,pixels,diagnostic,paired)
    print(json.dumps(dict(new_region={k:main[k] for k in ('effective_stable_area_ratio','sparse_reference_far','far_upper95_iid_descriptive','minimum_roi_event_refresh_ratio')},paired=paired['comparisons'],ambiguity_marginal=paired['ambiguity_marginal_rejection'],regression=regression,formal='No-Go'),indent=2))


if __name__=='__main__':main()
