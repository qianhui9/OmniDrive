"""Offline exploratory case forensics and independent frozen-grid risk replay."""
import argparse,copy,gzip,json,pickle
from collections import Counter,defaultdict
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from time import perf_counter,process_time
import numpy as np
from scipy.stats import beta
from worldstream.m1 import M1Config
from worldstream.m1.geometry import project_anchor,project_points
from worldstream.m1.bidirectional_geometry import verify_anchor_geometry
from worldstream.m1.state_types import json_value
from .audit import write

ROOT=Path('outputs/system_optimization_v3')


def upper95(errors,total):
    return float(beta.ppf(.95,errors+1,total-errors)) if total and errors<total else 1. if total else None


@lru_cache(maxsize=12)
def snapshot(path):
    with gzip.open(path,'rb') as f:return pickle.load(f)


def index_paths(root):
    # Preserve original run priority and original history when a frame repeats.
    rank={'long_stream':0,'single':0,'main':0,'long':0,'multi':1,'stride_refresh':2,'auto_validation':3,'auto':3,'cost_refresh_auto':4,'cost_refresh':5,'smoke':6}
    result={}
    for p in sorted(Path(root).glob('*/oracle_frames/*.pkl.gz'),key=lambda p:(rank.get(p.parents[1].name,8),str(p))):result.setdefault(p.name[:-7],p)
    return result


def proof(data,anchor,cid,config,paths):
    frame=data['frame'];previous=data.get('previous_cameras',{}).get(cid)
    evidence=data.get('temporal_evidence',{}).get(cid)
    if previous is None and anchor.source_frame_id in paths:
        previous=snapshot(str(paths[anchor.source_frame_id]))['frame'].cameras.get(cid)
    if evidence is None and previous is not None:
        from worldstream.m1.temporal_visibility import build_evidence
        # Original evidence has dynamic/occlusion exclusions. Intersecting
        # original acceptance is conservative; recomputation cannot approve
        # previously rejected anchors. Explicitly report this replay scope.
        evidence=build_evidence(previous,frame.cameras[cid],config)
    return verify_anchor_geometry(anchor,previous,frame.cameras[cid],evidence,
        project_anchor(anchor,frame.cameras[cid],config),config)


def diagnose():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    old=Path('outputs/system_optimization_v2');oracle=json.loads((old/'physical_oracle_results.json').read_text())
    errors=[r for r in oracle['physical_records'] if r['accepted'] and r['oracle_valid'] is False]
    paths=index_paths(old);config=M1Config.load('configs/worldstream_m1_coverage.yaml');cases=[]
    raw=pickle.load(open('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl','rb'))['infos']
    from .identity import token_index
    from tools.worldstream_m1.dataset import load_oracle
    from tools.worldstream_m1.evaluate import oracle_assignment
    infos=token_index(raw);selected={r['frame'] for r in errors}
    for r in errors:
        a=snapshot(str(paths[r['frame']]))['history'][r['anchor']]
        if a.source_frame_id:selected.add(a.source_frame_id)
    annotations=load_oracle([infos[t] for t in selected],'data/nuscenes')
    for i,row in enumerate(errors):
        data=snapshot(str(paths[row['frame']]));anchor=data['history'][row['anchor']];frame=data['frame']
        source=snapshot(str(paths[anchor.source_frame_id]))['frame'] if anchor.source_frame_id in paths else None
        assignment=oracle_assignment([anchor],annotations.get(anchor.source_frame_id,[]));identity=assignment.get(0)
        gt_source=next((x for x in annotations.get(anchor.source_frame_id,[]) if x['instance_id']==identity),None)
        gt_current=next((x for x in annotations[row['frame']] if x['instance_id']==identity),None)
        source_error=float(np.linalg.norm(anchor.geometry_world.center-gt_source['geometry'].center)) if gt_source else None
        camera_cases=[]
        for cid,check in data['result'].diagnostics['anchors'][row['anchor']]['cameras'].items():
            if not check.get('valid'):continue
            item=dict(camera=cid,source_region=anchor.observation_support[cid].region,original_validation=check,
                bidirectional=proof(data,anchor,cid,config,paths))
            def camera_record(c):
                return dict(timestamp=c.timestamp,intrinsic=c.intrinsic,world_to_camera=c.world_to_camera,
                    timing_reliable=c.timing_reliable,coordinate_space=c.coordinate_space) if c else None
            item['calibration']=dict(source=camera_record(source.cameras.get(cid) if source else None),current=camera_record(frame.cameras[cid]))
            if gt_current:
                uv,_,_=project_points(gt_current['geometry'].points,frame.cameras[cid].intrinsic,frame.cameras[cid].world_to_camera)
                item['oracle_projected_points']=uv
            camera_cases.append(item)
        label='historical_source_localization_bias' if source_error is not None and source_error>1 else 'source_association_or_oracle_uncertainty'
        mechanism=dict(identity_switch='not established by source assignment alone',geometry_projection='GT/predicted projected corners saved; appearance match is not metric localization',
            ego_motion_compensation='no evidence of pose corruption; independent GT world center comparison',dynamic_motion='not supported: static-object labels and zero/small GT displacement',
            descriptor_alias='plausible foreground/background alias; not independently proven',visibility='nuScenes visibility=4 and sensor support; not pixel segmentation',
            temporal_correspondence='measured forward/backward tracks saved; depth observability tested',historical_source_mismatch=label,
            oracle_uncertainty='2m source assignment can confuse adjacent cones; no guaranteed identity labels')
        case=dict(case=i,source_frame=anchor.source_frame_id,current_frame=row['frame'],anchor_id=anchor.anchor_id,
            semantic_class=anchor.semantic_class,oracle=row,source_center_error_m=source_error,primary_mechanism=label,
            mechanisms=mechanism,source_world_center=anchor.geometry_world.center,cameras=camera_cases)
        def annotation_record(a):
            return dict(instance_id=a['instance_id'],semantic_class=a['class'],visibility=a['visibility'],
                sensor_support=a['has_sensor_support'],world_center=a['geometry'].center,
                world_points=a['geometry'].points) if a else None
        case['independent_oracle']=dict(source=annotation_record(gt_source),current=annotation_record(gt_current),
            source_assignment_gate_m=2.,identity_is_assignment_not_guarantee=True)
        write(ROOT/'m1_case_visualizations'/f'case_{i:02d}.json',json_value(case));cases.append(json_value(case))
        view=camera_cases[0] if camera_cases else None
        fig,axes=plt.subplots(1,2,figsize=(13,4.5))
        for ax,camframe,title,region in ((axes[0],source,'Source',view['source_region'] if view else None),(axes[1],frame,'Current',view['original_validation']['predicted_region'] if view else None)):
            if view and camframe:
                ax.imshow(camframe.cameras[view['camera']].image)
                if region:ax.add_patch(Rectangle(region[:2],region[2]-region[0],region[3]-region[1],fill=False,color='yellow',lw=1))
            else:ax.text(.1,.5,'Source image unavailable in frozen snapshots')
            ax.set_title(title);ax.axis('off')
        if view:
            uv=np.asarray(view.get('oracle_projected_points',[]))
            if len(uv):axes[1].scatter(uv[:,0],uv[:,1],s=10,c='red',label='Oracle corners');axes[1].legend()
            for point in view['bidirectional'].get('points',[]):
                axes[0].plot(*point['source_uv'],marker='.',color='cyan');axes[1].plot(*point['current_uv'],marker='.',color='cyan')
        source_label='unknown' if source_error is None else f'{source_error:.2f}m'
        fig.suptitle(f"Case {i}: {anchor.semantic_class}; source error={source_label}; current={row['center_error_m']:.2f}m\n{label}",fontsize=10)
        fig.tight_layout();fig.savefig(ROOT/'m1_case_visualizations'/f'case_{i:02d}.png',dpi=140);plt.close(fig)
    write(ROOT/'m1_false_accept_diagnosis.json',dict(scope='Exploratory v2 cases only; no heldout claims',cases=cases,
        total=len(cases),scene_count=len({r['oracle']['scene'] for r in cases}),primary_mechanisms=dict(Counter(c['primary_mechanism'] for c in cases)),
        causal_limit='Source association and image correspondence do not independently prove object identity or foreground ownership'))


def evaluate():
    diagnosis=ROOT/'m1_false_accept_diagnosis.json'
    if not diagnosis.exists() or any('independent_oracle' not in r for r in json.loads(diagnosis.read_text())['cases']):
        diagnose()
    from .physical_oracle import main as physical
    physical(ROOT,query_subdirectory='long_task_supply')
    original=json.loads((ROOT/'physical_oracle_results.json').read_text());paths=index_paths(ROOT)
    # Align actual M1 log entries with the runner's executed persistent rows.
    # This verifies candidate invalidation masks, rather than inferring their
    # critical-change behavior from a smaller accepted-anchor set.
    import hashlib
    mask_checks=[];actual_valid={}
    for run in ('main','long'):
        from .v2_checks import rows
        executed=[r for r in rows(ROOT/run/'frames.jsonl') if r['frame']['service']=='persistent_state']
        actual={}
        count=0
        with (ROOT/run/'m1_state.jsonl').open() as f:
            for row,line in zip(executed,f):
                log=json.loads(line)['precheck'];assert log['frame_id']==row['frame']['identity']['frame_id']
                payload={cid:region['exact_invalidation_mask'] for cid,region in log['invalidation_regions'].items()}
                actual[(row['mode'],row['index'])]=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest();count+=1
                if row['mode'] in ('certified_serial','certified_overlap'):
                    actual_valid[(run,row['mode'],log['frame_id'])]=set(log['valid_anchor_ids'])
            assert f.readline()=='','Unexpected extra M1 log entry'
        assert count==len(executed),'Missing M1 log entries'
        for (mode,index),digest in actual.items():
            if mode in ('certified_serial','certified_overlap'):
                mask_checks.append(dict(run=run,mode=mode,index=index,equals_original_m1=digest==actual[('m1',index)]))
    protocol=json.loads((ROOT/'protocol.json').read_text());cfg=M1Config.load('configs/worldstream_m1_coverage.yaml');results=[]
    risk_runs=('main','long','auto')
    measured=[r for r in original['physical_records'] if r['run'] in risk_runs]
    for candidate in protocol['candidate_grid']:
        config=replace(cfg,depth_padding_m=candidate.get('depth_padding_m',.25),min_triangulation_angle_deg=candidate.get('min_angle_deg',.5))
        records=[];costs=[]
        for record in measured:
            r=dict(record)
            if record['accepted'] and candidate['bidirectional_geometry']:
                data=snapshot(str(paths[record['frame']]));anchor=data['history'][record['anchor']]
                tick=perf_counter();cpu=process_time()
                cameras=[cid for cid,v in data['result'].diagnostics['anchors'][record['anchor']]['cameras'].items() if v['geometry_valid']]
                checks={cid:proof(data,anchor,cid,config,paths) for cid in cameras}
                costs.append(dict(wall_ms=(perf_counter()-tick)*1000,cpu_ms=(process_time()-cpu)*1000))
                r['accepted']=bool(checks) and all(v['passed'] for v in checks.values());r['geometry_checks']=json_value(checks)
            records.append(r)
        from tools.worldstream_m1.refinement_common import summarize_rows
        splits={}
        for split in ('calibration','heldout'):
            selected=[r for r in records if not r.get('exploratory',False) and r['split']==split]
            stats=summarize_rows(selected);n=stats['accepted_labeled'];errors=stats['physical_errors'];scenes={r['scene'] for r in selected if r['accepted'] and r['oracle_valid'] is not None}
            clustered=[]
            for scene in sorted({r['scene'] for r in selected}):
                subset=summarize_rows([r for r in selected if r['scene']==scene])
                clustered.append([subset['physical_errors'],subset['accepted_labeled']])
            samples=np.asarray(clustered,dtype=float)
            boot=samples[np.random.RandomState(20261017).randint(0,len(samples),(10000,len(samples)))].sum(1) if len(samples) else np.empty((0,2))
            valid_boot=boot[:,1]>0 if len(boot) else np.empty(0,bool)
            interval=np.percentile(boot[valid_boot,0]/boot[valid_boot,1],[2.5,97.5]).tolist() if valid_boot.any() else None
            stats.update(far_upper95=upper95(errors,n),accepted_support_scenes=len(scenes),
                scene_cluster_bootstrap_far_95ci=interval,scene_bootstrap_zero_denominator_draws=int((~valid_boot).sum()),
                physical_risk_pass=n>=protocol['physical_risk']['min_accepted_scorable'] and len(scenes)>=protocol['physical_risk']['min_support_scenes'] and upper95(errors,n)<=protocol['physical_risk']['far_upper95_max'])
            critical=[e for e in original['critical_records'] if e['run'] in risk_runs and e['split']==split]
            critical_recall=sum(e['recalled'] for e in critical)/len(critical) if critical else None
            stats['critical_change_recall']=critical_recall
            stats['critical_change_gate_pass']=critical_recall is not None and critical_recall>=protocol['physical_risk']['min_critical_recall'] and all(m['equals_original_m1'] for m in mask_checks)
            stats['physical_risk_pass']=stats['physical_risk_pass'] and stats['critical_change_gate_pass']
            stats['by_category']={kind:summarize_rows([r for r in selected if (r['category']=='static_object')==(kind=='static')]) for kind in ('static','dynamic_or_map')}
            stats['by_measured_motion']={kind:summarize_rows([r for r in selected if r['gt_speed_mps'] is not None and
                (r['gt_speed_mps']<.5 if kind=='static_lt_0.5_mps' else r['gt_speed_mps']>=.5)]) for kind in ('static_lt_0.5_mps','dynamic_ge_0.5_mps')}
            splits[split]=stats
        results.append(dict(candidate=candidate,splits=splits,records=records,extra_verifier_costs=costs,
            scope='Rejection-only replay on fixed original history; GPU certified modes separately include altered history feedback'))
    reference_gate=next(r for r in results if r['candidate']['name']=='bidirectional_depth')
    expected=defaultdict(set)
    for row in reference_gate['records']:
        if row['accepted']:expected[row['frame']].add(row['anchor'])
    replay_comparisons=[dict(run=run,mode=mode,frame=token,actual_accepted_count=len(ids),
        replay_accepted_count=len(expected[token]),equal=ids==expected[token]) for (run,mode,token),ids in actual_valid.items()]
    # Select using calibration only; retain frozen declaration order on ties.
    replay_agrees=bool(replay_comparisons) and all(r['equal'] for r in replay_comparisons)
    eligible=[r for r in results if r['splits']['calibration']['physical_risk_pass'] and replay_agrees]
    chosen=max(eligible,key=lambda r:r['splits']['calibration']['valid_state_recovery_rate']) if eligible else None
    write(ROOT/'m1_risk_coverage_results.json',dict(protocol='protocol.json',candidates=results,
        selected=chosen['candidate']['name'] if chosen else None,
        independent_quality_pass=bool(chosen and chosen['splits']['heldout']['physical_risk_pass']),
        decision='No-Go: actual gate/replay disagreement' if not replay_agrees else 'No-Go: insufficient risk/coverage support' if chosen is None else 'Pass' if chosen['splits']['heldout']['physical_risk_pass'] else 'No-Go: heldout failed',
        confidence_scope='Binomial upper bound is descriptive under correlated frames; scene-cluster intervals and small-scene support limit generalization. No universal physical safety certification.',
        actual_certified_mask_comparisons=mask_checks,
        actual_gate_vs_replay=replay_comparisons,
        critical_scope='New gate only removes anchor certification; dense local pixel evidence is separate and unchanged. Original critical invalidation masks remain comparator, not newly inferred physical ownership.',
        original_critical_recall=original['critical_change_recall'],online_gt_used=False))
