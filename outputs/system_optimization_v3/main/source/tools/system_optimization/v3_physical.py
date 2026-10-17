"""Offline exploratory case forensics and independent frozen-grid risk replay."""
import argparse,copy,gzip,json,pickle
from collections import Counter
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
        fig.suptitle(f"Case {i}: {anchor.semantic_class}; source error={source_error}; current={row['center_error_m']:.2f}m\n{label}",fontsize=10)
        fig.tight_layout();fig.savefig(ROOT/'m1_case_visualizations'/f'case_{i:02d}.png',dpi=140);plt.close(fig)
    write(ROOT/'m1_false_accept_diagnosis.json',dict(scope='Exploratory v2 cases only; no heldout claims',cases=cases,
        total=len(cases),scene_count=len({r['oracle']['scene'] for r in cases}),primary_mechanisms=dict(Counter(c['primary_mechanism'] for c in cases)),
        causal_limit='Source association and image correspondence do not independently prove object identity or foreground ownership'))


def evaluate():
    from .physical_oracle import main as physical
    physical(ROOT)
    original=json.loads((ROOT/'physical_oracle_results.json').read_text());paths=index_paths(ROOT)
    protocol=json.loads((ROOT/'protocol.json').read_text());cfg=M1Config.load('configs/worldstream_m1_coverage.yaml');results=[]
    measured=original['physical_records']
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
            stats.update(far_upper95=upper95(errors,n),accepted_support_scenes=len(scenes),
                physical_risk_pass=n>=protocol['physical_risk']['min_accepted_scorable'] and len(scenes)>=protocol['physical_risk']['min_support_scenes'] and upper95(errors,n)<=protocol['physical_risk']['far_upper95_max'])
            stats['by_category']={kind:summarize_rows([r for r in selected if (r['category']=='static_object')==(kind=='static')]) for kind in ('static','dynamic_or_map')}
            splits[split]=stats
        results.append(dict(candidate=candidate,splits=splits,records=records,extra_verifier_costs=costs,
            scope='Rejection-only replay on fixed original history; GPU certified modes separately include altered history feedback'))
    # Select using calibration only; retain frozen declaration order on ties.
    eligible=[r for r in results if r['splits']['calibration']['physical_risk_pass']]
    chosen=max(eligible,key=lambda r:r['splits']['calibration']['valid_state_recovery_rate']) if eligible else None
    write(ROOT/'m1_risk_coverage_results.json',dict(protocol='protocol.json',candidates=results,
        selected=chosen['candidate']['name'] if chosen else None,
        independent_quality_pass=bool(chosen and chosen['splits']['heldout']['physical_risk_pass']),
        decision='No-Go: insufficient risk/coverage support' if chosen is None else 'Pass' if chosen['splits']['heldout']['physical_risk_pass'] else 'No-Go: heldout failed',
        critical_scope='New gate only removes certifications; original invalidation masks are conservative upper-support comparator, not a replacement for actual certified-mode masks',
        original_critical_recall=original['critical_change_recall'],online_gt_used=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--diagnose',action='store_true');args=parser.parse_args()
    if args.diagnose:diagnose()
    else:evaluate()
