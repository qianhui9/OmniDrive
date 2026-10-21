"""Offline read-only reduction of per-frame/Anchor/camera observation evidence.

Two populations are reported: all historical views and geometrically eligible
views of projection candidates. Dense pixels, motion identity and metric claims
remain different populations. No labels are read here.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def diagnose_scene(trace,output,config_path='configs/worldstream_m11_diagnostics.yaml'):
    from worldstream.m1 import M1Config
    config=M1Config.load(config_path)
    projection_failures={}
    rows=[json.loads(x) for x in Path(trace).read_text().splitlines()];funnel={};all_reasons=Counter();subgroups={};flat=[];frames=[]
    for row in rows:
        p=row['precheck'];frame_count=Counter();candidate=set(p['association_candidate_ids'])
        for aid,a in p['diagnostics'].get('anchors',{}).items():
            kind='map_or_region' if a['anchor_type']!='object' else 'static_class_prior' if a['motion_hypothesis']=='static_class_prior' else 'dynamic_or_motion_unqualified'
            group=subgroups.setdefault(kind,Counter(anchors=0,observations=0,eligible_camera_views=0))
            group['anchors']+=1;group['observations']+=a['valid']
            for cid,v in a['cameras'].items():
                eligible=aid in candidate and v['geometry_valid'];frame_count['all_historical_camera_views']+=1
                if eligible:group['eligible_camera_views']+=1;frame_count['eligible_candidate_camera_views']+=1
                if eligible and not v['valid']:all_reasons[v['reason']]+=1
                e=v.get('evidence',{});checks=e.get('observation_funnel',{})
                item=dict(frame_id=p['frame_id'],anchor_id=aid,camera=cid,subgroup=kind,
                    eligible_candidate_view=eligible,anchor_observation_supported=a['valid'],camera_observation_supported=v['valid'],
                    reason=v['reason'],descriptor_residual=v['residual'],source_frame_id=e.get('source_frame_id'),
                    source_frame_digest=e.get('source_frame_digest'),source_state_version=e.get('source_state_version'),
                    source_epoch=e.get('source_epoch'))
                for name,c in checks.items():
                    item[name]=c['passed']
                    if eligible:
                        counts=funnel.setdefault(name,Counter(total=0,passed=0,rejected=0,unknown=0))
                        counts['total']+=1;counts['passed' if c['passed'] is True else 'rejected' if c['passed'] is False else 'unknown']+=1
                projection_check=checks.get('current_projection',{})
                if projection_check.get('passed') is False:
                    projection_failures.setdefault(kind,Counter())[projection_check['reason']]+=1
                # These are the actual anchor_visibility outputs AFTER exclusion
                # and the predicted-box motion bound, not the raw local tracks.
                if eligible:
                    actual=dict(visibility_minimum_points=None if 'supported_points' not in e else e['supported_points']>=config.support_min_points,
                        visibility_support_fraction=None if 'support_fraction' not in e else e['support_fraction']>=config.support_point_fraction,
                        visibility_spatial_fraction=None if 'quadrant_fraction' not in e else e['quadrant_fraction']>=config.support_quadrant_fraction)
                    for name,value in actual.items():
                        item[name]=value
                        c=funnel.setdefault(name,Counter(total=0,passed=0,rejected=0,unknown=0));c['total']+=1
                        c['passed' if value is True else 'rejected' if value is False else 'unknown']+=1
                    for key in ('selected_points','supported_points','support_fraction','quadrant_fraction'):
                        item['actual_visibility_'+key]=e.get(key)
                item['prediction_kind']=e.get('prediction_kind')
                item['prediction_valid_points']=e.get('prediction_valid_points')
                item['visibility_status']=e.get('visibility_status')
                item['raw_projection_reason']=v.get('reason')
                measurements=e.get('observation_measurements',{})
                item['local_support_fraction']=measurements.get('local_support_fraction');item['quadrants']=measurements.get('quadrants')
                flat.append(item)
        frames.append(dict(frame_id=p['frame_id'],timestamp=p['timestamp'],counts=dict(frame_count),
            anchor_rejection_reasons=dict(Counter(reason for a in p['diagnostics'].get('anchors',{}).values() for reason in a['reasons']))))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    fields=sorted({k for x in flat for k in x})
    with (output/'anchor_camera_funnel.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(flat)
    result=dict(scene_id=rows[0]['precheck']['scene_id'],frames=frames,
        eligible_candidate_view_funnel={k:dict(v) for k,v in funnel.items()},eligible_view_rejection_reasons=dict(all_reasons),
        subgroups={k:dict(v) for k,v in subgroups.items()},
        projection_failure_reasons_by_subgroup={k:dict(v) for k,v in projection_failures.items()},
        finite_claim_boundaries=dict(static_class_prior_is_stationarity_proof=False,
            LK_or_stable_pixels_prove_foreground=False,identity_proves_metric_geometry=False,
            metric_geometry_qualified=False,neural_feature_equivalence=False))
    (output/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config',default='configs/worldstream_m11_diagnostics.yaml')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    root=Path(a.root);traces=sorted((root/'scenes').glob('*/repeat_0.jsonl')) or sorted((root/'scenes').glob('*/m1_state.jsonl'))
    summaries=[diagnose_scene(path,out/path.parent.name,a.config) for path in traces]
    (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')


if __name__=='__main__':main()
