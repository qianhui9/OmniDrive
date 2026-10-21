"""Read-only M1.2.1 depth/candidate/search funnels; independent scene summaries."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from .effective_acceptance import sha
from .scene_protocol import ratio,distribution


def run(root,output):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    all_counts=Counter();all_dist=defaultdict(list);summaries=[]
    traces=sorted((root/'scenes').glob('*/repeat_0.jsonl')) or sorted((root/'scenes').glob('*/m1_state.jsonl'))
    for trace in traces:
        scene=trace.parent.name;folder=out/scene;folder.mkdir();counts=Counter();values=defaultdict(list);reasons=Counter();statuses=Counter()
        def collect(name,x):
            a=np.asarray([] if x is None else x,float).ravel();values[name].extend(a[np.isfinite(a)].tolist())
        with (folder/'frame_camera_funnel.jsonl').open('w') as audit,trace.open() as stream:
            for line in stream:
                row=json.loads(line);p,c,s=row['precheck'],row['commit'],row['committed_state'];counts['frames']+=1
                assert p['diagnostics']['transaction_status']==s['transaction_status']=='committed'
                search=c.get('search_motion',{});statuses['search_'+search.get('status','Not Run')]+=1
                for k in ('attempts','generated','new_edges'):counts['dynamic_search_'+k]+=search.get(k,0)
                for h in search.get('hypotheses',{}).values():
                    reasons['search_'+h['reason']]+=1
                    if h['status']=='search_only':collect('search_uncertainty_m',[h['uncertainty_m']])
                for proposal in c['identity_continuity_proposals']:
                    source=search.get('hypotheses',{}).get(proposal['anchor_id'],{})
                    if source.get('status')=='search_only':
                        counts['search_source_identity_edges']+=1
                        counts['search_source_identity_candidates']+=proposal['status']=='candidate'
                    for view in proposal['measurements'].get('views',[]):
                        m=view.get('ego_motion',{});f=m.get('candidate_funnel',{})
                        for k,v in f.get('dynamic_search',{}).items():counts['dynamic_image_'+k]+=v
                        if f.get('dynamic_search',{}).get('selected',0):
                            counts['dynamic_selected_supported_identity_views']+=view['status']=='supported'
                            counts['dynamic_selected_candidate_identity_edges']+=proposal['status']=='candidate'
                for cid,e in p['diagnostics'].get('temporal_evidence',{}).items():
                    counts['historical_camera_pairs']+=1
                    h=e.get('depth_feature_seeding',{});g=e.get('ego_motion',{});association=g.get('depth_association',{})
                    m=e.get('point_measurements',{})
                    for k in ('redetected_points','depth_records','retained_points','duplicate_or_near_existing'):counts[k]+=h.get(k,0)
                    statuses['depth_retention_'+h.get('status','Not Run')]+=1
                    for k in ('records','query_points','within_radius','ambiguous','duplicate','one_to_one'):counts['depth_association_'+k]+=association.get(k,0)
                    if association:collect('depth_nearest_distance_px',association.get('nearest_distance_px'))
                    for k in ('points','bidirectional_points','geometric_points','supported_points','supported_points_after_exclusion'):counts[k]+=e.get(k,0)
                    for name,stages in e.get('candidate_funnel',{}).items():
                        for k,v in stages.items():counts['candidate_'+name+'_'+k]+=v
                    for k in ('seed_recovered','initialization_conflict'):
                        counts[k]+=int(np.asarray(m.get(k,[]),bool).sum())
                    for k in ('prediction_spread_px','prediction_residual_px','raw_fb_error_px','fb_error_px','source_depth_m'):
                        collect(k,m.get(k))
                    collect('depth_uncertainty_m',g.get('depth_uncertainty_m'))
                    audit.write(json.dumps(dict(frame_id=p['frame_id'],timestamp=p['timestamp'],camera_id=cid,
                        seeding=h,depth_association={k:v for k,v in association.items() if not isinstance(v,list)},
                        candidate_funnel=e.get('candidate_funnel',{}),post_exclusion_supported=e.get('supported_points_after_exclusion',0),
                        reason=e.get('reason'),depth_matching_state='Not Run' if not association else 'no_candidates' if not association.get('records') else 'passed' if association.get('one_to_one') else 'all_rejected'))+'\n')
                for h in s.get('depth_history',{}).values():
                    counts['generated_depth_records']+=h.get('points',0)
                    collect('triangulation_parallax_deg',h.get('parallax_deg'));collect('triangulation_reprojection_px',h.get('reprojection_px'))
                for a in p['diagnostics'].get('anchors',{}).values():
                    for v in a['cameras'].values():
                        aligned=v.get('evidence',{}).get('aligned_observation',{})
                        if aligned:
                            reasons['alignment_'+aligned.get('reason','not_applicable')]+=1
                            counts['alignment_measured_views']+=aligned.get('status')=='measured'
                            collect('aligned_descriptor_residual',[aligned.get('descriptor_residual')])
        def ratios(c):
            return dict(depth_match_rate=ratio(c['depth_association_one_to_one'],c['depth_association_query_points']),
                retained_ray_rate=ratio(c['retained_points'],c['depth_records']),local_support_rate=ratio(c['supported_points'],c['points']),
                post_exclusion_support_rate=ratio(c['supported_points_after_exclusion'],c['points']),
                dynamic_generation_rate=ratio(c['dynamic_search_generated'],c['dynamic_search_attempts']),
                dynamic_image_selection_rate=ratio(c['dynamic_image_selected'],c['dynamic_image_attempted']))
        summary=dict(scene_id=scene,counts=dict(counts),ratios=ratios(counts),statuses=dict(statuses),reasons=dict(reasons),
            distributions={k:distribution(v) for k,v in values.items()},trace_sha256=sha(trace))
        (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');summaries.append(summary);all_counts.update(counts)
        for k,v in values.items():all_dist[k]+=v
    result=dict(scenes=summaries,micro=dict(counts=dict(all_counts),ratios=ratios(all_counts),
        distributions={k:distribution(v) for k,v in all_dist.items()}),
        macro={k:distribution([s['ratios'][k] for s in summaries]) for k in summaries[0]['ratios']},
        no_formal_claim_from_search=True,script_sha256=sha(__file__))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.root,a.output)


if __name__=='__main__':main()
