"""Offline risk/coverage accounting; no changes to the independent oracle."""
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import beta
from .evidence_experiments import summary


def upper95(errors, count):
    if not count:return None
    return 1. if errors==count else float(beta.ppf(.95,errors+1,count-errors))


def log_risk(rows, accepted_key='candidate', wrong_key='correct'):
    units=defaultdict(list)
    for r in rows:
        if r[accepted_key]:units[r['log_token']].append(r)
    complete={k:v for k,v in units.items() if all(r[wrong_key] is not None for r in v)}
    errors=sum(any(r[wrong_key] is False for r in v) for v in complete.values())
    all_errors=sum(any(r[wrong_key] is False for r in v) for v in units.values())
    return dict(accepted_logs=len(units),fully_scorable_accepted_logs=len(complete),
        incomplete_accepted_logs=len(units)-len(complete),error_logs=all_errors,
        complete_error_logs=errors,complete_log_event_upper95=upper95(errors,len(complete)),
        worst_case_log_event_upper95=upper95(all_errors+sum(k not in complete and not any(r[wrong_key] is False for r in v) for k,v in units.items()),len(units)),
        events={k:dict(accepted=len(v),unscorable=sum(r[wrong_key] is None for r in v),
                       errors=sum(r[wrong_key] is False for r in v),scenes=sorted({r['scene'] for r in v})) for k,v in units.items()})


def identity_metrics(rows, original_denominator):
    selected=[r for r in rows if r['candidate']];scored=[r for r in selected if r['correct'] is not None]
    good=[r for r in scored if r['correct']];bad=[r for r in scored if not r['correct']]
    certified=[r for r in rows if r['certified']]
    return dict(geometric_candidates=original_denominator,evaluated_edges=len(rows),identity_proposals=len(selected),
        independently_correct_proposals=len(good),wrong_proposals=len(bad),unscorable_proposals=len(selected)-len(scored),
        proposal_coverage=len(selected)/original_denominator if original_denominator else None,
        scorable_correct_proposal_coverage=len(good)/original_denominator if original_denominator else None,
        unique_physical_identity_recovery=len({(r['log_token'],r['current_instance']) for r in good}),
        unique_correct_source_current_links=len({(r['scene'],r['source_frame'],r['frame'],r['source_instance'],r['current_instance']) for r in good}),
        correct_support_scenes=len({r['scene'] for r in good}),correct_support_logs=len({r['log_token'] for r in good}),
        qualified_certifications=len(certified),qualified_certification_coverage=len(certified)/original_denominator if original_denominator else None,
        conditional_proposal_far=len(bad)/len(scored) if scored else None,
        proposal_iid_upper95_descriptive=upper95(len(bad),len(scored)),
        qualified_FAR=(sum(r['correct'] is False for r in certified)/len(certified)) if certified and all(r['correct'] is not None for r in certified) else None,
        log_risk=log_risk(rows),qualified_log_risk=log_risk(rows,accepted_key='certified'),
        multilabel_rejections=dict(Counter(x for r in rows for x in set(r['rejection_reasons']))),
        primary_rejections=dict(Counter((r['rejection_reasons'] or ['proposal'])[0] for r in rows)),
        denominator_note='Original Hungarian geometric candidate pairs. Recovery edges additionally counted, never used to shrink the denominator.')


def region_metrics(rows):
    result=summary(rows)
    logs=defaultdict(list)
    for r in rows:logs[r['log_token']].append(r)
    accepted={k:v for k,v in logs.items() if sum(r.get('accepted_points',0) for r in v)>0}
    errors=sum(sum(r.get('false_points',0) for r in v)>0 for v in accepted.values())
    # Fixed seeded cluster bootstrap. Entire acquisition logs, never pixels, are
    # resampled. Zero-support replicates stay undefined and are counted explicitly.
    units=list(logs.values());rng=np.random.RandomState(20261019);areas=[];fars=[];undefined=0
    if units:
        stats=np.asarray([[sum(r.get(k,0) for r in v) for k in ('pixels','effective_pixels','accepted_points','false_points')] for v in units],float)
        sums=stats[rng.randint(len(units),size=(1000,len(units)))].sum(1)
        areas=(sums[:,1]/sums[:,0]).tolist();defined=sums[:,2]>0
        fars=(sums[defined,3]/sums[defined,2]).tolist();undefined=int((~defined).sum())
    events=[e for r in rows for e in r.get('events',[])]
    result.update(effective_stable_area_ratio=result['effective_stable_area'],
        independently_correct_stable_support=result['accepted_scorable_points']-result['false_stable_points'],
        independently_scorable_stable_support=result['accepted_scorable_points'],
        dense_unscorable_stable_pixels=None,
        unscorable_scope='Sparse LiDAR samples are not a pixel-area partition. Remaining stable area has no dense truth; never impute negative labels.',
        accepted_log_support=len(accepted),error_logs=errors,log_event_upper95=upper95(errors,len(accepted)),
        clustered_effective_area_ci95=np.percentile(areas,[2.5,97.5]).tolist() if areas else None,
        clustered_sparse_far_ci95=np.percentile(fars,[2.5,97.5]).tolist() if fars else None,
        cluster_bootstrap_undefined_zero_support=undefined,cluster_bootstrap_replicates=1000 if units else 0,
        cluster_bootstrap_note='Conditional sparse support uncertainty; sparse samples do not bound full dense area error',
        minimum_roi_event_refresh_ratio=min((e['invalidation_coverage'] for e in events),default=None),
        roi_event_refresh_ratio=[e['invalidation_coverage'] for e in events],
        risk_gates_passed=bool(result['critical_change_recall'] is not None and result['critical_change_recall']>=.998
            and result['critical_point_recall']>=.998 and events and all(e['invalidation_coverage']>=.9 for e in events)
            and result['sparse_reference_far'] is not None and result['sparse_reference_far']<=.05
            and result['far_upper95_iid_descriptive']<=.05 and result['accepted_scene_support']>=3))
    return result
