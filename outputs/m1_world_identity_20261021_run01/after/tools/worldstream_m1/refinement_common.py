"""Offline evaluation helpers; this module is never imported by the online M1 path."""
import numpy as np
from scipy.stats import beta


def ratio(a,b): return a/b if b else None


def interval(errors, n):
    if not n: return None
    return [float(beta.ppf(.025,errors,n-errors+1)) if errors else 0.,
            float(beta.ppf(.975,errors+1,n-errors)) if errors<n else 1.]


def summarize_rows(rows, decision='accepted'):
    labeled=[r for r in rows if r['oracle_valid'] is not None]
    accepted=[r for r in labeled if r[decision]]
    errors=sum(not r['oracle_valid'] for r in accepted)
    good=sum(bool(r['oracle_valid']) for r in labeled)
    return dict(trials=len(rows),scorable=len(labeled),accepted_all=sum(r[decision] for r in rows),
        accepted_labeled=len(accepted),physical_errors=errors,oracle_valid=good,
        valid_candidate_rate=ratio(sum(r[decision] for r in rows),len(rows)),
        physical_false_acceptance_rate=ratio(errors,len(accepted)),
        false_acceptance_95ci_trial_descriptive=interval(errors,len(accepted)),
        valid_state_recovery_rate=ratio(len(accepted)-errors,good),
        scorable_coverage=ratio(len(labeled),len(rows)),unknown_rate=ratio(len(rows)-len(labeled),len(rows)))


def physical_details(anchor, anns, identity, source_ann, timestamp):
    from worldstream.m1.geometry import predict_geometry
    ann=next((a for a in anns if a['instance_id']==identity),None)
    error=float(np.linalg.norm(predict_geometry(anchor,timestamp).center-ann['geometry'].center)) if ann else None
    displacement=float(np.linalg.norm(source_ann['geometry'].center-ann['geometry'].center)) if ann and source_ann else None
    age=timestamp-anchor.source_timestamp
    speed=displacement/age if displacement is not None and age>0 else None
    category=('map' if anchor.anchor_type=='map' else 'pedestrian' if anchor.semantic_class=='pedestrian'
              else 'static_object' if anchor.semantic_class in ('barrier','traffic_cone') or speed is not None and speed<.5
              else 'dynamic_vehicle')
    return dict(identity_known=identity is not None, same_instance_present=ann is not None,
        visibility=str(ann['visibility']) if ann else None,sensor_support=ann['has_sensor_support'] if ann else None,
        center_error_m=error,gt_displacement_m=displacement,gt_speed_mps=speed,category=category,
        speed_bin='unknown' if speed is None else '<0.5' if speed<.5 else '0.5-2' if speed<2 else '2-5' if speed<5 else '>=5',
        oracle_valid=None if identity is None else bool(ann and ann['has_sensor_support'] and str(ann['visibility'])=='4' and error<=1))
