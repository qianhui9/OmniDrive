"""Frozen scene-level statistical performance certification, exact quality keys."""
from collections import defaultdict
import hashlib,json
from pathlib import Path
import numpy as np
from .audit import write
from .v2_checks import rows


def upper95(values,seed=20261017):
    a=np.asarray(values,dtype=float)
    if not len(a):return None
    return float(np.percentile(a[np.random.RandomState(seed).randint(0,len(a),(10000,len(a)))].mean(1),95))


def calibrate(root,runs=('main',)):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text());frozen=protocol['performance']
    (root/'executed_cost_calibration.py').write_bytes(Path(__file__).read_bytes())
    data=[]
    for name in runs:
        for r in rows(root/name/'frames.jsonl'):r['source_run']=name;data.append(r)
    base={(r['source_run'],r['index'],r['repeat']):r for r in data if r['mode']=='m1' and not r['warmup']}
    groups=defaultdict(list)
    for r in data:
        if r['mode']!='overlap' or r['warmup']:continue
        ref=base[(r['source_run'],r['index'],r['repeat'])];key=r['frame']['admission']['workload_key']
        domain=frozen.get('frame_gap_domain_s')
        if key['cache_state']=='warm' and domain and not domain[0]<=key['frame_gap_s']<=domain[1]:continue
        normal=dict(key);normal.pop('frame_gap_s')
        q=r['frame']['quality'];equal=r['generated_ids']==ref['generated_ids'] and q['m1_state_digest']==ref['frame']['quality']['m1_state_digest'] and q['exact_perception_memory_digest']
        equal=equal and all(v['max_abs']==0 and not v['shape_mismatches'] for v in q['numerical'].values())
        groups[json.dumps(normal,sort_keys=True)].append(dict(key=key,scene=r['frame']['identity']['scene_id'],split=r['split'],
            delta_ms=r['frame']['e2e_wall_ms']-ref['frame']['e2e_wall_ms'],lookup_ms=r['frame']['admission']['lookup_wall_ms'],
            quality_pass=equal,peak_cache_bytes=r['frame']['memory']['tracked_cache_bytes']))
    evidence=hashlib.sha256(b''.join((root/name/'frames.jsonl').read_bytes() for name in runs)).hexdigest()
    amendment=(root/'cost_support_protocol.json').read_bytes() if (root/'cost_support_protocol.json').exists() else b''
    phash=hashlib.sha256((root/'protocol.json').read_bytes()+amendment).hexdigest()
    costs=[];quality=[];decisions=[]
    for samples in groups.values():
        cal=[s for s in samples if s['split']=='calibration'];held=[s for s in samples if s['split']=='heldout']
        cs=sorted({s['scene'] for s in cal});hs=sorted({s['scene'] for s in held})
        def scene_means(part):return [np.mean([s['delta_ms'] for s in part if s['scene']==scene]) for scene in sorted({s['scene'] for s in part})]
        cal_upper=upper95(scene_means(cal));held_upper=upper95(scene_means(held));reserve=max(s['lookup_ms'] for s in samples)
        domain=[min(s['key']['frame_gap_s'] for s in cal),max(s['key']['frame_gap_s'] for s in cal)] if cal else None
        in_domain=[s for s in held if domain and domain[0]<=s['key']['frame_gap_s']<=domain[1]]
        reason=None
        if len(cs)<frozen['min_calibration_scenes'] or len({s['scene'] for s in in_domain})<frozen['min_heldout_scenes']:reason='insufficient_independent_exact_domain_support'
        elif not all(s['quality_pass'] for s in samples):reason='strict_quality_failure'
        elif cal_upper+reserve>=0 or held_upper+reserve>=0:reason='scene_mean_upper_bound_not_beneficial'
        risk=float(np.mean([s['delta_ms']+reserve>0 for s in in_domain])) if in_domain else None
        if not reason and risk>frozen['max_positive_increment_rate']:reason='frozen_positive_increment_risk_exceeded'
        # Calibration support only: never expand frame gaps to force test hits.
        cost=dict(key=samples[0]['key'],cost_frame_gap_s_range=domain,range_calibrated=True,
            includes_all_incremental_costs=True,paired_increment_upper_ms=cal_upper,paired_samples=len(samples),
            independent_scene_bound_validated=not reason,peak_cache_bytes=max(s['peak_cache_bytes'] for s in samples),
            evidence_sha256=evidence,performance_model='scene_mean_ucb95_v1',
            statistical_not_realtime=True,heldout_mean_upper95_ms=held_upper,positive_increment_rate=risk)
        qual=dict(key=samples[0]['key'],cost_frame_gap_s_range=domain,range_calibrated=True,
            exact_targets_pass=all(s['quality_pass'] for s in samples),heldout_samples=len(in_domain),
            calibration_scenes=cs,heldout_scenes=hs,evidence_sha256=evidence,protocol_sha256=phash)
        if not reason:costs.append(cost);quality.append(qual)
        decisions.append(dict(approved=not reason,reason=reason or 'frozen_scene_mean_risk_pass',key=samples[0]['key'],
            calibrated_frame_gap_domain=domain,calibration_scenes=cs,heldout_scenes=hs,
            calibration_scene_mean_upper95_ms=cal_upper,heldout_scene_mean_upper95_ms=held_upper,
            worst_observed_increment_ms=max(s['delta_ms'] for s in cal) if cal else None,
            frozen_quantile_increment_ms=float(np.quantile([s['delta_ms'] for s in cal],frozen['frame_quantile'])) if cal else None,
            positive_increment_rate=risk,lookup_reserve_ms=reserve,samples=samples))
    write(root/'cost_certificate.json',dict(records=costs));write(root/'quality_certificate.json',dict(records=quality))
    write(root/'cost_calibration.json',dict(approved_keys=len(costs),decisions=decisions,source_runs=list(runs),
        protocol_sha256=phash,evidence_sha256=evidence,scope='Mean performance under fixed environment; never a deterministic deadline or physical-state certification',
        fallback='Runtime can disable overlap for a subsequent request; no same-frame replay after temporal heads advance'))
