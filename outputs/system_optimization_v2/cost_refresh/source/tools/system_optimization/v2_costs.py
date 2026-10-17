"""Conservative, bounded calibration from actual paired same-service frames."""
from collections import defaultdict
import hashlib,json
from pathlib import Path
from .audit import write
from .v2_checks import rows
from worldstream.runtime import UnifiedRuntime


def calibrate(root,runs=('single','long_stream')):
    root=Path(root);groups=defaultdict(list);sources=[]
    for name in runs:
        source=root/name/'frames.jsonl'
        if not source.exists():continue
        sources.append(source);data=[r for r in rows(source) if not r['warmup']]
        baseline={(r['index'],r['repeat']):r for r in data if r['mode']=='m1'}
        for row in data:
            if row['mode']!='overlap':continue
            ref=baseline[(row['index'],row['repeat'])];frame=row['frame']
            key=frame['admission']['workload_key'];normalized=dict(key);normalized.pop('frame_gap_s')
            quality=frame['quality']
            passed=quality['m1_state_digest']==ref['frame']['quality']['m1_state_digest'] and row['generated_ids']==ref['generated_ids']
            passed=passed and quality['exact_perception_memory_digest'] and all(v['max_abs']==0 and not v['shape_mismatches'] for v in quality['numerical'].values())
            groups[json.dumps(normalized,sort_keys=True)].append(dict(key=key,index=row['index'],scene=frame['identity']['scene_id'],
                split=row['split'],quality_pass=bool(passed),increment_ms=frame['e2e_wall_ms']-ref['frame']['e2e_wall_ms'],
                lookup_ms=frame['admission']['lookup_wall_ms'],peak_cache_bytes=frame['memory']['tracked_cache_bytes']))
    if not sources:raise ValueError('No actual formal paired measurements for cost calibration')
    evidence=hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()
    protocol=hashlib.sha256((root/'quality_targets.json').read_bytes()).hexdigest()
    approved_costs=[];approved_quality=[];decisions=[]
    for samples in groups.values():
        cal=[r for r in samples if r['split']=='calibration'];held=[r for r in samples if r['split']=='heldout']
        cal_scenes=sorted({r['scene'] for r in cal});held_scenes=sorted({r['scene'] for r in held})
        reason=None
        if len(cal_scenes)<2 or len(held_scenes)<2:reason='insufficient_independent_scenes_for_exact_key'
        elif not all(r['quality_pass'] for r in samples):reason='strict_quality_failed'
        bounds=[min(r['key']['frame_gap_s'] for r in cal),max(r['key']['frame_gap_s'] for r in cal)] if cal else None
        if not reason and any(not bounds[0]<=r['key']['frame_gap_s']<=bounds[1] for r in held):reason='heldout_gap_outside_calibrated_domain'
        upper=max((r['increment_ms'] for r in cal),default=None)
        if not reason and any(r['increment_ms']>upper for r in held):reason='heldout_pair_exceeds_calibrated_upper_bound'
        # Lookup is already charged in experimental E2E; retaining another
        # maximum lookup is deliberately conservative for online admission.
        overhead=max(r['lookup_ms'] for r in samples)
        if not reason and upper+overhead>=0:reason='no_conservative_net_benefit'
        common=dict(key=samples[0]['key'],cost_frame_gap_s_range=bounds,range_calibrated=True)
        costs=dict(common,includes_all_incremental_costs=True,paired_increment_upper_ms=upper,paired_samples=len(samples),
            peak_cache_bytes=max(r['peak_cache_bytes'] for r in samples),independent_scene_bound_validated=not reason,
            evidence_sha256=evidence)
        quality=dict(common,exact_targets_pass=all(r['quality_pass'] for r in samples),heldout_samples=len(held),
            calibration_scenes=cal_scenes,heldout_scenes=held_scenes,evidence_sha256=evidence,protocol_sha256=protocol)
        if not reason:
            allowed,runtime_reason=UnifiedRuntime.admit(samples[0]['key'],costs,quality,overhead)
            if not allowed:reason=runtime_reason
        if not reason:approved_costs.append(costs);approved_quality.append(quality)
        decisions.append(dict(key=samples[0]['key'],approved=not reason,reason=reason or 'independent_quality_and_conservative_paired_cost',
            calibrated_gap_range=bounds,calibration_scenes=cal_scenes,heldout_scenes=held_scenes,
            paired_increment_upper_ms=upper,reserved_admission_overhead_ms=overhead,samples=samples))
    write(root/'cost_certificate.json',dict(records=approved_costs))
    write(root/'quality_certificate.json',dict(records=approved_quality))
    result=dict(source_runs=list(runs),method='Calibration maximum per-frame paired increment; independent scene pairs must all remain below it; additional lookup reserve',
        no_extrapolation=True,approved_keys=len(approved_costs),decisions=decisions,
        caveat='Finite observed domain, not a universal worst-case real-time guarantee; no production approval outside exact deployment key',
        evidence_sha256=evidence,protocol_sha256=protocol)
    write(root/'cost_admission_results.json',result);return result
