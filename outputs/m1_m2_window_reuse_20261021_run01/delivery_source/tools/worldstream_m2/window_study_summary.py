"""Summarize observed native window-study runs; exclude complete warmup sequences."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np


def summarize(directory):
    root=Path(directory)
    status=json.loads((root/'status.json').read_text())
    if status.get('status')!='completed':
        raise ValueError('Only completed experiments may produce a final summary')
    runs=json.loads((root/'runs.json').read_text())
    quality=json.loads((root/'quality.json').read_text())
    warmup=json.loads((root/'protocol.json').read_text())['args']['warmup']
    result={}
    for mode in dict.fromkeys(run['mode'] for run in runs):
        selected=[run for run in runs if run['mode']==mode and not run['warmup']]
        frames=[frame for run in selected for frame in run['frames']]
        if not frames:continue
        diag=[frame['runtime'] for frame in frames]
        checks=[q for q in quality if q['mode']==mode and q['repeat']>=warmup]
        def latency(key):
            values=[f[key] for f in frames if f.get(key) is not None]
            return dict(p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95))) if values else None
        def count(key):
            return sum(d.get(key) or 0 for d in diag) if any(key in d for d in diag) else None
        result[mode]=dict(requests=len(frames),scenes=len(set(r['scene'] for r in selected)),
            E2E_wall_ms=latency('E2E_wall_ms'),M1_time_ms=latency('M1_time_ms'),
            candidate_block_windows=count('candidate_windows'),
            verified_exact_block_windows=count('verified_exact_windows'),
            dirty_hint_block_windows=count('dirty_hint_windows'),
            unknown_block_windows=count('unknown_windows'),
            actual_skipped_block_computations=sum(d.get('actual_skipped_block_computations',0) for d in diag),
            actual_reused_spatial_windows=sum(d.get('actual_reused_spatial_windows',0) for d in diag),
            executed_prefix_block_windows=(sum(d.get('recomputed_windows',0) for d in diag)
                if any('recomputed_windows' in d for d in diag) else None),
            fallback_reasons=dict(Counter(d.get('fallback_reason') or 'none' for d in diag)),
            prefix_fallback_reasons=dict(Counter(s.get('fallback_reason') or 'none' for d in diag
                for s in d.get('stages',[]) if 'executed_windows' in s)),
            peak_allocated_GiB=max(r['peak_gpu_allocated'] for r in selected)/1024**3,
            quality_checks=len(checks),quality_pass=all(q['frozen_targets_pass'] for q in checks) if checks else None,
            snapshot_reads=sum(bool(d.get('m1_snapshot_read')) for d in diag),
            physical_certifications=sum(sum(bool(c.get('certified')) for c in (f.get('m1_commit') or {}).get('post_perception_physical_certifications',[])) for f in frames),
            max_feature_abs=max((q['feature']['max_abs'] for q in checks),default=None))
        for key in ('M2_validation_ms','M2_execution_ms','M2_wall_ms','M1_M2_adapter_ms','M1_M2_state_read_ms'):
            values=[d[key] for d in diag if d.get(key) is not None]
            result[mode][key]=dict(p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95))) if values else None
    baseline={(run['scene'],run['repeat'],f['frame']):f['E2E_wall_ms']
        for run in runs if run['mode']=='full_maintain' and not run['warmup'] for f in run['frames']}
    reference_state={(run['scene'],run['repeat'],f['frame']):(f.get('state_epoch'),f.get('m1_anchor_ids'))
        for run in runs if run['mode']=='full_maintain' and not run['warmup'] for f in run['frames']}
    for mode,summary in result.items():
        if mode in ('original','full_maintain') or not baseline:continue
        paired=[(baseline[(run['scene'],run['repeat'],f['frame'])],f['E2E_wall_ms'])
            for run in runs if run['mode']==mode and not run['warmup'] for f in run['frames']]
        a,b=np.asarray(paired).T
        histories=[reference_state[(run['scene'],run['repeat'],f['frame'])]==(f.get('state_epoch'),f.get('m1_anchor_ids'))
            for run in runs if run['mode']==mode and not run['warmup'] for f in run['frames']]
        summary['M1_world_anchor_history_equal']=all(histories)
        summary['same_M1_full_pairing']=dict(pairs=len(paired),
            p50_improvement_percent=float(100*(np.percentile(a,50)-np.percentile(b,50))/np.percentile(a,50)),
            median_paired_delta_ms=float(np.median(a-b)),
            median_paired_improvement_percent=float(np.median(100*(a-b)/a)),
            interpretation='Descriptive paired timing; zero actual reuse cannot establish reuse acceleration')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory')
    args=p.parse_args()
    result=summarize(args.directory)
    Path(args.directory,'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
