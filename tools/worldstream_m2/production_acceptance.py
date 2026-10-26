"""Audit actual formal-route cache publication and window reuse, never approvals."""
import argparse
from collections import Counter
import json
from pathlib import Path

from tools.worldstream_m123.benchmark.common import write


def validate_rows(frames, require_reuse=True):
    errors=[]; total=executed=skipped=publications=0
    reasons=Counter()
    published=False
    previous_scene=None
    for frame in frames:
        diag=frame.get('m2') or {}
        if frame['scene_id']!=previous_scene: published=False
        previous_scene=frame['scene_id']
        if diag.get('offline_experiment',False): errors.append('offline_or_unknown_execution')
        if not diag.get('execution_completed') or not diag.get('m1_snapshot_read'):
            errors.append('incomplete_execution_or_missing_m1')
        stages=[s for s in diag.get('stages',[]) if 'total_windows' in s]
        if not stages: errors.append('missing_window_stages')
        frame_skipped=0
        for stage in stages:
            n,e,k=stage['total_windows'],stage['executed_windows'],stage.get('reused_windows',0)
            total+=n;executed+=e;skipped+=k;frame_skipped+=k
            if e+k!=n or len(set(stage.get('executed_window_ids',[])))!=e:
                errors.append('window_accounting_mismatch')
            if k:
                if not published: errors.append('reuse_without_prior_formal_publication')
            if stage.get('planned_window_ids')!=stage.get('executed_window_ids'):
                errors.append('plan_execution_mismatch')
            if not all(stage.get('selection_consistency',{}).values()):
                errors.append('invalid_selection_closure')
        if frame_skipped!=diag.get('actual_skipped_windows'): errors.append('actual_skipped_count_mismatch')
        if diag.get('cache_saved'):
            publications+=1;published=True
        else:
            published=False
            reasons.update(['cache_not_published'])
    if not frames: errors.append('no_frames')
    if require_reuse and not skipped: errors.append('no_actual_window_reuse')
    return dict(status='PASS' if not errors else 'FAIL',errors=dict(Counter(errors)),frames=len(frames),
        total_windows=total,executed_windows=executed,reused_windows=skipped,
        window_reuse_ratio=skipped/total if total else None,formal_cache_publications=publications,
        publication_rejections=dict(reasons),scope='Actual formal execution only; nonzero reuse is not a quality/speed guarantee')


def validate_run(run_dir, output):
    run=Path(run_dir)
    args=json.loads((run/'invocation.json').read_text())
    if args['condition']!='B3' or args.get('offline_wsgrr'):
        raise ValueError('Acceptance requires the actual formal B3 worker')
    frames=[json.loads(s) for s in (run/'per_frame_metrics.jsonl').read_text().splitlines()]
    sequence=json.loads((run/'sequence.json').read_text())
    if [r['frame_id'] for r in frames]!=sequence['tokens']: raise ValueError('Incomplete frame coverage')
    result=validate_rows(frames)
    result.update(run=str(run.resolve()))
    write(output,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=validate_run(args.run,args.output)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='PASS' else 1)
