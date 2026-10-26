"""Audit actual formal-route cache publication and window reuse, never approvals."""
import argparse
from collections import Counter
import json
from pathlib import Path

from worldstream.m2.cost_model import CostModel
from tools.worldstream_m123.benchmark.common import sha256, write


def validate_rows(frames, costs, require_reuse=True):
    errors=[]; total=executed=skipped=publications=0
    reasons=Counter()
    published=False
    previous_scene=None
    for frame in frames:
        diag=frame.get('m2') or {}
        if frame['scene_id']!=previous_scene: published=False
        previous_scene=frame['scene_id']
        if diag.get('offline_experiment') is not False: errors.append('offline_or_unknown_execution')
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
                if not stage.get('cost_admission_applied') or not stage.get('beneficial'):
                    errors.append('reuse_without_cost_admission')
                identities=[r for r in costs.records if r['block']==stage['block'] and r['updated']==e
                    and __import__('hashlib').sha256(r['key'].encode()).hexdigest()==stage.get('cost_key_sha256')]
                if len(identities)!=1 or not costs.beneficial(identities[0]['key'],e)[0]:
                    errors.append('reuse_without_matching_current_evidence')
        if frame_skipped!=diag.get('actual_skipped_windows'): errors.append('actual_skipped_count_mismatch')
        capture=diag.get('cache_capture_decision',{})
        if capture.get('policy')!='adaptive': errors.append('non_adaptive_cache_policy')
        if capture.get('published'):
            publications+=1;published=True
        else:
            published=False
            reasons.update(capture.get('rejection_reasons',[]))
    if not frames: errors.append('no_frames')
    if require_reuse and not skipped: errors.append('no_actual_window_reuse')
    return dict(status='PASS' if not errors else 'FAIL',errors=dict(Counter(errors)),frames=len(frames),
        total_windows=total,executed_windows=executed,reused_windows=skipped,
        window_reuse_ratio=skipped/total if total else None,formal_cache_publications=publications,
        publication_rejections=dict(reasons),scope='Actual formal execution only; nonzero reuse is not a quality/speed guarantee')


def validate_run(run_dir, output):
    run=Path(run_dir)
    args=json.loads((run/'invocation.json').read_text())
    binding=json.loads((run/'m2_cost_binding.json').read_text())
    if args['condition']!='B3' or args.get('offline_wsgrr') or binding['kind']!='formal':
        raise ValueError('Acceptance requires the actual formal B3 worker')
    if sha256(binding['path'])!=binding['sha256']: raise ValueError('Formal cost table changed during run')
    frames=[json.loads(s) for s in (run/'per_frame_metrics.jsonl').read_text().splitlines()]
    sequence=json.loads((run/'sequence.json').read_text())
    if [r['frame_id'] for r in frames]!=sequence['tokens']: raise ValueError('Incomplete frame coverage')
    result=validate_rows(frames,CostModel(binding['path']))
    result.update(cost_table=binding,run=str(run.resolve()))
    write(output,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=validate_run(args.run,args.output)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='PASS' else 1)
