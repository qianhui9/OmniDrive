"""Publish formal Auto quality admission only after inspecting actual downstream evidence."""
import hashlib
import json
from .refinement_common import ROOT
from .common import write


def main():
    data=json.loads((ROOT/'e2e_latency.json').read_text())
    targets=json.loads((ROOT/'quality_targets.json').read_text())
    cases=[q for q in data['quality'] if q['controlled'] and q['mode']=='optimized' and q['actual_skipped_windows']>0]
    audit_path=ROOT/'quality_audit_results.json'
    audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    changed=[q for q in audit.get('quality',[]) if q.get('input_case')=='changed' and
             q['mode']=='optimized' and q['actual_skipped_windows']>0]
    if changed:cases=changed
    failed_counts=set();evidence=[]
    for q in cases:
        # Deployed 2-block prefix has identical 54-window closure at both stages.
        count=54-q['actual_skipped_windows']//2
        coordinates=[x['max_abs'] for x in q['map_numeric']['leaves'] if x['path']=='root[0][0]']
        failed=(q['detection'].get('unmatched',0)!=0 or
            (q['detection'].get('max_center_difference_m') or 0)>targets['detection']['max_center_difference_m'] or
            max(coordinates,default=float('inf'))>targets['map_max_control_point_difference_m'] or not q['text_equal'])
        if failed:failed_counts.add(count)
        evidence.append(dict(scene=q['scene'],frame=q['frame'],count=count,target_failed=failed))
    table=json.loads((ROOT/'break_even_results.json').read_text())
    repeat=[q for q in audit.get('quality',[]) if q.get('input_case')=='repeat' and
            q['mode']=='optimized' and q['actual_skipped_windows']>0]
    repeat_pass=bool(repeat) and all(q['feature']['max_abs']==0 and q['output_numeric']['max_abs']==0 and
        q['memory_numeric']['max_abs']==0 and not q['memory_numeric']['shape_mismatches'] and q['text_equal'] for q in repeat)
    approved=[0] if repeat_pass else []
    for record in table['records']:
        base=record.get('backbone_quality_certified',record['quality_certified'])
        record['backbone_quality_certified']=base
        # Repeated identical inputs have a separate exact equality witness. Never
        # promote untested changed-input counts merely from a small feature L2.
        record['quality_certified']=base and record['updated'] in approved
        if record['updated']!=0:
            record['quality_rejection_reason']=('downstream_regression_target_failed' if
                record['updated'] in failed_counts else 'downstream_quality_unverified')
    table['formal_quality_policy']=dict(approved_counts=approved,changed_input_sparse='experimental_only',
        reason=('Identical-input full-model checks passed; changed inputs failed targets or lack broader certification.' if repeat_pass else
                'Formal admission withheld pending identical-input full-model verification; changed input quality not certified.'),
        targets_sha256=hashlib.sha256((ROOT/'quality_targets.json').read_bytes()).hexdigest())
    write(ROOT/'break_even_results.json',table)
    write(ROOT/'formal_quality_gate.json',dict(approved_counts=approved,repeat_full_model_pass=repeat_pass,
        repeat_nonzero_frame_checks=len(repeat),failed_tested_counts=sorted(failed_counts),evidence=evidence))


if __name__=='__main__':main()
