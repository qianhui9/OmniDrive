"""Final engineering delivery checks; research No-Go is never rewritten as Pass."""
import hashlib
import json
from pathlib import Path
import subprocess
from .io_utils import write_json
from .state_interface_gpu_report import read,junit,summarize


def main():
    root=Path('outputs/module1_source_state_refinement');gpu=root/'gpu_validation_v2'
    required=['implementation_changes.md','source_scope_validity.json','observation_link_recovery.json',
        'change_first_refresh.json','camera_selective_invalidation.json','candidate_quality.json',
        'm1_method_comparison.json','m1_m2_admission_funnel.json','runtime_breakdown.json',
        'm2_exact_reuse_results.json','e2e_comparison.json','regression_results.json',
        'original_model_parity.json','formal_go_nogo.json','commands.sh','experiment_report.md']
    missing=[name for name in required if not (root/name).exists()]
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    protected=['worldstream/m1/identity_temporal.py','worldstream/m1/identity_object.py',
        'worldstream/m1/physical_certifier.py','worldstream/m1/state_store.py',
        'worldstream/m2/window_executor.py','worldstream/m2/packing.py']
    protected+=subprocess.check_output(['git','ls-files','worldstream/m3/*.py',
        'projects/mmdet3d_plugin/models/**/*.py'],text=True).splitlines()
    protected=sorted(set(protected)-{'projects/mmdet3d_plugin/models/detectors/petr3d.py'})
    integrity={}
    for name in protected:
        baseline=subprocess.check_output(['git','show','HEAD:'+name])
        integrity[name]=dict(sha256=sha(Path(name)),unchanged_from_HEAD=Path(name).read_bytes()==baseline)
    write_json(root/'protected_source_integrity.json',integrity)
    files=set()
    for pattern in ('worldstream/m[123]/*.py','tests/worldstream_m[123]/*.py',
                    'tools/worldstream_m[12]/source_state_refinement*.py'):
        files.update(Path('.').glob(pattern))
    files.update(map(Path,('projects/mmdet3d_plugin/models/detectors/petr3d.py',
        'configs/worldstream_m1_source_state_refined.yaml',
        'projects/configs/OmniDrive/worldstream_m2_source_state_refined.py')))
    write_json(root/'delivery_source_manifest.json',{str(p):sha(p) for p in sorted(files)})
    complete=read(root/'replay_final/completion.json',{})
    replay_pins=complete.get('source_sha256',{})
    replay_match=bool(replay_pins) and all(Path(p).exists() and sha(Path(p))==value for p,value in replay_pins.items())
    natural=summarize(gpu/'natural');repeat=summarize(gpu/'repeat')
    current_natural=summarize(gpu/'current_source_recheck');current_repeat=summarize(gpu/'current_repeat_recheck')
    regression=junit(gpu/'regression_current_cuda.xml')
    errors=[]
    for name,result in (('primary natural',natural),('primary repeat',repeat),
            ('current natural',current_natural),('current repeat',current_repeat)):
        if result['status']!='completed' or not result['all_quality_pass']:errors.append(name+': quality incomplete')
        if result['source_audit']['status']!='passed':errors.append(name+': lineage audit failed/incomplete')
    for name,result in (('current natural',current_natural),('current repeat',current_repeat)):
        if not result['all_runtime_sources_match']:errors.append(name+': final code hashes differ')
    if regression['status']!='passed':errors.append('Final CUDA regression incomplete')
    if not replay_match:errors.append('Final replay source hashes differ')
    if not all(v['unchanged_from_HEAD'] for v in integrity.values()):errors.append('Protected code changed')
    gate=read(root/'formal_go_nogo.json',{});q=gate.get('qualification',{})
    if not (gate.get('decision')=='No-Go' and q.get('Qualified_Certification')==0 and q.get('Qualified_FAR') is None
            and all(q.get(k)=='Not Run' for k in ('Calibration','Heldout','Final_Test'))):
        errors.append('Independent qualification contract changed')
    def reject_constant(value):raise ValueError('Non-finite JSON constant '+value)
    for path in root.glob('*.json'):json.loads(path.read_text(),parse_constant=reject_constant)
    window=read(gpu/'natural/window_scope_probe.json',{})
    if window.get('status')!='completed' or not window.get('records'):errors.append('Real EVA window scope probe incomplete')
    scoped=read(root/'replay_final/scope_validity.json',[])
    refined_methods={'scope_validity','link_recovery','change_first','optimized'}
    # The source_linked control intentionally retains the historical any-pixel
    # boolean and has no scope ratios. Audit the new interface independently.
    false_scope=[s for s in scoped if s['method'] in refined_methods and s.get('observation_valid') and
        (s.get('observation_support_ratio') is None or s['observation_support_ratio']<1)]
    if false_scope:errors.append('Partial scope incorrectly marked complete')
    # Ensure the actual final-model recheck reaches receipt-only source recovery.
    receipt_only=[]
    for run in read(gpu/'current_repeat_recheck/runs.json',[]):
        for frame in run['frames']:
            receipt_only.extend(s for s in (frame.get('m1_computation_state') or {}).get('source_records',[])
                if s.get('source_representation')=='committed_M2_receipt_without_M1_feature_slot')
    if not receipt_only:errors.append('Final real-model receipt-only source recovery was not exercised')
    figure_count=len(list((root/'figures').glob('*.png')))
    if figure_count<6:errors.append('Required six figures incomplete')
    result=dict(passed=not missing and not errors,missing=missing,errors=errors,
        final_CUDA_regression=regression,final_replay_source_hashes_match=replay_match,
        final_replay_frames=complete.get('frames'),final_replay_camera_observations=complete.get('camera_observations'),
        refined_scope_overclaims=len(false_scope),scope_methods_audited=sorted(refined_methods),
        real_window_scope_records=len(window.get('records',[])),receipt_only_sources_exercised=len(receipt_only),
        protected_source_files=len(integrity),figures=figure_count,
        primary_quality_checks=dict(natural=sum(v['checks'] for v in natural['quality'].values()),
            repeat=sum(v['checks'] for v in repeat['quality'].values())),
        current_quality_checks=dict(natural=sum(v['checks'] for v in current_natural['quality'].values()),
            repeat=sum(v['checks'] for v in current_repeat['quality'].values())),
        research_decision=gate.get('decision'),M1_freeze=gate.get('M1_freeze'))
    write_json(root/'artifact_validation.json',result)
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
