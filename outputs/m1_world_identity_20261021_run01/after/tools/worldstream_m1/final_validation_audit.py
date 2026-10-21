"""Deterministic delivery audit, separate engineering and research gates."""
import hashlib
import json
from pathlib import Path
import subprocess
from .io_utils import write_json
from .state_interface_gpu_report import read,junit,summarize
from .final_validation_sampling import PROTOCOL

ROOT=Path('outputs/module1_final_validation_refinement')


def main():
    r=ROOT;g=r/'gpu_validation';errors=[]
    required=['implementation_changes.md','critical_event_root_cause.json','motion_risk_ablation.json',
        'candidate_physical_quality.json','candidate_sampling_protocol.json','candidate_paired_analysis.json',
        'refresh_quality_comparison.json','runtime_hotspot_analysis.json','runtime_optimization_ablation.json',
        'm1_m2_integration.json','original_model_parity.json','regression_results.json','formal_go_nogo.json',
        'm1_freeze_decision.md','commands.sh','experiment_report.md']
    missing=[n for n in required if not (r/n).exists()]
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    pins=read(r/'replay/completion.json',{}).get('source_sha256',{})
    exact=bool(pins) and all(Path(p).exists() and sha(Path(p))==v for p,v in pins.items())
    if not exact:errors.append('Replay source hashes differ')
    checks=junit(g/'regression_cuda.xml')
    if checks['status']!='passed':errors.append('CUDA suite incomplete')
    configuration_evidence={}
    for case in ('natural','repeat'):
        s=summarize(g/case)
        if not s['all_quality_pass'] or s['source_audit']['status']!='passed':errors.append(case+' full-task/source check failed')
        if not s['all_runtime_sources_match']:errors.append(case+' current runtime hashes mismatch')
        for mode,q in s['quality'].items():
            if q['text_equal']!=q['checks'] or q['planning_equal']!=q['checks'] or q['feature_max_abs']!=0 or q['memory_max_abs']!=0 or q['output_max_abs']!=0:
                errors.append(case+'/'+mode+' original outputs differ')
        decisions={}
        for run in read(g/case/'runs.json',[]):
            if run['mode']=='original':continue
            flags=set()
            for frame in run['frames']:
                commit=frame.get('m1_commit') or {}
                for proposal in commit.get('identity_continuity_proposals',[]):
                    cfg=proposal.get('measurements',{}).get('producer_config',{})
                    if cfg:flags.add((cfg.get('motion_risk_revocation'),cfg.get('measurement_reuse')))
                key=(run['scene'],run['repeat'],frame['frame'])
                value=dict(anchor_ids=frame.get('m1_anchor_ids'),association_pairs=commit.get('association_pairs'),
                    selected_prediction_indices=commit.get('selected_prediction_indices'))
                if key in decisions and decisions[key]!=value:errors.append(case+' real-model state decision mismatch '+str(key))
                decisions[key]=value
            expected=((False,True) if run['mode']=='measurement_only' else (False,False)
                if run['mode'] in ('old_source_exact','current_optimized') else (True,True))
            if flags and flags!={expected}:errors.append(case+'/'+run['mode']+' actual configuration flags mismatch')
            configuration_evidence[case+'/'+run['mode']]=sorted(flags)
    protected=subprocess.check_output(['git','ls-files','worldstream/m2','worldstream/m3',
        'projects/mmdet3d_plugin/models','worldstream/m1/identity_temporal.py','worldstream/m1/identity_object.py',
        'worldstream/m1/physical_certifier.py','worldstream/m1/state_store.py'],text=True).splitlines()
    integrity={p:Path(p).read_bytes()==subprocess.check_output(['git','show','HEAD:'+p]) for p in protected}
    if not all(integrity.values()):errors.append('Protected runtime/model/identity source changed')
    write_json(r/'protected_source_integrity.json',integrity)
    historical=subprocess.run(['git','diff','--exit-code','HEAD','--','outputs/module1_source_state_refinement',
        'outputs/module1_state_interface_rearchitecture','outputs/module1_quality_coverage_refinement'],capture_output=True)
    if historical.returncode:errors.append('Historical outputs changed')
    if read(r/'candidate_sampling_protocol.json')!=PROTOCOL:errors.append('Prespecified sampling protocol changed')
    root=read(r/'critical_event_root_cause.json',{})
    if len(root.get('events',[]))!=14:errors.append('14 failure root causes incomplete')
    if len(list((r/'critical_event_overlays').glob('paired_*.png')))!=14:errors.append('14 paired event overlays incomplete')
    if len(list((r/'figures').glob('*.png')))<4:errors.append('Required risk/runtime/funnel figures missing')
    runtime=read(r/'runtime_optimization_ablation.json',{})
    if not runtime.get('state_decisions_equal'):errors.append('Persisted state decisions differ')
    gate=read(r/'formal_go_nogo.json',{});q=gate.get('qualification',{})
    if q.get('Qualified_Certification')!=0 or q.get('Qualified_FAR') is not None or any(q.get(k)!='Not Run' for k in ('Calibration','Heldout','Final_Test')):
        errors.append('Formal qualification isolation failed')
    if gate.get('conditional_reuse_authorization')!='No-Go' or gate.get('formal_physical_certification')!='No-Go':errors.append('Unqualified authorization granted')
    def reject_constant(value):raise ValueError('Non-finite JSON constant '+value)
    for path in r.glob('*.json'):json.loads(path.read_text(),parse_constant=reject_constant)
    files=[*Path('worldstream').rglob('*.py'),*Path('tools/worldstream_m1').glob('final_validation*.py'),
        *Path('tools/worldstream_m2').glob('final_validation*.py'),Path('tests/worldstream_m1/test_final_validation.py'),
        Path('configs/worldstream_m1_final_validation.yaml'),Path('configs/worldstream_m1_engineering_frozen.yaml'),
        Path('projects/configs/OmniDrive/worldstream_m2_final_validation.py')]
    write_json(r/'delivery_source_manifest.json',{str(p):sha(p) for p in files})
    result=dict(passed=not errors and not missing,errors=errors,missing=missing,replay_hashes_match=exact,
        actual_runtime_configuration_flags=configuration_evidence,
        CUDA=checks,protected_sources=len(integrity),historical_outputs_unchanged=historical.returncode==0,
        paired_overlays=14,engineering_interface_freeze=gate.get('engineering_interface_freeze'),
        candidate_quality_freeze=gate.get('candidate_quality_freeze'),handoff_to_M2=gate.get('handoff_to_M2'))
    write_json(r/'artifact_validation.json',result);print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
