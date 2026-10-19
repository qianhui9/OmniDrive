"""Verify the completed delivery against raw runs and current production sources."""
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from .io_utils import write_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('outputs/module1_state_interface_rearchitecture')
    required = [
        'implementation_changes.md', 'state_contract.md', 'source_provenance_audit.json',
        'validity_lease_results.json', 'refresh_contract_results.json', 'm1_candidate_quality.json',
        'm1_method_comparison.json', 'm1_m2_admission_funnel.json', 'm2_dependency_correctness.json',
        'm2_actual_reuse_results.json', 'runtime_overhead.json', 'e2e_comparison.json',
        'identity_certification_regression.json', 'original_model_parity.json',
        'regression_results.json', 'formal_go_nogo.json', 'commands.sh', 'experiment_report.md']
    missing = [name for name in required if not (root/name).is_file()]
    files = set()
    for pattern in ('worldstream/m[123]/*.py', 'tests/worldstream_m[123]/*.py',
                    'tools/worldstream_m[12]/state_interface*.py'):
        files.update(Path('.').glob(pattern))
    files.update(Path(p) for p in (
        'projects/mmdet3d_plugin/models/detectors/petr3d.py',
        'configs/worldstream_m1_source_linked_state.yaml',
        'projects/configs/OmniDrive/worldstream_m2_source_linked.py'))
    pins = {str(p): sha(p) for p in sorted(files)}
    write_json(root/'delivery_source_manifest.json', pins)
    protected = json.loads((root/'protected_source_integrity.json').read_text())
    for name in protected:
        baseline = subprocess.check_output(['git', 'show', 'HEAD:'+name])
        protected[name] = dict(sha256=sha(Path(name)),
            unchanged_from_HEAD=Path(name).read_bytes() == baseline)
    write_json(root/'protected_source_integrity.json', protected)
    replay = json.loads((root/'replay/completion.json').read_text())
    replay_match = all(Path(p).exists() and sha(Path(p)) == value
        for p, value in replay['source_sha256'].items())
    def reject_constant(value):
        raise ValueError('Non-finite JSON constant: '+value)
    json_paths = list(root.glob('*.json'))
    for path in json_paths:
        json.loads(path.read_text(), parse_constant=reject_constant)
    evidence = json.loads((root/'gpu_followup_results.json').read_text())
    gate = json.loads((root/'formal_go_nogo.json').read_text())
    regression = evidence['regression']
    cases = list(ET.parse(regression['source']).iter('testcase'))
    errors = []
    for case in ('natural', 'controlled_repeat'):
        result = evidence[case]
        if result['status'] != 'completed' or not result['all_quality_pass']:
            errors.append(case+': incomplete full-task quality evidence')
        if result['source_audit']['status'] != 'passed' or not result['all_runtime_sources_match']:
            errors.append(case+': source lineage or production-code mismatch')
    if regression['status'] != 'passed' or len(cases) != regression['passed']:
        errors.append('CUDA-visible regression incomplete')
    qualification = gate['qualification']
    if not (qualification['Qualified_Certification'] == 0 and
            qualification['Qualified_FAR'] is None and gate['decision'] == 'No-Go'):
        errors.append('Formal independent-qualification contract changed')
    result = dict(required_artifacts_present=not missing, missing=missing,
        strict_JSON=True, checked_top_level_JSON=len(json_paths),
        delivery_source_hashes_match=all(sha(Path(p)) == value for p,value in pins.items()),
        replay_source_hashes_match=replay_match,
        protected_sources_unchanged=all(v['unchanged_from_HEAD'] for v in protected.values()),
        protected_source_files=len(protected), figures=len(list((root/'figures').glob('*.png'))),
        CUDA_regression_passed=regression['passed'], CUDA_quality_and_lineage_errors=errors,
        engineering_observations=sum(len(json.loads(prop.get('value'))) for c in cases
            for prop in c.findall('./properties/property') if prop.get('name') == 'engineering_observations'))
    result['passed'] = (not missing and replay_match and result['delivery_source_hashes_match']
        and result['protected_sources_unchanged'] and result['figures'] >= 4 and not errors)
    write_json(root/'artifact_validation.json', result)
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
