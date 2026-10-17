"""Verify completed GPU artifacts after an independently recorded exit fault."""
import hashlib,json
from pathlib import Path
from .audit import write
from .v2_checks import rows,check


def main():
    root=Path('outputs/system_optimization_v2');verified={}
    for name in ('single','multi','long_stream','stride_refresh','auto_validation'):
        path=root/name;status=json.loads((path/'status.json').read_text());assert status['status']=='completed' and status['weights_unchanged']
        protocol=json.loads((path/'protocol.json').read_text());args=protocol['args'];data=rows(path/'frames.jsonl')
        expected=args['scenes']*args['frames']*len(args['modes'])*(args['warmup']+args['repeats'])
        assert len(data)==expected and all(r['frame']['status']=='completed' for r in data)
        if any(r['mode']=='m1' for r in data):assert check(path,required=False)['passed']
        verified[name]=dict(expected_rows=expected,actual_rows=len(data),weights_unchanged=True,
            frame_file_sha256=hashlib.sha256((path/'frames.jsonl').read_bytes()).hexdigest())
    for name,key in [('formal_probes/overlap_lifecycle/results.json','passed'),
            ('long_state_supply/state_supply_results.json','weights_unchanged')]:
        record=json.loads((root/name).read_text());assert record[key] is True
    backend=json.loads((root/'formal_probes/decode_backend_results.json').read_text())
    assert len(backend['quality'])==4 and all(r['complete_token_equality'] and r['cpu_rng_equal'] and r['cuda_rng_equal'] for r in backend['quality'])
    profiles=json.loads((root/'formal_probes/profile_manifest.json').read_text())
    assert len(profiles)==3 and all(Path(r['path']).stat().st_size>0 for r in profiles)
    batch=json.loads((root/'batch_probes/batch_quality_results.json').read_text())
    assert all(batch['sampling_fallback']['complete_token_equality'])
    write(root/'formal_process_failure.json',dict(process_exit_code=139,
        phase='After the complete GPU stage marker; no native stack trace available',
        user_reported_hardware_cause='Faulty physical CPU cores 4/5; subsequent execution uses taskset -c 0-3,6-31',
        raw_compute_artifacts_eligible=True,eligibility_reason='All expected rows, weight checks, strict state gates, lifecycle controls and standalone probes exist and pass',
        original_process_not_marked_successful=True,verified_runs=verified))
    print('Verified complete GPU artifacts; preserve exit 139 separately and resume offline analysis')


if __name__=='__main__':main()
