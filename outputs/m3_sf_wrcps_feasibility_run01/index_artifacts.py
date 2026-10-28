"""Index actual raw state/tensors and final sources; protect all prior work."""
import json
from pathlib import Path
import subprocess
from tools.worldstream_m123.benchmark.common import sha256,write

root=Path(__file__).resolve().parent
base='ec4c667736a72afb2cafb1635540d9db613f333c'
raw=[p for p in root.rglob('*') if p.is_file() and (p.suffix=='.pt' or p.name=='checkpoint_validation.json'
    or p==root/'state_audit/frames.jsonl' or 'answers' in p.relative_to(root).parts)]
write(root/'raw_artifacts_sha256.json',{str(p.relative_to(Path.cwd())):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in raw})
sources=[p for folder in ('worldstream/m1','worldstream/m2','worldstream/m3','tools/worldstream_m123') for p in Path(folder).rglob('*.py')]
sources += [Path('tests/worldstream_m3/test_sf_feasibility.py'),Path('.aris/compute/local.md'),Path('.gitignore')]
write(root/'final_source_hashes.json',{str(p):sha256(p) for p in sorted(set(sources))})
protected=['worldstream/m1','worldstream/m2','worldstream/m3','tools/worldstream_m123/wrcps_train.py',
    'projects/mmdet3d_plugin/models/detectors/petr3d.py','projects/configs/OmniDrive/worldstream_m123_auto.py',
    'outputs/m3_sf_wrcps_optimization_run01','outputs/m3_semantic_core_wrcps_run01',
    'outputs/m3_wrcps_optimization_run01','outputs/m3_crossframe_e2e_optimization_run01']
changed=subprocess.check_output(['git','diff','--name-only',base,'--']+protected,text=True).splitlines()
assert not changed,changed
write(root/'protected_artifacts.json',dict(base_commit=base,protected_paths=protected,changed_paths=changed,
    formal_configuration_sha256=sha256(protected[5]),prefill_executor_sha256=sha256('worldstream/m3/prefill_executor.py')))
assert sha256(root/'semantic_qualification.json')==sha256('outputs/m3_sf_wrcps_optimization_run01/semantic_qualification.json')
assert sha256(root/'target_sequence.json')==sha256('outputs/m3_sf_wrcps_optimization_run01/target_sequence.json')
micro=json.loads((root/'controlled_micro/summary.json').read_text())
checkpoint=Path('ckpts/iter_10548.pth')
write(root/'model_artifacts.json',dict(base_checkpoint=dict(path=str(checkpoint),bytes=checkpoint.stat().st_size,
    sha256=micro['base_checkpoint_sha256'],verification='Rehashed in completed actual LLM micro'),
    semantic_qualification=dict(path=str(root/'semantic_qualification.json'),sha256=sha256(root/'semantic_qualification.json'),
        source='Unchanged prior independently validated policy, no new threshold fitting'),
    training_dataset_manifest_sha256=sha256(root/'training_dataset_manifest.json'),
    target_sequence_sha256=sha256(root/'target_sequence.json'),new_trained_SF_adapter=None,
    random_probe=dict(trained=False,seed=20261028,artifact_emitted=False,usage='K64 structure cost and explicit counterfactual only'),
    actual_optimizer_updates=0))
prefix=['taskset','-c','0-3,6-31','env','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','/home/qh/miniconda3/envs/OmniDrive/bin/python']
runroot='outputs/m3_sf_wrcps_feasibility_run01'
write(root/'commands.json',dict(command_prefix=prefix,
    executions=[dict(module='tools.worldstream_m123.sf_feasibility',stage='prepare',root=runroot,returncode=0),
        dict(module='tools.worldstream_m123.sf_feasibility',stage='audit',root=runroot,returncode=0,log='state_audit_driver.log'),
        dict(module='tools.worldstream_m123.sf_feasibility_analysis',stage='data',root=runroot,returncode=0,log='data_analysis_driver.log'),
        dict(module='tools.worldstream_m123.wrcps_train',args=['--manifest',runroot+'/training_dataset_manifest.json','--output',runroot+'/training_preflight',
            '--spatial','--stable-tokens','64','--rank','128','--epochs','3','--seed','20261028','--preflight-only'],returncode=0,log='training_preflight.log'),
        dict(module='tools.worldstream_m123.sf_feasibility_micro',root=runroot,attempt_returncodes=[1,1,0],logs=['micro_failed01.log','micro_failed02.log','micro_driver.log']),
        dict(module='tools.worldstream_m123.sf_feasibility_analysis',stage='estimate',root=runroot,returncode=0,log='estimate_driver.log'),
        dict(module='tools.worldstream_m123.sf_feasibility_report',root=runroot,returncode=0,log='report_driver.log')],
    final_cuda_regression=json.loads((root/'final_test_result.json').read_text()),
    note='Every module command uses -m MODULE --root ROOT and --stage STAGE when present; explicit training-preflight argv shown. Tool failures preserved; no optimizer training or formal E2E command executed.'))
print('Indexed',len(raw),'raw files,',len(sources),'sources; protected production and prior experiment paths unchanged.')
