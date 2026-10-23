"""Verify archived completion and provenance without modifying experiment data."""
from pathlib import Path
import hashlib
import json
import subprocess
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent.parent

def read(path):
    return json.loads(path.read_text()) if path.exists() else {}

def main():
    checks={}
    for name in ('Native_full','D3_full','NoM1Region_full'):
        inference=read(ROOT/name/'status.json');official=read(ROOT/name/'official_eval/status.json')
        checks[name+'/inference']=inference.get('status')=='COMPLETED' and inference.get('measured_frames')==6019
        checks[name+'/official']=official.get('status')=='COMPLETED' and official.get('frames')==6019 and official.get('full_validation') is True
    for name in ('tests_gpu_final','tests_gpu_final_verification','executor_regression','executor_regression_baseline','baseline_control'):
        checks[name]=read(ROOT/(name+'_status.json')).get('returncode')==0
    queue=read(ROOT/'baseline_task_queue_status.json')
    checks['baseline_task_queue']=all(queue.get(name)==0 for name in ('BaselineNative_r0','BaselineD3_r0','BaselineNative_r1','BaselineD3_r1','BaselineD3_official_subset'))
    checks['map_gt_audit40']=all(read(ROOT/name/'map_gt_audit40/status.json').get('status')=='COMPLETED' for name in ('Native_r0','D3_r0','NoM1Region_r0','NoM1RegionMatched_r0_retry01'))
    checks['map_gt_audit_full']=all(read(ROOT/name/'map_gt_audit_full/status.json').get('status')=='COMPLETED' for name in ('Native_full','D3_full','NoM1Region_full'))
    checks['checkpoint_unchanged']=read(ROOT/'checkpoint_recheck.json').get('unchanged') is True
    timeline=read(ROOT/'core_source_timeline_verification.json').get('core_files',{})
    checks['core_source_unchanged']=bool(timeline) and all(hashlib.sha256((REPO/name).read_bytes()).hexdigest()==value['sha256'] for name,value in timeline.items())
    protected=read(ROOT/'protected_hash_check.json')
    for name,value in protected.get('files',{}).items():
        current=hashlib.sha256((REPO/name).read_bytes()).hexdigest()
        value.update(current=current,unchanged=current==value['initial'])
    protected['changed']=[name for name,value in protected.get('files',{}).items() if not value['unchanged']]
    (ROOT/'protected_hash_check.json').write_text(json.dumps(protected,indent=2)+'\n')
    checks['protected_files_unchanged']=not protected['changed']
    modified=subprocess.check_output(['git','diff','--name-only'],cwd=REPO,text=True).splitlines()
    added=['docs/worldstream_m2_followup.md','tests/worldstream_m2/test_task_diagnostics.py']+[str(p.relative_to(REPO)) for p in (REPO/'tools/worldstream_m2').glob('*.py') if not subprocess.run(['git','ls-files','--error-unmatch',str(p.relative_to(REPO))],cwd=REPO,capture_output=True).returncode==0]
    manifest=read(ROOT/'final_implementation_manifest.json')
    manifest['files']={name:hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in sorted(set(modified+added))}
    manifest['experiment_helpers']={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    manifest['updated_utc']=datetime.now(timezone.utc).isoformat()
    (ROOT/'final_implementation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    result=dict(status='COMPLETED' if all(checks.values()) else 'INCOMPLETE',checks=checks,scope='Required archived runs and provenance only; not research quality/speed approval',uncompleted_research_goals=['Full VLM validation and independently adjudicated semantic error rates','Stable net speedup vs Native','Detection accuracy preservation','Bitwise numerical equivalence of selected-pointwise v2'],updated_utc=datetime.now(timezone.utc).isoformat())
    (ROOT/'delivery_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
