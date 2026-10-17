"""Guarded, deferred cost-key revision after the loaded formal program exits.

This changes key coverage and timing descriptions, not model computations.
Its fresh calibration is mandatory; initial evidence remains archived.
"""
import argparse,difflib,hashlib,json
from pathlib import Path
from .audit import write

ROOT=Path('outputs/system_optimization_v2')


def prepared():
    runtime=Path('worldstream/runtime.py');old=runtime.read_text()
    line='functions=[type(self).execute,type(self).admit,type(self).workload_key,M1Worker.submit,M1Worker.join,model.simple_test]'
    replacement='functions=[type(self).execute,type(self).admit,type(self).workload_key,type(self).identity,type(self).m1_signature_suffix,M1Worker.submit,M1Worker.join,model.simple_test]\n        if hasattr(model,"simple_test_pts"):functions.append(model.simple_test_pts)'
    assert old.count(line)==1
    new=old.replace(line,replacement)
    line="code_hash=hashlib.sha256(b''.join(marshal.dumps(getattr(f,'__func__',f).__code__) for f in functions)).hexdigest()"
    replacement="""codes=[]
        for function in functions:
            function=getattr(function,'__func__',function);seen=set()
            while function is not None and id(function) not in seen:
                seen.add(id(function));code=getattr(function,'__code__',None)
                if code is not None:codes.append(marshal.dumps(code))
                function=getattr(function,'__wrapped__',None)
        code_hash=hashlib.sha256(b''.join(codes)).hexdigest()"""
    assert new.count(line)==1;new=new.replace(line,replacement)
    line='        return json.loads(json.dumps(key,sort_keys=True,allow_nan=False))'
    replacement="""        import numpy as np
        key['strategy']['runtime']['cpu_resources'].update(opencv_version=cv2.__version__,numpy_version=np.__version__,
            openblas_num_threads=os.environ.get('OPENBLAS_NUM_THREADS'))
        trace_parent=Path(self.config.trace_path).parent if self.config.trace_path else None
        if trace_parent is not None:
            while not trace_parent.exists():trace_parent=trace_parent.parent
        key['strategy']['runtime']['policy']=dict(profile_stages=self.config.profile_stages,
            timeout_s=self.config.m1_timeout_s,capture_cost_key=self.config.capture_cost_key,
            freshness_limit_s=self.config.max_frame_gap_s,trace_logging=bool(self.config.trace_path),
            trace_filesystem_device=os.stat(trace_parent).st_dev if trace_parent is not None else None,
            cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic)
        return json.loads(json.dumps(key,sort_keys=True,allow_nan=False))"""
    assert new.count(line)==1;new=new.replace(line,replacement)
    experiment=Path('tools/system_optimization/experiment.py');before=experiment.read_text()
    line='Preprocess/disk/collate/scatter plus synchronized VLM inference; audit CPU copies included equally; reference I/O excluded'
    replacement='Preprocess/disk/collate/scatter plus synchronized VLM inference; generated-ID CPU capture and native/runtime I/O included; post-timer bulk tensor audits, digests/snapshots and reference I/O excluded'
    assert before.count(line)==1
    test=Path('tests/worldstream_m3/test_cost_policy_scope.py')
    assert not test.exists()
    body='''from dataclasses import replace
from types import SimpleNamespace
import torch
from worldstream.runtime import UnifiedRuntime
from worldstream.m1 import M1Config,WorldStateManager


def test_measured_policy_cannot_certify_different_timeout_profiling_or_logging(tmp_path,llm):
    model=SimpleNamespace(lm_head=llm,m1_manager=WorldStateManager(M1Config()),simple_test=lambda:None)
    config=dict(model_version='weights-sha',lora_version='lora-sha',m1_overlap='experimental',profile_stages=True,
        capture_cost_key=True,trace_path=str(tmp_path/'trace.jsonl'))
    runtime=UnifiedRuntime(config)
    data=dict(img=torch.zeros(1,6,3,8,8),input_ids=[[torch.tensor([1,2])]],timestamp=torch.tensor([1.]))
    key=runtime.workload_key(model,data)
    quality=dict(key=key,exact_targets_pass=True,heldout_samples=2,calibration_scenes=['a'],heldout_scenes=['b'],
        evidence_sha256='raw',protocol_sha256='frozen')
    costs=dict(key=key,includes_all_incremental_costs=True,full_wall_ms=1000.,optimized_wall_ms=800.,peak_cache_bytes=0)
    assert runtime.admit(key,costs,quality)[0]
    for updates in (dict(profile_stages=False),dict(m1_timeout_s=.01),dict(trace_path=''),dict(capture_cost_key=False)):
        changed=UnifiedRuntime(replace(runtime.config,**updates)).workload_key(model,data)
        assert not runtime.admit(changed,costs,quality)[0]
'''
    return [(runtime,old,new),(experiment,before,before.replace(line,replacement)),(test,'',body)]


def main(dry=False):
    revision=ROOT/'policy_key_revision.json'
    if revision.exists():
        record=json.loads(revision.read_text())
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==value for p,value in record['after'].items())
        return
    changes=prepared();snapshot=json.loads((ROOT/'single/source_manifest.json').read_text())
    for path,before,after in changes:
        if before:assert hashlib.sha256(before.encode()).hexdigest()==snapshot[str(path)],'Loaded formal source changed before guarded revision'
        compile(after,str(path),'exec')
    patch=''.join(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+str(path),tofile='b/'+str(path))) for path,before,after in changes)
    (ROOT/'policy_key_revision.patch').write_text(patch)
    if dry:
        print('Deferred policy patch prepared; active inference sources unchanged');return
    assert json.loads((ROOT/'formal_status.json').read_text())['status']=='completed'
    for path,before,after in changes:
        if before:
            dest=ROOT/'baseline_source/before_policy'/path;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert dest.read_text()==before
            else:dest.write_text(before)
        path.write_text(after)
    write(revision,dict(status='applied_after_original_formal_completed',model_arithmetic_changed=False,
        fresh_cost_calibration_required=True,before={str(p):hashlib.sha256(a.encode()).hexdigest() for p,a,b in changes if a},
        after={str(p):hashlib.sha256(b.encode()).hexdigest() for p,a,b in changes}))
    print('Policy key coverage revised; original E2E evidence preserved; fresh calibration required')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dry-run',action='store_true');main(p.parse_args().dry_run)
