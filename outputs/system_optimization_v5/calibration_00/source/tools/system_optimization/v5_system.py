"""Fresh uninstrumented Overlap/Auto evidence; never imports v4 certificates."""
from dataclasses import replace
import json
import os
from pathlib import Path

from .audit import write
from .v5 import ROOT, sha256


def identity_probe(model):
    """Actual model metadata, controlled CPU-only probe; never an E2E result."""
    import ast
    import subprocess
    from time import perf_counter
    import numpy as np
    import torch
    import worldstream.runtime as runtime_module
    from .streaming_replay import distribution
    source=subprocess.check_output(['git','show','HEAD:worldstream/runtime.py'],text=True)
    cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='UnifiedRuntime')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='identity')
    namespace=dict(vars(runtime_module))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'v4_identity_reference','exec'),namespace)
    old=namespace['identity']
    runtime=runtime_module.UnifiedRuntime(dict(model_version='probe',lora_version='probe'))
    meta=[dict(scene_token='metadata-probe',sample_idx='metadata-probe',filename=['CAM_FRONT/probe'])]
    data=dict(img=torch.zeros(1,1,3,1,1),timestamp=torch.tensor([1.]))
    timings={'v4_identity':[],'v5_identity':[]};equal=True
    for i in range(61):
        results={}
        methods=[('v4_identity',old),('v5_identity',type(runtime).identity)]
        if i%2:methods.reverse()
        for name,function in methods:
            tick=perf_counter();results[name]=function(runtime,model,meta,data)
            if i:timings[name].append((perf_counter()-tick)*1000)
        equal=equal and results['v4_identity']==results['v5_identity']
    write(ROOT/'identity_signature_microbenchmark.json',dict(scope='Controlled CPU metadata probe on actual loaded OmniDrive model; synthetic frame metadata; not GPU/E2E speedup',
        repeated_signatures_equal=equal,latency_ms={k:distribution(v) for k,v in timings.items()},
        source_reference='git HEAD runtime identity before v5',parameter_count=sum(1 for _ in model.parameters())))
    if not equal:raise AssertionError('Signature optimization changed identity semantics')


def main():
    import cv2
    import torch
    from . import v2, v3
    from .v3_costs import calibrate
    from tools.worldstream_m2.rearchitecture.integration import load

    if set(os.sched_getaffinity(0)) & {4,5}:raise RuntimeError('Forbidden CPUs')
    torch.set_num_threads(1);cv2.setNumThreads(1);torch.backends.cuda.matmul.allow_tf32=False
    protocol=json.loads((ROOT/'experiment_protocol.json').read_text())
    # calibrate consumes protocol.json; create a byte-exact copy of the frozen
    # v5 experiment contract, not the previous observed experiment protocol.
    path=ROOT/'protocol.json'
    if path.exists():raise FileExistsError('Fresh system experiment required')
    path.write_bytes((ROOT/'experiment_protocol.json').read_bytes())
    write(ROOT/'runtime_environment_before.json',v3.environment())
    manifest=json.loads((ROOT/'model_manifest.json').read_text())
    manifest['sha256']=manifest['checkpoint_sha256']
    write(ROOT/'model_manifest.json',manifest)
    v2.ROOT=ROOT;v3.ROOT=ROOT
    model,dataset=load()
    identity_probe(model)

    class Driver(v3.Driver):
        def setup(self,model,mode,*args):
            super().setup(model,mode,*args)
            model.stream_runtime.config=replace(model.stream_runtime.config,
                profile_stages=False,capture_cost_key=mode in ('overlap','unified_auto'))

    driver=Driver(model,dataset)
    scenes=sorted({i['scene_token'] for i in dataset.data_infos})
    try:
        for split,specs in protocol['selections'].items():
            for ordinal,spec in enumerate(specs):
                name=f'{split}_{ordinal:02d}'
                modes=['m1','overlap'] if split!='final_test' else ['original','m1','overlap','unified_auto']
                if split!='final_test' and ordinal%2:modes.reverse()
                args=v2.arguments(name,'single',scenes.index(spec['scene']),1,len(spec['tokens']),modes,warmup=0)
                args.profile_stages=False;args.scene_splits={spec['scene']:split}
                print('SYSTEM',name,spec['scene'],modes,flush=True)
                driver.run(args);v3.quality(ROOT/name)
                write(ROOT/'system_progress.json',dict(split=split,last_completed=name))
            if split=='heldout':
                runs=tuple(f'{s}_{i:02d}' for s in ('calibration','heldout') for i in range(6))
                calibrate(ROOT,runs)
                print('APPROVED',json.loads((ROOT/'cost_calibration.json').read_text())['approved_keys'],flush=True)
        write(ROOT/'runtime_environment_after.json',v3.environment())
        write(ROOT/'system_weights_preservation.json',dict(sha256_after=sha256(protocol['checkpoint']),
            unchanged=sha256(protocol['checkpoint'])==manifest['sha256']))
    finally:driver.close()


if __name__=='__main__':
    write(ROOT/'system_status.json',dict(status='running'))
    try:main()
    except BaseException as error:
        write(ROOT/'system_status.json',dict(status='failed',error=repr(error)));raise
    else:write(ROOT/'system_status.json',dict(status='completed'))
