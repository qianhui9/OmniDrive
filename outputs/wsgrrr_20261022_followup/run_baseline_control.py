"""Exact Git-baseline core with the final SAME measurement harness.

Only experiment strategy enumeration and the old export API signature differ;
the native/candidate timing scopes, warmup, repeats, data and kernels are intact.
The complete adapted function is archived before execution for review.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
REPOSITORY=ROOT.parent.parent


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--output',default=str(ROOT/'baseline_final40_control'))
    args=parser.parse_args()
    if not args.prepare_only:
        # Keep CPU map auditing outside the independent latency measurement.
        while True:
            status_path=ROOT/'map_gt_audit_full_queue_status.json'
            status=json.loads(status_path.read_text()) if status_path.exists() else {}
            if any(value != 0 for value in status.values()):
                raise RuntimeError('Full Map audit failed; resolve it before performance measurement')
            names=('Native_full','D3_full','NoM1Region_full')
            official_done=all((ROOT/name/'official_eval/status.json').exists() and
                json.loads((ROOT/name/'official_eval/status.json').read_text()).get('status')=='COMPLETED'
                for name in names)
            if official_done and all(status.get(name)==0 for name in names):break
            time.sleep(10)
    snapshot=ROOT/'baseline_source_snapshot'
    sys.path.insert(0,str(REPOSITORY));sys.path.insert(0,str(snapshot))
    os.chdir(snapshot)
    from tools.worldstream_m2 import window_validate as harness
    import worldstream.m1.state_manager as state_manager
    import worldstream.m2.runtime as runtime
    assert Path(state_manager.__file__).is_relative_to(snapshot)
    assert Path(runtime.__file__).is_relative_to(snapshot)
    source=inspect.getsource(harness.run_wsgrr)
    start,end=source.index('    specs=('),source.index('    runtimes=')
    source=source[:start]+"    specs=[('D3','D',3)]\n"+source[end:]
    old_call='expected_context=context,computation_only=True'
    assert source.count(old_call)==2
    source=source.replace(old_call,'expected_context=context')
    adapted=ROOT/'baseline_measurement_function.py';adapted.write_text(source)
    manifest=dict(core_commit=json.loads((snapshot/'manifest.json').read_text())['commit'],
        core_modules=dict(state_manager=state_manager.__file__,runtime=runtime.__file__),
        original_harness=str(Path(harness.__file__).resolve()),
        original_harness_sha256=hashlib.sha256(Path(harness.__file__).read_bytes()).hexdigest(),
        adapted_function_sha256=hashlib.sha256(source.encode()).hexdigest(),
        adaptations=['Only D3 candidate, Native still measured every frame','Original full M1 export API: remove unsupported computation_only keyword'],
        frames=40,repeats=5,warmup=2,ordinary_budget=3,production_approvals_added=0)
    (ROOT/'baseline_control_manifest.json').write_text(json.dumps(manifest,indent=2))
    if args.prepare_only:return
    exec(compile(source,str(adapted),'exec'),harness.__dict__)
    params=SimpleNamespace(output=str(Path(args.output).resolve()),checkpoint=str(REPOSITORY/'ckpts/iter_10548.pth'),
        frames=40,scene=None,m1_predictions=str(REPOSITORY/'outputs/final_route/benchmark/run_20261021_233721/experiments/B0_full_planning/detection/pts_bbox/results_nusc.json'),
        followup=False,followup_profile=False,profile_block_costs=False,repeats=5,warmup=2,budgets=[3])
    harness.run_wsgrr(params)


if __name__=='__main__':main()
