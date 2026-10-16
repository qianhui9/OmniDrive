"""Sequential after main pipeline: offline probes, gate overhead and GPU checks."""
import json
import gc
import subprocess
import torch
from .common import ROOT,write


@torch.no_grad()
def main():
    from .numerical_shadow import main as shadow
    from .approximate import main as approximate
    if not (ROOT/'output_layout_profile.json').exists():shadow()
    gc.collect();torch.cuda.empty_cache()
    if not (ROOT/'approximate_probe_results.json').exists():approximate()
    gc.collect();torch.cuda.empty_cache()
    from .pipeline_probe import main as pipeline_probe
    if not (ROOT/'carrier_profile.json').exists():pipeline_probe()
    gc.collect();torch.cuda.empty_cache()
    from worldstream.m2.cheap_admission import CheapAdmission
    from worldstream.m2.profiler import measure
    torch.set_num_threads(1);torch.manual_seed(9011)
    base=torch.randn(6,3,640,640,device='cuda');different=base+.01
    context=dict(scene_id='gate_cost',camera_ids=list(range(6)))
    gate=CheapAdmission();gate.check(base,context,(256,256))
    def reject():
        # Alternate actual sources so every sample differs from the previous frame.
        image=base if gate.version%2==0 else different
        return gate.check(image,context,(256,256))
    rejected=measure(reject,warmup=5,repeats=30)
    identical=measure(lambda:gate.check(base,context,(256,256)),warmup=5,repeats=30)
    write(ROOT/'cheap_gate_microbenchmark.json',dict(rejection=rejected,identical_candidate=identical,
        shape=list(base.shape),includes='GPU reads, scalar device-to-host synchronization and full owned snapshot copy',
        admission='sampling never accepts; candidate continues full certification'))
    completed=subprocess.run(['/home/qh/miniconda3/envs/OmniDrive/bin/python','-m','pytest',
        'tests/worldstream_m1','tests/worldstream_m2','-q','--junitxml='+str(ROOT/'logs/unit_tests.xml')],
        stdout=(ROOT/'logs/unit_tests.txt').open('w'),stderr=subprocess.STDOUT)
    if completed.returncode:raise RuntimeError('Final CUDA tests failed')
    from .report import main as report
    report()
    from .plot import main as plot
    plot()
    write(ROOT/'completion.json',dict(status='completed'))

if __name__=='__main__':main()
