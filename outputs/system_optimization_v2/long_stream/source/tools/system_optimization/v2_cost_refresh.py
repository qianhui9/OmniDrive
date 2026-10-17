"""Fresh exact-key calibration after a coordinator policy-key revision.

Run only after the original formal stage; preserve its certificates and rows.
"""
import json,shutil
from pathlib import Path
import torch
from .audit import write
from .v2 import ROOT,Driver,arguments
from .v2_checks import check
from .v2_costs import calibrate


@torch.no_grad()
def main():
    import cv2
    torch.set_num_threads(1);cv2.setNumThreads(1);torch.backends.cuda.matmul.allow_tf32=False
    if json.loads((ROOT/'formal_status.json').read_text())['status']!='completed':raise ValueError('Original formal stage must finish first')
    archive=ROOT/'initial_cost_certificates';archive.mkdir(exist_ok=False)
    for name in ('cost_certificate.json','quality_certificate.json','cost_admission_results.json'):
        shutil.copyfile(ROOT/name,archive/name)
    write(ROOT/'cost_refresh_status.json',dict(status='starting'))
    from tools.worldstream_m2.rearchitecture.integration import load
    model,dataset=load();driver=Driver(model,dataset)
    try:
        driver.run(arguments('cost_refresh','single',102,4,4,['m1','opt_serial','overlap'],warmup=0))
        check(ROOT/'cost_refresh');calibrate(ROOT,('cost_refresh',))
        # New scenes chosen before this revision's calibration outcomes.
        driver.run(arguments('cost_refresh_auto','single',114,2,4,['m1','unified_auto'],warmup=0))
        check(ROOT/'cost_refresh_auto')
    except Exception as error:
        write(ROOT/'cost_refresh_status.json',dict(status='failed',error=repr(error)));raise
    finally:driver.close()
    write(ROOT/'cost_refresh_status.json',dict(status='completed',new_policy_key_measured=True))


if __name__=='__main__':main()
