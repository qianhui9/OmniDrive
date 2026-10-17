"""Add fixed new scenes for frozen minimum exact-stratum support, no replacement."""
import importlib,json,shutil
from .v3 import ROOT,arguments,quality
from .audit import write


def run(driver):
    protocol=json.loads((ROOT/'cost_support_protocol.json').read_text())
    archive=ROOT/'initial_cost_calibration';archive.mkdir(exist_ok=False)
    for name in ('cost_calibration.json','cost_certificate.json','quality_certificate.json','executed_cost_calibration.py'):
        shutil.copyfile(ROOT/name,archive/name)
    scenes=sorted({i['scene_token'] for i in driver.dataset.data_infos})
    if len(scenes)<148:raise ValueError('Fixed cost-support/Auto scenes unavailable; no scene replacement allowed')
    args=arguments('cost_support','single',136,8,4,['m1','overlap'])
    args.scene_splits={s:'calibration' if i<4 else 'heldout' for i,s in enumerate(scenes[136:144])}
    driver.run(args);quality(ROOT/'cost_support')
    from . import v3_costs
    importlib.reload(v3_costs)
    v3_costs.calibrate(ROOT,('main','cost_support'))
    driver.run(arguments('auto_supported','single',144,4,4,['m1','unified_auto']))
    quality(ROOT/'auto_supported')
    write(ROOT/'cost_support_status.json',dict(status='completed',original_scenes_preserved=True,
        risk_thresholds_unchanged=True,new_auto_scenes_independent=True,
        amendment='Additional fixed scenes for support; enforce originally frozen warm frame-gap domain in revised analysis; initial empty certificates preserved'))
