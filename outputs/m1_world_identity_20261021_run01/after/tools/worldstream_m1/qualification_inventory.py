"""Read-only offline feasibility audit. Never emits a producer qualification.

Repeated samples and scenes in one log do not increase independent log units.
No labels are passed to online M1. A scene/log disjoint split is saved before
any scoring; known previously exposed scenes remain exploratory.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
from scipy.stats import beta


def inventory(info_path, table_root):
    with Path(info_path).open('rb') as h:
        infos=pickle.load(h)['infos']
    scenes={i['scene_token'] for i in infos}
    table=json.loads((Path(table_root)/'scene.json').read_text())
    logs={s['token']:s['log_token'] for s in table if s['token'] in scenes}
    by_log={}
    for scene,log in sorted(logs.items()):
        by_log.setdefault(log,[]).append(scene)
    log_ids=sorted(by_log)
    # Deterministic feasibility split, not a claim of unexposed validation data.
    calibration=log_ids[:max(1,len(log_ids)//5)]
    validation=log_ids[len(calibration):]
    n=len(validation)
    best=float(beta.ppf(.95,1,n)) if n else None
    return dict(info_path=str(info_path),info_sha256=hashlib.sha256(Path(info_path).read_bytes()).hexdigest(),
        frames=len(infos),scenes=len(scenes),mapped_scenes=len(logs),independent_logs=len(log_ids),
        split=dict(calibration={k:by_log[k] for k in calibration},validation={k:by_log[k] for k in validation}),
        split_use='Feasibility only; historical exposure must be excluded before qualification',
        validation_log_units=n,best_case_zero_error_upper_95=best,
        absolute_best_case_all_logs_upper_95=float(beta.ppf(.95,1,len(log_ids))) if log_ids else None,
        risk_limit=.05,minimum_zero_error_independent_units=59,
        qualification='BLOCKED',
        reasons=['No independently scored current-protocol producer qualification exists']+
            (['Insufficient independent validation logs even with zero errors'] if best is None or best>.05 else []),
        actual_error_acceptance_risk=None,actual_unscorable_acceptances=None,
        labels_present={name:(Path(table_root)/name).is_file() for name in
                        ('sample_annotation.json','instance.json','sample.json','scene.json')})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--infos',default='data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl')
    p.add_argument('--tables',default='data/nuscenes/v1.0-trainval')
    p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('Choose fresh output')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(inventory(a.infos,a.tables),indent=2)+'\n')


if __name__=='__main__':main()
