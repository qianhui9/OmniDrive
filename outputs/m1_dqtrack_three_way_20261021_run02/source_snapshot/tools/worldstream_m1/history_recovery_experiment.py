"""M1.2.1 predeclared fixed cohort, followed by deterministic expansion.

Uses the existing M1 CPU replay, never a new inference runtime or executor.
"""
import argparse
import json
import pickle
from pathlib import Path
from .effective_acceptance import replay,sha,source_manifest
from .scene_protocol import select_scenes
from .dataset import log_context_from_root

VARIANTS=('baseline','old_b','a','b','c','aligned')


def freeze(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    prior=Path('outputs/m12_ego_motion_20261020/run03/frozen_protocol.json')
    old=json.loads(prior.read_text())
    with open(old['source_info'],'rb') as f:infos=pickle.load(f)['infos']
    detection='outputs/module1_evidence_optimization/model_observations.pkl'
    with open(detection,'rb') as f:outputs=pickle.load(f)
    sequences,selection=select_scenes(infos,set(outputs),0,12,log_context_from_root('data/nuscenes'))
    fixed=old['selected_scene_ids'];expanded=sorted(set(sequences)-set(fixed))[:2]
    configs={v:dict(path=f'configs/worldstream_m121_{v}.yaml',sha256=sha(f'configs/worldstream_m121_{v}.yaml')) for v in VARIANTS}
    spec=dict(frozen_before_evaluation=True,online_GT=False,qualification_population=False,
        fixed_scene_ids=fixed,expanded_scene_ids=expanded,frames=12,repeats=2,
        expansion_rule='After fixed regression, two lexicographically first available scenes outside fixed M1.2 cohort; no result/label selection',
        variants=configs,expanded_variants=['baseline','c','aligned'],
        old_fixed_protocol=str(prior),old_fixed_protocol_sha256=sha(prior),
        selection=selection,source_sha256=source_manifest(),
        full_model_target_ids=old['GPU_target_scene_ids'],
        scripts_sha256={str(p):sha(p) for p in Path('tools/worldstream_m1').glob('*.py')},
        m2_enabled=False,m3_enabled=False,execution_mode='observe_only')
    (out/'frozen_protocol.json').write_text(json.dumps(spec,indent=2)+'\n')


def run(root,cohort,variant):
    root=Path(root);spec=json.loads((root/'frozen_protocol.json').read_text())
    if spec['source_sha256']!=source_manifest():raise ValueError('Frozen online source changed')
    if variant not in spec['variants']:raise ValueError('Unknown frozen variant')
    if cohort=='expanded':
        if variant not in spec['expanded_variants']:raise ValueError('Variant not scheduled in expansion')
        for v in spec['variants']:
            s=json.loads((root/'fixed'/v/'summary.json').read_text())
            if not s['reproduced'] or not s['source_unchanged']:raise ValueError('Fixed cohort incomplete or failed reproducibility')
    config=spec['variants'][variant]
    if sha(config['path'])!=config['sha256']:raise ValueError('Frozen config changed')
    replay(root/cohort/variant,12,0,config['path'],2,spec[cohort+'_scene_ids'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['freeze','run'])
    p.add_argument('--output',required=True);p.add_argument('--cohort',choices=['fixed','expanded'])
    p.add_argument('--variant',choices=VARIANTS)
    a=p.parse_args()
    if a.action=='freeze':freeze(a.output)
    else:run(a.output,a.cohort,a.variant)


if __name__=='__main__':main()
