"""Official nuScenes tracking algorithm on explicit complete-scene SUBSET only.

Only the split scene resolver is scoped in this process. Metric algorithms,
threshold sweeps and association code are unchanged. Never a full-val score.
"""
import argparse
import json
from pathlib import Path
from unittest.mock import patch


def merge_subset(protocol,root):
    results={}
    for spec in protocol['development']:
        data=json.loads((Path(root)/spec['scene_id']/'normalized_tracking.json').read_text())
        if set(data['results'])!=set(spec['samples']):raise ValueError('Scene not complete in exported subset')
        for token,rows in data['results'].items():
            if token in results:raise ValueError('Duplicate sample across scenes')
            results[token]=[dict(row,tracking_id=spec['scene_id']+':'+row['tracking_id']) for row in rows]
    return dict(meta=dict(use_camera=True,use_lidar=False,use_radar=False,use_map=False,use_external=False),results=results)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('protocol','dqtrack-root','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--data-root',default='data/nuscenes')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(Path(a.protocol).read_text());payload=merge_subset(protocol,a.dqtrack_root)
    (out/'subset_predictions.json').write_text(json.dumps(payload,allow_nan=False))
    from nuscenes.utils.splits import create_splits_scenes
    import nuscenes.eval.common.loaders as common
    import nuscenes.eval.tracking.loaders as tracking
    from nuscenes.eval.tracking.evaluate import TrackingEval
    from nuscenes.eval.common.config import config_factory
    splits=create_splits_scenes();splits['val']=[s['name'] for s in protocol['development']]
    with patch.object(common,'create_splits_scenes',lambda:splits),patch.object(tracking,'create_splits_scenes',lambda:splits):
        evaluator=TrackingEval(config_factory('tracking_nips_2019'),str(out/'subset_predictions.json'),
            'val',str(out/'official_algorithm'),'v1.0-trainval',a.data_root,verbose=True)
        if set(evaluator.sample_tokens)!=set(payload['results']):raise ValueError('Subset token validation failed')
        result=evaluator.main(render_curves=False)
    # Preserve official raw files; publish an explicitly scoped nullable summary.
    from worldstream.m1.state_types import json_value
    (out/'SUBSET_DIAGNOSTIC.json').write_text(json.dumps(json_value(dict(
        scope='OFFICIAL_ALGORITHM_SUBSET_DIAGNOSTIC_NOT_FULL_VAL',scenes=len(protocol['development']),
        frames=len(payload['results']),split_override='val scene membership only; metrics unchanged',metrics=result)),indent=2,allow_nan=False))


if __name__=='__main__':main()
