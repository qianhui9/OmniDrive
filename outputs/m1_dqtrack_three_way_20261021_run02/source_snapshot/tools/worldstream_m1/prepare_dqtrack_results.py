"""Add an audited frame contract to existing official nuScenes tracking JSON.

This does not run DQTrack. Only sample/scene metadata is read, never annotations.
"""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(results, samples, run_id, model_config, checkpoint_sha256):
    if not run_id or not model_config or len(checkpoint_sha256) != 64:
        raise ValueError('Run, verified model config and checkpoint digest are required')
    by_token = {s['token']: s for s in samples}
    if len(by_token) != len(samples):
        raise ValueError('Duplicate sample tokens')
    frames = {}
    for token, rows in results['results'].items():
        s = by_token[token]
        if any(r['sample_token'] != token for r in rows):
            raise ValueError('Row/key sample mismatch')
        frames[token] = dict(sample_token=token, scene_id=s['scene_token'],
                             timestamp=s['timestamp']/1e6)
    return dict(results, m1_manifest=dict(coordinate_system='world', size_order='wlh',
        timestamp_unit='seconds', run_id=run_id, model_config=model_config,
        checkpoint_sha256=checkpoint_sha256, frames=frames,
        frame_mapping_source='nuscenes sample.json; OmniDrive dataset sample_idx=info[token]'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',required=True)
    p.add_argument('--sample-json',required=True)
    p.add_argument('--model-config',required=True)
    p.add_argument('--checkpoint-sha256',required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--confirmed-nuscenes-world-output',action='store_true',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    result=prepare(json.loads(Path(a.results).read_text()),json.loads(Path(a.sample_json).read_text()),
                   a.run_id,a.model_config,a.checkpoint_sha256)
    result['m1_manifest']['sample_json_sha256']=hashlib.sha256(Path(a.sample_json).read_bytes()).hexdigest()
    with open(a.output,'x') as handle:json.dump(result,handle,allow_nan=False)


if __name__=='__main__':main()
