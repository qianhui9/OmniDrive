"""Conservatively freeze development/reserved scene metadata before outcomes.

A token mentioned in an old artifact is potentially exposed, including unexecuted
candidate lists. Reserved tokens require an additional history audit before they
can be described as independent. No GT/annotation table is read here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def validate_sequence(infos,spec,samples):
    if len(spec['samples'])!=len(set(spec['samples'])):raise ValueError('Duplicate samples')
    by={s['token']:s for s in samples}
    sequence=sorted((i for i in infos if i['scene_token']==spec['scene_id']),key=lambda i:i['timestamp'])[:len(spec['samples'])]
    if [i['token'] for i in sequence]!=spec['samples']:raise ValueError('Not the frozen scene prefix')
    for index,info in enumerate(sequence):
        s=by[info['token']]
        if s['scene_token']!=spec['scene_id'] or s['timestamp']!=info['timestamp']:
            raise ValueError('Token/scene/timestamp mismatch')
        if s['prev']!=(sequence[index-1]['token'] if index else ''):
            raise ValueError('Non-continuous scene prefix')
    return sequence


def freeze(scenes,samples,existing_root,output):
    from nuscenes.utils.splits import val
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    scene_by={s['token']:s for s in scenes};exposed=set();evidence={}
    for p in sorted(Path(existing_root).rglob('*')):
        if not p.is_file() or output in p.parents or p.suffix not in ('.json','.jsonl','.md','.csv'):continue
        if p.stat().st_size>8*1024*1024:continue
        if not any(k in p.name for k in ('protocol','selection','summary','report','scene','provenance','evaluation')):continue
        found=set(re.findall(r'\b[0-9a-f]{32}\b',p.read_text(errors='replace')))&set(scene_by)
        if found:exposed.update(found);evidence[str(p)]=sorted(found)
    vs={s['token'] for s in scenes if s['name'] in val}
    key=lambda token:hashlib.sha256(('dqtrack-policy-v1:'+token).encode()).hexdigest()
    dev=sorted(vs&exposed,key=key)[:4]
    reference='01452fbfbf4543af8acdfd3e8a1ee806'
    if reference in vs&exposed and reference not in dev:dev=[reference]+dev[:3]
    reserved=sorted(vs-exposed,key=key)[:4];by={}
    for sample in samples:by.setdefault(sample['scene_token'],[]).append(sample)
    def specs(tokens):
        return [dict(scene_id=t,name=scene_by[t]['name'],samples=[s['token'] for s in sorted(by[t],key=lambda s:s['timestamp'])]) for t in tokens]
    protocol=dict(version=1,created_before_new_results=True,development=specs(dev),holdout_reserved=specs(reserved),
        exposed_scene_ids=sorted(exposed),exposure_evidence=evidence,
        exposure_scope='conservative metadata mentions, not proof each scene was used for tuning',
        independent_status='reserved candidates only; independent history audit still required',full_validation_authorized=False)
    path=output/'protocol.json';path.write_text(json.dumps(protocol,indent=2))
    (output/'protocol.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')
    candidates=output/'omnidrive';candidates.mkdir()
    (candidates/'data_selection.json').write_text(json.dumps(dict(info_file='data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl',splits={'exposed_development':protocol['development']}),indent=2))
    return protocol


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tables',default='data/nuscenes/v1.0-trainval')
    p.add_argument('--existing-outputs',default='outputs');p.add_argument('--output',required=True)
    a=p.parse_args();tables=Path(a.tables)
    freeze(json.loads((tables/'scene.json').read_text()),json.loads((tables/'sample.json').read_text()),a.existing_outputs,a.output)


if __name__=='__main__':main()
