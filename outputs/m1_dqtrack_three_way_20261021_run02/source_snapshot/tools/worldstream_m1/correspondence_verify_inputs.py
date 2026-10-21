"""Stream and verify all pinned nuScenes, model and prediction-cache inputs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from .correspondence_diagnostics import ROOT
from .io_utils import write_json


def check(item):
    path,expected=item;digest=hashlib.sha256()
    with open(path,'rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):digest.update(block)
    actual=digest.hexdigest()
    return path,dict(expected=expected,actual=actual,match=actual==expected)


def main():
    manifest=json.loads((ROOT/'data_manifest.json').read_text())
    inputs={**manifest['used_development_files_sha256'],**manifest['metadata_sha256'],
        manifest['checkpoint']['path']:manifest['checkpoint']['sha256']}
    cache={}
    for row in manifest['cache_provenance']:
        cache[row['path']]=row['sha256'];cache[row['provenance_path']]=row['provenance_sha256']
    for name,pins in [('data_hash_verification.json',inputs),('cache_hash_verification.json',cache)]:
        with ThreadPoolExecutor(max_workers=4) as pool:checks=dict(pool.map(check,pins.items()))
        write_json(ROOT/name,dict(files=checks,all_match=all(v['match'] for v in checks.values())))
        assert all(v['match'] for v in checks.values()),name
        print('Verified',len(checks),name,flush=True)
    from pathlib import Path
    selection=Path('outputs/module1_evidence_optimization/data_selection.json')
    spec=json.loads(selection.read_text())
    extra=[selection,Path(spec['info_file']),Path('outputs/module1_evidence_optimization/model_observations.pkl')]
    digests={}
    for p in extra:
        _,row=check((str(p),None));digests[str(p)]=row['actual']
    write_json(ROOT/'replay_input_hashes.json',dict(files=digests,
        scope='Actual replay selection, serialized calibration/poses and original-model observation objects; freshly hashed during delivery verification'))


if __name__=='__main__':main()
