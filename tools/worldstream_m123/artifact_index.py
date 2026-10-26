"""Index immutable local experiment artifacts after every writer has finished."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def build_index(root):
    reassessment=json.loads((root/'historical_quality_reassessment/reassessment.json').read_text())
    old=Path(reassessment['evidence_root'])
    checks={name:digest(old/name)==expected for name,expected in reassessment['evidence_sha256'].items()}
    if not all(checks.values()):raise ValueError('Historical evidence changed')
    files={};links={}
    for p in sorted(root.rglob('*')):
        if '__pycache__' in p.parts or p.name=='raw_artifacts_sha256.json' or p.suffix in ('.pyc','.tmp'):continue
        name=str(p.relative_to(root))
        if p.is_symlink():links[name]=str(p.readlink())
        elif p.is_file():files[name]=dict(bytes=p.stat().st_size,sha256=digest(p))
    output=dict(files=files,symlinks=links,historical_evidence_unchanged=checks,
        scope='Local raw artifacts including ignored large traces/tensors; symlink target directories are not recursively duplicated; index excludes itself and bytecode')
    (root/'raw_artifacts_sha256.json').write_text(json.dumps(output,indent=2))
    return len(files)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);a=p.parse_args();print('Indexed',build_index(Path(a.root)),'files')
