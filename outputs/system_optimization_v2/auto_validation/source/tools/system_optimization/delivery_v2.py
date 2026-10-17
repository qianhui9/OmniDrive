"""Preserve source patches and independently verify unchanged old artifacts."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from .audit import write


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(32*1024**2),b''):digest.update(chunk)
    return digest.hexdigest()


def main(full=True):
    root=Path('outputs/system_optimization_v2');old=json.loads((root/'previous_results_manifest.json').read_text())
    verified={p:sha(p)==value for p,value in old['sha256'].items()}
    if not all(verified.values()):raise AssertionError('A previous result artifact changed: '+str(verified))
    manifest=json.loads((root/'model_manifest.json').read_text());checkpoint=Path(manifest['checkpoint'])
    weight_check=dict(path=str(checkpoint),size_equal=checkpoint.stat().st_size==manifest['bytes'],
        content_sha256_equal=sha(checkpoint)==manifest['sha256'] if full else None,
        note='Full file rehash' if full else 'Quick source audit; does not replace actual run parameter-version checks')
    if not weight_check['size_equal'] or weight_check['content_sha256_equal'] is False:raise AssertionError('Checkpoint changed')
    scopes=['worldstream','tools/system_optimization','tests/worldstream_m1','tests/worldstream_m2','tests/worldstream_m3',
        'projects/mmdet3d_plugin/models/detectors/petr3d.py','docs/worldstream_system_optimization.md']
    changed=subprocess.check_output(['git','diff','--name-only','--',*scopes],text=True).splitlines()
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',*scopes],text=True).splitlines()
    files=sorted(set(changed+untracked));source={}
    for name in files:
        path=Path(name)
        if not path.is_file():continue
        dest=root/'delivery_source'/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
        source[name]=sha(path)
    patch=subprocess.check_output(['git','diff','--binary','--',*scopes])
    for name in untracked:
        result=subprocess.run(['git','diff','--no-index','--binary','--','/dev/null',name],capture_output=True)
        if result.returncode not in (0,1):raise RuntimeError(result.stderr.decode())
        patch+=result.stdout
    (root/'implementation.patch').write_bytes(patch)
    probe_sources={}
    for path in root.glob('*_probes/executed_source.json'):
        record=json.loads(path.read_text());target=Path(record['path'])
        matches=target.is_file() and sha(target)==record['sha256']
        if matches:(path.parent/'executed_probe.py').write_bytes(target.read_bytes())
        archived=path.parent/'executed_probe.py'
        probe_sources[str(path)]=dict(matches_current=matches,actual_sha256=record['sha256'],
            archived_source_matches=archived.exists() and sha(archived)==record['sha256'])
    write(root/'delivery_manifest.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        baseline_head=old['head'],previous_results_unchanged=verified,checkpoint=weight_check,
        source=source,late_loaded_probe_sources=probe_sources,patch_sha256=sha(root/'implementation.patch')))
    print('Delivery:',len(source),'source files;',len(verified),'old artifacts unchanged; checkpoint',weight_check)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true');main(not p.parse_args().quick)
