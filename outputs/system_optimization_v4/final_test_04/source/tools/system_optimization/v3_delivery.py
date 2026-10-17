"""Final immutable-baseline and full checkpoint-content verification."""
import hashlib,json,subprocess
from pathlib import Path
from .audit import write
from .delivery_v2 import sha
from .v3 import ROOT


def main():
    old=json.loads((ROOT/'previous_results_manifest.json').read_text())
    previous={p:sha(p)==value for p,value in old['sha256'].items()}
    assert all(previous.values()),'Old v2 artifact changed'
    weights=json.loads((ROOT/'model_manifest.json').read_text());actual=sha(weights['checkpoint'])
    assert actual==weights['sha256'],'Checkpoint content changed'
    scopes=['worldstream','tools/system_optimization','tests/worldstream_m1','tests/worldstream_m2','tests/worldstream_m3','configs',
        'projects/mmdet3d_plugin/models/detectors/petr3d.py','docs/worldstream_system_optimization.md','docs/setup.md','AGENTS.md']
    changed=subprocess.check_output(['git','diff',old['head'],'--name-only','--',*scopes],text=True).splitlines()
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',*scopes],text=True).splitlines()
    source={}
    for name in sorted(set(changed+untracked)):
        p=Path(name)
        if not p.is_file():continue
        target=ROOT/'delivery_source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes());source[name]=sha(p)
    patch=subprocess.check_output(['git','diff',old['head'],'--binary','--',*scopes])
    for name in untracked:
        r=subprocess.run(['git','diff','--no-index','--binary','--','/dev/null',name],capture_output=True)
        assert r.returncode in (0,1);patch+=r.stdout
    (ROOT/'implementation.patch').write_bytes(patch)
    report=json.loads((ROOT/'report_manifest.json').read_text());assert report['status']=='completed'
    assert 'passed' in (ROOT/'logs/final_gpu_tests.log').read_text()
    write(ROOT/'delivery_manifest.json',dict(status='completed',baseline_head=old['head'],source=source,
        previous_results_unchanged=previous,checkpoint_sha256=actual,checkpoint_content_unchanged=True,
        report_sha256=sha(ROOT/'experiment_report.md'),report=report,
        gpu_status=json.loads((ROOT/'gpu_status.json').read_text()),patch_sha256=sha(ROOT/'implementation.patch')))
    print('Verified',len(source),'source files;',len(previous),'old artifacts unchanged; full checkpoint SHA256 unchanged')


if __name__=='__main__':main()
