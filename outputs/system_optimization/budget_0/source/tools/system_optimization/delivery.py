"""Freeze delivery diffs and disclose per-run source revisions without rewriting runs."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
from .audit import write


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root='outputs/system_optimization'):
    root=Path(root)
    tracked=subprocess.check_output(['git','diff','--name-only'],text=True).splitlines()
    extra=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],text=True).splitlines()
    changed=sorted(p for p in set(tracked+extra) if not p.startswith('outputs/') and Path(p).is_file())
    patches={key:[] for key in ('BC','DE')}
    changelog=['# 插桩与代码变更清单','','旧模块实验不改写；profile_stages=False 关闭详细插桩；不配置 stream_config 保留原入口。',
        '时间类型与开销边界详见 docs/worldstream_system_optimization.md；实际运行版本见 delivery_source_audit.json。','',
        '| 文件 | 类型 | 首个修改行 |','|---|---|---:|']
    for name in changed:
        if name in tracked:
            diff=subprocess.check_output(['git','diff','--',name],text=True)
            match=re.search(r'^@@ .*?\+(\d+)',diff,re.M)
            line=int(match.group(1)) if match else 1
            kind='modified'
        else:
            result=subprocess.run(['git','diff','--no-index','--','/dev/null',name],capture_output=True,text=True)
            if result.returncode not in (0,1):raise RuntimeError(result.stderr)
            diff=result.stdout;line=1;kind='created'
        stage='BC' if name.startswith(('worldstream/m1/','worldstream/m2/','tests/worldstream_m1/','tests/worldstream_m2/')) else 'DE'
        patches[stage].append(diff)
        changelog.append(f'| [{name}](../../{name}) | {kind} | {line} |')
    for stage,diffs in patches.items():(root/f'stage_{stage}.patch').write_text(''.join(diffs))
    (root/'instrumentation_changelog.md').write_text('\n'.join(changelog)+'\n')
    files=list(Path('worldstream').rglob('*.py'))+list(Path('tools/system_optimization').glob('*.py'))
    files+=[Path('projects/mmdet3d_plugin/models/detectors/petr3d.py')]
    files+=[Path(p) for p in changed if p.startswith(('tests/','docs/'))]
    manifest={str(p):sha(p) for p in sorted(set(files))}
    write(root/'implementation_manifest.json',manifest)
    runs=[]
    for path in sorted(root.glob('*/source_manifest.json')):
        old=json.loads(path.read_text());status_path=path.parent/'status.json'
        runs.append(dict(run=path.parent.name,status=json.loads(status_path.read_text()) if status_path.exists() else None,
            source_manifest=str(path),manifest_sha256=sha(path),
            differs_from_delivery=[p for p,s in old.items() if manifest.get(p)!=s],
            absent_from_run=sorted(set(manifest)-set(old))))
    frozen=json.loads((root/'audit_manifest.json').read_text())
    integrity={p:Path(p).exists() and sha(Path(p))==s for p,s in frozen['sources'].items()}
    write(root/'delivery_source_audit.json',dict(runs=runs,historical_evidence_unchanged=integrity,
        notes=[
            'Run snapshots preserve loaded revisions; different SHA values are disclosed, never silently replaced.',
            'Early single/multi used old memory accounting and backbone capture; actual camera reuse was zero.',
            'Language candidate suffix labels are deterministically reclassified from explicit KV branches; raw durations remain unchanged.',
            'Late-loaded lifecycle control has its own executed_probe.py and source hash.',
            'Failed/smoke/control runs remain separate from completed formal performance results.']))
    traces=[dict(path=str(p),bytes=p.stat().st_size) for p in sorted(root.glob('*/pipeline_trace.json'))]
    write(root/'traces/index.json',dict(traces=traces,note='Original Chrome traces kept in each run; not duplicated here.'))
    assert all(integrity.values()),'Historical evidence changed'
    print(f'Delivery: {len(changed)} changed/new source files; {len(runs)} run snapshots; historical hashes all preserved')


if __name__=='__main__':main()
