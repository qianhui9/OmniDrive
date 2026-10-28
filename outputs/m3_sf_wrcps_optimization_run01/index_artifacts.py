"""Index server-local raw captures and final source/protected artifacts."""
import json
from pathlib import Path
import subprocess
from tools.worldstream_m123.benchmark.common import sha256,write

root=Path(__file__).resolve().parent
files=[p for p in root.rglob('*') if p.is_file() and '__pycache__' not in str(p)]
raw=[p for p in files if p.suffix in ('.pt','.pkl') or p.name in ('m1_state.jsonl','per_frame_metrics.jsonl','checkpoint_validation.json')
     or any(part in ('answers','measured_answers') for part in p.relative_to(root).parts)]
write(root/'raw_artifacts_sha256.json',{str(p.relative_to(Path.cwd())):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in raw})
folders=['worldstream/m1','worldstream/m2','worldstream/m3','tools/worldstream_m123']
sources=[p for folder in folders for p in Path(folder).rglob('*.py')]
sources += [Path('projects/mmdet3d_plugin/models/detectors/petr3d.py'),Path('.aris/compute/local.md'),Path('.gitignore')]
write(root/'final_source_hashes.json',{str(p):sha256(p) for p in sorted(set(sources))})
base='0932a377b199e6249b87e15881c6e3c960f3cf3b'
protected=['worldstream/m1','worldstream/m2','worldstream/m3/prefill_executor.py',
    'projects/configs/OmniDrive/worldstream_m123_auto.py',
    'outputs/m3_semantic_core_wrcps_run01','outputs/m3_wrcps_optimization_run01']
changes=subprocess.check_output(['git','diff','--name-only',base,'--']+protected,text=True).splitlines()
assert not changes,changes
write(root/'protected_artifacts.json',dict(base_commit=base,protected_paths=protected,changed_paths=changes,
    formal_configuration_sha256=sha256(protected[3]),prefill_executor_sha256=sha256(protected[2]),
    old_qualification_sha256=sha256('outputs/m3_semantic_core_wrcps_run01/semantic_qualification.json')))
base_path=Path('ckpts/iter_10548.pth')
write(root/'model_artifacts.json',dict(base_checkpoint=dict(path=str(base_path),bytes=base_path.stat().st_size,
    sha256=json.loads((root/'no_core_equivalence/summary.json').read_text())['base_checkpoint_sha256'],
    verification='Rehashed during actual no-core diagnostic; unchanged base checkpoint'),
    old_adapter=dict(path='outputs/m3_semantic_core_wrcps_run01/adapter_training/adapter.pt',
        sha256=sha256('outputs/m3_semantic_core_wrcps_run01/adapter_training/adapter.pt'),used_for_new_SF_inference=False),
    new_qualification=dict(path=str(root/'semantic_qualification.json'),sha256=sha256(root/'semantic_qualification.json')),
    new_SF_adapter=None,reason='Training data admission failed before optimizer creation; no SF checkpoint exists'))
print('Indexed',len(raw),'raw artifacts;',len(sources),'source files; protected paths unchanged')
