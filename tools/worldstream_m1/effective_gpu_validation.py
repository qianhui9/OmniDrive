"""Final full Planning validation through the existing native model entry point.

Adds source/checkpoint pins, language-file comparison and independent M1 log
verification. Does not alter model, generation settings, M2 or M3 mechanisms.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from .effective_acceptance import sha, verify


def source_manifest():
    paths=[*Path('worldstream/m1').glob('*.py'),
           Path('projects/mmdet3d_plugin/models/detectors/petr3d.py'),
           Path('configs/worldstream_m1_effective.yaml'),
           Path('projects/configs/OmniDrive/worldstream_m1_effective.py'),
           Path('tools/worldstream_m1/model_smoke.py'),Path(__file__),
           Path('tools/worldstream_m1/effective_acceptance.py')]
    return {str(p):sha(p) for p in paths}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--frames',type=int,default=8);p.add_argument('--scene')
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    write=lambda name,value:(root/name).write_text(json.dumps(value,indent=2)+'\n')
    manifest=source_manifest();write('source_before.json',manifest)
    from mmcv import Config
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m1_effective.py')
    checkpoint_paths=[Path('ckpts/iter_10548.pth'),*sorted(Path('ckpts/pretrain_qformer').glob('*'))]
    checkpoints={}
    for path in checkpoint_paths:
        if not path.is_file():continue
        print('Hashing',str(path),flush=True)
        checkpoints[str(path)]=dict(bytes=path.stat().st_size,sha256=sha(path))
        write('checkpoint_digests.json',checkpoints)
    print('Checkpoint hashes completed; starting full model',flush=True)
    write('provenance.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        affinity=sorted(os.sched_getaffinity(0)),python=sys.executable,
        checkpoint=checkpoints,dataset=cfg.data.test.ann_file,dataset_sha256=sha(cfg.data.test.ann_file),
        model_config_sha256=__import__('hashlib').sha256(cfg.pretty_text.encode()).hexdigest(),
        population='Exploratory native configured dataset; no independent qualification or quality claim',
        m2_enabled=False,m3_enabled=False,planning_task_unchanged=True))
    from . import model_smoke
    old=sys.argv
    sys.argv=['model_smoke','--config','projects/configs/OmniDrive/worldstream_m1_effective.py',
              '--m1-config','configs/worldstream_m1_effective.yaml','--compare','--frames',str(a.frames),'--output',str(root)]
    if a.scene:sys.argv+=['--scene',a.scene]
    try:model_smoke.main()
    finally:sys.argv=old
    baseline=root/'baseline/language';observed=root/'observe_only/language'
    names=sorted(p.name for p in baseline.iterdir())
    comparisons=[dict(frame=n,equal=(baseline/n).read_bytes()==(observed/n).read_bytes(),
                      baseline_sha256=sha(baseline/n),observed_sha256=sha(observed/n)) for n in names]
    language_equal=names==sorted(p.name for p in observed.iterdir()) and all(c['equal'] for c in comparisons)
    write('language_comparison.json',dict(actual_language_files_equal=language_equal,frames=len(names),files=comparisons))
    write('m1_independent_acceptance.json',verify(root/'m1_state.jsonl'))
    after=source_manifest();write('source_after.json',after)
    write('final_verification.json',dict(sources_unchanged=manifest==after,language_equal=language_equal,
        native_model_parity=json.loads((root/'results.json').read_text())['observe_only_equal']))
    assert manifest==after and language_equal


if __name__=='__main__':main()
