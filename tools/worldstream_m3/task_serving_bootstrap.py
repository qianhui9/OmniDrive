"""Create a fresh reproduction directory; never overwrite prior experiments."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile


def main():
    root=Path(os.environ.get('OMNIDRIVE_TASK_SERVING_OUTPUT','outputs/m2_freeze_m3_task_serving'))
    root.mkdir(exist_ok=False)
    original=Path('outputs/m2_freeze_m3_task_serving')
    for name in ('commands.sh','requirements.txt'):
        if (original/name).is_file():(root/name).write_bytes((original/name).read_bytes())
    files=[*Path('worldstream').rglob('*.py'),Path('configs/worldstream_m1_engineering_frozen.yaml'),*Path('projects/mmdet3d_plugin').rglob('*.py')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (root/'baseline.json').write_text(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),sha256=hashes),indent=2))
    with tarfile.open(root/'baseline_source.tar.gz','w:gz') as archive:
        for p in files:archive.add(p)

if __name__=='__main__':main()
