"""Read-only inventory; never unpickle arbitrary model objects or infos."""
import argparse
import hashlib
import json
from pathlib import Path
import pickletools
import zipfile


def inspect_file(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(block)
    result = dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())
    if path.suffix == '.pth' and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.endswith('/data.pkl')]
            if names:
                # Static opcode reading: GLOBAL/REDUCE are never executed.
                payload = archive.read(names[0])
                strings = [arg for op, arg, _ in pickletools.genops(payload)
                           if op.name in ('BINUNICODE', 'SHORT_BINUNICODE', 'UNICODE')]
                keys = [s for s in strings if isinstance(s,str) and
                        (s.endswith('.weight') or s.endswith('.bias'))]
                result['parameter_keys'] = keys
                result['metadata_strings'] = [s for s in strings if s not in keys and
                    any(x in s.lower() for x in ('config','dqtrack','tracker','mmdet','mmcv','epoch','version'))]
                result['inspection'] = 'static_pickle_opcodes_no_deserialization'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    files = [inspect_file(p) for p in sorted(Path(args.root).rglob('*')) if p.is_file()]
    with open(args.output, 'x') as handle:
        json.dump(files, handle, indent=2)


if __name__ == '__main__':
    main()
