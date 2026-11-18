"""Extract matching official driver userspace libraries for this process only.

No package installation, kernel module reload, GPU reset or system symlink changes.
The CUDA and NVML libraries must match the currently loaded kernel module exactly.
"""
import argparse
from datetime import datetime
import gzip
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import urllib.request

from .run import ROOT, read, sha, write

REPOSITORY = 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/'
PACKAGES = ('libnvidia-compute-580', 'libnvidia-gpucomp-580', 'libnvidia-cfg1-580', 'nvidia-utils-580')
DEFAULT_ROOT = ROOT / 'outputs/worldstream_driver_compat'


def kernel_version():
    text = Path('/proc/driver/nvidia/version').read_text()
    match = re.search(r'\b(\d{3}\.\d+(?:\.\d+)?)\b', text)
    if not match:
        raise RuntimeError('Cannot identify loaded NVIDIA kernel module version')
    return match.group(1)


def select_packages(index, version):
    result = {}
    for block in index.split('\n\n'):
        fields = {}
        for line in block.splitlines():
            if ': ' in line and not line.startswith(' '):
                key, value = line.split(': ', 1)
                fields[key] = value
        name = fields.get('Package')
        if (name in PACKAGES and fields.get('Architecture') == 'amd64'
                and fields.get('Version', '').startswith(version + '-')):
            filename = fields['Filename'].removeprefix('./')
            if '/' in filename or not re.fullmatch(r'[A-Za-z0-9_.+~-]+\.deb', filename):
                raise RuntimeError('Unexpected package path')
            if name in result:
                raise RuntimeError('Ambiguous package version: ' + name)
            result[name] = dict(filename=filename, sha256=fields['SHA256'],
                                size=int(fields['Size']), version=fields['Version'])
    if set(result) != set(PACKAGES):
        raise RuntimeError('Official repository lacks matching packages: ' + repr(set(PACKAGES) - set(result)))
    return result


def verify(bundle):
    manifest = read(bundle / 'manifest.json')
    if manifest['kernel_version'] != kernel_version():
        raise RuntimeError('Loaded kernel module changed; prepare matching libraries again')
    for name, digest in manifest['files'].items():
        if sha(bundle / name) != digest:
            raise RuntimeError('Local compatibility library changed: ' + name)
    for name, target in manifest['symlinks'].items():
        path = bundle / name
        if not path.is_symlink() or os.readlink(path) != target:
            raise RuntimeError('Local compatibility symlink changed: ' + name)
    return manifest


def prepare(root):
    if platform.machine() != 'x86_64':
        raise RuntimeError('This repository is for Ubuntu 22.04 x86_64')
    version = kernel_version()
    if not version.startswith('580.'):
        raise RuntimeError('This compatibility bundle supports the installed 580 driver branch')
    bundle = root / version
    if (bundle / 'manifest.json').exists():
        verify(bundle)
        print('MATCHING_DRIVER_BUNDLE_VERIFIED', bundle, flush=True)
        return bundle
    bundle.mkdir(parents=True, exist_ok=True)
    print('Fetching official NVIDIA package metadata for', version, flush=True)
    with urllib.request.urlopen(REPOSITORY + 'Packages.gz', timeout=45) as response:
        index_bytes = response.read()
    (bundle / 'Packages.gz').write_bytes(index_bytes)
    packages = select_packages(gzip.decompress(index_bytes).decode(), version)
    for name, package in packages.items():
        archive = bundle / package['filename']
        if not archive.exists() or sha(archive) != package['sha256']:
            temp = archive.with_suffix('.deb.part')
            print('Downloading', name, package['size'], 'bytes', flush=True)
            with urllib.request.urlopen(REPOSITORY + package['filename'], timeout=60) as response, temp.open('wb') as handle:
                for chunk in iter(lambda: response.read(1024**2), b''):
                    handle.write(chunk)
            if temp.stat().st_size != package['size'] or sha(temp) != package['sha256']:
                raise RuntimeError('Official package SHA256/size mismatch: ' + name)
            temp.replace(archive)
        # dpkg-deb -x only extracts files; package maintainer scripts are not run.
        subprocess.run(['dpkg-deb', '-x', str(archive), str(bundle / 'extracted')], check=True)
    library = bundle / 'extracted/usr/lib/x86_64-linux-gnu'
    for name in ('libcuda.so.1', 'libnvidia-ml.so.1'):
        path = library / name
        if not path.exists() or version not in path.resolve().name or not path.resolve().is_relative_to(bundle):
            raise RuntimeError('Missing exact-version local library: ' + name)
    files, symlinks = {}, {}
    for path in (bundle / 'extracted').rglob('*'):
        relative = str(path.relative_to(bundle))
        if path.is_symlink():
            symlinks[relative] = os.readlink(path)
        elif path.is_file():
            files[relative] = sha(path)
    write(bundle / 'manifest.json', dict(kernel_version=version, repository=REPOSITORY,
        created_at=datetime.now().astimezone().isoformat(), packages=packages,
        metadata_sha256=sha(bundle / 'Packages.gz'), files=files, symlinks=symlinks,
        library_directory=str(library), bin_directory=str(bundle / 'extracted/usr/bin'),
        scope='Process environment only; no system packages installed'))
    print('MATCHING_DRIVER_BUNDLE_READY', bundle, flush=True)
    return bundle


def environment(bundle):
    manifest = verify(bundle)
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = manifest['library_directory'] + (
        ':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    env['PATH'] = manifest['bin_directory'] + ':' + env['PATH']
    env['OMNIDRIVE_DRIVER_COMPAT_MANIFEST'] = str(bundle / 'manifest.json')
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'exec'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if {4, 5} & os.sched_getaffinity(0):
        parser.error('Use taskset -c 0-3,6-31')
    os.chdir(ROOT)
    bundle = args.root.resolve() / kernel_version()
    if args.action == 'prepare':
        prepare(args.root.resolve())
    else:
        cmd = args.command[1:] if args.command[:1] == ['--'] else args.command
        if not cmd:
            parser.error('exec requires a command after --')
        env = environment(bundle)
        print('PROCESS_DRIVER_COMPAT', bundle, flush=True)
        os.execvpe(cmd[0], cmd, env)


if __name__ == '__main__':
    main()
