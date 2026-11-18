"""Process-local driver selection must not silently mix library versions."""
import os

import pytest

from . import driver_compat as compat
from .run import sha, write


def test_selects_exact_kernel_version_and_all_required_components():
    entries = []
    for package in compat.PACKAGES:
        for version in ('580.173.02', '580.178.04'):
            entries.append(f'Package: {package}\nVersion: {version}-1ubuntu1\nArchitecture: amd64\n'
                           f'Filename: ./{package}_{version}_amd64.deb\nSize: 100\nSHA256: abc')
    selected = compat.select_packages('\n\n'.join(entries), '580.173.02')
    assert set(selected) == set(compat.PACKAGES)
    assert all(v['version'] == '580.173.02-1ubuntu1' for v in selected.values())
    with pytest.raises(RuntimeError, match='lacks matching'):
        compat.select_packages('\n\n'.join(entries[2:]), '580.173.02')


def test_rejects_duplicate_matching_packages_and_paths():
    item = 'Package: libnvidia-compute-580\nVersion: 580.173.02-1\nArchitecture: amd64\nFilename: ./ok.deb\nSize: 1\nSHA256: abc'
    with pytest.raises(RuntimeError, match='Ambiguous'):
        compat.select_packages(item + '\n\n' + item, '580.173.02')
    with pytest.raises(RuntimeError, match='Unexpected package path'):
        compat.select_packages(item.replace('./ok.deb', '../escape.deb'), '580.173.02')


def bundle(tmp_path, monkeypatch):
    library = tmp_path / 'libs'
    library.mkdir()
    binary = library / 'libcuda.so.580.173.02'
    binary.write_bytes(b'test fixture only')
    (library / 'libcuda.so.1').symlink_to(binary.name)
    write(tmp_path / 'manifest.json', dict(kernel_version='580.173.02',
          files={'libs/' + binary.name: sha(binary)}, symlinks={'libs/libcuda.so.1': binary.name},
          library_directory=str(library), bin_directory=str(tmp_path / 'bin')))
    monkeypatch.setattr(compat, 'kernel_version', lambda: '580.173.02')
    return binary


def test_environment_isolated_and_existing_paths_preserved(tmp_path, monkeypatch):
    bundle(tmp_path, monkeypatch)
    monkeypatch.setenv('LD_LIBRARY_PATH', '/existing')
    original_path = os.environ['PATH']
    env = compat.environment(tmp_path)
    assert env['LD_LIBRARY_PATH'] == str(tmp_path / 'libs') + ':/existing'
    assert env['PATH'] == str(tmp_path / 'bin') + ':' + original_path
    assert os.environ['LD_LIBRARY_PATH'] == '/existing'
    assert os.environ['PATH'] == original_path


def test_kernel_version_change_and_library_tampering_rejected(tmp_path, monkeypatch):
    binary = bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(compat, 'kernel_version', lambda: '580.178.04')
    with pytest.raises(RuntimeError, match='kernel module changed'):
        compat.environment(tmp_path)
    monkeypatch.setattr(compat, 'kernel_version', lambda: '580.173.02')
    binary.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='library changed'):
        compat.environment(tmp_path)


def test_symlink_retargeting_is_rejected(tmp_path, monkeypatch):
    bundle(tmp_path, monkeypatch)
    link = tmp_path / 'libs/libcuda.so.1'
    link.unlink()
    link.symlink_to('/usr/lib/x86_64-linux-gnu/libcuda.so.1')
    with pytest.raises(RuntimeError, match='symlink changed'):
        compat.environment(tmp_path)
