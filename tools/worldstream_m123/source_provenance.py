"""Verified source provenance for a checkout without a local Git database."""
import hashlib
import json
from pathlib import Path
import subprocess

REPOSITORY = Path(__file__).resolve().parents[2]
MANIFEST = REPOSITORY/'docs/provenance/git_detachment.json'
LAUNCHER = 'tools/worldstream_m123/frozen_main_benchmark.py'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest():
    return json.loads(MANIFEST.read_text())


def historical_source(commit, path):
    """Read an exported, SHA256-checked source; never require Git for exports."""
    entry = manifest()['historical_sources'].get(commit+':'+path)
    if entry is not None:
        source = REPOSITORY/entry['path']
        data = source.read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry['sha256'], source
        return data
    if (REPOSITORY/'.git').exists():
        return subprocess.check_output(['git', 'show', commit+':'+path], cwd=REPOSITORY)
    raise FileNotFoundError(f'Historical source not exported: {commit}:{path}')


def source_revision(repository=REPOSITORY):
    """Identify live Git revisions, or the explicit standalone source inventory."""
    git = Path(repository)/'.git'
    if git.is_file() or (git/'HEAD').is_file():
        try:
            return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                           cwd=repository, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return 'standalone-sha256:'+manifest()['source_inventory_sha256']


def validate_executed_source(path, expected):
    """Keep original execution hashes and permit only the audited launcher edit."""
    current = REPOSITORY/path
    if digest(current) == expected:
        return
    # Model, algorithm and configuration changes must always fail the old freeze.
    assert current.resolve() == (REPOSITORY/LAUNCHER).resolve(), path
    migration = manifest()['launcher_migration']
    assert migration['path'] == LAUNCHER and migration['original_sha256'] == expected, path
    assert digest(REPOSITORY/migration['original_snapshot']) == expected, path
    assert digest(current) == migration['standalone_sha256'], path
    assert digest(Path(__file__)) == migration['provenance_helper_sha256'], path
