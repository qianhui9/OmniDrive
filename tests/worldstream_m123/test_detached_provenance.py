"""Git removal must preserve Native auditing and strict execution provenance."""
import hashlib

import pytest

from tools.worldstream_m123 import source_provenance as provenance


def test_exported_native_source_needs_no_git(monkeypatch):
    def unavailable(*args, **kwargs):
        raise AssertionError('Git must not be called for an exported source')
    monkeypatch.setattr(provenance.subprocess, 'check_output', unavailable)
    for key, entry in provenance.manifest()['historical_sources'].items():
        commit, path = key.split(':', 1)
        assert hashlib.sha256(provenance.historical_source(commit, path)).hexdigest() == entry['sha256']


def test_standalone_revision_uses_explicit_inventory(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise AssertionError('Git must not be called in a standalone directory')
    monkeypatch.setattr(provenance.subprocess, 'check_output', unavailable)
    assert provenance.source_revision(tmp_path) == 'standalone-sha256:'+provenance.manifest()['source_inventory_sha256']
    # Execution environments can recreate an empty .git mount point.
    (tmp_path/'.git').mkdir()
    assert provenance.source_revision(tmp_path) == 'standalone-sha256:'+provenance.manifest()['source_inventory_sha256']


def test_only_exact_audited_launcher_migration_is_accepted():
    migration = provenance.manifest()['launcher_migration']
    provenance.validate_executed_source(provenance.LAUNCHER, migration['original_sha256'])
    provenance.validate_executed_source(str(provenance.REPOSITORY/provenance.LAUNCHER),
                                        migration['original_sha256'])
    with pytest.raises(AssertionError):
        provenance.validate_executed_source(provenance.LAUNCHER, '0'*64)
    with pytest.raises(AssertionError):
        provenance.validate_executed_source('worldstream/r0.py', '0'*64)


def test_exported_snapshot_tampering_fails(tmp_path, monkeypatch):
    entry = dict(next(iter(provenance.manifest()['historical_sources'].values())))
    bad = tmp_path/'tampered.snapshot'
    bad.write_bytes(b'tampered')
    monkeypatch.setattr(provenance, 'REPOSITORY', tmp_path)
    entry['path'] = bad.name
    monkeypatch.setattr(provenance, 'manifest', lambda: {'historical_sources': {'x:y': entry}})
    with pytest.raises(AssertionError):
        provenance.historical_source('x', 'y')
