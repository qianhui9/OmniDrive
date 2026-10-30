"""Retired outputs may use Git; live model code may never silently use old bytes."""
import hashlib
import json

import pytest

from tools.worldstream_m123 import consolidated_outputs as store


def fixture_store(tmp_path, monkeypatch, retired):
    monkeypatch.setattr(store, 'REPO', tmp_path)
    main=tmp_path/'main'; record=main/'consolidation';record.mkdir(parents=True)
    (record/'plan.json').write_text(json.dumps(dict(source_commit='pinned',files=retired)))
    (record/'source_snapshots.json').write_text('{}')
    return main


def test_live_model_mutation_is_not_hidden_by_git(tmp_path, monkeypatch):
    main=fixture_store(tmp_path,monkeypatch,[])
    path=tmp_path/'model.py';path.write_bytes(b'changed')
    monkeypatch.setattr(store,'git_hashes',lambda *args:pytest.fail('Live code must be checked live'))
    with pytest.raises(AssertionError):
        store.historical_check({'model.py':hashlib.sha256(b'original').hexdigest()},main)


def test_retired_link_verifies_original_target_bytes(tmp_path, monkeypatch):
    h=hashlib.sha256(b'tensor').hexdigest()
    files=[dict(relative_path='outputs/old/link',kind='symlink',target='data',sha256='link-bytes'),
           dict(relative_path='outputs/old/data',sha256=h)]
    main=fixture_store(tmp_path,monkeypatch,files)
    calls=[]
    def git(commit,names):
        calls.append((commit,dict(names)))
        return {'outputs/old/data':h}
    monkeypatch.setattr(store,'git_hashes',git)
    assert store.historical_check({'outputs/old/link':h},main)['retired_in_git']==1
    assert calls==[('pinned',{'outputs/old/data':h})]


def test_retired_git_corruption_is_rejected(tmp_path, monkeypatch):
    main=fixture_store(tmp_path,monkeypatch,[dict(relative_path='outputs/old/data',sha256='expected')])
    monkeypatch.setattr(store,'git_hashes',lambda *args:{'outputs/old/data':'wrong'})
    with pytest.raises(AssertionError):
        store.historical_check({'outputs/old/data':'expected'},main)
