"""Safety properties for authorized removal of obsolete generated files."""
import gzip
import hashlib
import json

import pytest

from tools.worldstream_m123 import freeze_cleanup as cleanup


def test_scope_rejects_outside_path_and_symbolic_link(tmp_path):
    scope=tmp_path/'outputs';scope.mkdir()
    outside=tmp_path/'weight.pth';outside.write_bytes(b'keep')
    link=scope/'dump';link.symlink_to(outside)
    assert not cleanup.scoped_regular(outside,(scope,))
    assert not cleanup.scoped_regular(link,(scope,))


def test_scope_rejects_symbolic_link_directory(tmp_path):
    scope=tmp_path/'outputs';scope.mkdir()
    dataset=tmp_path/'dataset';dataset.mkdir();(dataset/'data').write_bytes(b'keep')
    (scope/'linked').symlink_to(dataset,target_is_directory=True)
    assert not cleanup.scoped_regular(scope/'linked/data',(scope,))
    assert list(cleanup.file_inventory(scope))==[]


def test_tracked_file_aborts_entire_candidate_check(tmp_path):
    p=tmp_path/'receipt';p.write_bytes(b'keep')
    plan=dict(scopes=[str(tmp_path)],files=[cleanup.metadata(p)])
    with pytest.raises(ValueError,match='Protected'):
        cleanup.check_candidates(plan,{p})
    assert p.read_bytes()==b'keep'


def test_changed_candidate_is_never_removed(tmp_path):
    p=tmp_path/'dump';p.write_bytes(b'old')
    plan=dict(scopes=[str(tmp_path)],files=[cleanup.metadata(p)])
    p.write_bytes(b'new data from an active experiment')
    with pytest.raises(ValueError,match='changed'):
        cleanup.check_candidates(plan,set())
    assert p.exists()


def test_valid_preflight_has_no_mutations(tmp_path):
    p=tmp_path/'dump';p.write_bytes(b'old')
    before=cleanup.metadata(p)
    cleanup.check_candidates(dict(scopes=[str(tmp_path)],files=[before]),set())
    assert cleanup.metadata(p)==before


def test_raw_receipt_cannot_be_deleted_when_it_differs_from_archive(tmp_path,monkeypatch):
    monkeypatch.setattr(cleanup,'REPO',tmp_path)
    for folder in (cleanup.MAIN,cleanup.PARENT):
        root=tmp_path/folder;root.mkdir(parents=True)
        payload=b'{"real_receipt": 1}\n'
        raw=root/'requests.jsonl';raw.write_bytes(payload)
        archive=root/'requests.jsonl.gz';archive.write_bytes(gzip.compress(payload))
        manifest=dict(files=[dict(raw=raw.name,archive=archive.name,
            raw_sha256=hashlib.sha256(payload).hexdigest(),archive_sha256=cleanup.digest(archive))])
        (root/'receipt_archive.json').write_text(json.dumps(manifest))
    assert len(cleanup.validate_archives())==2
    changed=tmp_path/cleanup.MAIN/'requests.jsonl';changed.write_bytes(b'unarchived new result')
    with pytest.raises(AssertionError):cleanup.validate_archives()
    assert changed.exists()


def test_frozen_replay_has_full_controls_without_parameter_search(tmp_path):
    policy=dict(selected='R0',M1=dict(min_detection_score=.3),M2=dict(ordinary_budget=3))
    templates=[dict(workload=w,frequency=f,condition='A',scene=dict(tokens=['f0','f1']))
        for w,f in [('addressable','every4'),('addressable','every2'),('addressable','one'),('addressable','three'),('mixed','three')]]
    jobs=cleanup.frozen_jobs(policy,templates,tmp_path)
    assert len(jobs)==15
    assert all(j['m1']==policy['M1'] and j['m2']==policy['M2'] for j in jobs)
    assert sum(j['native_without_m1'] for j in jobs)==5
    assert all(j['mode']=='online' for j in jobs if j['condition']=='C')
    assert all('m1' not in t for t in templates)


def test_frozen_replay_rejects_alternative_parameters(tmp_path):
    with pytest.raises(ValueError,match='frozen R0'):
        cleanup.frozen_jobs(dict(selected='R1',M1=dict(min_detection_score=.4),M2=dict(ordinary_budget=1)),[],tmp_path)
