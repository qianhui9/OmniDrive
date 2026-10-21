import hashlib
import json
from tools.worldstream_m2.state_interface_integration import archive_sources


def test_absolute_source_is_read_only_and_archived_under_destination(tmp_path):
    repo=tmp_path/'repository';source=repo/'src'/'example.py'
    source.parent.mkdir(parents=True);source.write_text("print('original')\n")
    source.chmod(0o444)
    before=source.stat().st_mtime_ns;payload=source.read_bytes()
    destination=tmp_path/'result'
    manifest=archive_sources(destination,[source.resolve()],repo)
    assert source.read_bytes()==payload and source.stat().st_mtime_ns==before
    assert (destination/'source'/'src'/'example.py').read_bytes()==payload
    assert manifest=={'src/example.py':hashlib.sha256(payload).hexdigest()}
    assert json.loads((destination/'source_manifest.json').read_text())==manifest


def test_archive_cannot_alias_the_original_file(tmp_path):
    import pytest
    repo=tmp_path/'source';source=repo/'example.py'
    repo.mkdir();source.write_text('original\n');source.chmod(0o444)
    before=source.stat().st_mtime_ns
    with pytest.raises(ValueError,match='aliases'):
        archive_sources(tmp_path,[source],repo)
    assert source.read_text()=='original\n' and source.stat().st_mtime_ns==before
