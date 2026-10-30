"""Keep only the frozen main output tree, with explicit archival provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from .freeze_cleanup import digest, experiment_lock, file_inventory, metadata, read, write

REPO = Path(__file__).resolve().parents[2]
MAIN = REPO / 'outputs/m123_final_quality_validation_run01'
RECORD = MAIN / 'consolidation'
OLD_NAME = 'outputs/m123_critical_path_and_joint_tuning_run01'
CHANGED_SOURCES = (
    'tools/worldstream_m123/critical_path.py',
    'tools/worldstream_m123/final_quality_analysis.py',
    'tools/worldstream_m123/freeze_cleanup.py',
    'tools/worldstream_m123/benchmark/preflight.py',
    'projects/configs/OmniDrive/worldstream_m123_auto.py',
    'tests/worldstream_m123/test_critical_path.py',
)


def git_hashes(commit, paths):
    """Hash committed bytes through one cat-file process; never restore old trees."""
    process = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=REPO,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    result = {}
    try:
        for name in paths:
            process.stdin.write((commit + ':' + name + '\n').encode())
            process.stdin.flush()
            header = process.stdout.readline().decode().strip().split()
            if len(header) != 3 or header[1] != 'blob':
                raise ValueError('Missing committed file: ' + name)
            remaining = int(header[2]); h = hashlib.sha256()
            while remaining:
                block = process.stdout.read(min(remaining, 1024 * 1024))
                if not block:
                    raise ValueError('Truncated Git blob: ' + name)
                h.update(block); remaining -= len(block)
            assert process.stdout.read(1) == b'\n'
            result[name] = h.hexdigest()
        process.stdin.close()
        assert process.wait() == 0
    finally:
        if process.poll() is None:
            process.kill(); process.wait()
    return result


def historical_check(original, root):
    """Distinguish live evidence, exact source snapshots and retired Git evidence."""
    record = root / 'consolidation'
    if not (record / 'plan.json').exists():
        assert all(Path(p).is_file() and digest(p) == h for p, h in original.items())
        return dict(live=len(original), source_snapshots=0, retired_in_git=0)
    plan = read(record / 'plan.json')
    retired = {i['relative_path']: i for i in plan['files']}
    snapshots = read(record / 'source_snapshots.json')
    counts = dict(live=0, source_snapshots=0, retired_in_git=0)
    archived = {}
    for name, expected in original.items():
        path = REPO / name
        if name in retired:
            evidence_name = name
            seen = set()
            while retired[evidence_name].get('kind') == 'symlink':
                assert evidence_name not in seen, name
                seen.add(evidence_name)
                target = Path(retired[evidence_name]['target'])
                target = target if target.is_absolute() else (REPO/evidence_name).parent/target
                evidence_name = str(Path(os.path.normpath(str(target))).relative_to(REPO))
                assert evidence_name in retired, name
            assert retired[evidence_name]['sha256'] == expected, name
            archived[evidence_name] = expected
            counts['retired_in_git'] += 1
        elif name in snapshots and digest(path) != expected:
            snap = root / snapshots[name]['snapshot']
            assert digest(snap) == expected, name
            counts['source_snapshots'] += 1
        else:
            assert path.is_file() and digest(path) == expected, name
            counts['live'] += 1
    assert git_hashes(plan['source_commit'], archived) == archived
    return counts


def prepare():
    if (RECORD / 'migrated_files.json').exists() and not (RECORD / 'plan.json').exists():
        return inventory(read(RECORD / 'migrated_files.json')['source_commit'])
    RECORD.mkdir(exist_ok=False)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    snapshots = {}
    for name in CHANGED_SOURCES:
        dst = RECORD / 'original_sources' / name
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(REPO / name, dst)
        snapshots[name] = dict(snapshot=str(dst.relative_to(MAIN)), sha256=digest(dst))
    write(RECORD / 'source_snapshots.json', snapshots)
    original = {str(p.relative_to(MAIN)): digest(p) for p in file_inventory(MAIN)
                if not p.is_relative_to(RECORD)}
    write(RECORD / 'original_main_files.json', original)
    parent = MAIN / 'provenance/parent_evidence'; parent.mkdir(parents=True)
    copied = []
    def copy(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(dict(original=str(source.relative_to(REPO)),
                           retained=str(destination.relative_to(MAIN)), sha256=digest(destination)))
    for name in ('extended_frame_selection.json', 'current_gt_annotations.json', 'reference_policy.json',
                 'data_roles.json', 'task_quality.json', 'severe_error_audit.json', 'abc_comparison.json'):
        copy(REPO / OLD_NAME / name, parent / name)
    shutil.copytree(MAIN / 'baseline_sources', parent / 'baseline_sources')
    archives = read(REPO / OLD_NAME / 'receipt_archive.json')['files']
    needed = [e for e in archives if e['raw'].split('/')[0] in ('tune_m1', 'tune_m2')]
    for entry in needed:
        copy(REPO / OLD_NAME / entry['archive'], parent / entry['archive'])
    write(parent / 'receipt_archive.json', dict(files=needed))
    for p in sorted((REPO / 'outputs/final_route').iterdir()):
        assert p.is_file() and not p.is_symlink()
        copy(p, MAIN / 'resources/final_route' / p.name)
    copy(REPO / 'outputs/m3_final_fact_bypass_run01/checkpoint_verification.json',
         MAIN / 'resources/checkpoint_verification.json')
    freeze = MAIN / 'freeze'; freeze.mkdir()
    for name in ('protected_checkpoints.json', 'environment_versions.json', 'freeze_manifest.json',
                 'cleanup_result.json', 'completion.json', 'preservation_validation.json'):
        copy(REPO / 'outputs/main_experiment_freeze_run01' / name, freeze / name)
    (freeze / 'freeze_manifest.json').rename(freeze / 'previous_freeze_manifest.json')
    write(RECORD / 'migrated_files.json', dict(source_commit=commit, files=copied,
        note='Only dependencies needed by the current frozen study; old directories are not retained.'))
    inventory(commit)


def inventory(commit):
    dirs = sorted(p for p in (REPO / 'outputs').iterdir() if p != MAIN)
    files = []
    for folder in dirs:
        assert folder.is_dir() and not folder.is_symlink(), folder
        for p in file_inventory(folder):
            item = metadata(p); item.update(relative_path=str(p.relative_to(REPO)), sha256=digest(p))
            files.append(item)
        # Record links themselves, never their targets (some old cache links dangle).
        known = {i['path'] for i in files}
        for base, subdirs, names in os.walk(folder, followlinks=False):
            for name in subdirs + names:
                p = Path(base, name)
                if p.is_symlink():
                    item = metadata(p)
                    assert item['uid'] == os.getuid()
                    item.update(relative_path=str(p.relative_to(REPO)), kind='symlink',
                        target=os.readlink(p), sha256=hashlib.sha256(os.readlink(p).encode()).hexdigest())
                    files.append(item); known.add(str(p))
            assert all(str(Path(base, n)) in known for n in names)
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=REPO).decode().split('\0'))
    expected = {i['relative_path']: i['sha256'] for i in files if i['relative_path'] in tracked}
    assert git_hashes(commit, expected) == expected
    write(RECORD / 'plan.json', dict(source_commit=commit, directories=[str(p.relative_to(REPO)) for p in dirs],
        files=files, files_count=len(files), logical_bytes=sum(i['size'] for i in files),
        allocated_bytes=sum(i['allocated_bytes'] for i in files), tracked_git_bytes_verified=True,
        untracked_regenerable_files=len(files)-len(expected),
        user_authorized_tracked_output_deletion=True, git_history_rewritten=False,
        dataset_weights_environment_untouched=True))
    print('CONSOLIDATION_PREPARED', len(dirs), len(files), flush=True)


def clean():
    plan = read(RECORD / 'plan.json')
    validation=read(RECORD / 'pre_delete_validation.json')
    assert validation['status'] == 'PASS'
    assert validation['plan_sha256'] == digest(RECORD / 'plan.json')
    assert validation['policy_sha256'] == digest(MAIN / 'final_research_policy.json')
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == plan['source_commit']
    from .freeze_cleanup import scoped_regular
    scopes = tuple(REPO / d for d in plan['directories'])
    with experiment_lock():
        # Validate all candidates before the first deletion; protect changed/open files.
        candidates = {i['path'] for i in plan['files']}
        open_files = []
        for proc in Path('/proc').glob('[0-9]*'):
            try:
                for fd in (proc / 'fd').iterdir():
                    try:
                        name = os.readlink(fd)
                        if name in candidates: open_files.append(name)
                    except OSError: pass
            except OSError: pass
        assert not open_files, open_files
        for i in plan['files']:
            p = Path(i['path'])
            if i.get('kind') == 'symlink':
                assert p.is_symlink() and os.readlink(p) == i['target'], p
                assert any(p.is_relative_to(s) for s in scopes), p
                assert not any(a.is_symlink() for a in p.parents), p
            else:
                assert scoped_regular(p, scopes), p
            current = metadata(p)
            assert all(current[k] == i[k] for k in ('device', 'inode', 'size', 'mtime_ns')), p
            if i.get('kind') != 'symlink': assert digest(p) == i['sha256'], p
        disk_before = shutil.disk_usage(REPO)._asdict()
        for i in plan['files']: Path(i['path']).unlink()
        for folder in scopes:
            for base, _, _ in os.walk(folder, topdown=False): Path(base).rmdir()
        disk_after = shutil.disk_usage(REPO)._asdict()
    assert sorted(p.name for p in (REPO / 'outputs').iterdir()) == [MAIN.name]
    write(RECORD / 'cleanup_result.json', dict(status='COMPLETED', deleted_directories=len(scopes),
        deleted_files=len(plan['files']), deleted_logical_bytes=plan['logical_bytes'],
        deleted_allocated_bytes=plan['allocated_bytes'], disk_before=disk_before, disk_after=disk_after,
        only_output_directory=MAIN.name, GPU_inference_calls=0, git_history_rewritten=False,
        weights_deleted=0, dataset_files_deleted=0, processes_terminated=0))
    print('OUTPUTS_CONSOLIDATED', flush=True)


def verify():
    for entry in read(RECORD / 'migrated_files.json')['files']:
        retained = entry['retained']
        if retained == 'freeze/freeze_manifest.json': retained = 'freeze/previous_freeze_manifest.json'
        assert digest(MAIN / retained) == entry['sha256'], retained
    for name, h in read(RECORD / 'original_main_files.json').items():
        # Live navigation documents are explicitly revised; original bytes remain in Git.
        if name in ('REPRODUCE.md', 'e2e_performance_report.md'): continue
        assert digest(MAIN / name) == h, name
    from .final_quality_analysis import verify as verify_study
    result = verify_study(MAIN)
    write(RECORD / 'preservation_validation.json', dict(status='PASS', study=result,
        main_policy_and_raw_evidence_unchanged=True, migrated_dependencies_byte_identical=True,
        current_output_directories=sorted(p.name for p in (REPO / 'outputs').iterdir())))
    print('CONSOLIDATION_PRESERVATION_PASS', flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=('prepare', 'clean', 'verify'))
    args = parser.parse_args()
    assert not {4, 5} & os.sched_getaffinity(0)
    globals()[args.action]()


if __name__ == '__main__': main()
