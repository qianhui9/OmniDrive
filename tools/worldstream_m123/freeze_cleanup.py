"""Freeze the current research route and remove an explicit, reviewed file list.

Never traverses symlinks or removes tracked files, weights, dataset files, Git
objects, installed environments, or process memory. No model inference occurs.
"""
import argparse
import copy
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import shutil

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / 'outputs/main_experiment_freeze_run01'
MAIN = 'outputs/m123_final_quality_validation_run01'
PARENT = 'outputs/m123_critical_path_and_joint_tuning_run01'
TMP_SCOPES = tuple(Path(p) for p in (
    '/tmp/omnidrive-final-quality-features',
    '/tmp/omnidrive-critical-full-features',
    '/tmp/omnidrive-critical-equivalence-features',
    '/tmp/omnidrive_wheels',
))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read(path):
    return json.loads(Path(path).read_text())


def tracked_paths():
    return {REPO / p for p in subprocess.check_output(
        ['git', 'ls-files', '-z'], cwd=REPO).decode().split('\0') if p}


def metadata(path):
    s = path.lstat()
    return dict(path=str(path), size=s.st_size, allocated_bytes=s.st_blocks * 512,
                device=s.st_dev, inode=s.st_ino, mtime_ns=s.st_mtime_ns,
                uid=s.st_uid, nlink=s.st_nlink)


def scoped_regular(path, scopes):
    """Reject lexical escapes and any symlink in the entire candidate path."""
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        return False
    scope = next((s for s in scopes if path != s and path.is_relative_to(s)), None)
    if scope is None:
        return False
    for part in (path, *path.parents):
        if part.is_symlink():
            return False
        if part == scope:
            break
    try:
        s = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISREG(s.st_mode) and s.st_uid == os.getuid()


def file_inventory(scope):
    if scope.is_symlink() or not scope.is_dir():
        return
    for base, dirs, files in os.walk(scope, followlinks=False):
        dirs[:] = [d for d in dirs if not Path(base, d).is_symlink()]
        for name in files:
            p = Path(base, name)
            if scoped_regular(p, (scope,)):
                yield p


def check_candidates(plan, tracked):
    """Validate the ENTIRE plan before unlinking anything."""
    scopes = tuple(Path(p) for p in plan['scopes'])
    for item in plan['files']:
        path = Path(item['path'])
        if path in tracked or not scoped_regular(path, scopes):
            raise ValueError('Protected, missing, or out-of-scope candidate: ' + str(path))
        current = metadata(path)
        if any(current[k] != item[k] for k in ('size', 'device', 'inode', 'mtime_ns', 'uid', 'nlink')):
            raise ValueError('Candidate changed since inventory: ' + str(path))


@contextmanager
def experiment_lock():
    with Path('/tmp/omnidrive_m123_latency_gpu.lock').open('a+') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('An experiment holds the GPU lock; do not clean active files.')
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def validate_archives():
    checked = []
    for folder in (MAIN, PARENT):
        root = REPO / folder
        for entry in read(root / 'receipt_archive.json')['files']:
            raw, archive = root / entry['raw'], root / entry['archive']
            assert digest(archive) == entry['archive_sha256'], archive
            h = hashlib.sha256()
            with gzip.open(archive, 'rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            assert h.hexdigest() == entry['raw_sha256'], archive
            if raw.exists():
                assert digest(raw) == entry['raw_sha256'], raw
            checked.append(dict(root=folder, raw=entry['raw'], archive=entry['archive'],
                                raw_sha256=entry['raw_sha256'], archive_sha256=entry['archive_sha256']))
    return checked


def prepare():
    ROOT.mkdir(exist_ok=False)
    tracked = tracked_paths()
    protected = {str(p.relative_to(REPO)): digest(p) for p in sorted(tracked) if p.is_file()}
    write(ROOT / 'protected_tracked_files.json', protected)
    checkpoints = {}
    for p in sorted(file_inventory(REPO / 'ckpts')):
        print('PIN_WEIGHT', p.relative_to(REPO), flush=True)
        checkpoints[str(p.relative_to(REPO))] = dict(bytes=p.stat().st_size, sha256=digest(p))
    write(ROOT / 'protected_checkpoints.json', checkpoints)
    assert checkpoints['ckpts/iter_10548.pth']['sha256'] == '6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d'
    write(ROOT / 'archive_validation.json', dict(status='PASS', files=validate_archives()))
    print('ARCHIVES_VERIFIED', flush=True)
    scopes = (REPO / 'outputs', REPO / 'test', REPO / 'results_planning_only', *TMP_SCOPES)
    files = []
    for scope in scopes:
        for p in file_inventory(scope):
            if p in tracked or p.is_relative_to(ROOT):
                continue
            item = metadata(p)
            item['reason'] = ('regenerable_feature_or_installation_cache' if scope in TMP_SCOPES else
                              'losslessly_archived_duplicate_or_generated_cache' if p.is_relative_to(REPO / MAIN) or p.is_relative_to(REPO / PARENT) else
                              'unversioned_obsolete_experiment_output')
            files.append(item)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    plan = dict(source_commit=commit, scopes=[str(p) for p in scopes], files=files,
                candidate_files=len(files), logical_bytes=sum(x['size'] for x in files),
                allocated_bytes=sum(x['allocated_bytes'] for x in files),
                no_tracked_deletions=True, no_dataset_or_weight_deletions=True,
                no_git_history_rewrite=True, no_process_memory_cleanup=True)
    write(ROOT / 'cleanup_plan.json', plan)
    write(ROOT / 'freeze_manifest.json', dict(source_commit=commit, main_directory=MAIN,
        policy=MAIN + '/final_research_policy.json',
        policy_sha256=digest(REPO / MAIN / 'final_research_policy.json'),
        policy_parameters=dict(M1_min_detection_score=.3, M2_ordinary_budget=3,
                               M3='current_only/C-basic/.45/.75'),
        protected_tracked_sha256=digest(ROOT / 'protected_tracked_files.json'),
        protected_checkpoints_sha256=digest(ROOT / 'protected_checkpoints.json'),
        plan_sha256=digest(ROOT / 'cleanup_plan.json'),
        dataset_symlink=dict(path='data/nuscenes', target=str((REPO/'data/nuscenes').resolve()), touched=False),
        keep_reasons='Tracked historical artifacts remain prerequisites of the exact 17715-file integrity audit and historical NO_GO provenance; inactive parameters are audit records, not alternate active policies.',
        risk='FAILED_OBSERVED_RESIDUAL_SEVERE; INDEPENDENT_VALIDATION_INCOMPLETE',
        production_deployment_allowed=False))
    write(ROOT / 'environment_versions.json', dict(python=sys.version, executable=sys.executable,
        distributions=sorted((d.metadata.get('Name', ''), d.version) for d in importlib.metadata.distributions())))
    print(json.dumps({k: plan[k] for k in ('candidate_files', 'logical_bytes', 'allocated_bytes')}), flush=True)


def clean():
    frozen = read(ROOT / 'freeze_manifest.json')
    assert digest(ROOT / 'cleanup_plan.json') == frozen['plan_sha256']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == frozen['source_commit']
    plan = read(ROOT / 'cleanup_plan.json')
    with experiment_lock():
        tracked = tracked_paths()
        check_candidates(plan, tracked)
        validate_archives()
        deleted = 0
        with (ROOT / 'deletion_receipts.jsonl').open('x') as log:
            for item in plan['files']:
                p = Path(item['path'])
                # Recheck immediately before unlink, even after global preflight.
                check_candidates(dict(scopes=plan['scopes'], files=[item]), tracked)
                p.unlink()
                log.write(json.dumps(item) + '\n')
                deleted += 1
        for name in plan['scopes']:
            scope = Path(name)
            if not scope.exists() or scope.is_symlink():
                continue
            for base, dirs, files in os.walk(scope, topdown=False, followlinks=False):
                p = Path(base)
                if p == REPO / 'outputs' or p.is_relative_to(ROOT) or p.is_symlink():
                    continue
                try:
                    p.rmdir()
                except OSError:
                    pass
    write(ROOT / 'cleanup_result.json', dict(status='COMPLETED', deleted_files=deleted,
        deleted_logical_bytes=plan['logical_bytes'], deleted_allocated_bytes=plan['allocated_bytes'],
        tracked_files_deleted=0, weights_deleted=0, dataset_files_deleted=0,
        installed_environment_modified=False, process_terminated=False, GPU_inference_calls=0))
    print('CLEANED', deleted, flush=True)


def verify():
    frozen = read(ROOT / 'freeze_manifest.json')
    assert digest(ROOT / 'protected_tracked_files.json') == frozen['protected_tracked_sha256']
    assert digest(ROOT / 'protected_checkpoints.json') == frozen['protected_checkpoints_sha256']
    for path, h in read(ROOT / 'protected_tracked_files.json').items():
        assert digest(REPO / path) == h, path
    for path, entry in read(ROOT / 'protected_checkpoints.json').items():
        print('VERIFY_WEIGHT', path, flush=True)
        assert digest(REPO / path) == entry['sha256'], path
    assert digest(REPO / frozen['policy']) == frozen['policy_sha256']
    assert str((REPO / 'data/nuscenes').resolve()) == frozen['dataset_symlink']['target']
    regenerated=[]
    for item in read(ROOT/'cleanup_plan.json')['files']:
        p=Path(item['path'])
        if not p.exists():continue
        current=metadata(p)
        assert any(current[k]!=item[k] for k in ('device','inode','size','mtime_ns')),p
        regenerated.append(str(p))
    validate_archives()
    write(ROOT / 'preservation_validation.json', dict(status='PASS', all_frozen_tracked_files_unchanged=True,
        all_checkpoint_bytes_unchanged=True, main_policy_unchanged=True,
        dataset_link_unchanged=True, compressed_receipts_verified=True,
        original_deleted_files_absent=True,newly_regenerated_files=regenerated))
    print('PRESERVATION_PASS', flush=True)


def frozen_jobs(policy, templates, destination):
    """A/B/C controls, one R0 parameter set; no tuning or selection stage."""
    if policy['selected'] != 'R0' or policy['M1']['min_detection_score'] != .3 or policy['M2']['ordinary_budget'] != 3:
        raise ValueError('Only the frozen R0 configuration is supported')
    jobs=[]
    for workload,frequency in (('addressable','every4'),('addressable','every2'),
                              ('addressable','one'),('addressable','three'),('mixed','three')):
        for condition in ('A','B','C'):
            matches=[j for j in templates if (j['workload'],j['frequency'])==(workload,frequency)]
            template=next((j for j in matches if j['condition']==condition),matches[0])
            j=copy.deepcopy(template)
            j.update(label='frozen_'+condition,condition=condition,mode='online' if condition=='C' else 'off',
                policy=str(destination/'reference_policy.json') if condition=='C' else None,
                m1=copy.deepcopy(policy['M1']),m2=copy.deepcopy(policy['M2']),profile=False,
                native_without_m1=condition=='A',m2_full=False,audit_stage='main',audit_map_capture=True)
            jobs.append(j)
    return jobs


def prepare_replay(destination):
    """Create a fresh full 15-cell main replay without running a GPU job."""
    frozen=read(ROOT/'freeze_manifest.json')
    main=REPO/MAIN
    assert digest(main/'final_research_policy.json')==frozen['policy_sha256']
    protected=read(ROOT/'protected_tracked_files.json')
    for p,h in protected.items():
        if p.startswith(('worldstream/','projects/','configs/','tools/worldstream_m123/critical_path',
                         'tools/worldstream_m123/final_quality_capture.py')):
            assert digest(REPO/p)==h,p
    destination=Path(destination).absolute()
    policy=read(main/'final_research_policy.json')
    templates=read(main/'selected_confirm_jobs.json')
    jobs=frozen_jobs(policy,templates,destination)
    destination.mkdir(parents=True,exist_ok=False)
    for name in ('extended_frame_selection.json','protocol_freeze.json','reference_policy.json',
                 'final_research_policy.json','current_gt_annotations.json'):
        shutil.copy2(main/name,destination/name)
    target=destination/'baseline_sources/worldstream/m1';target.mkdir(parents=True)
    for name in ('input_contract.py','source_digest.py'):
        shutil.copy2(main/'baseline_sources/worldstream/m1'/name,target/name)
    write(destination/'main_jobs.json',jobs)
    write(destination/'frozen_route.json',dict(source_commit=frozen['source_commit'],
        policy_sha256=frozen['policy_sha256'],only_parameter_set='R0',conditions=['A','B','C'],
        jobs=len(jobs),planned_full_model_calls=sum(len(j['scene']['tokens']) for j in jobs),
        tuning=False,automatic_selection=False,independent_TEST=False,production_allowed=False,
        note='One exposed DEV log, full15-cell repeat. Unlike original208 new+32 reused receipts, all240 calls execute anew. Never merge their means or call this independent validation.'))
    print('FROZEN_MAIN_REPLAY_PREPARED',destination,flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=('prepare', 'clean', 'verify', 'prepare-replay'))
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    assert not {4, 5} & os.sched_getaffinity(0)
    if args.action=='prepare-replay':
        if args.output is None:parser.error('--output is required')
        prepare_replay(args.output)
    else:
        globals()[args.action]()


if __name__ == '__main__':
    main()
