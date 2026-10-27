"""Scene-disjoint natural M3 experiments using the existing benchmark worker.

No question rotation, rewritten inputs, alternate runtime, or deployment writes.
The first labelled questions retain their native Conv/VQA file order.
"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess

from tools.worldstream_m123.benchmark.common import sha256, write
from tools.worldstream_m123.benchmark.independent_validation import historical_audit
from tools.worldstream_m123.benchmark.sequence import validate_sequence
from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
from tools.worldstream_m2.global_residual_benchmark import execute, wait_gpu, read, rows, evaluate
from worldstream.m3.task_conditioner import task_profile

POLICY = Path('outputs/final_route/m3_reuse_policy.json')
ORDERS = [('native', 'zero', 'reuse'), ('reuse', 'zero', 'native'), ('zero', 'native', 'reuse')]


def native_pairs(token):
    pairs = []
    for source in ('conv', 'vqa'):
        path = Path('data/nuscenes')/source/'val'/(token+'.json')
        if path.exists():
            pairs.extend(dict(p, source=str(path), source_sha256=sha256(path)) for p in read(path))
    counts = Counter(p['question'] for p in pairs)
    # Never silently remove or reorder a duplicated/unlabelled first question.
    if any(counts[p['question']] != 1 or not p.get('answer') for p in pairs[:3]):
        return []
    return pairs


def prepare(root):
    from mmcv import Config
    if (root/'experiment_plan.json').exists():
        raise FileExistsError('Preserve frozen plan; use a fresh directory')
    root.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py')
    infos = pickle.load(open(cfg.data.test.ann_file, 'rb'))['infos']
    audit_path = root/'historical_data_audit.json'
    audit = read(audit_path) if audit_path.exists() else historical_audit('outputs', root, infos)
    write(audit_path, audit)
    grouped = defaultdict(list)
    for info in infos:
        if info['scene_token'] not in audit['excluded_scenes']:
            grouped[info['scene_token']].append(info)
    eligible = []
    for scene, values in sorted(grouped.items()):
        segment = sorted(values, key=lambda x:x['timestamp'])[:6]
        if len(segment) < 6 or any(len(native_pairs(i['token'])) < 3 for i in segment):
            continue
        validate_sequence(segment, scene)
        if any(a.get('next') != b['token'] or b.get('prev') != a['token'] for a,b in zip(segment, segment[1:])):
            continue
        eligible.append((scene, segment))
    if len(eligible) < 10:
        raise ValueError('Need four development and six independent scenes')
    development, validation = eligible[:4], eligible[4:10]
    # Cover motion without changing a question or its order. Selection uses
    # only pre-execution question metadata, never model output or reuse rate.
    def profiles(scenes):
        return Counter(task_profile(p['question'])[1] for _,ss in scenes for i in ss for p in native_pairs(i['token'])[:3])
    if not profiles(validation)['object_ego_motion']:
        replacement = next((s for s in eligible[10:] if profiles([s])['object_ego_motion']), None)
        if replacement is None:
            raise ValueError('No independent natural motion coverage')
        validation[-1] = replacement
    evidence = []
    for name, scenes in [('development', development), ('single', validation), ('multi', validation)]:
        count = 1 if name == 'single' else 3
        manifest = dict(tokens=[], questions={}, scene_splits={}, continuity={}, question_selection='first labelled pairs in original Conv then VQA order')
        for ordinal, (scene, segment) in enumerate(scenes):
            manifest['scene_splits'][scene] = ('calibration' if ordinal < 2 else 'validation') if name == 'development' else 'independent_acceptance'
            manifest['continuity'][scene] = validate_sequence(segment, scene)
            for info in segment:
                token = info['token']; pairs = native_pairs(token)[:count]
                manifest['tokens'].append(token)
                manifest['questions'][token] = [p['question'] for p in pairs]
                evidence.extend(dict(workload=name, frame_id=token, scene_id=scene, **p) for p in pairs)
        manifest['profiles'] = dict(Counter(task_profile(q)[1] for qq in manifest['questions'].values() for q in qq))
        write(root/(name+'_sequence.json'), manifest)
    write(root/'question_sources.json', evidence)
    shutil.copy2(POLICY, root/'frozen_m3_policy.json')
    shutil.copytree('worldstream/m3', root/'before_m3', ignore=shutil.ignore_patterns('__pycache__'))
    write(root/'experiment_plan.json', dict(initial_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        checkpoint='ckpts/iter_10548.pth', quality_budget=.02,
        task_quality_budgets=dict(mAP=.1,NDS=.1,DET_l=.1,Token_F1=.1),
        development_scenes=[s for s,_ in development], acceptance_scenes=[s for s,_ in validation],
        performance=dict(rounds=3,orders=ORDERS,bootstrap=dict(seed=20261027,repetitions=10000)),
        modes=['free','fixed'], kinds=['native','zero','reuse'],
        manifest_sha256={p.name:sha256(p) for p in [root/(s+'_sequence.json') for s in ('development','single','multi')]+[root/'frozen_m3_policy.json']},
        rule='No formal policy change without independent semantic/quality and paired latency acceptance. Historical critical records remain binding.'))
    print('PREPARED', root, {s:read(root/(s+'_sequence.json'))['profiles'] for s in ('development','single','multi')}, flush=True)


def worker(root, name, workload, kind, repeat=0, fixed=False, diagnostic=False):
    target = root/name
    if target.exists():
        if not (target/'status.json').exists() or read(target/'status.json')['status'] != 'COMPLETED':
            raise ValueError('Preserve partial run: '+str(target))
        return
    sequence = root/(workload+'_sequence.json')
    args = ['-m','tools.worldstream_m123.benchmark.worker','--output',str(target),'--run-root',str(root),
        '--condition','B0' if kind == 'native' else 'B3','--workload','language','--limit',str(len(read(sequence)['tokens'])),
        '--warmup','1','--seed','0','--repeat',str(repeat),'--sequence-file',str(sequence),
        '--questions-per-frame','1' if workload=='single' else '3','--measurement-set','efficiency']
    if kind != 'native':
        args += ['--m3-strategy','wsdr_kv','--m3-policy',str(root/'frozen_m3_policy.json')]
    if kind == 'zero': args += ['--m3-zero-reuse']
    if fixed: args += ['--fixed-token-reference',str(root/(workload+'_fixed_reference.json'))]
    if diagnostic: args += ['--capture-m3-inputs','--trace-m3-decisions','--runtime-profile']
    wait_gpu()
    execute(root, name, args)


def run(root, stage):
    plan = read(root/'experiment_plan.json')
    for name, digest in plan['manifest_sha256'].items():
        if sha256(root/name) != digest: raise ValueError('Frozen input changed: '+name)
    if stage == 'diagnostic':
        worker(root,'development_diagnostic','development','zero',diagnostic=True)
        return
    if stage == 'calibrate':
        wait_gpu()
        execute(root,'calibration128',['-m','tools.worldstream_m123.benchmark.prefill_validate',
            '--output',str(root/'calibration128'),'--calibrate-wsdr',
            '--capture',str(root/'development_diagnostic'),'--sequence-file',str(root/'development_sequence.json'),
            '--max-visual-tokens','128'])
        return
    if stage == 'micro':
        # This function already holds the same process lock.
        from tools.worldstream_m123.crossframe_diagnostics import run as diagnostics
        wait_gpu();diagnostics(root)
        return
    if stage == 'post':
        if not (root/'development_fixed_control_summary.json').exists():
            wait_gpu()
            execute(root,'development_fixed_control',['-c',
                'from pathlib import Path\nfrom tools.worldstream_m123.crossframe_diagnostics import prefill_control\nprefill_control(Path('+repr(str(root))+'))'])
        profile_status=root/'full_request_profile/status.json'
        if not profile_status.exists() or read(profile_status)['status']!='COMPLETED':
            wait_gpu()
            execute(root,'full_request_profile',['-m','cProfile','-o',str(root/'full_request_cpu.prof'),
                '-m','tools.worldstream_m123.benchmark.worker','--output',str(root/'full_request_profile'),
                '--run-root',str(root),'--condition','B3','--workload','language','--limit','6','--warmup','1',
                '--sequence-file',str(root/'development_sequence.json'),'--m3-strategy','wsdr_kv',
                '--m3-policy',str(root/'frozen_m3_policy.json')])
        if not (root/'environment.json').exists():
            wait_gpu()
            execute(root,'environment',['-c',
                'from pathlib import Path\nfrom tools.worldstream_m123.crossframe_e2e import environment\nenvironment(Path('+repr(str(root))+'))'])
        from tools.worldstream_m123.crossframe_report import analyze
        analyze(root)
        from tools.worldstream_m123.artifact_index import build_index
        build_index(root)
        return
    if stage in ('all','tests'):
        attempt=0
        while (root/('regression_tests'+('_r'+str(attempt) if attempt else '')+'.log')).exists():attempt+=1
        name='regression_tests'+('_r'+str(attempt) if attempt else '')
        wait_gpu()
        write(root/'regression_test_result.json',dict(status='RUNNING',log=name+'.log'))
        try:
            execute(root,name,['-m','pytest','tests/worldstream_m1','tests/worldstream_m2','tests/worldstream_m3','tests/worldstream_m123','-q'])
        except Exception:
            write(root/'regression_test_result.json',dict(status='FAILED',log=name+'.log'));raise
        write(root/'regression_test_result.json',dict(status='PASS',log=name+'.log'))
    if stage in ('all','free','fixed'):
        for workload in ('single','multi'):
            for mode in (('free','fixed') if stage=='all' else (stage,)):
                for repeat, order in enumerate(ORDERS):
                    for kind in order:
                        worker(root,f'{workload}_{mode}_r{repeat}_{kind}',workload,kind,repeat,mode=='fixed')
                    if mode=='free' and repeat==0:
                        path=root/'{}_free_r0_native/per_question_metrics.jsonl'.format(workload)
                        frames={}
                        for q in rows(path):
                            ids=q['token_ids'][0]
                            if ids[0]!=1 or len(ids)-1!=q['generated_tokens']:raise ValueError('Invalid native token reference')
                            frames.setdefault(q['frame_id'],[]).append(dict(question=q['question'],tokens=ids[1:]))
                        write(root/(workload+'_fixed_reference.json'),dict(source='Native natural generation, same frozen original questions',source_sha256=sha256(path),frames=frames))
    if stage in ('all','quality'):
        for workload in ('single','multi'):
            for kind in ('native','reuse'):
                name=f'{workload}_free_r0_{kind}'
                if not (root/(name+'_official/summary.json')).exists():evaluate(root,name)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--stage',choices=['prepare','diagnostic','calibrate','micro','tests','free','fixed','quality','all','post'],required=True)
    a=p.parse_args()
    if set(os.sched_getaffinity(0)) & {4,5}:p.error('Use taskset -c 0-3,6-31')
    root=Path(a.root)
    if a.stage=='prepare':prepare(root)
    else:
        with gpu_job_lock():run(root,a.stage)


def environment(root):
    import importlib.metadata
    import platform
    import sys
    import torch
    from datetime import datetime
    from zoneinfo import ZoneInfo
    torch.manual_seed(20261027)
    x=torch.arange(64,device='cuda',dtype=torch.float32).reshape(8,8)
    witness=float((x@x.T).sum())
    if witness!=510720.:raise ValueError('Seeded CUDA kernel witness failed')
    packages={}
    for name in ('torch','transformers','mmcv-full','mmdet','mmdet3d','numpy','nuscenes-devkit','flash-attn'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    checkpoint='ckpts/iter_10548.pth';digest=sha256(checkpoint)
    if digest!=read(root/'frozen_m3_policy.json')['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    write(root/'environment.json',dict(time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        executable=sys.executable,python=sys.version,platform=platform.platform(),cpu_affinity=sorted(os.sched_getaffinity(0)),
        packages=packages,cuda_runtime=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(0),capability=torch.cuda.get_device_capability(0),
        nvidia_smi=subprocess.check_output(['taskset','-c','0-3,6-31','nvidia-smi'],text=True),
        thread_environment={key:os.environ.get(key) for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG')},
        torch_cpu_threads=torch.get_num_threads(),environment_rebuilt=False,
        checkpoint=checkpoint,checkpoint_sha256=digest,seeded_kernel_witness=dict(value=witness,expected=510720.,passed=True)))


if __name__=='__main__':main()
