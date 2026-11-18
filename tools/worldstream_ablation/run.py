"""Sequential, fresh-process ablations on identical chronological nuScenes scenes."""
import argparse
from dataclasses import asdict
from datetime import datetime
import faulthandler
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback
from .registry import VARIANTS, GROUPS, select, inventory

ROOT = Path(__file__).resolve().parents[2]
PYTHON = '/home/qh/miniconda3/envs/OmniDrive/bin/python'
CHECKPOINT_SHA = '6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d'


def write(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def stage(out,name,**details):
    value = dict(stage=name,time=datetime.now().astimezone().isoformat(),pid=os.getpid(),**details)
    write(out/'stage.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True)


def preflight(args):
    from tools.worldstream_m123.r0_task_evaluation import selected_scenes, facts_questions
    required = [args.infos,args.checkpoint,args.main_manifest/'scene_selection.json',
        Path('ckpts/pretrain_qformer/config.json'),Path('ckpts/pretrain_qformer/pytorch_model-00001-of-00002.bin'),
        Path('ckpts/pretrain_qformer/pytorch_model-00002-of-00002.bin'),Path('configs/worldstream_r0.json'),
        Path('outputs/m123_final_quality_validation_run01/reference_policy.json')]
    required += [Path('data/nuscenes/v1.0-trainval')/n for n in
                 ('sample_annotation.json','category.json','instance.json')]
    missing = [str(p) for p in required if not p.exists()]
    missing += ['module:'+n for n in ('torch','mmcv','mmdet','mmdet3d','transformers','peft','cv2','nuscenes','scipy')
                if importlib.util.find_spec(n) is None]
    if missing:
        raise FileNotFoundError(repr(missing))
    infos = pickle.load(args.infos.open('rb'))['infos']
    scenes = selected_scenes(infos,args.scope,args.main_manifest,1 if args.smoke else args.max_scenes)
    by = {i['token']:i for i in infos}
    if args.smoke:
        scenes = [dict(s, tokens=s['tokens'][:args.smoke_frames]) for s in scenes]
    data_paths = set()
    for s in scenes:
        for token in s['tokens']:
            info = by[token]
            if len(info['cams']) != 6:
                raise ValueError('Expected six cameras:'+token)
            for cam in info['cams'].values():
                p = Path(cam['data_path'])
                if not p.is_file():
                    raise FileNotFoundError(p)
                data_paths.add(p)
            for folder in ('conv','vqa'):
                p = Path('data/nuscenes')/folder/'val'/(token+'.json')
                if not p.exists():
                    raise FileNotFoundError(p)
                data_paths.add(p)
            qs,_ = facts_questions(token,'data/nuscenes',args.other_questions)
            if len(qs) < 3:
                raise ValueError('Factual workload incomplete')
    source = [p for folder in ('tools/worldstream_ablation','worldstream','projects') for p in Path(folder).rglob('*.py')]
    source += [Path('configs/worldstream_r0.json'),Path('tools/worldstream_ablation/run.sh'),
               Path('tools/worldstream_m123/r0_task_evaluation.py')]
    source += list(Path('tools/worldstream_m123').rglob('*.py'))
    hashes = {str(p):sha(p) for p in set(source+required) if p.is_file() and p != args.checkpoint}
    # Bind raw RGB and query files as well as official annotations. No output
    # from previous model experiments enters the new inference receipts.
    data_hashes = {str(p):sha(p) for p in sorted(data_paths)}
    actual_checkpoint = sha(args.checkpoint)
    if actual_checkpoint != CHECKPOINT_SHA:
        raise ValueError('Checkpoint differs from frozen full R0; do not compare changed weights')
    return dict(status='PREFLIGHT_PASS',scenes=scenes,scope=args.scope,smoke=args.smoke,
        frames=sum(len(s['tokens']) for s in scenes),seed=args.seed,other_questions=args.other_questions,
        checkpoint=str(args.checkpoint),checkpoint_sha256=actual_checkpoint,infos=str(args.infos),
        variants=select(args.group),semantic_threshold=args.semantic_threshold,inventory=inventory(),
        source_hashes=hashes,data_hashes=data_hashes,cpu_affinity=sorted(os.sched_getaffinity(0)),
        independent_TEST=False,latency_unit='one frame and all its queries; includes cold scene frames',
        compute_definition='Executed Linear/Conv2d and attention MACs, including EVA functional QKV/proj, LoRA and fused attention; excludes elementwise ops, memory movement and sampling. 1 MAC = 2 arithmetic FLOPs. Independently measured CUDA latency.',
        accuracy_definition='Binary fully-correct factual answers on supported fact grammar; UNKNOWN excluded and counted. Other original QA limited-fact scores separate.',
        stale_definition='Same-question adjacent observations with known changed GT answer semantics. Stale if current answer is incorrect now and fully correct for preceding GT. This measures behavior, not inferred internal cause. Cached-source disagreement is also recorded.',
        selected_before_inference=True)


def wait_gpu(out,once=False,own_process=False):
    """Check physical GPU 0, never kill or modify an existing workload."""
    while True:
        try:
            result = subprocess.run(['nvidia-smi','--id=0','--query-compute-apps=pid',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            usage = subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'],text=True).strip().split(',')
        except (OSError,subprocess.CalledProcessError) as exc:
            raise RuntimeError('Cannot access NVIDIA driver. Run in the GPU-enabled host environment; no inference performed.') from exc
        active = '\n'.join(p for p in result.stdout.splitlines() if p.strip() and
                           (not own_process or p.strip()!=str(os.getpid())))
        if not active and (own_process or (int(usage[0]) < 1000 and int(usage[1]) < 10)):
            return
        stage(out,'WAIT_GPU_IDLE',active_compute_pids=active,memory_MiB=int(usage[0]),utilization=int(usage[1]))
        if once:
            raise RuntimeError('GPU busy; check-only did not wait or stop any workload')
        time.sleep(30)


def worker(args,shared=None):
    out = args.output/args.variant
    manifest = read(args.output/'manifest.json')
    with (out/'native_crash.log').open('a') as crash:
        faulthandler.enable(file=crash,all_threads=True)
        stage(out,'FRESH_WORKER',cpu_affinity=sorted(os.sched_getaffinity(0)))
        # Do not import pickle-backed dataset/plugins until CUDA is initialized.
        import torch
        from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
        with gpu_job_lock():
            stage(out,'CHECK_GPU')
            wait_gpu(out,own_process=shared is not None and 'model' in shared)
            torch.cuda.set_device(0)
            torch.cuda.init()
            stage(out,'CUDA_READY',device=torch.cuda.get_device_name(0))
            import projects.mmdet3d_plugin
            from mmcv.parallel import DataContainer, collate, scatter
            from mmdet3d.datasets import build_dataset
            from tools.worldstream_m123.benchmark.worker import effective_config,load_model,reset
            from tools.worldstream_m123.r0_task_evaluation import facts_questions
            from tools.worldstream_m123.selective_answer_reuse import encode_questions
            from tools.worldstream_m123.frozen_main_benchmark import frame_rng
            from tools.worldstream_m123.benchmark.common import diagnostic_json
            from worldstream.m3.request_rng import request_seed
            from .runtime import attach
            from .instrumentation import Meter
            cfg = effective_config('B0' if args.variant=='native' and shared is None else 'B3','planning',out)
            cfg.data.test.ann_file = manifest['infos']
            stage(out,'LOAD_WEIGHTS')
            if shared is not None and 'model' in shared:
                model = shared['model']
                model.m1_manager,model.m2_runtime,model.m3_runtime = shared['runtimes']
                model.worldstream_require_m123 = True
                reset(model)
                write(out/'checkpoint_validation.json',dict(status='PASS_SHARED_SMOKE_MODEL',
                    strict_validation_source='worldstream/checkpoint_validation.json',
                    weights_reloaded=False,states_reset=True))
            else:
                model = load_model(cfg,manifest['checkpoint'],out).eval().requires_grad_(False)
                if shared is not None:
                    shared.update(model=model,runtimes=(model.m1_manager,model.m2_runtime,model.m3_runtime))
            effective = attach(model,args.variant,manifest['semantic_threshold'])
            model.worldstream_audit_identity = dict(checkpoint_sha256=manifest['checkpoint_sha256'])
            model.worldstream_question_rng_enabled = True
            model.worldstream_request_rng_context = dict(repeat=0,experiment_seed=manifest['seed'])
            model.worldstream_profile = False  # Meter records independent synchronized timings.
            effective['runtime_configs'] = {label:asdict(getattr(model,name).config)
                for label,name in (('ogsm','m1_manager'),('davp','m2_runtime'),('qcse','m3_runtime'))
                if getattr(model,name) is not None and hasattr(getattr(model,name).config,'__dataclass_fields__')}
            write(out/'effective_runtime.json',effective)
            stage(out,'MODEL_READY')
            dataset = build_dataset(cfg.data.test)
            original = {i['token']:i for i in dataset.data_infos}
            tokens = [t for s in manifest['scenes'] for t in s['tokens']]
            dataset.data_infos = [original[t] for t in tokens]
            index = {info['token']:i for i,info in enumerate(dataset.data_infos)}
            meter = Meter(model)
            with (out/'receipts.jsonl').open('x') as f:
                calls = 0
                for si,scene in enumerate(manifest['scenes']):
                    reset(model)
                    dest = out/'answers'/scene['scene_id']
                    dest.mkdir(parents=True,exist_ok=False)
                    model.save_path = str(dest)+'/'
                    for fi,token in enumerate(scene['tokens']):
                        meter.clear()
                        context = dict(scene_id=scene['scene_id'],frame_id=token,repeat=0,experiment_seed=manifest['seed'])
                        stage(out,'FRAME',scene_index=si,frame_index=fi,frame_id=token)
                        with frame_rng(request_seed(context,-1)):
                            torch.cuda.synchronize()
                            tick = time.perf_counter()
                            data = scatter(collate([dataset[index[token]]],samples_per_gpu=1),[0])[0]
                            questions,groups = facts_questions(token,'data/nuscenes',manifest['other_questions'])
                            data['input_ids'][0][0] = encode_questions(model,questions,original[token]['location'])
                            data['img_metas'][0][0]['vlm_labels'] = DataContainer(questions,stack=False)
                            with torch.no_grad():
                                outputs = model(return_loss=False,rescale=True,**data)
                            torch.cuda.synchronize()
                            elapsed = (time.perf_counter()-tick)*1000
                        answers = outputs[0]['text_out']
                        if len(answers)!=len(questions) or any(a['Q']!=q for a,q in zip(answers,questions)):
                            raise RuntimeError('Wrong/missing answers for requested workload')
                        trace = getattr(model,'worldstream_execution',{})
                        if not trace.get('completed'):
                            raise RuntimeError('Model execution incomplete')
                        if model.m1_manager and not trace['m1'].get('commit_completed'):
                            raise RuntimeError('State commit missing')
                        if model.m2_runtime and not model.m2_diagnostics.get('execution_completed'):
                            raise RuntimeError('Visual executor incomplete')
                        if model.m3_runtime and not model.m3_diagnostics.get('execution_completed'):
                            raise RuntimeError('Semantic executor incomplete')
                        stats = meter.collect()
                        m1 = trace.get('m1',{})
                        commit = m1.get('commit_timings_ms',{})
                        ogsm_ms = m1.get('timings_ms',{}).get('precheck_total',0.)+commit.get('commit_total_ms',0.)
                        requests = (model.m3_diagnostics.get('requests',[]) if model.m3_runtime else [])
                        row = dict(variant=args.variant,scene_id=scene['scene_id'],frame_id=token,
                            frame_index=fi,questions=questions,groups=groups,answers=[a['A'][0] for a in answers],
                            e2e_ms=elapsed,ogsm_precheck_commit_ms=ogsm_ms,
                            ogsm_including_adapters_ms=commit.get('m1_total_including_adapters_ms'),
                            **stats,requests=diagnostic_json(requests),
                            visual_diagnostics=diagnostic_json(model.m2_diagnostics) if model.m2_runtime else None,
                            state_validity=diagnostic_json(model.m1_manager.last_result.diagnostics) if model.m1_manager else None)
                        f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
                        f.flush()
                        calls += 1
                        write(out/'progress.json',dict(frames=calls,total=len(tokens)))
                meter.close()
            write(out/'inference_complete.json',dict(status='COMPLETED',frames=calls,
                receipts_sha256=sha(out/'receipts.jsonl'),smoke=manifest['smoke']))
            stage(out,'COMPLETED',frames=calls)
        faulthandler.disable()


def command(args,name):
    return ['taskset','-c','0-3,6-31',PYTHON,'-B','-u','-X','faulthandler','-m',
            'tools.worldstream_ablation.run','--worker','--variant',name,'--output',str(args.output.resolve())]


def launch(args,manifest):
    if manifest['smoke']:
        # Startup/mechanism smoke only: one checked weight load, all 14 variants
        # sequentially with resets. Full experiments below use independent execs.
        for name in manifest['variants']:
            (args.output/name).mkdir(exist_ok=False)
        cmd = ['taskset','-c','0-3,6-31',PYTHON,'-B','-u','-X','faulthandler','-m',
               'tools.worldstream_ablation.run','--shared-smoke-worker','--output',str(args.output.resolve())]
        write(args.output/'smoke_command.json',cmd)
        with (args.output/'smoke_runtime.log').open('x') as f:
            proc = subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            write(args.output/'smoke_worker.json',dict(pid=proc.pid,command=cmd))
            for line in proc.stdout:
                f.write(line);f.flush()
                print(line,end='',flush=True)
            code = proc.wait()
        if code:
            write(args.output/'smoke_failure.json',dict(returncode=code,log='smoke_runtime.log'))
            raise RuntimeError(f'Shared smoke worker exited {code}; see smoke_runtime.log')
        from .report import report
        report(args.output)
        return
    for name in manifest['variants']:
        out = args.output/name
        if args.resume and (out/'inference_complete.json').exists():
            done = read(out/'inference_complete.json')
            if done['status']=='COMPLETED' and done['receipts_sha256']==sha(out/'receipts.jsonl'):
                print('RESUME_COMPLETED',name,flush=True)
                continue
        # A failed variant is never silently restarted over partial receipts.
        out.mkdir(exist_ok=False)
        cmd = command(args,name)
        write(out/'command.json',cmd)
        with (out/'runtime.log').open('x') as f:
            proc = subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            write(out/'worker.json',dict(pid=proc.pid,command=cmd))
            for line in proc.stdout:
                f.write(line);f.flush()
                print(line,end='',flush=True)
            code = proc.wait()
        if code:
            write(out/'failure.json',dict(returncode=code,stage=read(out/'stage.json') if (out/'stage.json').exists() else None))
            raise RuntimeError(f'{name} exited {code}: see {out}/runtime.log; remaining variants not started')
    from .report import report
    report(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--group',choices=['all',*GROUPS],default='all')
    parser.add_argument('--variant',choices=VARIANTS,help=argparse.SUPPRESS)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--shared-smoke-worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--scope',choices=('main-scenes','full-val'),default='main-scenes')
    parser.add_argument('--main-manifest',type=Path,default=Path('outputs/m123_r0_only_reproduction_run01'))
    parser.add_argument('--infos',type=Path,default=Path('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl'))
    parser.add_argument('--checkpoint',type=Path,default=Path('ckpts/iter_10548.pth'))
    parser.add_argument('--seed',type=int,default=20261030)
    parser.add_argument('--max-scenes',type=int)
    parser.add_argument('--other-questions',type=int,default=2)
    parser.add_argument('--semantic-threshold',type=float,default=.95)
    parser.add_argument('--smoke-frames',type=int,default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke',action='store_true',help='Real checkpoint, first scene, 3 chronological frames per variant; never paper results')
    mode.add_argument('--check-only',action='store_true',help='Verify assets, data, configurations; no GPU inference')
    mode.add_argument('--report-only',action='store_true')
    mode.add_argument('--resume',action='store_true',help='Skip hash-verified completed variants only; failed partial variants require a new output root')
    args = parser.parse_args()
    os.chdir(ROOT)
    if {4,5} & os.sched_getaffinity(0):
        parser.error('Use taskset -c 0-3,6-31')
    if args.other_questions<0 or not 0<=args.semantic_threshold<=1 or args.smoke_frames<2 or (args.max_scenes is not None and args.max_scenes<1):
        parser.error('Invalid question count, threshold, smoke frames or max-scenes')
    if args.output is None:
        args.output = Path('outputs')/('worldstream_ablation_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    if args.worker:
        try:
            worker(args)
        except BaseException as exc:
            write(args.output/args.variant/'exception.json',dict(error=repr(exc),traceback=traceback.format_exc()))
            raise
        return
    if args.shared_smoke_worker:
        shared = {}
        manifest = read(args.output/'manifest.json')
        # Full R0 must load first; Native subsequently uses its disabled branch
        # with the exact same weights and complete shared metadata pipeline.
        order = list(dict.fromkeys(['worldstream',*manifest['variants']]))
        if 'worldstream' not in manifest['variants']:
            raise RuntimeError('Smoke groups must include the full reference')
        for name in order:
            args.variant = name
            try:
                worker(args,shared)
            except BaseException as exc:
                write(args.output/name/'exception.json',dict(error=repr(exc),traceback=traceback.format_exc()))
                raise
        return
    if args.report_only:
        from .report import report
        report(args.output)
        return
    if args.resume:
        manifest = read(args.output/'manifest.json')
        for key in ('source_hashes','data_hashes'):
            for path,digest in manifest[key].items():
                if sha(path)!=digest:
                    raise RuntimeError('Cannot resume with changed source/data:'+path)
        if sha(manifest['checkpoint']) != manifest['checkpoint_sha256']:
            raise RuntimeError('Cannot resume with changed checkpoint')
    else:
        print('PREFLIGHT assets / chronological scenes / source and data SHA256',flush=True)
        manifest = preflight(args)
        print(json.dumps({k:manifest[k] for k in ('status','frames','variants','smoke')},indent=2),flush=True)
        if args.check_only:
            return
        args.output.mkdir(parents=True,exist_ok=False)
        write(args.output/'manifest.json',manifest)
        write(args.output/'planned_commands.json',[command(args,n) for n in manifest['variants']])
    launch(args,manifest)


if __name__=='__main__':
    main()
