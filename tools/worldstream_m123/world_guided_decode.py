"""WSG-SD staged experiment on frozen real nuScenes streams (no online GT)."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import subprocess
import re
from collections import Counter

import torch

from tools.worldstream_m123.benchmark.common import write, sha256

ROOT = Path('outputs/m3_world_guided_decode_run01')
SOURCE = Path('outputs/m3_compact_visual_memory_run01/natural_live/bridge_sequence.json')


def question_key(question, template=False):
    q = ' '.join(str(question).lower().split())
    return re.sub(r'<[^>]*>|\b\d+(?:\.\d+)?\b', '<value>', q) if template else q


def overlap(root):
    profile = json.loads((root/'native_decode_profile.json').read_text())
    assert profile['status'] == 'COMPLETED'
    history, events = {}, []
    for frame in profile['frames']:
        pending = []
        for request in frame['requests']:
            key = (frame['scene_id'], question_key(request['question']))
            old = history.get(key)
            lcp = 0
            if old:
                for a, b in zip(old['tokens'], request['tokens']):
                    if a != b:
                        break
                    lcp += 1
            template = [v for (s,q), v in history.items() if s == frame['scene_id'] and
                        question_key(q, True) == question_key(request['question'], True)]
            events.append(dict(scene_id=frame['scene_id'], frame_id=frame['frame_id'],
                question=request['question'], exact_question=old is not None,
                question_origin=request.get('question_origin','unknown'),
                template_match=bool(template), historical_frame=old['frame_id'] if old else None,
                historical_tokens=old['tokens'] if old else None,
                native_tokens=request['tokens'], token_lcp=lcp,
                historical_length=len(old['tokens']) if old else None,
                object_persistence='NOT_RUN', target_verified_acceptance=None))
            pending.append((key, dict(tokens=request['tokens'], frame_id=frame['frame_id'])))
        history.update(pending)
    write(root/'historical_answer_overlap.json', dict(status='COMPLETED', workload='native_loader_mixed',
        scope=profile['scope'], requests=len(events), exact_available=sum(e['exact_question'] for e in events),
        template_available=sum(e['template_match'] for e in events), events=events,
        original_dataset_questions=sum(e['question_origin']=='dataset_conversation' for e in events),
        original_dataset_exact_available=sum(e['question_origin']=='dataset_conversation' and e['exact_question'] for e in events),
        warning='Native-loader overview templates and original QA must be separated. Token overlap between independently sampled native answers is not target verification acceptance. Fixed repeated workload measured separately by CUDA stage.'))
    print('OVERLAP', Counter(exact=sum(e['exact_question'] for e in events), total=len(events)), flush=True)


def archive_failure(root):
    """Keep an aborted strict gate intact before an explicit approximate run."""
    result=json.loads((root/'decode_cuda_benchmark.json').read_text())
    if result['status']!='FAIL_GREEDY_IDENTITY':
        raise ValueError('Archive only a recorded greedy identity failure')
    dest=root/'strict_gate_failure';dest.mkdir(exist_ok=False)
    for name in ('cuda_gate','cuda_tasks.json','gpu_verifier_correctness.json',
                 'decode_trials.jsonl','decode_cuda_benchmark.json'):
        (root/name).rename(dest/name)


def native_profile(root, question_manifest=None):
    import projects.mmdet3d_plugin  # noqa
    from mmcv.parallel import collate, scatter, DataContainer
    from mmdet3d.datasets import build_dataset
    from tools.worldstream_m123.benchmark.worker import effective_config, load_model, reset
    from tools.worldstream_m123.benchmark.observe import Observer
    from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
    from tools.worldstream_m2.global_residual_benchmark import wait_gpu
    dest = root/'native_profile'
    dest.mkdir(parents=True, exist_ok=False)
    write(root/'status.json', dict(stage='native_profile', status='RUNNING'))
    sequence = json.loads(SOURCE.read_text())
    frozen_questions=json.loads(question_manifest.read_text())['frames'] if question_manifest else None
    write(root/'sequence.json', sequence)
    with gpu_job_lock():
        wait_gpu()
        torch.manual_seed(20261028)
        x = torch.arange(64, device='cuda', dtype=torch.float32).reshape(8, 8)
        assert float((x@x.T).sum()) == 510720.
        cfg = effective_config('B3', 'language', dest, 'native_full')
        dataset = build_dataset(cfg.data.test)
        lookup = {r['token']: r for r in dataset.data_infos}
        dataset.data_infos = [lookup[t] for t in sequence['tokens']]
        model = load_model(cfg, 'ckpts/iter_10548.pth', dest).eval().requires_grad_(False)
        observer = Observer(model)
        rows, previous = [], None
        for i, info in enumerate(dataset.data_infos):
            if previous != info['scene_token']:
                reset(model)
            data = scatter(collate([dataset[i]], samples_per_gpu=1), [0])[0]
            meta = data['img_metas'][0][0]
            if frozen_questions is not None:
                saved=frozen_questions[info['token']]
                device=data['input_ids'][0][0][0].device
                data['input_ids'][0][0]=[torch.tensor(t,device=device,dtype=torch.long) for t in saved['ids']]
                meta['vlm_labels']=DataContainer(saved['questions'],stack=False)
            timestamp = float(info['timestamp']) / 1e6
            questions = list(meta['vlm_labels'].data)[:3]
            data['input_ids'][0][0] = data['input_ids'][0][0][:3]
            meta['vlm_labels'] = DataContainer(questions, stack=False)
            observer.clear()
            torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(20261028+i)
            torch.cuda.synchronize()
            tick = perf_counter()
            with torch.no_grad():
                model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
            frame_wall = (perf_counter()-tick)*1000
            assert len(observer.native_generations) == len(questions)
            cap = dict(scene_id=info['scene_token'], frame_id=info['token'],
                timestamp=timestamp, questions=questions,
                ids=[g['audit_input_ids'].cpu() for g in observer.native_generations],
                carriers=observer.native_generations[0]['audit_carriers'].cpu(),
                generation=observer.native_generations[0]['audit_generation'])
            assert cap['carriers'].shape[1] == 513
            torch.save(cap, dest/(info['token']+'.pt'))
            stages = observer.collect()
            requests = []
            for qi, (q, g) in enumerate(zip(questions, observer.native_generations)):
                phase = g['phase_events']
                dec = [e for e in phase if e[0] == 'decode']
                pref = [e for e in phase if e[0] == 'prefill']
                requests.append(dict(question=q,
                    question_origin='native_loader_overview_template' if qi==0 else 'dataset_conversation',
                    tokens=g['output'][0, 1:].cpu().tolist(),
                    generated_tokens=g['generated_tokens'], decode_forward_calls=len(dec),
                    prefill_cuda_ms=sum(a.elapsed_time(b) for _, a, b, _ in pref),
                    decode_cuda_ms=sum(a.elapsed_time(b) for _, a, b, _ in dec),
                    decode_wall_ms=phase[0][2].elapsed_time(g['end']),
                    decode_wall_scope='CUDA stream span after prefill including host gaps',
                    decode_host_enqueue_ms=sum(e[3] for e in dec),
                    m3_wall_ms=g['host_wall_ms'], m3_cuda_span_ms=g['start'].elapsed_time(g['end']),
                    mean_token_decode_cuda_ms=sum(a.elapsed_time(b) for _, a, b, _ in dec)/max(1,len(dec))))
            rows.append(dict(scene_id=info['scene_token'], frame_id=info['token'],
                frame_wall_ms=frame_wall, requests=requests, stages=stages,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                m1_wall_ms=sum(stages.get(k, {}).get('host_wall_ms', 0) for k in ('m1_precheck','m1_commit')),
                m2_wall_ms=stages.get('m2_total', {}).get('host_wall_ms'),
                head_wall_ms=sum(stages.get(k, {}).get('host_wall_ms',0) for k in ('detection','map')),
                warmup=i==0))
            write(root/'native_decode_profile.json', dict(status='RUNNING', frames=rows))
            print('PROFILE', i+1, info['token'], [r['generated_tokens'] for r in requests], flush=True)
            previous = info['scene_token']
        write(root/'native_decode_profile.json', dict(status='COMPLETED', frames=rows,
            scope='First three native loader questions: one loader-generated overview template plus first two original conversation questions; 24 distinct frames, three development scenes; one instrumented pass, not formal E2E.',
            generation=cap['generation'], model_sha256=sha256('ckpts/iter_10548.pth'),
            code_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            gpu=torch.cuda.get_device_name(0), visual_tokens=513))
    write(root/'status.json',dict(stage='native_profile',status='COMPLETED'))


@torch.no_grad()
def benchmark(root, numerical_diagnostic=False):
    """A bounded cost gate, not full-pipeline acceptance or coverage estimation."""
    import projects.mmdet3d_plugin  # noqa
    from functools import wraps
    from tools.worldstream_m123.benchmark.worker import effective_config, load_model
    from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
    from tools.worldstream_m2.global_residual_benchmark import wait_gpu
    from worldstream.m3 import expand_input, CarrierLayout
    from worldstream.m3.visual_state import base_model
    from worldstream.m3.prefill_executor import PrefillExecutor
    from worldstream.m3.prefix_kv_cache import PrefixKVCache
    profile = json.loads((root/'native_decode_profile.json').read_text())
    overlaps = json.loads((root/'historical_answer_overlap.json').read_text())
    tasks = []
    frames = profile['frames']
    for event in overlaps['events']:
        if event['exact_question']:
            f = next(f for f in frames if f['frame_id']==event['frame_id'])
            tasks.append(dict(workload='natural',frame_id=f['frame_id'],scene_id=f['scene_id'],
                source_frame=event['historical_frame'],question_index=next(i for i,q in enumerate(f['requests']) if q['question']==event['question']),
                question=event['question'],draft=event['historical_tokens']))
    for scene in dict.fromkeys(f['scene_id'] for f in frames):
        group = [f for f in frames if f['scene_id']==scene]
        for f in group[1:3]:
            for qi,q in enumerate(group[0]['requests']):
                tasks.append(dict(workload='fixed',scene_id=scene,frame_id=f['frame_id'],
                    source_frame=group[0]['frame_id'],question_index=qi,question=q['question'],draft=q['tokens']))
    write(root/'cuda_tasks.json',dict(tasks=tasks,
        selection='All mixed-loader exact-question opportunities; first two subsequent distinct frames per scene for fixed questions, all first-three native loader source questions. Index zero is a generated overview template; report it separately from actual dataset QA. Conditional availability diagnostic, not deployment-weighted coverage.',
        rounds=3,modes=['native','history4','history8'],contracts=['greedy','sampling']))
    dest = root/'cuda_gate'
    dest.mkdir(exist_ok=False)
    with gpu_job_lock():
        wait_gpu()
        cfg=effective_config('B3','language',dest,'native_full')
        model=load_model(cfg,'ckpts/iter_10548.pth',dest).eval().requires_grad_(False)
        base=base_model(model.lm_head)
        executor=PrefillExecutor(PrefixKVCache(0))
        forward_events=[]
        original=base.forward
        @wraps(original)
        def observed(*args,**kwargs):
            a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2));a.record()
            value=original(*args,**kwargs);b.record()
            tokens=kwargs['inputs_embeds'].shape[1] if kwargs.get('inputs_embeds') is not None else kwargs['input_ids'].shape[1]
            forward_events.append((a,b,tokens,kwargs.get('past_key_values') is None))
            return value
        observed._worldstream_read_only_observer=True
        base.forward=observed
        def measure(fn):
            forward_events.clear()
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            a,b=(torch.cuda.Event(enable_timing=True) for _ in range(2));a.record();tick=perf_counter()
            output=fn();b.record();b.synchronize()
            wall=(perf_counter()-tick)*1000
            events=[dict(tokens=k,prefill=p,cuda_ms=s.elapsed_time(e)) for s,e,k,p in forward_events]
            pre=[(s,e) for s,e,k,p in forward_events if p]
            return output,dict(wall_ms=wall,cuda_stream_ms=a.elapsed_time(b),
                prefill_cuda_ms=sum(e['cuda_ms'] for e in events if e['prefill']),
                decode_cuda_ms=sum(e['cuda_ms'] for e in events if not e['prefill']),
                decode_wall_ms=pre[-1][1].elapsed_time(b),
                decode_forward_calls=sum(not e['prefill'] for e in events),forward_events=events,
                peak_gpu_bytes=torch.cuda.max_memory_allocated())
        rows=[];correctness=[];greedy_pairs=[]
        # Native identity, adversarial proposals, current-vision and cache lifecycle
        # witnesses on the actual target before accepting any performance evidence.
        cap=torch.load(root/'native_profile'/(tasks[0]['frame_id']+'.pt'),weights_only=False)
        images=cap['carriers'].cuda();ids=cap['ids'][0].cuda()
        ex=expand_input(base,ids,images,carrier_layout=CarrierLayout(256,1,256))
        gen=dict(cap['generation'],do_sample=False)
        expected=model.lm_head.generate(inputs=ids,images=images,**gen)
        golden=expected[0,1:].tolist()
        proposals=dict(correct=golden,first=[(golden[0]+1)%base.config.vocab_size]+golden[1:],
            middle=golden[:5]+[(golden[5]+1)%base.config.vocab_size]+golden[6:],
            wrong=[7]*len(golden),early_eos=[base.generation_config.eos_token_id]+golden,
            invalid=[-1,base.config.vocab_size])
        for name,draft in proposals.items():
            (actual,diag),timing=measure(lambda:executor.generate(base,ex,capture=False,profile=True,
                draft_tokens=draft,draft_length=8,**gen))
            ok=torch.equal(actual,expected)
            correctness.append(dict(case=name,identical=ok,native_tokens=golden,
                speculative_tokens=actual[0,1:].cpu().tolist(),diagnostics=diag,timing=timing))
            write(root/'gpu_verifier_correctness.json',dict(status='RUNNING',cases=correctness))
            if not ok:
                write(root/'gpu_verifier_correctness.json',dict(status='FAIL',cases=correctness))
                raise RuntimeError('Real-model greedy tokens differ; deployment must stop')
        write(root/'gpu_verifier_correctness.json',dict(status='PASS',cases=correctness))
        for sampling in (False,True):
            for repeat in range(3):
                modes=['native','history4','history8']
                modes=modes[repeat:]+modes[:repeat]
                for ti,task in enumerate(tasks):
                    cap=torch.load(root/'native_profile'/(task['frame_id']+'.pt'),weights_only=False)
                    src=torch.load(root/'native_profile'/(task['source_frame']+'.pt'),weights_only=False)
                    qi=task['question_index']
                    ids=(src if task['workload']=='fixed' else cap)['ids'][qi].cuda()
                    images=cap['carriers'].cuda()
                    ex=expand_input(base,ids,images,carrier_layout=CarrierLayout(256,1,256))
                    gen=dict(cap['generation'],do_sample=sampling)
                    reference=None
                    outputs={}
                    for mode in modes:
                        torch.manual_seed(20261028+repeat*1000+ti)
                        tick=perf_counter();draft=tuple(task['draft']);selection_ms=(perf_counter()-tick)*1000
                        if mode=='native':
                            output,timing=measure(lambda:model.lm_head.generate(inputs=ids,images=images,**gen))
                            diag={}
                        else:
                            (output,diag),timing=measure(lambda:executor.generate(base,ex,capture=False,profile=True,
                                draft_tokens=draft,draft_length=int(mode[-1]),experimental_sampling=sampling,**gen))
                        outputs[mode]=output.cpu()
                        record=dict(task,contract='sampling' if sampling else 'greedy',round=repeat,mode=mode,
                            tokens=output[0,1:].cpu().tolist(),generated_tokens=output.shape[-1]-1,
                            draft_selection_wall_ms=selection_ms,diagnostics=diag,**timing)
                        rows.append(record)
                        with (root/'decode_trials.jsonl').open('a') as handle:
                            handle.write(json.dumps(record,ensure_ascii=False)+'\n')
                    if not sampling:
                        for mode in ('history4','history8'):
                            same=torch.equal(outputs['native'],outputs[mode])
                            greedy_pairs.append(dict(task_index=ti,round=repeat,mode=mode,
                                workload=task['workload'],frame_id=task['frame_id'],identical=same))
                            if not same and not numerical_diagnostic:
                                write(root/'decode_cuda_benchmark.json',dict(status='FAIL_GREEDY_IDENTITY',trials=rows))
                                raise RuntimeError('Greedy identity failed on real paired workload')
                    print('GATE',sampling,repeat,ti+1,'/',len(tasks),flush=True)
        base.forward=original
        write(root/'decode_cuda_benchmark.json',dict(status='COMPLETED',trials=rows,
            scope='Real 3090 same-current-carrier M3 cost gate, three rotating orders. No full M123 E2E claim.',
            diagnostic_only=numerical_diagnostic,greedy_pairs=greedy_pairs,
            greedy_strict_pass=all(p['identical'] for p in greedy_pairs),
            model_sha256=profile['model_sha256'],code_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            visual_tokens=513))
    write(root/'status.json',dict(stage='cuda_gate',status='COMPLETED'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--stage', choices=['profile', 'overlap', 'benchmark','archive-failure'], required=True)
    parser.add_argument('--allow-numerical-diagnostic',action='store_true',
        help='Retain nonidentical real greedy runs as approximate diagnostics; never deploy or claim Exact')
    parser.add_argument('--question-manifest',type=Path,
        help='Freeze actual captured question IDs; excludes answers and stabilizes the native random overview template')
    args = parser.parse_args()
    if args.stage=='benchmark':benchmark(args.root,args.allow_numerical_diagnostic)
    elif args.stage=='profile':native_profile(args.root,args.question_manifest)
    else:{'overlap': overlap,'archive-failure':archive_failure}[args.stage](args.root)


if __name__ == '__main__':
    main()
