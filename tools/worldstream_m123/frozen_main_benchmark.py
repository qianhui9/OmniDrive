"""Frozen R0 versus Native, full chronological scene pairs on all readable logs.

This module is an external experiment launcher. Serving algorithms are untouched.
"""
import argparse
import ast
from collections import defaultdict
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import time
import traceback

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '4'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/omnidrive-r0-main-mpl')

from .benchmark.common import sha256, write, diagnostic_json, typed_digest
from .continuous_stream import question_schedule
from .selective_answer_reuse import encode_questions, validate_native_prompt_context, append

ROOT = Path('outputs/m123_r0_frozen_main_benchmark_run01')
AUTHORITY = Path('outputs/m123_final_quality_validation_run01')
INFOS = Path('data/nuscenes/nuscenes2d_ego_temporal_infos_val.pkl')
SEED = 20261030
WORKLOADS = [('S1', 'Every-4', 'addressable', 'every4'),
             ('S2', 'Every-2', 'addressable', 'every2'),
             ('S3', 'Every-1', 'addressable', 'one'),
             ('S4', 'Three-per-frame', 'addressable', 'three'),
             ('S5', 'Mixed-three', 'mixed', 'three')]


def read(path):
    return json.loads(Path(path).read_text())


def content_hash(value):
    return hashlib.sha256(json.dumps(diagnostic_json(value), sort_keys=True,
        separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def source_hashes():
    paths = [p for folder in ('worldstream', 'projects') for p in Path(folder).rglob('*.py')]
    paths += [Path(__file__), Path('tools/worldstream_m123/continuous_stream.py'),
              Path('tools/worldstream_m123/selective_answer_reuse.py'),
              Path('tools/worldstream_m123/critical_path.py')]
    paths += list(Path('tools/worldstream_m123/benchmark').glob('*.py'))
    paths += list(Path('configs').glob('worldstream*'))
    return {str(p): sha256(p) for p in sorted(set(paths)) if p.is_file()}


def historical_m3_config(policy_path):
    """Evaluate the exact historical constructor's kwargs, then materialize ALL defaults."""
    from worldstream.m3.config import M3Config
    tree = ast.parse(Path('tools/worldstream_m123/critical_path.py').read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == 'M3Runtime')
    expression = ast.Expression(node.args[0]); ast.fix_missing_locations(expression)
    kwargs = eval(compile(expression, '<historical M3 constructor>', 'eval'),
                  {'dict': dict, 'str': str, 'policy': str(policy_path), 'job': {'mode': 'online'}})
    return asdict(M3Config(**kwargs))


def select_complete_scenes(infos, scenes):
    """One full first-readable scene per log; selection never uses model/answer quality."""
    by_scene = defaultdict(list)
    for info in infos: by_scene[info['scene_token']].append(info)
    logs = defaultdict(list); rejected = []
    for sid, frames in sorted(by_scene.items()):
        frames.sort(key=lambda f: f['timestamp']); scene = scenes[sid]
        complete = (len(frames) == scene['nbr_samples']
                    and frames[0]['token'] == scene['first_sample_token']
                    and frames[-1]['token'] == scene['last_sample_token']
                    and all(a['next'] == b['token'] and b['prev'] == a['token']
                            for a, b in zip(frames, frames[1:])))
        missing = []
        for f in frames:
            for cam in f['cams'].values():
                if not Path(cam['data_path']).is_file(): missing.append(cam['data_path'])
            qpath = Path('data/nuscenes/conv/val') / (f['token'] + '.json')
            if not qpath.is_file() or len(read(qpath)) < 4: missing.append(str(qpath))
        if not complete or missing:
            rejected.append(dict(scene_id=sid, complete=complete, missing=missing)); continue
        logs[scene['log_token']].append(dict(scene_id=sid, log_id=scene['log_token'],
            tokens=[f['token'] for f in frames], description=scene['description'],
            true_first_sample=scene['first_sample_token'], true_last_sample=scene['last_sample_token'],
            timestamps=[f['timestamp']/1e6 for f in frames]))
    selected = [sorted(part, key=lambda s: s['scene_id'])[0] for _, part in sorted(logs.items())]
    return selected, rejected, sum(len(v) for v in logs.values())


def baseline_audit(root):
    initial = '0a644635'
    path = 'projects/mmdet3d_plugin/models/detectors/petr3d.py'
    raw = subprocess.check_output(['git', 'show', initial + ':' + path])
    dst = root/'baseline_sources'/path; dst.parent.mkdir(parents=True); dst.write_bytes(raw)
    original = ast.parse(raw); current = ast.parse(Path(path).read_text())
    def methods(tree):
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Petr3D')
        return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    before, after = methods(original), methods(current)
    equal = [n for n in before if n in after and ast.dump(before[n]) == ast.dump(after[n])]
    def generation(tree):
        return [ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == 'generate'
                and isinstance(n.func.value, ast.Attribute) and n.func.value.attr == 'lm_head']
    assert generation(original) == generation(current), 'Native generation arguments changed'
    write(root/'baseline_source_audit.json', dict(status='PASS_SOURCE_AND_DISABLED_BRANCH_AUDIT',
        original_commit=initial, current_source=path, original_sha256=hashlib.sha256(raw).hexdigest(),
        current_sha256=sha256(path), identical_methods=equal,
        native_generate_AST_identical=True,
        baseline='A uses the current Petr3D disabled-M123 branch: M1/M2/M3 None; original backbone, heads and lm_head.generate.',
        fairness_overrides=['identical city-aware Conv prompts and six-camera preprocessing',
            'request-local fork_rng for both arms; unchanged generation distribution',
            'fresh answer export directory; no existing-answer cache hits', 'read-only timing/feature/forward observers'],
        note='Audit is explicit source/branch equivalence, not a claim of whole-file byte equivalence. Original source snapshot is retained. No native model source was edited.'))


def prepare(root):
    import numpy as np
    from tools.worldstream_m1.io_utils import iter_json_array
    from .answer_validity_quality import load_current_annotations
    root.mkdir(parents=True, exist_ok=False)
    policy = read(AUTHORITY/'final_research_policy.json')
    assert policy['selected'] == 'R0' and policy['M1']['min_detection_score'] == .3 and policy['M2']['ordinary_budget'] == 3
    shutil.copy2(AUTHORITY/'final_research_policy.json', root/'authoritative_R0_policy.json')
    shutil.copy2(AUTHORITY/'reference_policy.json', root/'reference_policy.json')
    m3 = historical_m3_config(root/'reference_policy.json')
    for key, value in policy['M3'].items():
        if key in m3: assert m3[key] == value, key
        else: assert read(root/'reference_policy.json')[key] == value, key
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    config = dict(status='FINAL_MAIN_CONFIGURATION', name='R0', source_commit=commit,
        original_research_commit='c8ddb66bf90ee2c4d4627751d5f9d3961133f810',
        authority=str(AUTHORITY/'final_research_policy.json'), authority_sha256=sha256(root/'authoritative_R0_policy.json'),
        M1=policy['M1'], M2=policy['M2'], M3=m3,
        M3_fact_policy=read(root/'reference_policy.json'), fact_policy_sha256=sha256(root/'reference_policy.json'),
        generation=dict(do_sample=True, temperature=.1, top_p=.75, num_beams=1, max_new_tokens=320, use_cache=True),
        overrides_source='Exact critical_path.py historical M3 constructor and current dataclass defaults, not the parameter summary',
        production_deployment_allowed=False, production_default=False, auto_apply=False,
        known_risk=policy['severe_error_status'], independent_TEST=False)
    write(root/'frozen_main_config.json', config)
    (root/'frozen_main_config.sha256').write_text(sha256(root/'frozen_main_config.json')+'\n')
    def flatten(value, prefix=''):
        result = {}
        for k,v in value.items():
            key = prefix+k
            if isinstance(v,dict): result.update(flatten(v,key+'.'))
            else: result[key] = v
        return result
    write(root/'parameter_inventory.json', flatten(config))
    baseline_audit(root)
    infos = pickle.load(INFOS.open('rb'))['infos']; by = {f['token']:f for f in infos}
    scenes = {s['token']:s for s in iter_json_array('data/nuscenes/v1.0-trainval/scene.json')}
    selected, rejected, eligible = select_complete_scenes(infos, scenes)
    assert len(selected) >= 3
    write(root/'scene_selection.json', dict(rule='First lexicographic complete readable Scene per Log; all samples from official first to last. Fixed before inference. No answer/detection filter.',
        available_frames=len(infos), available_scenes=len({i['scene_token'] for i in infos}),
        available_logs=len({scenes[i['scene_token']]['log_token'] for i in infos}),
        eligible_full_scenes=eligible, rejected=rejected, selected=selected,
        sample_reason='6019 frames x 10 arms would require 60190 full model calls on one RTX3090; use one complete scene per available log, retain every selected frame.'))
    tokens = {t for s in selected for t in s['tokens']}
    print('LOADING_CURRENT_GT',len(tokens),flush=True)
    objects = load_current_annotations(str(INFOS),tokens,'data/nuscenes')
    ann_by_id = {g['annotation_id']:g for gs in objects.values() for g in gs}
    for ann in iter_json_array('data/nuscenes/v1.0-trainval/sample_annotation.json'):
        if ann['token'] in ann_by_id:
            ann_by_id[ann['token']]['visibility_token'] = int(ann.get('visibility_token') or 0)
    write(root/'current_gt_annotations.json',objects)
    old = read(AUTHORITY/'extended_frame_selection.json')['selected']
    dev_logs = {s['log_id'] for s in old if s['role']=='extended_DEV'}
    diagnostic_logs = {s['log_id'] for s in old if s['role']!='extended_DEV'}
    roles = [dict(scene_id=s['scene_id'],log_id=s['log_id'],tokens=s['tokens'],
        role='EXPOSED_DEV' if s['log_id'] in dev_logs else 'HISTORICAL_DATA',
        parameter_selection_log=s['log_id'] in dev_logs,
        known_error_analysis_or_evaluation_log=s['log_id'] in diagnostic_logs,
        historical_visual_exposure='All official validation logs were used historically; prior local environment ledger and Git retain exposure records',
        independent_TEST=False) for s in selected]
    write(root/'data_role_manifest.json',dict(independent_TEST_logs=0,roles=roles,
        conservative_rule='No official validation Log is relabeled independent, even if its selected Scene was not previously captured.',
        historical_ledger_sha256=sha256('.aris/compute/local.md')))
    frame_conditions = {}; density = float(np.median([sum(g['visible'] for g in objects[t]) for t in tokens]))
    for scene in selected:
        previous = None
        for fi, token in enumerate(scene['tokens']):
            obs = objects[token]; visible = [g for g in obs if g['visible']]
            ids = {g['instance_id']:g for g in visible}
            common = set(ids)&set(previous or {})
            change = 1-len(common)/len(set(ids)|set(previous or {})) if ids or previous else 0.
            displacement = float(np.median([np.linalg.norm(np.array(ids[i]['center_ego'])-np.array(previous[i]['center_ego'])) for i in common])) if common else 0.
            gap = scene['timestamps'][fi]-scene['timestamps'][fi-1] if fi else 0.
            phase = 'cold' if fi==0 else 'transition' if gap>.75 or change>.2 or displacement>2. else 'steady'
            vel=np.asarray(by[token]['gt_velocity'],dtype=float); valid=np.isfinite(vel).all(axis=1)
            moving=float((np.linalg.norm(vel[valid],axis=1)>2.).mean()) if valid.any() else None
            projected=[g for g in obs if g['cameras']];occluded=sum(g['visibility_token']<=2 for g in projected)/len(projected) if projected else None
            description=scene['description'].lower()
            frame_conditions[token]=dict(scene_id=scene['scene_id'],log_id=scene['log_id'],frame_index=fi,phase=phase,
                timestamp=scene['timestamps'][fi],gap_s=gap,visible_GT_Jaccard_change=change,matched_GT_ego_displacement_m=displacement,
                visible_objects=len(visible),density='dense' if len(visible)>density else 'sparse',
                targets='multi' if len(visible)>1 else 'single' if visible else 'none',
                moving_fraction=moving,dynamics='high' if moving is not None and moving>=.25 else 'low' if moving is not None else 'UNKNOWN',
                low_visibility_fraction=occluded,occlusion='high' if occluded is not None and occluded>=.25 else 'low' if occluded is not None else 'UNKNOWN',
                target_change='rapid' if change>.2 else 'stable',
                cameras_with_visible_GT=sorted({c for g in visible for c in g['cameras']}),
                environment=('night_tagged' if 'night' in description else 'rain_tagged' if 'rain' in description else 'not_tagged'),
                description=scene['description'])
            previous=ids
    write(root/'scene_condition_manifest.json',dict(scope='Offline current GT/official metadata; no inference/admission input',
        rules=dict(cold='official first frame',transition='gap>.75s OR visible-instance Jaccard change>.2 OR median matched GT ego displacement>2m',
            steady='remaining noncold frames',density_median_visible_GT=density,
            high_dynamics='>=25% finite annotated velocities >2m/s',high_occlusion='>=25% camera-projected GT visibility_token<=2',
            environment='Only explicit night/rain description tags; no implicit clear/day label'),frames=frame_conditions))
    pairs=[]
    for si,s in enumerate(selected):
        for wi,(wid,name,workload,frequency) in enumerate(WORKLOADS):
            pairs.append(dict(pair_id=f'{wid}_{si:02d}',scene=s,workload_id=wid,workload_name=name,
                workload=workload,frequency=frequency,order=['A','C'] if (si+wi)%2==0 else ['C','A']))
    write(root/'workload_manifest.json',dict(definitions=[dict(id=i,name=n,workload=w,frequency=f) for i,n,w,f in WORKLOADS],
        schedule_source='tools.worldstream_m123.continuous_stream.question_schedule (unchanged)',
        mixed='existing rotating addressable question plus first two current original natural questions',pairs=pairs))
    protocol=dict(status='FROZEN_BEFORE_PILOT',seed=SEED,source_commit=commit,
        config_sha256=sha256(root/'frozen_main_config.json'),GT_sha256=sha256(root/'current_gt_annotations.json'),
        selected_scenes_sha256=sha256(root/'scene_selection.json'),workloads_sha256=sha256(root/'workload_manifest.json'),
        conditions_sha256=sha256(root/'scene_condition_manifest.json'),roles_sha256=sha256(root/'data_role_manifest.json'),
        arms=['A: Original Native (M1/M2/M3 None)','C: Frozen R0 full M123'],
        paired_order='alternate by scene ordinal + workload ordinal; all frames of an arm remain chronological',
        RNG='existing question_rng fork_rng, identical (scene_id,frame_id,question_index,repeat0,seed) per A/C, outer frame fork_rng restored independently',
        reset='reset BOTH task heads and all saved M1/M2/M3 runtimes before every arm',
        timing=dict(primary_E2E='CPU image/metadata loading+preprocessing+H2D+question encoding, then synchronized complete model call incl answer export; observer hashing/GT/feature transfer excluded',
            model_E2E='synchronized model(return_loss=False,rescale=True) including M1/M2/M3/task heads and answer export',
            question_service='existing per-question generation/fast-answer timer; excludes shared vision; not frame E2E',
            arrival='open-loop-free sequential replay; same original timestamps, QA schedule and zero artificial queue; not a real-time SLA simulation',
            GPU='CUDA events; nested stages overlap and are not summed into E2E',FPS='1000/mean frame E2E',QPS='QA requests / summed frame E2E seconds'),
        frames_per_arm=sum(len(s['tokens']) for s in selected)*5,
        planned_main_model_calls=sum(len(s['tokens']) for s in selected)*10,
        pilot=dict(scene=selected[0],frames=4,workload_id='S4',main_result=False),
        feature_error='Actual returned Native backbone NCHW vs returned M2 NCHW on identical current frames; transfer/analysis outside timers',
        quality='existing conservative visible current GT matcher/parser; UNKNOWN remains unscored; known-answer accuracy and full denominators both retained',
        clusters='paired Log bootstrap 2000 resamples, seed fixed; logs are sampling clusters, not independent TEST qualification',
        no_search=True,no_training=True,no_native_or_algorithm_source_edits=True,production_default=False)
    write(root/'experiment_protocol.json',protocol)
    write(root/'executed_source_hashes.json',source_hashes())
    write(root/'historical_output_hashes.json',{str(p):sha256(p) for p in AUTHORITY.rglob('*') if p.is_file()})
    write(root/'environment_versions.json',dict(python=subprocess.check_output([os.sys.executable,'--version'],text=True).strip(),
        executable=os.sys.executable,distributions=sorted((d.metadata.get('Name',''),d.version) for d in importlib.metadata.distributions()),
        reuse='Existing OmniDrive conda environment, no install/rebuild/download',ledger_sha256=sha256('.aris/compute/local.md')))
    print('FROZEN_PROTOCOL',len(selected),'logs',len(tokens),'unique frames',protocol['planned_main_model_calls'],'model calls',flush=True)


@contextmanager
def frame_rng(seed):
    import numpy as np
    import torch
    py=random.getstate(); nr=np.random.get_state()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed)
        try: yield
        finally: random.setstate(py);np.random.set_state(nr)


def validate_frozen(root):
    protocol=read(root/'experiment_protocol.json')
    for name,key in [('frozen_main_config.json','config_sha256'),('current_gt_annotations.json','GT_sha256'),
                     ('scene_selection.json','selected_scenes_sha256'),('workload_manifest.json','workloads_sha256'),
                     ('scene_condition_manifest.json','conditions_sha256'),('data_role_manifest.json','roles_sha256')]:
        assert sha256(root/name)==protocol[key],name
    for p,h in read(root/'executed_source_hashes.json').items():assert sha256(p)==h,p
    assert sha256(AUTHORITY/'final_research_policy.json')==read(root/'frozen_main_config.json')['authority_sha256']
    return protocol


def run(root,stage):
    import numpy as np
    import torch
    import projects.mmdet3d_plugin
    from mmcv.parallel import collate,scatter,DataContainer
    from mmdet3d.datasets import build_dataset
    from .benchmark.worker import effective_config,load_model,reset
    from .benchmark.observe import Observer
    from .benchmark.gpu_lock import gpu_job_lock
    from tools.worldstream_m2.global_residual_benchmark import wait_gpu
    from .critical_path import visual_receipt
    from worldstream.m3 import M3Runtime
    from worldstream.m3.request_rng import request_seed,rng_digest
    from worldstream.m3.visual_state import base_model
    protocol=validate_frozen(root); frozen=read(root/'frozen_main_config.json')
    out=root/stage;out.mkdir(exist_ok=False)
    pairs=read(root/'workload_manifest.json')['pairs']
    if stage=='pilot':
        pair=dict(pairs[3]);pair['scene']=dict(pair['scene']);pair['scene']['tokens']=pair['scene']['tokens'][:4];pair['order']=['A','C'];pairs=[pair]
    else: assert read(root/'pilot/run_status.json')['status']=='COMPLETED'
    with gpu_job_lock():
        wait_gpu(); assert torch.cuda.is_available() and '3090' in torch.cuda.get_device_name(0)
        torch.manual_seed(SEED);x=torch.arange(256.,device='cuda').reshape(16,16)
        write(out/'kernel_witness.json',dict(result=float((x@x.T).sum()),GPU=torch.cuda.get_device_name(0),
            cpu_affinity=sorted(os.sched_getaffinity(0)),torch=torch.__version__,CUDA=torch.version.cuda))
        cfg=effective_config('B3','language',out,'native_full')
        model=load_model(cfg,'ckpts/iter_10548.pth',out).eval().requires_grad_(False)
        actual=sha256('ckpts/iter_10548.pth')
        assert actual=='6468af836dbdbc41034545d2a4669abc3b5288a669b4e13ac9a99bf9bb53c30d'
        model.worldstream_audit_identity=dict(checkpoint_sha256=actual)
        dataset=build_dataset(cfg.data.test); original={i['token']:i for i in dataset.data_infos}
        tokens=list(dict.fromkeys(t for p in pairs for t in p['scene']['tokens']))
        dataset.data_infos=[original[t] for t in tokens]
        index={i['token']:k for k,i in enumerate(dataset.data_infos)}
        scenes=list({p['scene']['scene_id']:p['scene'] for p in pairs}.values())
        validate_native_prompt_context(model,dataset,index,scenes,out)
        saved_m1,saved_m2=model.m1_manager,model.m2_runtime
        saved_m1.config=replace(saved_m1.config,**frozen['M1'])
        saved_m2.config=replace(saved_m2.config,**frozen['M2'])
        observer=Observer(model);holder={};calls=[];first_rng={};bound=[]
        def native_feature(module,args,result):
            if model.m2_runtime is None:holder['tensor']=result[0].detach()
        native_handle=model.img_backbone.register_forward_hook(native_feature)
        old=saved_m2.forward_from_m1
        def returned(*args,**kwargs):
            result,diag=old(*args,**kwargs);holder['tensor']=result[0].detach();return result,diag
        saved_m2.forward_from_m1=returned
        def llm(module,args):
            runtime=model.m3_runtime
            qi=runtime.last_diagnostics.get('active_request_index') if runtime else len(observer.native_generations)
            if qi not in first_rng:first_rng[qi]=rng_digest('cuda')
            calls.append(qi)
        handle=base_model(model.lm_head).register_forward_pre_hook(llm)
        all_count=0;conditions=read(root/'scene_condition_manifest.json')['frames']
        tempbase=Path('/tmp')/('omnidrive-r0-main-features-'+stage+'-'+protocol['config_sha256'][:12])
        tempbase.mkdir(exist_ok=False)
        for pair_index,pair in enumerate(pairs):
            scene=pair['scene']; pair_rows={}
            for arm in pair['order']:
                model.m1_manager=saved_m1;model.m2_runtime=saved_m2
                reset(model)
                model.m3_runtime=None if arm=='A' else M3Runtime(frozen['M3'])
                model.m1_manager=None if arm=='A' else saved_m1
                model.m2_runtime=None if arm=='A' else saved_m2
                model.worldstream_require_m123=arm=='C';model.worldstream_allow_idle_frames=True
                model.worldstream_question_rng_enabled=True;model.worldstream_profile=True
                model.worldstream_request_rng_context=dict(repeat=0,experiment_seed=SEED,rng_audit=True)
                effective=dict(arm=arm,M1=asdict(saved_m1.config) if arm=='C' else None,
                    M2=asdict(saved_m2.config) if arm=='C' else None,M3=asdict(model.m3_runtime.config) if arm=='C' else None,
                    fact_policy_sha256=sha256(root/'reference_policy.json') if arm=='C' else None,
                    generation=frozen['generation'],checkpoint_sha256=actual,config_sha256=protocol['config_sha256'])
                if arm=='C':
                    for k in ('M1','M2','M3'):assert diagnostic_json(effective[k])==frozen[k],k
                append(out/'effective_runtime_configs.jsonl',dict(pair_id=pair['pair_id'],**effective))
                config_hash=content_hash(effective)
                dest=out/'answers'/pair['pair_id']/arm;dest.mkdir(parents=True);model.save_path=str(dest)+'/'
                pair_rows[arm]=[]
                for fi,token in enumerate(scene['tokens']):
                    context=dict(scene_id=scene['scene_id'],frame_id=token,repeat=0,experiment_seed=SEED)
                    outer_before=rng_digest('cuda')
                    with frame_rng(request_seed(context,-1)):
                        prep=time.perf_counter()
                        data=scatter(collate([dataset[index[token]]],samples_per_gpu=1),[0])[0]
                        meta=data['img_metas'][0][0];natural=list(meta['vlm_labels'].data)[1:4]
                        questions=question_schedule(pair['workload'],pair['frequency'],fi,natural,natural)
                        ids=encode_questions(model,questions,original[token]['location']) if questions else []
                        data['input_ids'][0][0]=ids;meta['vlm_labels']=DataContainer(questions,stack=False)
                        torch.cuda.synchronize();input_ms=(time.perf_counter()-prep)*1000
                        input_digest=typed_digest(data['img'])
                        prompt_ids=[v.cpu().tolist() for v in ids]
                        observer.clear();holder.clear();calls.clear();first_rng.clear();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                        tick=time.perf_counter()
                        with torch.no_grad():outputs=model(return_loss=False,rescale=True,**data)
                        torch.cuda.synchronize();wall=(time.perf_counter()-tick)*1000
                        stages=observer.collect();diag=model.m3_diagnostics or {};requests=[]
                        if arm=='A':
                            assert model.m1_manager is None and model.m2_runtime is None and model.m3_runtime is None
                            assert observer.backbone_cameras==6 and len(observer.native_generations)==len(questions)
                            for qi,(q,g) in enumerate(zip(questions,observer.native_generations)):
                                phases=g['phase_events'];pre=[e for e in phases if e[0]=='prefill'];dec=[e for e in phases if e[0]=='decode']
                                assert pre
                                requests.append(dict(question=q,output_token_ids=g['output'][0,1:].cpu().tolist(),
                                    llm_forward_calls=len(pre)+len(dec),llm_skipped=False,fast_answer=None,
                                    prefill_cuda_ms=sum(a.elapsed_time(b) for _,a,b,_ in pre),decode_cuda_ms=sum(a.elapsed_time(b) for _,a,b,_ in dec),
                                    end_to_end_ms=g['host_wall_ms'],request_seed=request_seed(context,qi),
                                    observed_forward_calls=calls.count(qi),sampling_first_forward_rng=first_rng.get(qi),
                                    reused_tokens=0,answer_source='current_target_model_generation'))
                        else:
                            assert diag['execution_completed'] and len(diag['requests'])==len(questions)
                            assert model.worldstream_execution['m1']['commit_completed'] and model.m2_diagnostics['execution_completed']
                            assert diag['config']==frozen['M3']
                            for qi,(q,d) in enumerate(zip(questions,diag['requests'])):
                                assert calls.count(qi)==d['llm_forward_calls']
                                if d['llm_skipped']:assert calls.count(qi)==0 and not d['prefill_completed'] and not d['decode_completed']
                                else:
                                    assert d['question_rng']['seed']==request_seed(context,qi)
                                    assert d['question_rng']['outer_before']==d['question_rng']['outer_after']
                                requests.append(dict(d,question=q,prefill_cuda_ms=d['prefill_including_cache_gpu_ms'],
                                    decode_cuda_ms=d['decode_forward_cuda_ms'],request_seed=request_seed(context,qi),
                                    observed_forward_calls=calls.count(qi),sampling_first_forward_rng=first_rng.get(qi)))
                        assert sum(r['llm_forward_calls'] for r in requests)==len(calls)
                        assert not (not questions and (calls or outputs[0]['text_out']))
                        assert 'tensor' in holder and holder['tensor'].ndim==4
                        feature=holder['tensor'].cpu()
                        fp=tempbase/f'{arm}_{fi:03d}.pt';torch.save(feature,fp)
                        snapshot=getattr(model,'m3_committed_state',None)
                        row=dict(stage=stage,pair_id=pair['pair_id'],pair_ordinal=pair_index,arm_order=pair['order'],
                            condition=arm,workload_id=pair['workload_id'],workload=pair['workload'],frequency=pair['frequency'],
                            scene_id=scene['scene_id'],log_id=scene['log_id'],frame_id=token,frame_index=fi,
                            timestamp=original[token]['timestamp']/1e6,phase=conditions[token]['phase'],
                            camera_ids=list(meta['cam_infos']),questions=questions,requests=requests,answers=outputs[0]['text_out'],
                            input_preparation_ms=input_ms,model_e2e_wall_ms=wall,e2e_wall_ms=input_ms+wall,
                            m1_including_adapters_ms=saved_m1.last_commit_diagnostics.get('m1_total_including_adapters_ms',0.) if arm=='C' else 0.,
                            m3_wall_ms=diag.get('runtime_host_elapsed_ms',sum(g['host_wall_ms'] for g in observer.native_generations)),
                            stages=stages,m2=diagnostic_json(model.m2_diagnostics),
                            m1_precheck=diagnostic_json(saved_m1.last_result.diagnostics) if arm=='C' else {},
                            m1_commit=diagnostic_json(saved_m1.last_commit_diagnostics) if arm=='C' else {},
                            m1_commit_count=saved_m1.store.commit_count if arm=='C' else 0,
                            committed_world=asdict(snapshot) if snapshot else None,
                            observed_llm_forward_calls=len(calls),peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                            config_sha256=protocol['config_sha256'],effective_config_sha256=config_hash,
                            checkpoint_sha256=actual,input_digest=input_digest,prompt_token_ids=prompt_ids,
                            request_seeds=[request_seed(context,qi) for qi in range(len(questions))],
                            feature_digest=typed_digest(feature),execution_completed=True,
                            module_execution=diagnostic_json(model.worldstream_execution))
                        row['visual_quality']=visual_receipt(model,outputs,{},dict(),fi,root)
                        lanes,scores=outputs[0]['lane_results'][0]
                        row['visual_quality']['lane_predictions']=dict(points=[p.tolist() for p in lanes],scores=scores.tolist())
                    assert rng_digest('cuda')==outer_before,'Frame RNG leaked into another arm'
                    pair_rows[arm].append(row);append(out/'raw_gpu_receipts.jsonl',row);all_count+=1
                    if fi%8==0:print('PROGRESS',stage,pair_index+1,len(pairs),arm,fi+1,len(scene['tokens']),all_count,flush=True)
                bound.append(dict(pair_id=pair['pair_id'],arm=arm,config_sha256=config_hash))
            for fi,(a,c) in enumerate(zip(pair_rows['A'],pair_rows['C'])):
                for key in ('frame_id','timestamp','camera_ids','questions','prompt_token_ids','request_seeds','input_digest'):
                    assert a[key]==c[key],(pair['pair_id'],fi,key)
                for aq,cq in zip(a['requests'],c['requests']):
                    if not cq['llm_skipped']:
                        assert aq['sampling_first_forward_rng']==cq['sampling_first_forward_rng'],'Unpaired sampling RNG'
                native=torch.load(tempbase/f'A_{fi:03d}.pt').float();ours=torch.load(tempbase/f'C_{fi:03d}.pt').float()
                assert native.shape==ours.shape
                delta=ours-native;err=dict(relative_l2=float(delta.norm()/native.norm().clamp_min(1e-12)),
                    rmse=float(delta.square().mean().sqrt()),max_abs=float(delta.abs().max()),
                    per_camera_relative_l2=[float(d.norm()/n.norm().clamp_min(1e-12)) for d,n in zip(delta,native)])
                token_err=delta.square().sum(1).sqrt()/native.square().sum(1).sqrt().clamp_min(1e-12)
                err['token_relative_l2_quantiles']={str(q):float(torch.quantile(token_err.flatten(),q)) for q in (.5,.9,.99)}
                for arm in ('A','C'):(tempbase/f'{arm}_{fi:03d}.pt').unlink()
                receipt=dict(stage=stage,pair_id=pair['pair_id'],frame_id=a['frame_id'],log_id=scene['log_id'],
                    workload_id=pair['workload_id'],A=a,C=c,feature_error=err,fair_inputs_verified=True)
                append(out/'paired_gpu_receipts.jsonl',receipt)
                if stage=='main':append(root/'paired_gpu_receipts.jsonl',receipt)
            write(out/'progress.json',dict(completed_pairs=pair_index+1,total_pairs=len(pairs),full_model_calls=all_count,
                last_pair=pair['pair_id'],status='RUNNING'))
        tempbase.rmdir();handle.remove();native_handle.remove();validate_frozen(root)
        write(out/'run_status.json',dict(status='COMPLETED',stage=stage,full_model_calls=all_count,
            new_only=True,no_historical_receipt_reuse=True,paired_frames=all_count//2,
            runtime_bindings=bound,GPU=torch.cuda.get_device_name(0),checkpoint_sha256=actual,
            config_sha256=protocol['config_sha256']))


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('prepare','pilot','main','verify'))
    parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args()
    assert not {4,5}&os.sched_getaffinity(0)
    if args.stage=='prepare':prepare(args.root)
    elif args.stage=='verify':
        validate_frozen(args.root)
        for p,h in read(args.root/'historical_output_hashes.json').items():assert sha256(p)==h,p
        print('FROZEN_CONFIG_SOURCE_HISTORY_PASS')
    else:
        with (args.root/(args.stage+'_execution.log')).open('x') as log,redirect_stdout(log),redirect_stderr(log):
            try:run(args.root,args.stage)
            except BaseException:
                write(args.root/(args.stage+'_failure.json'),dict(status='FAILED_REAL_EXECUTION',error=traceback.format_exc()))
                traceback.print_exc();raise


if __name__=='__main__':main()
