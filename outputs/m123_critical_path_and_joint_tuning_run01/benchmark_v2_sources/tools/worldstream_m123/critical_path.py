"""Critical-path studies using the existing complete OmniDrive model and observer.

This worker is adapted from continuous_stream.run; inference remains entirely
inside the existing Petr3D/M1/M2/M3 runtimes. Job manifests are frozen inputs.
"""
from .continuous_stream import *
from .critical_path_probe import CriticalPathProbe,profile_frame
from dataclasses import replace
import copy
import hashlib
import types

ROOT=Path('outputs/m123_critical_path_and_joint_tuning_run01')

def execution_sources():
    return {p:h for p,h in source_hashes().items() if not p.endswith('critical_path_analysis.py')}


def configure_job(model,job,root,observer):
    from worldstream.m1.config import M1Config
    from worldstream.m2.config import M2Config
    select_m1_executor(root,job.get('baseline_m1',False))
    if not hasattr(model,'_critical_base_m1'):model._critical_base_m1=model.m1_manager.config
    if not hasattr(model,'_critical_base_m2'):model._critical_base_m2=model._continuous_m2.config
    model.m1_manager.config=replace(model._critical_base_m1,**job.get('m1',{}))
    model._continuous_m2.config=replace(model._critical_base_m2,**job.get('m2',{}))
    rt=model._continuous_m2
    if not hasattr(rt,'_critical_normal_forward'):rt._critical_normal_forward=rt.forward_from_m1
    rt.forward_from_m1=rt._critical_normal_forward
    if job.get('m2_full'):
        # Sole serving executor; override only reuse on a private M2 input copy.
        # M1 state/observation computation remains untouched.
        def full(this,images,manager,context):
            start=time.perf_counter()
            snapshot=copy.copy(manager.export_m2_inputs(expected_context=context,computation_only=True))
            snapshot.full_refresh_hint=True
            result,diag=this.forward(images,m1_result=snapshot,execution_context=context)
            diag.update(m1_snapshot_read=True,full_control=True,M1_M2_state_read_ms=(time.perf_counter()-start)*1000)
            return result,diag
        rt.forward_from_m1=types.MethodType(full,rt)
        # Same observer outer CUDA boundary for Full and Incremental.
        observer.wrap(rt,'forward_from_m1','m2_total')


_m1_executors=None
def select_m1_executor(root,baseline):
    """Paired historical function control inside the same state manager."""
    global _m1_executors
    import importlib.util
    import worldstream.m1.input_contract as contract
    import worldstream.m1.source_digest as digest
    import worldstream.m1.state_manager as manager
    if _m1_executors is None:
        reference={}
        for name in ('input_contract','source_digest'):
            path=root/'baseline_sources/worldstream/m1'/(name+'.py')
            spec=importlib.util.spec_from_file_location('worldstream.m1._critical_reference_'+name,path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);reference[name]=module
        _m1_executors=(dict(frame_digest=contract.frame_digest,camera_snapshot_digest=contract.camera_snapshot_digest,source_digest=digest.source_digest),
            dict(frame_digest=reference['input_contract'].frame_digest,camera_snapshot_digest=reference['input_contract'].camera_snapshot_digest,source_digest=reference['source_digest'].source_digest))
    chosen=_m1_executors[int(baseline)]
    contract.frame_digest=manager.frame_digest=chosen['frame_digest']
    contract.camera_snapshot_digest=chosen['camera_snapshot_digest'];digest.source_digest=chosen['source_digest']


def visual_receipt(model,outputs,holder,job,fi,root):
    from .benchmark.common import typed_digest
    result=outputs[0]['pts_bbox'];boxes=result['boxes_3d']
    receipt=dict(boxes=boxes.tensor.detach().cpu().tolist(),labels=result['labels_3d'].cpu().tolist(),
        scores=result['scores_3d'].cpu().tolist(),commit_count=model.m1_manager.store.commit_count if model.m1_manager else 0,
        committed_anchor_ids=model.m1_manager.last_commit_diagnostics.get('committed_anchor_ids',[]) if model.m1_manager else [],
        association_pairs=model.m1_manager.last_commit_diagnostics.get('association_pairs',[]) if model.m1_manager else [])
    if 'tensor' in holder:
        import torch
        x=holder['tensor'];receipt['feature']=typed_digest(x)
        if job.get('save_full_features'):
            dest=Path('/tmp/omnidrive-critical-full-features');dest.mkdir(exist_ok=True)
            torch.save(x.cpu(),dest/f'{fi:02d}.pt')
        if job.get('compare_full_features'):
            reference=torch.load(Path('/tmp/omnidrive-critical-full-features')/f'{fi:02d}.pt',map_location=x.device)
            delta=(x-reference).float();receipt['full_feature_difference']=dict(rmse=float(delta.square().mean().sqrt()),
                relative_l2=float(delta.norm()/reference.float().norm().clamp_min(1e-12)),max_abs=float(delta.abs().max()))
    return receipt

def run(root,stage):
    import torch
    import projects.mmdet3d_plugin
    from mmcv.parallel import collate,scatter,DataContainer
    from mmdet3d.datasets import build_dataset
    from .benchmark.worker import effective_config,load_model,reset
    from .benchmark.observe import Observer
    from .benchmark.gpu_lock import gpu_job_lock
    from tools.worldstream_m2.global_residual_benchmark import wait_gpu
    from worldstream.m3 import M3Runtime
    from worldstream.m3.visual_state import base_model
    selection=json.loads((root/'extended_frame_selection.json').read_text());protocol=json.loads((root/'protocol_freeze.json').read_text())
    assert sha256(root/'extended_frame_selection.json')==protocol['selection_sha256']
    jobs=json.loads((root/(stage+'_jobs.json')).read_text());scenes=list({j['scene']['scene_id']:j['scene'] for j in jobs}.values())
    out=root/stage;out.mkdir(exist_ok=False)
    hashes=execution_sources();write(out/'execution_sources.json',hashes)
    with gpu_job_lock():
        wait_gpu();assert torch.cuda.is_available() and not {4,5}&os.sched_getaffinity(0)
        torch.manual_seed(20261030);x=torch.arange(256.,device='cuda').reshape(16,16);witness=float((x@x.T).sum().item())
        write(out/'kernel_witness.json',dict(gpu=torch.cuda.get_device_name(0),seed=20261030,result=witness,cpu_affinity=sorted(os.sched_getaffinity(0))))
        cfg=effective_config('B3','language',out,'native_full')
        model=load_model(cfg,'ckpts/iter_10548.pth',out).eval().requires_grad_(False)
        cp=json.loads(Path('outputs/m3_final_fact_bypass_run01/checkpoint_verification.json').read_text())['actual_sha256']
        actual=sha256('ckpts/iter_10548.pth');assert cp==actual
        model.worldstream_audit_identity=dict(checkpoint_sha256=actual)
        dataset=build_dataset(cfg.data.test);original={i['token']:i for i in dataset.data_infos}
        dataset.data_infos=[original[t] for s in scenes for t in s['tokens']];index={f['token']:i for i,f in enumerate(dataset.data_infos)}
        validate_native_prompt_context(model,dataset,index,scenes,out)
        probe=CriticalPathProbe();probe.install(model);observer=Observer(model);calls=[]
        hook=base_model(model.lm_head).register_forward_pre_hook(lambda *a:calls.append(model.m3_runtime.last_diagnostics.get('active_request_index') if model.m3_runtime else None))
        completed=0;jobindex=0
        with torch.no_grad():
            while jobindex<len(jobs):
                job=jobs[jobindex];condition=job['condition'];scene=job['scene'];workload=job['workload'];frequency=job['frequency'];policy=job['policy']
                policy=Path(policy) if policy else None
                if not hasattr(model,'_critical_m1'):model._critical_m1=model.m1_manager
                model.m1_manager=model._critical_m1
                model.m3_runtime=None if condition=='A' else M3Runtime(dict(reuse_strategy='exact_prefix',decode_strategy='world_guided',
                    target_aware_exact_only=True,experimental_speculative_sampling=True,selective_answer_mode=job['mode'],
                    selective_answer_policy=str(policy) if policy else '',question_rng_enabled=True,target_binding_enabled=True,
                    temporal_risk_enabled=False,target_completeness_enabled=False,current_detection_support=True))
                model.worldstream_question_rng_enabled=True;model.worldstream_require_m123=condition!='A';model.worldstream_allow_idle_frames=True
                model.worldstream_profile=True;model.worldstream_capture_m3_inputs=False
                # A is the native visual+language reference. B/C share the same
                # deployed M2; retain M1 in A for honest state-maintenance cost.
                if not hasattr(model,'_continuous_m2'):model._continuous_m2=model.m2_runtime
                model.m2_runtime=None if condition=='A' else model._continuous_m2
                configure_job(model,job,root,observer)
                if job.get('native_without_m1'):
                    assert condition=='A' and model.m2_runtime is None and model.m3_runtime is None
                    model.m1_manager=None
                dest=out/f'{jobindex:03d}_{workload}_{frequency}_{condition}';dest.mkdir();model.save_path=str(dest)+'/'
                reset(model);fixed=None
                feature_holder={}
                if model.m2_runtime is not None:
                    def capture_feature(i,x):
                        if i==len(model.img_backbone.blocks)-1:feature_holder['tensor']=x.detach()
                    model.m2_runtime.stage_observer=capture_feature
                for fi,token in enumerate(scene['tokens']):
                    torch.manual_seed(20261030+fi);random.seed(20261030+fi)
                    model.worldstream_request_rng_context=dict(repeat=0,experiment_seed=20261030)
                    prep=time.perf_counter();data=scatter(collate([dataset[index[token]]],samples_per_gpu=1),[0])[0]
                    meta=data['img_metas'][0][0];natural=list(meta['vlm_labels'].data)[1:4]
                    if fixed is None:fixed=list(natural)
                    questions=question_schedule(workload,frequency,fi,natural,fixed)
                    ids=encode_questions(model,questions,original[token]['location']) if questions else []
                    data['input_ids'][0][0]=ids;meta['vlm_labels']=DataContainer(questions,stack=False)
                    input_ms=(time.perf_counter()-prep)*1000
                    observer.clear();calls.clear();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                    probe.clear();probe.enabled=job.get('profile',stage=='profile')
                    heavy=probe.enabled and fi in job.get('profile_frames',(0,6,7))
                    with profile_frame(heavy,out/'profile_output'/f'{jobindex:03d}_{fi:02d}.json'):
                        tick=time.perf_counter();outputs=model(return_loss=False,rescale=True,**data);torch.cuda.synchronize();wall=(time.perf_counter()-tick)*1000
                    probe.enabled=False
                    stages=observer.collect();answers=outputs[0]['text_out'];diag=model.m3_diagnostics or {};requests=[]
                    if condition=='A':
                        assert len(observer.native_generations)==len(questions)
                        for q,g in zip(questions,observer.native_generations):
                            phases=g['phase_events'];pre=[e for e in phases if e[0]=='prefill'];dec=[e for e in phases if e[0]=='decode']
                            requests.append(dict(question=q,tokens=g['output'][0,1:].cpu().tolist(),llm_forward_calls=len(pre)+len(dec),
                                llm_skipped=False,answer_source='current_target_model_generation',fast_answer=None,
                                prefill_cuda_ms=sum(a.elapsed_time(b) for _,a,b,_ in pre),decode_cuda_ms=sum(a.elapsed_time(b) for _,a,b,_ in dec),
                                prefill_wall_ms=sum(e[3] for e in pre),decode_wall_ms=sum(e[3] for e in dec),end_to_end_ms=g['host_wall_ms']))
                    else:
                        assert diag['execution_completed'] and len(diag['requests'])==len(questions)
                        for qi,(q,d) in enumerate(zip(questions,diag['requests'])):
                            assert calls.count(qi)==d['llm_forward_calls']
                            if d['llm_skipped']:assert calls.count(qi)==0 and d['layout_preparation_ms']==0. and not d['prefill_completed'] and not d['decode_completed']
                            requests.append(dict(d,question=q,tokens=d['output_token_ids'],prefill_cuda_ms=d['prefill_including_cache_gpu_ms'],decode_cuda_ms=d['decode_forward_cuda_ms'],
                                prefill_wall_ms=d.get('prefill_including_cache_cpu_enqueue_ms',0.),decode_wall_ms=d['decode_loop_gpu_span_ms']))
                    trace=model.worldstream_execution;snapshot=getattr(model,'m3_committed_state',None)
                    assert trace['completed'] and (job.get('native_without_m1') or trace['m1']['commit_completed'])
                    if condition!='A':assert trace['m2']['execution_completed']
                    if not questions:assert len(calls)==0 and not requests and not answers
                    row=dict(stage=stage,scope=SCOPE,job=jobindex,condition=condition,workload=workload,frequency=frequency,
                        scene_id=scene['scene_id'],log_id=scene['log_id'],frame_id=token,frame_index=fi,timestamp=original[token]['timestamp']/1e6,
                        phase='cold' if fi==0 else 'steady' if fi>=4 else 'transition',questions=questions,question_count=len(questions),requests=requests,answers=answers,
                        e2e_wall_ms=wall,input_preparation_ms=input_ms,m3_wall_ms=diag.get('runtime_host_elapsed_ms',sum(g['host_wall_ms'] for g in observer.native_generations)),
                        m1_wall_ms=sum(stages.get(k,{}).get('host_wall_ms',0.) for k in ('m1_precheck','m1_commit')),
                        m1_including_adapters_ms=model.m1_manager.last_commit_diagnostics.get('m1_total_including_adapters_ms') if model.m1_manager else 0.,
                        m2_wall_ms=stages.get('m2_total' if condition!='A' else 'eva_native',{}).get('host_wall_ms'),
                        heads_wall_ms=sum(stages.get(k,{}).get('host_wall_ms',0.) for k in ('detection','map')),
                        snapshot_wall_ms=getattr(model,'m3_committed_snapshot_wall_ms',0.),answer_export_ms=model.language_export_wall_ms,
                        peak_gpu_bytes=torch.cuda.max_memory_allocated(),committed_world=asdict(snapshot) if snapshot else None,
                        camera_ids=list(meta['cam_infos']),observed_llm_forward_calls=len(calls),execution_completed=True,
                        m1_commit_completed=bool(model.m1_manager),m2_completed=condition!='A',m2=diagnostic_json(model.m2_diagnostics),
                        m1_commit_count=model.m1_manager.store.commit_count if model.m1_manager else 0,stages=stages,policy_file=str(policy) if policy else None)
                    row.update(heavy_profiler=heavy,job_spec=job,probe=probe.summary(),m1_precheck=diagnostic_json(model.m1_manager.last_result.diagnostics) if model.m1_manager else {},m1_commit=diagnostic_json(model.m1_manager.last_commit_diagnostics) if model.m1_manager else {})
                    row['visual_quality']=visual_receipt(model,outputs,feature_holder,job,fi,root)
                    append(out/'requests.jsonl',row);completed+=1
                    if fi%4==0:print('PROGRESS',stage,jobindex+1,len(jobs),condition,workload,frequency,fi+1,len(scene['tokens']),flush=True)
                jobindex+=1
        hook.remove();assert all(sha256(p)==h for p,h in hashes.items()),'Source changed during GPU stage'
        write(out/'run_status.json',dict(status='COMPLETED',gpu=torch.cuda.get_device_name(0),full_M123_frame_calls=completed,
            question_requests=sum(len(r['requests']) for r in lines(out/'requests.jsonl')),source_hashes=hashes,checkpoint_sha256=actual))



def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage');p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args()
    assert not {4,5}&os.sched_getaffinity(0)
    with (a.root/(a.stage+'_execution.log')).open('x') as log:
        from contextlib import redirect_stdout,redirect_stderr
        with redirect_stdout(log),redirect_stderr(log):
            try:run(a.root,a.stage)
            except BaseException:
                traceback.print_exc();write(a.root/(a.stage+'_failure.json'),dict(error=traceback.format_exc()));raise

if __name__=='__main__':main()
