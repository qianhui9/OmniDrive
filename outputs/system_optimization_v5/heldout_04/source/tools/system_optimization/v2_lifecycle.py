"""Real GPU overlap failure/timeout controls, excluded from all speed samples."""
import copy,json,hashlib
from dataclasses import replace
from pathlib import Path
from threading import Event
from time import sleep
from types import SimpleNamespace
import torch
from .audit import write
from .experiment import seed
from .v2_probes import frozen_question
from .lifecycle_probe import head_memory_status


@torch.no_grad()
def run(driver,source,dest):
    from mmcv.parallel import collate,scatter
    from tools.worldstream_m3.refinement import freeze_sample_questions
    from tools.worldstream_m2.refinement_common import arrays,numeric_diff
    from tools.worldstream_m2.rearchitecture.common import memory
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False);source=Path(source)
    executed=Path(__file__).read_bytes();(dest/'executed_probe.py').write_bytes(executed)
    write(dest/'source_manifest.json',dict(path=__file__,sha256=hashlib.sha256(executed).hexdigest()))
    questions=json.loads((source/'questions.json').read_text());protocol=json.loads((source/'protocol.json').read_text())
    model,dataset=driver.model,driver.dataset;first,second=protocol['scenes'][0]['indices'][:2]
    other=protocol['scenes'][1]['indices'][0];sink=SimpleNamespace(ids=[])
    import inspect
    native=inspect.getclosurevars(driver.native_generate).nonlocals['generate'];feature={}
    original_extract=model.extract_img_feat;owned='extract_img_feat' in vars(model)
    def capture(*a,**kw):
        output=original_extract(*a,**kw);feature['x']=output.detach();return output
    model.extract_img_feat=capture
    def generate(*a,**kw):
        output=native(*a,**kw);sink.ids.append(output.cpu().tolist());return output
    model.lm_head.generate=generate
    def execute(index,label):
        seed(index);sample=freeze_sample_questions(dataset[index],frozen_question(questions[str(index)]))
        batch=scatter(collate([sample],samples_per_gpu=1),[0])[0]
        seed(index+10000);sink.ids=[];model.save_path=str(dest/label/'language')+'/'
        out=model(return_loss=False,rescale=True,**batch);torch.cuda.synchronize()
        return dict(output=arrays(out),feature=arrays(feature.pop('x')),memory=memory(model),ids=copy.deepcopy(sink.ids))
    results=[]
    try:
        for fault in ('worker_failure','worker_timeout','eva_failure'):
            driver.setup(model,'overlap',sink,dest,512,10,False)
            execute(first,fault+'_warm')
            manager=model.m1_manager;precheck=manager.precheck;finished=Event();release=Event()
            prior=dict(commits=manager.store.commit_count,cameras=len(manager._previous_cameras),features=len(manager.store.features))
            assert all(prior.values()),'Fault control did not begin with actual nonempty history'
            def check(frame):
                try:
                    if fault=='worker_timeout':
                        assert release.wait(30);sleep(.05)
                    result=precheck(frame)
                    if fault=='worker_failure':raise RuntimeError('controlled_overlap_worker_failure')
                    return result
                finally:finished.set()
            manager.precheck=check
            join=model.stream_runtime.join_precheck
            if fault=='worker_timeout':
                model.stream_runtime.config=replace(model.stream_runtime.config,m1_timeout_s=.0001)
                def signal_join():release.set();return join()
                model.stream_runtime.join_precheck=signal_join
            if fault=='eva_failure':
                def fail(*a,**kw):capture(*a,**kw);raise RuntimeError('controlled_overlap_eva_failure')
                model.extract_img_feat=fail
            try:execute(second,fault+'_injected')
            except (RuntimeError,TimeoutError) as error:
                if fault=='worker_timeout':assert isinstance(error,TimeoutError)
                else:assert 'controlled_overlap_' in str(error)
            else:raise AssertionError('Expected controlled fault')
            finally:
                manager.precheck=precheck;model.stream_runtime.join_precheck=join;model.extract_img_feat=capture
                model.stream_runtime.config=replace(model.stream_runtime.config,m1_timeout_s=30.)
            torch.cuda.synchronize();heads=head_memory_status(model)
            cleared=dict(worker_finished=finished.is_set(),pending_cleared=manager._pending is None,
                anchors=len(manager.store.anchors),features=len(manager.store.features),cameras=len(manager._previous_cameras),
                previous_cleared=model.stream_runtime.previous is None,verified_state_cleared=model.m3_verified_state is None,
                future_cleared=model.stream_runtime._m1_worker.future is None,heads=heads,status=model.stream_runtime.last_record.status)
            assert all(cleared[k] for k in ('worker_finished','pending_cleared','previous_cleared','verified_state_cleared','future_cleared'))
            assert cleared['status']=='failed' and all(all(v.values()) for v in heads.values())
            assert not any(cleared[k] for k in ('anchors','features','cameras'))
            # Exercise coordinator recovery with no manual reset.
            recovered=execute(second,fault+'_recovered')
            switched=execute(other,fault+'_scene_switch');resets=model.stream_runtime.last_record.lifecycle_resets
            assert 'scene_change' in resets
            driver.setup(model,'original',sink,dest,512,10,False)
            reference=execute(second,fault+'_native_reference')
            driver.setup(model,'original',sink,dest,512,10,False)
            switch_reference=execute(other,fault+'_switch_native_reference')
            def compare(a,b):
                diff={k:numeric_diff(a[k],b[k]) for k in ('output','feature','memory')}
                passed=a['ids']==b['ids'] and all(d['max_abs']==0 and not d['shape_mismatches'] for d in diff.values())
                return dict(pass_equal=passed,numerical=diff,complete_token_equality=a['ids']==b['ids'])
            recovery=compare(recovered,reference);switch=compare(switched,switch_reference)
            assert recovery['pass_equal'] and switch['pass_equal']
            results.append(dict(fault=fault,prior=prior,cleared=cleared,recovery=recovery,scene_switch=switch))
            write(dest/'results.json',dict(scope='Controlled actual GPU frames, not independent samples or performance',cases=results,passed=len(results)==3))
    finally:
        if owned:model.extract_img_feat=original_extract
        else:delattr(model,'extract_img_feat')
        model.stream_runtime.close()
