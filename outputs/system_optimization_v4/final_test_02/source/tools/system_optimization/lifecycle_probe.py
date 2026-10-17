"""Real-weight controlled failure AFTER native temporal heads have advanced.

Uses repeated real frames solely as a lifecycle control, never independent
driving samples or performance measurements. No GT or final answers are cached.
"""
from pathlib import Path
import copy
import hashlib
from functools import wraps
import torch
from .audit import write


def head_memory_status(model):
    # memory_len is immutable configuration, not temporal state to clear.
    fields=('memory_embedding','memory_reference_point','memory_timestamp',
        'memory_egopose','memory_velo','sample_time','memory_canbus','memory_mask')
    return {branch:{name:getattr(getattr(model,branch),name) is None
        for name in fields if hasattr(getattr(model,branch),name)}
        for branch in ('pts_bbox_head','map_head')}


@torch.no_grad()
def run(model,dataset,frozen,indices,sink,dest,args,configure,seed,recorded_m3):
    from mmcv.parallel import collate,scatter
    from tools.worldstream_m3.refinement import freeze_sample_questions
    from tools.worldstream_m2.refinement_common import arrays,numeric_diff
    from tools.worldstream_m2.rearchitecture.common import memory
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).read_bytes()
    (dest/'executed_probe.py').write_bytes(source)
    write(dest/'source.json',dict(path=__file__,sha256=hashlib.sha256(source).hexdigest(),
        note='Probe imported at control execution; main inference source snapshot is in the parent run'))
    if len(indices)<3:raise ValueError('Lifecycle control needs three real source frames')
    feature={}
    repeated={}
    extract_owned='extract_img_feat' in vars(model)
    original_extract=model.extract_img_feat
    @wraps(original_extract)
    def capture_extract(*a,**kw):
        result=original_extract(*a,**kw);feature['value']=result.detach();return result
    model.extract_img_feat=capture_extract
    def execute(index,label,repeat_input=False,capture_input=False):
        seed(index)
        sample=freeze_sample_questions(dataset[index],frozen[index])
        batch=scatter(collate([sample],samples_per_gpu=1),[torch.cuda.current_device()])[0]
        if repeat_input:batch['img'][0].copy_(repeated['image'])
        if capture_input:repeated['image']=batch['img'][0].detach().clone()
        seed(index+10000);sink.ids=[]
        model.save_path=str(dest/label/'language')+'/'
        output=model(return_loss=False,rescale=True,**batch)
        torch.cuda.synchronize()
        return dict(output=arrays(output),feature=arrays(feature.pop('value')),
            memory=memory(model),ids=copy.deepcopy(sink.ids))
    try:
        configure(model,'m123',sink,dest,args.cache_budget_mb,args.refresh_interval,False)
        from worldstream.m2 import IncrementalVisionRuntime
        model.m2_runtime=IncrementalVisionRuntime(model.img_backbone,dict(mode='exact',reuse_boundary='camera',
            memory_budget_mb=512,cache_capture='always',cheap_gate=True,m1_service='maintain'))
        # Populate a real source KV, plus real camera features and world history.
        # This is an explicit offline materialization, not a quality approval.
        model.m3_runtime=recorded_m3(dict(mode='exact',offline_experiment=True,
            cross_frame=True,borrow_prefix=True,memory_budget_mb=512),sink)
        warm=execute(indices[0],'warm_real_sources',capture_input=True)
        prior=model.m1_manager.store.features['backbone']
        prior_source=dict(frame=prior.source_frame_id,timestamp=prior.computation_timestamp,
            version=prior.computation_version)
        repeated_output=execute(indices[1],'controlled_repeat_input',repeat_input=True)
        repeat_diag=model.m2_diagnostics
        carried=model.m1_manager.store.features['backbone']
        reuse_proof=dict(actual_reused_cameras=repeat_diag.get('actual_reused_cameras',0),
            source_frame=carried.source_frame_id,source_timestamp=carried.computation_timestamp,
            source_version=carried.computation_version,prior=prior_source,
            current_heads_timestamp=model.m1_manager.store.features['object_carrier'].computation_timestamp,
            feature_difference=numeric_diff(repeated_output['feature'],warm['feature']))
        assert reuse_proof['actual_reused_cameras']==len(carried.source_camera_ids)
        assert (carried.source_frame_id,carried.computation_timestamp,carried.computation_version)==(
            prior_source['frame'],prior_source['timestamp'],prior_source['version'])
        assert reuse_proof['feature_difference']['max_abs']==0
        populated=dict(m1_anchors=len(model.m1_manager.store.anchors),
            m1_history_cameras=len(model.m1_manager._previous_cameras),m1_commits=model.m1_manager.store.commit_count,
            m2_bytes=model.m2_runtime.camera_executor.cache.bytes,m3_bytes=model.m3_runtime.cache.nbytes)
        if not all(populated[k]>0 for k in ('m1_history_cameras','m1_commits','m2_bytes','m3_bytes')):
            raise AssertionError('Controlled failure must start with actual nonempty state/cache sources')
        original=model.m3_runtime.generate
        def fail_after_heads(*a,**kw):
            raise RuntimeError('controlled_after_native_temporal_heads')
        model.m3_runtime.generate=fail_after_heads
        try:
            execute(indices[2],'fault_after_heads')
        except RuntimeError as error:
            if str(error)!='controlled_after_native_temporal_heads':raise
        else:
            raise AssertionError('Expected controlled post-head failure')
        finally:
            model.m3_runtime.generate=original
        heads=head_memory_status(model)
        cleared=dict(heads=heads,m1_anchors=len(model.m1_manager.store.anchors),
            m1_pending=model.m1_manager._pending,m1_camera_history=len(model.m1_manager._previous_cameras),
            m2_bytes=model.m2_runtime.cache.bytes+model.m2_runtime.camera_executor.cache.bytes,
            m3_bytes=model.m3_runtime.cache.nbytes,stream_previous=model.stream_runtime.previous,
            failed_record_status=model.stream_runtime.last_record.status)
        write(dest/'fault_state.json',cleared)
        assert all(all(values.values()) for values in heads.values())
        assert all(cleared[k]==0 for k in ('m1_anchors','m1_camera_history','m2_bytes','m3_bytes'))
        assert cleared['m1_pending'] is None and cleared['stream_previous'] is None
        assert cleared['failed_record_status']=='failed'
        # No manual head reset here: exercise the coordinator's real recovery.
        model.m3_runtime=recorded_m3(dict(mode='auto'),sink)
        recovered=execute(indices[2],'recovered_request')
        other=next((i for i in frozen if dataset.data_infos[i]['scene_token']!=
            dataset.data_infos[indices[0]]['scene_token']),None)
        switched=execute(other,'real_scene_switch') if other is not None else None
        switch_resets=list(model.stream_runtime.last_record.lifecycle_resets) if switched is not None else []
        if switched is not None:assert 'scene_change' in switch_resets
        configure(model,'original',sink,dest,args.cache_budget_mb,args.refresh_interval,False)
        reference=execute(indices[2],'fresh_native_reference')
        differences={key:numeric_diff(recovered[key],reference[key]) for key in ('output','feature','memory')}
        token_equal=recovered['ids']==reference['ids']
        equal=token_equal and all(not d['shape_mismatches'] and d['max_abs']==0 for d in differences.values())
        switch_proof=None
        if switched is not None:
            configure(model,'original',sink,dest,args.cache_budget_mb,args.refresh_interval,False)
            switch_reference=execute(other,'fresh_native_scene_switch_reference')
            switch_diff={key:numeric_diff(switched[key],switch_reference[key]) for key in ('output','feature','memory')}
            switch_equal=switched['ids']==switch_reference['ids'] and all(
                not d['shape_mismatches'] and d['max_abs']==0 for d in switch_diff.values())
            switch_proof=dict(source_index=other,lifecycle_resets=switch_resets,
                matches_fresh_native=switch_equal,numerical=switch_diff)
        write(dest/'results.json',dict(scope='controlled repeated real-input failure/recovery, not latency or independent sample data',
            vision_policy='explicit exact offline control, finite whole-camera equality still required',production_approved=False,
            source_indices=indices[:3],controlled_reuse_source= reuse_proof,populated_sources=populated,cleared=cleared,
            recovered_complete_token_equality=token_equal,recovered_numerical=differences,
            recovered_matches_fresh_native=equal,real_scene_switch=switch_proof,
            reference_ids=reference['ids'],recovered_ids=recovered['ids']))
        assert equal,'Recovery differs from fresh native heads/memory/vision/generation'
        if switch_proof is not None:assert switch_proof['matches_fresh_native'],'Scene switch differs from fresh native'
    finally:
        if extract_owned:model.extract_img_feat=original_extract
        else:delattr(model,'extract_img_feat')
