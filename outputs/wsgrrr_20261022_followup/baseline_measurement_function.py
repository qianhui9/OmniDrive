@torch.no_grad()
def run_wsgrr(args):
    """Continuous committed-cache evaluation using the production runtime/executor.

    Replay contains actual full-model predictions, never annotation identities.
    Each ablation owns both its M1 transactions and its M2 cache lineage.
    """
    import copy
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    import projects.mmdet3d_plugin
    from worldstream.m1 import WorldStateManager, M1Config, Frame, CameraFrame
    from worldstream.m1.omnidrive_adapter import quaternion_transform
    from worldstream.m1.geometry import inverse_transform
    from tools.worldstream_m1.dataset import prediction_outputs
    from tools.worldstream_m1.io_utils import read_prediction_subset
    from worldstream.m2 import IncrementalVisionRuntime
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    source_hashes={str(p):sha256(p) for directory in ('worldstream/m1','worldstream/m2','tools/worldstream_m2')
                   for p in Path(directory).glob('*.py')}
    write(out/'source_hashes.json',source_hashes)
    cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m123_auto.py')
    cfg.data.test.test_mode=True
    cfg.data.test.pipeline=[n for n in cfg.data.test.pipeline if n['type'] in
        ('LoadMultiViewImageFromFiles','ResizeCropFlipRotImage','ResizeMultiview3D','NormalizeMultiviewImage','PadMultiViewImage')]
    dataset=build_dataset(cfg.data.test)
    dataset.data_infos.sort(key=lambda i:(i['scene_token'],i['timestamp']))
    if args.scene:dataset.data_infos=[i for i in dataset.data_infos if i['scene_token']==args.scene]
    infos=dataset.data_infos[:args.frames]
    if len(infos)<2:raise ValueError('Need a continuous sequence with >=2 frames')
    predictions=read_prediction_subset(args.m1_predictions,{i['token'] for i in infos})
    missing=[i['token'] for i in infos if i['token'] not in predictions]
    if missing:raise ValueError('Missing actual M1 replay predictions: '+str(missing))
    model=backbone(args.checkpoint)
    specs=[('D3','D',3)]
    runtimes={name:IncrementalVisionRuntime(model,dict(cache_capture='always',ordinary_budget=b)) for name,_,b in specs}
    managers={name:WorldStateManager(M1Config.load('configs/worldstream_m1_final.yaml')) for name,_,_ in specs}
    rows=[];native_rows=[];cost_records=[];previous=None;old_info=None
    # The model kernels are warmed before any reported frames.
    for index,info in enumerate(infos):
        sample=dataset[index]
        image=torch.from_numpy(np.stack(sample['img']).transpose(0,3,1,2).copy()).cuda()
        x0=block0_input(model,image)
        native_output,dense=measure_call(lambda:model(image)[0],args.repeats,args.warmup)
        native_rows.append(dict(ordinal=index,frame_id=info['token'],scene_id=info['scene_token'],timing=dense))
        norm=sample['img_norm_cfg'];cameras={}
        for c,(cid,cam) in enumerate(info['cams'].items()):
            rgb=sample['img'][c]*np.asarray(norm['std'])+np.asarray(norm['mean'])
            if not norm.get('to_rgb',False):rgb=rgb[...,::-1]
            c2w=quaternion_transform(cam['ego2global_rotation'],cam['ego2global_translation']) @ quaternion_transform(cam['sensor2ego_rotation'],cam['sensor2ego_translation'])
            raw_k=np.asarray(cam['cam_intrinsic']);k=sample['intrinsics'][c][:3,:3]
            cameras[cid]=CameraFrame(cid,np.clip(rgb,0,255).astype(np.uint8),k,inverse_transform(c2w),
                cam['timestamp']/1e6,'preprocessed_image',k@np.linalg.inv(raw_k),True,raw_intrinsic=raw_k)
        frame=Frame(info['scene_token'],info['token'],info['timestamp']/1e6,
            quaternion_transform(info['ego2global_rotation'],info['ego2global_translation']),cameras,'wsgrr_offline_real_v1')
        continuous=old_info is not None and old_info['scene_token']==info['scene_token']
        exact_ratio=float(exact_token_changes(x0,previous).float().mean()) if continuous else 1.
        for name,variant,budget in (specs if index%2==0 else list(reversed(specs))):
            rt=runtimes[name];manager=managers[name]
            if not continuous:rt.reset()
            tick=perf_counter();result=manager.precheck(frame);precheck_ms=(perf_counter()-tick)*1000
            context=dict(scene_id=frame.scene_id,frame_id=frame.frame_id,timestamp=frame.timestamp,
                execution_signature=frame.execution_signature,state_epoch=result.diagnostics['state_epoch'],
                coordinate_space='preprocessed_image',camera_ids=list(cameras))
            tick=perf_counter();snapshot=manager.export_m2_inputs(expected_context=context)
            read_ms=(perf_counter()-tick)*1000
            if getattr(args,'followup_profile',False) and name=='D3' and index==1:
                import cProfile,io,pstats
                profiler=cProfile.Profile();profiler.enable()
                manager.export_m2_inputs(expected_context=context)
                profiler.disable();profiler.dump_stats(str(out/'m1_export_cpu.prof'))
                stream=io.StringIO();pstats.Stats(profiler,stream=stream).sort_stats('cumtime').print_stats(50)
                (out/'m1_export_cpu.txt').write_text(stream.getvalue())
            source_cache=rt.cache
            block_inputs={}
            execute=rt.executor.execute
            if args.profile_block_costs:
                def observe(block,x,cached,ids):
                    block_inputs[id(block)]=(x,cached,ids)
                    return execute(block,x,cached,ids)
                rt.executor.execute=observe
            def candidate():
                # commit replaces owned fields atomically; the source snapshot is
                # immutable, so repeats see identical history without GPU clones.
                rt.cache=copy.copy(source_cache)
                return rt.evaluate_offline(image,variant=variant,m1_result=snapshot,execution_context=context)
            measured,timing=measure_call(candidate,args.repeats,args.warmup)
            rt.executor.execute=execute
            output,diag=measured
            next_cache=rt.cache
            if getattr(args,'followup_profile',False) and name=='D3' and index==1:
                from .profiling import profile_call
                rt.cache=source_cache
                # Profiling mutates this shallow copy only; selected history and
                # the measured publication remain exactly the ones above.
                profile_call(rt,candidate,out/'profile_output')
                rt.cache=next_cache
            delta=difference(output[0],native_output)
            outputs=prediction_outputs(predictions[info['token']],manager.config)
            outputs.execution_receipt=diag
            manager.commit(frame,outputs,result)
            rt.cache=next_cache
            selected=diag['stages'][0]['executed_window_ids']
            selected_mask=torch.zeros(len(diag['output_window_provenance'][0]),device=x0.device,dtype=torch.bool)
            selected_mask[selected]=True
            token_ratio=float(rt.planner.windows_to_tokens(selected_mask,x0.shape[:3],model.blocks[0].window_size).float().mean())
            row=dict(case=name+'_'+str(index),strategy=name,variant=variant,budget=budget,ordinal=index,frame_id=frame.frame_id,scene_id=frame.scene_id,
                timestamp=frame.timestamp,continuous=continuous,exact_token_update_ratio=exact_ratio,
                selected_token_update_ratio=token_ratio,stages=diag['stages'],region_refresh=diag['region_refresh'],
                full_update_reason=diag['full_update_reason'],difference=delta,
                incremental=timing,native=dense,M1_precheck_ms=precheck_ms,M1_state_read_ms=read_ms,
                M1_precheck_components_ms=result.diagnostics.get('timings_ms',{}),
                region_scoring_host_ms=diag['M2_region_scoring_ms'],window_and_suffix_host_ms=diag['M2_window_and_suffix_ms'],
                cache_publication_host_ms=diag['M2_cache_publication_ms'],cache_saved=diag['cache_saved'],
                m1_regions=len(snapshot.computation_state.region_states),
                region_states=snapshot.computation_state.region_states,
                real_m1_expiry_events=snapshot.computation_state.region_events,
                m1_replay_source='historical_full_model_detection_replay',
                downstream_quality='NOT_MEASURED: replay supplies states, not candidate task predictions')
            rows.append(row)
            if args.profile_block_costs and variant=='D' and continuous:
                publication_cache=StateCache(rt.cache.budget_bytes)
                publication_cache.version=next_cache.version
                def publish():
                    publication_cache.commit(image,next_cache.outputs,context,next_cache.signature,
                        output[0].stride(),next_cache.sources,block0_input=x0,
                        region_states=next_cache.region_states,window_ages=next_cache.window_ages,
                        descriptors=next_cache.descriptors)
                _,row['synchronized_cache_publication']=measure_call(publish,args.repeats,args.warmup)
                for i in rt.prefix:
                    block=model.blocks[i];x,cached,ids=block_inputs[id(block)]
                    (partial,full),a,b=paired_measure(lambda:execute(block,x,cached,ids),lambda:block(x),args.repeats,args.warmup)
                    local_delta=difference(partial[0],full)
                    cost_records.append(dict(key=cost_key(block,x,i,model=model,refresh_policy=rt.config.refresh_signature()),
                        block=i,updated=len(ids),numerical_verified=all(np.isfinite(v) for k,v in local_delta.items() if k!='exact'),
                        difference=local_delta,quality_certified=False,quality_rejection_reason='downstream_quality_unverified',
                        incremental_gpu_ms=a['gpu_span_ms'],full_gpu_ms=b['gpu_span_ms'],
                        incremental_wall_ms=a['wall_ms'],full_wall_ms=b['wall_ms'],
                        pipeline_incremental_wall_ms=timing['wall_ms'],pipeline_full_wall_ms=dense['wall_ms'],
                        pipeline_incremental_gpu_ms=timing['gpu_span_ms'],pipeline_full_gpu_ms=dense['gpu_span_ms'],
                        evidence_cases=[row['case']],numerical_evidence=dict(path='results.json')))
            with (out/'frames.jsonl').open('a') as handle:handle.write(json.dumps(row,allow_nan=False)+'\n')
            print(index,name,diag['recomputed_windows'],delta,'wall',round(timing['wall_ms'],2),'native',round(dense['wall_ms'],2),flush=True)
        previous=x0;old_info=info
    aggregate={}
    for name,_,budget in specs:
        data=[r for r in rows if r['strategy']==name and r['continuous']]
        stages=[s for r in data for s in r['stages'] if 'total_windows' in s]
        updated=sum(s['executed_windows'] for s in stages);total=sum(s['total_windows'] for s in stages)
        aggregate[name]=dict(frames=len(data),window_update_ratio=updated/total,reuse_ratio=1-updated/total,
            exact_token_update_ratio=float(np.mean([r['exact_token_update_ratio'] for r in data])),
            selected_token_update_ratio=float(np.mean([r['selected_token_update_ratio'] for r in data])),
            wall_ms=float(np.median([r['incremental']['wall_ms'] for r in data])),
            gpu_span_ms=float(np.median([r['incremental']['gpu_span_ms'] for r in data])),
            native_wall_ms=float(np.median([r['native']['wall_ms'] for r in data])),
            native_gpu_span_ms=float(np.median([r['native']['gpu_span_ms'] for r in data])),
            mean_abs=float(np.mean([r['difference']['mean_abs'] for r in data])),
            max_abs=max(r['difference']['max_abs'] for r in data),
            event_regions=sum(len(r['region_refresh'].get('event_regions',[])) for r in data))
    result=dict(aggregate=aggregate,frames=len(infos),sample_tokens=[i['token'] for i in infos],
        timestamp_span_s=(infos[-1]['timestamp']-infos[0]['timestamp'])/1e6,
        checkpoint=args.checkpoint,backbone_fingerprint=block_fingerprint(model),numerical_policy=NUMERICAL_POLICY,
        validation_kind='wsgrr_quantitative_v1',natural=rows,native_measurements=native_rows,
        controlled_microbenchmarks=[],m1_predictions=args.m1_predictions,
        m1_prediction_sha256=sha256(args.m1_predictions),production_approvals_added=0,production_ready=False,
        downstream_quality='PENDING independent Detection, Map, full VLM evaluation',
        coverage=dict(classes=sorted({str(n) for i in infos for n in i.get('gt_names',[])}),
            confirmed_disappearance='UNAVAILABLE: M1 absence is not confirmation',
            traffic_signal_state='UNAVAILABLE: no M1 signal-state producer',
            visibility='M1 RGB support and occlusion hypotheses, not independent ground truth'),
        timing_scope='Embedding, M1 admission, scoring, prefix + full suffix, atomic publication, diagnostics; excludes M1 precheck/read (separate) and downstream tasks',
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
            cpu_affinity=sorted(__import__('os').sched_getaffinity(0))))
    write(out/'results.json',result)
    if args.profile_block_costs:
        groups={}
        for r in cost_records:groups.setdefault((r['key'],r['updated']),[]).append(r)
        records=[]
        for group in groups.values():
            record=dict(max(group,key=lambda r:r['incremental_wall_ms']/max(r['full_wall_ms'],1e-9)))
            record.update(measurements=group,evidence_cases=[c for r in group for c in r['evidence_cases']],
                numerical_verified=all(r['numerical_verified'] for r in group),
                numerical_evidence=dict(path='results.json',sha256=sha256(out/'results.json')))
            records.append(record)
        write(out/'window_costs_provisional.json',dict(schema_version=SCHEMA_VERSION,records=records,
            validation_sha256=sha256(out/'results.json'),backbone_fingerprint=block_fingerprint(model),
            formal_quality_policy=dict(approved_counts=[]),production_ready=False))
