"""Audit actual cached window producers, including displaced research roots."""


def window_lineage(runs):
    errors=[];units=0;displaced=0
    for run in runs:
        roots={}
        for frame in run['frames']:
            diag=frame['runtime'];published=diag.get('output_window_provenance',{})
            for block,values in published.items():
                for index,unit in enumerate(values):
                    origin=unit.get('computation_frame_id');root=unit.get('computation_window_index',index)
                    key=(str(block),origin,root)
                    detail=dict(mode=run['mode'],repeat=run['repeat'],frame=frame['frame'],block=block,index=index,root=root,origin=origin)
                    units+=1;displaced+=int(root!=index)
                    if origin==frame['frame']:
                        if root!=index:errors.append(dict(reason='new_computation_has_wrong_root_window',**detail))
                        roots[key]=(unit.get('computation_version'),unit.get('computation_timestamp'),unit.get('camera_id'))
                    actual=(unit.get('computation_version'),unit.get('computation_timestamp'),unit.get('camera_id'))
                    if roots.get(key)!=actual:errors.append(dict(reason='root_producer_not_recorded_or_changed',**detail))
                    if root!=index and unit.get('reuse_mode')!='experimental_conditional':
                        errors.append(dict(reason='displaced_context_not_conditional',**detail))
                new=sum(u.get('computation_frame_id')==frame['frame'] for u in values)
                executed=sum(w['shape'][0] for w in frame['actual_GPU_work'] if str(w['block'])==str(block) and w['component']=='attention')
                if new!=executed:errors.append(dict(reason='producer_count_differs_from_actual_attention_work',mode=run['mode'],frame=frame['frame'],block=block,new=new,executed=executed))
    return dict(status='passed' if units and not errors else 'failed',units=units,displaced_root_units=displaced,errors=errors,
        scope='Checks root frame/version/timestamp/camera/window identity against recorded actual computation; distinct from physical, numerical or task certification.')
