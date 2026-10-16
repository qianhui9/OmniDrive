"""Frozen reference, protocol selection, and offline-only refinement diagnostics."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path('outputs/module2_refinement')


def reference():
    path=ROOT/'baseline_source/worldstream/m2/__init__.py'
    spec=importlib.util.spec_from_file_location('m2_refinement_reference',path,
        submodule_search_locations=[str(path.parent)])
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def select_protocol(infos):
    groups={}
    for index,info in enumerate(infos):groups.setdefault(info['scene_token'],[]).append(index)
    candidates=[]
    def yaw(q):
        w,x,y,z=q;return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
    for scene,indices in sorted(groups.items()):
        indices=sorted(indices,key=lambda j:infos[j]['timestamp'])
        for start in range(0,len(indices)-4,4):
            picked=[indices[start+k] for k in (0,1,2,4)]
            items=[infos[j] for j in picked]
            angles=np.unwrap([yaw(v['ego2global_rotation']) for v in items])
            turn=float(np.ptp(angles));travel=float(np.linalg.norm(np.asarray(items[-1]['ego2global_translation'])-items[0]['ego2global_translation']))
            speeds=np.concatenate([np.linalg.norm(np.nan_to_num(v['gt_velocity']),axis=-1) for v in items])
            lowvis=[str(z) in ('1','2') for v in items for camera in v['visibilities'] for z in camera]
            candidates.append(dict(scene=scene,indices=picked,turn_rad=turn,travel_m=travel,
                max_object_speed_mps=float(speeds.max()) if len(speeds) else 0.,
                low_visibility_fraction=float(np.mean(lowvis)) if lowvis else 0.,
                description=items[0]['description'],scene_name=items[0]['scene_name']))
    selected=[];used=set()
    rules=[('turn',lambda r:r['turn_rad']),
           ('straight',lambda r:r['travel_m'] if r['turn_rad']<.03 else -1),
           ('moving_objects',lambda r:r['max_object_speed_mps']),
           ('occlusion',lambda r:r['low_visibility_fraction'])]
    for name,key in rules:
        row=max((r for r in candidates if r['scene'] not in used),key=key)
        row=dict(row,category=name,sample_ids=[infos[j]['token'] for j in row['indices']],
            gaps_s=np.diff([infos[j]['timestamp']/1e6 for j in row['indices']]).tolist())
        used.add(row['scene']);selected.append(row)
    return selected


def arrays(value):
    if hasattr(value,'tensor'):value=value.tensor
    if hasattr(value,'detach'):
        import torch
        value=value.detach().cpu()
        if value.dtype==torch.bfloat16:value=value.float()
        return value.numpy().copy()
    if isinstance(value,np.ndarray):return value.copy()
    if isinstance(value,dict):return {k:arrays(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [arrays(v) for v in value]
    return value


def numeric_diff(a,b):
    leaves=[];shape_mismatches=[]
    def visit(x,y,path):
        if isinstance(x,np.ndarray) and isinstance(y,np.ndarray):
            if x.shape!=y.shape:shape_mismatches.append(path);return
            if x.size:
                delta=x.astype(float)-y.astype(float)
                leaves.append(dict(path=path,max_abs=float(np.max(np.abs(delta))),
                    relative_l2=float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(y.ravel()),1e-12))))
        elif isinstance(x,dict) and isinstance(y,dict):
            for k in x.keys()&y.keys():visit(x[k],y[k],path+'.'+k)
        elif isinstance(x,list) and isinstance(y,list):
            if len(x)!=len(y):shape_mismatches.append(path);return
            for i,(u,v) in enumerate(zip(x,y)):visit(u,v,path+f'[{i}]')
    visit(a,b,'root')
    return dict(max_abs=max((r['max_abs'] for r in leaves),default=0.),
        leaves=leaves,shape_mismatches=shape_mismatches)


def detection_diff(actual,baseline):
    a,b=actual[0]['pts_bbox'],baseline[0]['pts_bbox']
    aa,bb=a['boxes_3d'],b['boxes_3d']
    if len(aa) and len(bb):
        distance=np.linalg.norm(aa[:,None,:3]-bb[None,:,:3],axis=-1)
        penalty=(a['labels_3d'][:,None]!=b['labels_3d'][None,:])*1e6
        rows,cols=linear_sum_assignment(distance+penalty)
        good=penalty[rows,cols]==0
        distances=distance[rows[good],cols[good]]
        return dict(actual_count=len(aa),baseline_count=len(bb),matched_same_class=int(good.sum()),
            max_center_difference_m=float(distances.max()) if len(distances) else None,
            mean_center_difference_m=float(distances.mean()) if len(distances) else None,
            pairs_over_2m=int((distances>2).sum()),unmatched=abs(len(aa)-len(bb))+2*int((~good).sum()))
    return dict(actual_count=len(aa),baseline_count=len(bb),unmatched=abs(len(aa)-len(bb)))


def error_location(a,b):
    import torch
    delta=(a.float()-b.float()).abs()
    flat=int(delta.argmax());coord=np.unravel_index(flat,tuple(a.shape))
    return dict(max_abs=float(delta.flatten()[flat]),index=[int(v) for v in coord],
        actual=float(a[coord]),baseline=float(b[coord]),
        relative_l2=float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)))


def window_statistics(manager,projection,window_size=16):
    result={}
    for cid,mask in manager.last_masks.items():
        h,w=mask.shape;ph,pw=projection.kernel_size;th,tw=h//ph,w//pw
        stable=(~mask[:th*ph,:tw*pw]).reshape(th,ph,tw,pw).all(axis=(1,3))
        evidence=manager.last_evidence.get(cid)
        changed=np.zeros_like(stable)
        if evidence is not None:
            changed=evidence.changed_mask[:th*ph,:tw*pw].reshape(th,ph,tw,pw).any(axis=(1,3))
        stable &= ~changed;unknown=~(stable|changed)
        rows=[]
        for y in range(0,th,window_size):
            for x in range(0,tw,window_size):
                sl=(slice(y,y+window_size),slice(x,x+window_size))
                rows.append(dict(row=y//window_size,col=x//window_size,
                    stable=int(stable[sl].sum()),changed=int(changed[sl].sum()),unknown=int(unknown[sl].sum()),
                    valid_tokens=int(stable[sl].size),candidate=bool(stable[sl].all())))
        from scipy.ndimage import label
        components,n=label(stable)
        counts=np.bincount(components.ravel())[1:]
        result[cid]=dict(stable_patch_mask=stable.astype(int).tolist(),windows=rows,
            stable_patches=int(stable.sum()),changed_patches=int(changed.sum()),unknown_patches=int(unknown.sum()),
            largest_stable_component=int(counts.max()) if n else 0,
            update_closure_patch=int((~stable).sum()),
            update_closure_window=sum(not r['candidate'] for r in rows),
            global_closure_tokens=int(stable.size) if not stable.all() else 0,
            transformer_dependency_certified=False)
    return result


def memory_alignment(actual,baseline):
    """Read-only same-age slot matching; never rewrites or repairs model memory."""
    results={}
    for branch in actual:
        a,b=actual[branch],baseline[branch]
        aa,bb=a.get('memory_reference_point'),b.get('memory_reference_point')
        if aa is None or bb is None:continue
        if aa.shape!=bb.shape:
            results[branch]=dict(shape_mismatch=True);continue
        n=aa.shape[1];permutation=np.full(n,-1,int)
        ta=np.asarray(a['memory_timestamp'])[0,:,0]
        tb=np.asarray(b['memory_timestamp'])[0,:,0]
        pa=np.asarray(a['memory_egopose'])[0,:,3,3]>.5
        pb=np.asarray(b['memory_egopose'])[0,:,3,3]>.5
        point_errors=[];groups=[]
        for age in np.unique(ta):
            for valid in (False,True):
                left=np.flatnonzero((ta==age)&(pa==valid));right=np.flatnonzero((tb==age)&(pb==valid))
                if not len(left):continue
                if len(left)!=len(right):groups.append(dict(age=float(age),shape_mismatch=True));continue
                av=aa[0,left].reshape(len(left),-1,3);bv=bb[0,right].reshape(len(right),-1,3)
                distance=np.linalg.norm(av[:,None]-bv[None,:],axis=-1).mean(-1)
                rows,cols=linear_sum_assignment(distance)
                permutation[left[rows]]=right[cols];point_errors.extend(distance[rows,cols].tolist())
                groups.append(dict(age=float(age),valid=bool(valid),count=len(left),
                    reordered=int((left[rows]!=right[cols]).sum()),max_mean_point_error_m=float(distance[rows,cols].max())))
        matched=np.flatnonzero(permutation>=0);leaves={}
        for name,value in a.items():
            other=b.get(name)
            if (isinstance(value,np.ndarray) and isinstance(other,np.ndarray) and value.ndim>=2
                    and value.shape[1]==n and value.shape==other.shape):
                delta=value[:,matched].astype(float)-other[:,permutation[matched]].astype(float)
                leaves[name]=dict(max_abs=float(np.max(np.abs(delta))) if delta.size else 0.,
                    relative_l2=float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(other[:,permutation[matched]].ravel()),1e-12)))
        results[branch]=dict(matched=len(matched),total=n,groups=groups,permutation=permutation.tolist(),
            max_matched_point_error_m=max(point_errors,default=0),leaves=leaves)
    return results


def map_alignment(actual,baseline):
    a,b=actual[0],baseline[0]
    ap,asc=a;bp,bsc=b
    av=np.asarray(ap).reshape(len(ap),-1,3);bv=np.asarray(bp).reshape(len(bp),-1,3)
    distances=np.linalg.norm(av[:,None]-bv[None,:],axis=-1).mean(-1)
    rows,cols=linear_sum_assignment(distances)
    return dict(actual_count=len(ap),baseline_count=len(bp),
        reordered=int((rows!=cols).sum()),max_mean_curve_difference_m=float(distances[rows,cols].max()),
        max_matched_control_coordinate_abs_m=float(np.abs(av[rows]-bv[cols]).max()),
        max_matched_score_abs=float(np.abs(np.asarray(asc)[rows]-np.asarray(bsc)[cols]).max()))
