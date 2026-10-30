"""Request-local projected-box occlusion; exact original strict comparisons."""
import numpy as np


def projected_occlusion(projections, camera_id, config):
    keys=list(projections)
    rows=[projections[k][camera_id] for k in keys]
    keep=[i for i,p in enumerate(rows) if p.region is not None]
    result={k:False for k in keys}
    if not keep:return result
    boxes=np.asarray([rows[i].region for i in keep],dtype=float)
    depths=np.asarray([rows[i].depth for i in keep],dtype=float)
    lo=np.maximum(boxes[:,None,:2],boxes[None,:,:2])
    hi=np.minimum(boxes[:,None,2:],boxes[None,:,2:])
    side=np.maximum(0.,hi-lo)
    overlap=side[...,0]*side[...,1]
    area=np.maximum(0.,boxes[:,2]-boxes[:,0])*np.maximum(0.,boxes[:,3]-boxes[:,1])
    nearer=depths[None,:]<depths[:,None]-config.occlusion_depth_margin_m
    mask=nearer & (overlap>config.occlusion_overlap_fraction*area[:,None])
    np.fill_diagonal(mask,False)
    flags=mask.any(1)
    result.update((keys[i],bool(v)) for i,v in zip(keep,flags))
    return result
