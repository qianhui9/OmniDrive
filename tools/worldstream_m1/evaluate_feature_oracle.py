"""Offline GPU hooks: spatially aligned feature differences, never reuse authorization."""
from collections import defaultdict
import numpy as np
from .io_utils import write_json


def bilinear(feature, points, shape):
    """HWC grid centres, corresponding to patch centres in the processed image."""
    h,w=feature.shape[:2]; ih,iw=shape
    xy=points*np.array([w/iw,h/ih])-.5
    inside=(xy[:,0]>=0)&(xy[:,0]<=w-1)&(xy[:,1]>=0)&(xy[:,1]<=h-1)
    xy=np.clip(xy,[0,0],[w-1,h-1]); lo=np.floor(xy).astype(int); hi=np.minimum(lo+1,[w-1,h-1]); alpha=xy-lo
    x,y=lo[:,0],lo[:,1]; xx,yy=hi[:,0],hi[:,1]
    value=(feature[y,x].astype(np.float32)*(1-alpha[:,0,None])*(1-alpha[:,1,None])
          +feature[y,xx].astype(np.float32)*alpha[:,0,None]*(1-alpha[:,1,None])
          +feature[yy,x].astype(np.float32)*(1-alpha[:,0,None])*alpha[:,1,None]
          +feature[yy,xx].astype(np.float32)*alpha[:,0,None]*alpha[:,1,None])
    return value,inside


def errors(a,b):
    l2=np.linalg.norm(a-b,axis=-1)/np.maximum(np.linalg.norm(b,axis=-1),1e-8)
    cosine=(a*b).sum(-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-8)
    return l2,cosine


class FeatureOracle:
    def __init__(self,model,root):
        self.model=model; self.root=root; self.previous={}; self.rows=[]; self.branch=[]; self.hooks=[]
        backbone=model.img_backbone
        layers={'patch_embed':backbone.patch_embed,'window_block_0':backbone.blocks[0],
                'global_block_2':backbone.blocks[2],'backbone_final':backbone}
        for name,module in layers.items():
            self.hooks.append(module.register_forward_hook(self.spatial_hook(name)))
        for name,module in [('object_carrier',model.pts_bbox_head),('map_carrier',model.map_head)]:
            self.hooks.append(module.register_forward_hook(self.branch_hook(name)))

    def spatial_hook(self,name):
        def callback(module,inputs,out):
            import torch
            x=out[0] if isinstance(out,(list,tuple)) else out
            if name=='backbone_final': x=x.permute(0,2,3,1)
            current=x.detach().to(device='cpu',dtype=torch.float16).numpy().copy()
            manager=self.model.m1_manager; result=manager.last_result
            previous=self.previous.get(name)
            order=result.diagnostics['execution_context']['camera_order']
            if previous is not None and previous[0]==result.scene_id:
                old=previous[1]; old_timestamp=previous[2]
                for ci,cid in enumerate(order):
                    evidence=manager.last_evidence.get(cid)
                    if evidence is None or not len(evidence.source_points): continue
                    reliable=evidence.correspondence_valid & evidence.geometry_valid
                    # Comparability gate is common to accepted and rejected states;
                    # failed correspondences are excluded, never counted as zero error.
                    p=evidence.source_points; q=evidence.current_points
                    shape=evidence.stable_mask.shape
                    oldv,inside0=bilinear(old[ci],p,shape); newv,inside1=bilinear(current[ci],q,shape)
                    valid=reliable&inside0&inside1
                    l2,cosine=errors(oldv,newv)
                    if evidence.dense_source_map is not None:
                        # Independent full-forward measurement at actually accepted pixels.
                        # Fixed 8px evaluation grid; does not change online evidence.
                        yy,xx=np.mgrid[4:shape[0]:8,4:shape[1]:8]
                        xy=np.stack([xx.ravel(),yy.ravel()],axis=1)
                        keep=~manager.last_masks[cid][xy[:,1],xy[:,0]]
                        xy=xy[keep]
                        if len(xy):
                            mapped=evidence.dense_source_map[xy[:,1],xy[:,0]]
                            a,ia=bilinear(old[ci],mapped,shape); b,ib=bilinear(current[ci],xy,shape)
                            comparable=ia&ib
                            if comparable.any():
                                dl2,dcos=errors(a[comparable],b[comparable])
                                self.rows.append(dict(scene_id=result.scene_id,frame_id=result.frame_id,
                                    anchor_id=None,camera=cid,layer=name,patch_group='dense_stable_pixel',
                                    accepted=True,points=int(comparable.sum()),relative_l2=float(dl2.mean()),
                                    relative_l2_p50=float(np.median(dl2)),relative_l2_p95=float(np.percentile(dl2,95)),
                                    cosine_similarity=float(dcos.mean()),
                                    fraction_l2_le_01=float((dl2<=.1).mean()),
                                    fraction_l2_le_02=float((dl2<=.2).mean()),
                                    transformer_dependency_certified=False))
                    for aid,check in result.diagnostics['anchors'].items():
                        anchor=manager.store.anchors[aid]; obs=anchor.observation_support.get(cid)
                        if obs is None or abs(anchor.source_timestamp-old_timestamp)>1e-5: continue
                        r=obs.region
                        selected=valid&(p[:,0]>=r[0])&(p[:,0]<r[2])&(p[:,1]>=r[1])&(p[:,1]<r[3])
                        if not selected.any(): continue
                        self.rows.append(dict(scene_id=result.scene_id,frame_id=result.frame_id,anchor_id=aid,
                            camera=cid,layer=name,accepted=check['valid'],points=int(selected.sum()),
                            relative_l2=float(np.mean(l2[selected])),cosine_similarity=float(np.mean(cosine[selected])),
                            mean_observation_residual=float(np.mean(evidence.patch_residual[selected])),
                            transformer_dependency_certified=False))
                    for label,selected in [('stable_patch',valid & evidence.supported),('rejected_patch',valid & ~evidence.supported)]:
                        if selected.any():
                            self.rows.append(dict(scene_id=result.scene_id,frame_id=result.frame_id,anchor_id=None,
                                camera=cid,layer=name,patch_group=label,accepted=label=='stable_patch',points=int(selected.sum()),
                                relative_l2=float(np.mean(l2[selected])),cosine_similarity=float(np.mean(cosine[selected]))))
            self.previous[name]=(result.scene_id,current,result.timestamp)
        return callback

    def branch_hook(self,name):
        def callback(module,inputs,out):
            x=out[1].detach().float().mean(dim=1).cpu().numpy()
            manager=self.model.m1_manager; result=manager.last_result; key='branch_'+name
            if key in self.previous and self.previous[key][0]==result.scene_id:
                l2,cos=errors(self.previous[key][1],x)
                self.branch.append(dict(frame_id=result.frame_id,branch=name,mean_pool_relative_l2=float(l2.mean()),
                    mean_pool_cosine=float(cos.mean()),comparison='branch mean pool; no object/token one-to-one alignment'))
            self.previous[key]=(result.scene_id,x)
        return callback

    def close(self):
        for h in self.hooks: h.remove()
        write_json(self.root/'feature_oracle_records.json',self.rows)
        write_json(self.root/'carrier_oracle_records.json',self.branch)
        summary={}
        for layer in sorted({r['layer'] for r in self.rows}):
            groups={}
            for accepted in (True,False):
                rs=[r for r in self.rows if r['layer']==layer and r['anchor_id'] is not None and r['accepted']==accepted]
                groups['accepted' if accepted else 'rejected']=dict(anchor_camera_records=len(rs),
                    relative_l2_mean=float(np.mean([r['relative_l2'] for r in rs])) if rs else None,
                    cosine_mean=float(np.mean([r['cosine_similarity'] for r in rs])) if rs else None)
            summary[layer]=groups
        write_json(self.root/'feature_oracle_metrics.json',dict(status='measured',layers=summary,branch=self.branch,
            spatial_comparison='Bilinear sample at same measured correspondences, only forward/backward and camera geometry valid; no index reuse.',
            dependencies={'patch_embed':'16x16 local convolution; bilinear comparison is approximate and footprints may differ',
                          'window_block_0':'16x16 tokens in window, positional embeddings and cross-token attention',
                          'global_block_2':'global all spatial tokens per camera',
                          'backbone_final':'global and window dependencies accumulated across 24 blocks',
                          'carriers':'no object/token bijection; branch mean pool only'},
            limitation='Spatial feature difference is not downstream quality. Overlapping anchor/camera records and frames are correlated; small-scene experiment.'))
