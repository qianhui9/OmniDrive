"""R0-required utilities; historical alternative runners have been retired."""

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
        if job.get('save_equivalence_reference'):
            dest=Path('/tmp/omnidrive-critical-equivalence-features');dest.mkdir(exist_ok=True)
            torch.save(x.cpu(),dest/f'{fi:02d}.pt')
        if job.get('compare_equivalence_reference'):
            reference=torch.load(Path('/tmp/omnidrive-critical-equivalence-features')/f'{fi:02d}.pt',map_location=x.device)
            delta=(x-reference).float();receipt['equivalence_feature_difference']=dict(rmse=float(delta.square().mean().sqrt()),
                relative_l2=float(delta.norm()/reference.float().norm().clamp_min(1e-12)),max_abs=float(delta.abs().max()))
    return receipt
