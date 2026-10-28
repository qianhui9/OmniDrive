"""Frozen native teacher/LLM distillation of the fixed-budget WRCPS adapter.

Consumes real, hashed native captures and original language answer labels.
Training is deliberately refused when M1 cannot produce stable world facts.
Validation loss is development evidence, never the four-metric quality gate.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import torch
import torch.nn.functional as F
from tools.worldstream_m123.benchmark.common import sha256, write
from worldstream.m3.stable_world import StableAnchorManager, StableWorldSnapshot
from worldstream.m3.world_prefix_composer import WorldPrefixComposer,SpatiallyFaithfulComposer
from tools.worldstream_m123.latency_controls import load_capture


def validate_manifest(manifest):
    splits = manifest['scene_splits']
    allowed = {'train', 'validation', 'calibration', 'independent_acceptance'}
    if not splits or set(splits.values())-allowed or not all(isinstance(s,str) and s for s in splits):
        raise ValueError('Invalid scene partition')
    if not {'train', 'validation'} <= set(splits.values()):
        raise ValueError('Disjoint training and validation scenes are required')
    samples = manifest['samples']
    if not samples: raise ValueError('Real training captures are missing')
    from nuscenes.utils.splits import create_splits_scenes
    scene_table = json.loads(Path(manifest['nuscenes_scene_table']).read_text())
    names = {s['token']:s['name'] for s in scene_table}
    if manifest.get('semantic_qualification'):
        logs = {s['token']:s['log_token'] for s in scene_table}
        from worldstream.m3.semantic_core import SemanticQualifier
        qualifier=SemanticQualifier(manifest['semantic_qualification'])
        reserved=set(qualifier.policy['calibration_log_ids']+qualifier.policy['validation_log_ids'])
        reserved.update(logs[s] for s,p in splits.items() if p in ('calibration','independent_acceptance'))
        train_logs={logs[s] for s,p in splits.items() if p=='train'}
        validation_logs={logs[s] for s,p in splits.items() if p=='validation'}
        if train_logs&validation_logs or (train_logs|validation_logs)&reserved:
            raise ValueError('Adapter train/validation/qualification/acceptance logs must be disjoint')
    else:qualifier=None
    official_train = set(create_splits_scenes()['train'])
    prior = {}; seen = set(); valid_pairs = Counter(); represented = Counter()
    for sample in samples:
        scene = sample['scene_id']; split = splits[scene]
        if split not in ('train', 'validation'):
            raise ValueError('Calibration/acceptance data must not enter adapter training')
        represented[split] += 1
        if names.get(scene) not in official_train:
            raise ValueError('Adapter development must use official nuScenes training scenes')
        key = (scene, sample['frame_id'])
        if key in seen: raise ValueError('Duplicate frame capture')
        seen.add(key)
        if sha256(sample['capture']) != sample['capture_sha256']:
            raise ValueError('Capture digest mismatch')
        snapshot = StableWorldSnapshot(**dict(sample['stable_snapshot'],
            facts=tuple(tuple(f) for f in sample['stable_snapshot']['facts'])))
        if snapshot.content_sha256 != sample['stable_snapshot']['content_sha256']:
            raise ValueError('Stable snapshot digest mismatch')
        if (snapshot.scene_id,snapshot.frame_id) != key:
            raise ValueError('Training snapshot/frame binding mismatch')
        if not snapshot.facts and qualifier is None:
            raise ValueError('No qualified stable facts: adapter training cannot establish WRCPS feasibility')
        if qualifier is not None and snapshot.qualification_id!=qualifier.sha256:
            raise ValueError('Adapter snapshot has different semantic qualification')
        context = dict(scene_id=scene,frame_id=sample['frame_id'],timestamp=snapshot.timestamp,
            state_epoch=snapshot.state_epoch,state_version=snapshot.state_version,
            state_execution_signature=snapshot.execution_signature)
        manager = prior.setdefault(scene,StableAnchorManager())
        _, hit = manager.update(snapshot,context); valid_pairs[split] += int(hit)
        questions = json.loads(Path(sample['question_source']).read_text())
        if sha256(sample['question_source']) != sample['question_source_sha256']:
            raise ValueError('Natural question source changed')
        for pair in sample['pairs']:
            if not any(p.get('question') == pair['question'] and p.get('answer') == pair['answer'] for p in questions):
                raise ValueError('Training labels must be original real natural-language labels')
    if any(not represented[s] or not valid_pairs[s] for s in ('train','validation')):
        raise ValueError('Both train and validation need real frames and cross-frame stable anchors')
    return dict(frames=len(samples),stable_transitions=dict(valid_pairs),
        training_scene_ids=sorted(s for s,p in splits.items() if p=='train'),
        validation_scene_ids=sorted(s for s,p in splits.items() if p=='validation'))


def multimodal_embeddings(base, ids, carriers, answer_ids):
    ids = ids.reshape(-1)
    markers = (ids == -200).nonzero().flatten()
    if len(markers) != 1: raise ValueError('Training requires one native visual marker')
    pos = int(markers[0])
    embedding = base.get_input_embeddings()
    # Match LlavaMetaForCausalLM's actual native assembly: detection/map heads
    # publish FP32 carriers, then the assembler casts them to the LLM dtype.
    carriers=carriers.to(device=embedding.weight.device,dtype=embedding.weight.dtype)
    answer_ids = answer_ids.reshape(-1).to(ids.device)
    prompt = torch.cat((embedding(ids[:pos]).unsqueeze(0),carriers,
                        embedding(ids[pos+1:]).unsqueeze(0)),1)
    full = torch.cat((prompt,embedding(answer_ids).unsqueeze(0)),1)
    return full, prompt.shape[1]-1, answer_ids


def distillation_loss(student_logits, teacher_logits, labels, task_weight=1., critical=False,
                      spatial_mask=None,spatial_weight=0.):
    # Answer positions only: no task supervision on visual pseudo-token IDs.
    kl = F.kl_div(F.log_softmax(student_logits.float(),-1),
        F.softmax(teacher_logits.float(),-1),reduction='batchmean')
    ce = F.cross_entropy(student_logits.float(),labels)
    loss = kl+task_weight*(2. if critical else 1.)*ce
    spatial=kl.new_zeros(());aligned=conflicts=0
    if spatial_mask is not None:
        agrees=teacher_logits.argmax(-1)==labels
        selected=spatial_mask & agrees
        aligned=int(selected.sum());conflicts=int((spatial_mask & ~agrees).sum())
        if aligned:
            spatial=F.kl_div(F.log_softmax(student_logits[selected].float(),-1),
                F.softmax(teacher_logits[selected].float(),-1),reduction='batchmean')
            loss=loss+spatial_weight*spatial
    return loss, dict(kl=float(kl.detach()),task_ce=float(ce.detach()),critical=critical,
        spatial_kl=float(spatial.detach()),spatial_aligned_tokens=aligned,teacher_reference_direction_conflicts=conflicts)


def train(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(Path(args.manifest).read_text())
    evidence=validate_manifest(manifest)
    write(root/'data_validation.json',evidence)
    if args.preflight_only:return
    if args.epochs < 1 or args.lr <= 0 or args.task_weight < 0:raise ValueError('Invalid training parameters')
    import projects.mmdet3d_plugin
    from tools.worldstream_m123.benchmark.worker import effective_config,load_model
    from worldstream.m3.visual_state import base_model
    from worldstream.m3.task_conditioner import task_profile
    torch.manual_seed(args.seed);random.seed(args.seed)
    cfg=effective_config('B0','language',root)
    model=load_model(cfg,args.checkpoint,root).eval().requires_grad_(False)
    base=base_model(model.lm_head)
    # Captured carriers make the native perception modules unnecessary on GPU
    # for distillation. Move them without modifying any weights.
    for name,module in model.named_children():
        if name!='lm_head':module.cpu()
    torch.cuda.empty_cache()
    base_hash=sha256(args.checkpoint)
    if manifest['base_weights_sha256'] != base_hash:raise ValueError('Teacher checkpoint mismatch')
    composer_class=SpatiallyFaithfulComposer if args.spatial else WorldPrefixComposer
    composer=composer_class(base.config.hidden_size,args.stable_tokens,args.rank).to('cuda')
    optimizer=torch.optim.AdamW(composer.parameters(),lr=args.lr)
    write(root/'training_config.json',dict(vars(args),**evidence,manifest_sha256=sha256(args.manifest),
        base_weights_sha256=base_hash,loss='answer KL + task CE (critical query weight 2)',
        temporal_loss='zero by construction for the same immutable world facts',
        spatial_supervision='0.5 answer-position KL on directional reference tokens where frozen Teacher argmax agrees; conflicts not promoted to truth' if args.spatial else 'none',
        frozen='all native perception, heads, and LLM parameters',spec=composer.spec))
    steps=0;best=float('inf')
    for epoch in range(args.epochs):
        for split in ('train','validation'):
            composer.train(split=='train');managers={};losses=[];core_losses=[]
            for sample in manifest['samples']:
                if manifest['scene_splits'][sample['scene_id']] != split:continue
                capture=load_capture(sample['capture'])
                snapshot=StableWorldSnapshot(**dict(sample['stable_snapshot'],
                    facts=tuple(tuple(f) for f in sample['stable_snapshot']['facts'])))
                if snapshot.content_sha256 != capture['stable_snapshot'].content_sha256:
                    raise ValueError('Training snapshot differs from the real causal capture')
                snapshot.validate(capture['stable_context'])
                manager=managers.setdefault(snapshot.scene_id,StableAnchorManager())
                anchor,_=manager.update(snapshot,capture['stable_context'])
                native=capture['carriers'].cuda().detach()
                for pair in sample['pairs']:
                    torch.cuda.reset_peak_memory_stats()
                    index=pair['question_index']
                    if capture['questions'][index] != pair['question']:raise ValueError('Capture question ordering mismatch')
                    ids=capture['ids'][index].cuda()
                    answer=model.tokenizer(pair['answer'],add_special_tokens=False,return_tensors='pt').input_ids.cuda()
                    eos=model.tokenizer.eos_token_id
                    answer=torch.cat((answer,torch.tensor([[eos]],device='cuda')),1)
                    # Teacher retains original layout and original checkpoint.
                    with torch.no_grad():
                        teacher,start,labels=multimodal_embeddings(base,ids,native,answer)
                        target=base(inputs_embeds=teacher,use_cache=False).logits[:,start:start+labels.numel()].detach()
                    with torch.set_grad_enabled(split=='train'):
                        composed=composer(native.float(),anchor).to(native.dtype)
                        student,offset,_=multimodal_embeddings(base,ids,composed,answer)
                        decoder=base.model
                        decoder.gradient_checkpointing=bool(args.gradient_checkpointing and split=='train')
                        # Only the decoder checkpoint dispatch flag changes.
                        # Frozen LoRA/dropout children stay in eval mode.
                        decoder.training=bool(args.gradient_checkpointing and split=='train')
                        logits=base(inputs_embeds=student,use_cache=False).logits[:,offset:offset+labels.numel()]
                        critical=task_profile(pair['question'])[1] in ('object_ego_motion','map_relations','full_visual')
                        spatial_mask=None
                        if args.spatial:
                            directional={'left','right','front','behind','ahead','straight','opposite','rear'}
                            spatial_mask=torch.tensor([model.tokenizer.decode([int(t)]).strip().lower() in directional for t in labels],device=labels.device)
                        loss,details=distillation_loss(logits[0],target[0],labels,args.task_weight,critical,
                            spatial_mask=spatial_mask,spatial_weight=.5 if args.spatial else 0.)
                        if not torch.isfinite(loss):raise ValueError('Nonfinite adapter distillation loss')
                    updated=split=='train' and loss.requires_grad
                    if updated:
                        optimizer.zero_grad(set_to_none=True)
                        try:loss.backward()
                        except torch.cuda.OutOfMemoryError:
                            write(root/'memory_probe.json',dict(status='OOM',step=steps,
                                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                gradient_checkpointing=args.gradient_checkpointing));raise
                        if not steps:
                            write(root/'memory_probe.json',dict(status='PASS',step=0,
                                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                input_tokens=student.shape[1],answer_tokens=labels.numel(),
                                gradient_checkpointing=args.gradient_checkpointing))
                        torch.nn.utils.clip_grad_norm_(composer.parameters(),1.);optimizer.step();steps+=1
                    decoder.training=False;decoder.gradient_checkpointing=False
                    losses.append(float(loss.detach()))
                    if anchor.stable_state_valid:core_losses.append(losses[-1])
                    with (root/'training_log.jsonl').open('a') as f:
                        f.write(json.dumps(dict(epoch=epoch,split=split,step=steps,frame_id=snapshot.frame_id,
                            scene_id=snapshot.scene_id,loss=losses[-1],core_valid=anchor.stable_state_valid,
                            optimizer_updated=updated,
                            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),**details))+'\n')
            if not losses:raise ValueError('Empty training/validation partition')
            selection_losses=core_losses if args.spatial else losses
            if not selection_losses:raise ValueError('No real core-valid validation/training questions')
            mean=sum(selection_losses)/len(selection_losses)
            if split=='validation' and mean < best:
                best=mean
                torch.save(dict(spec=composer.spec,layout_version=composer.layout.order_contract,state_dict=composer.state_dict(),
                    trained_steps=steps,training_scene_ids=evidence['training_scene_ids'],
                    validation_scene_ids=evidence['validation_scene_ids'],base_weights_sha256=base_hash,
                    data_manifest_sha256=sha256(args.manifest),seed=args.seed,validation_loss=best,
                    semantic_qualification_sha256=(sha256(manifest['semantic_qualification'])
                        if manifest.get('semantic_qualification') else '')),root/'adapter.pt')
    write(root/'training_result.json',dict(trained_steps=steps,best_validation_loss=best,
        adapter_sha256=sha256(root/'adapter.pt'),deployment_qualified=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--checkpoint',default='ckpts/iter_10548.pth')
    p.add_argument('--stable-tokens',type=int,default=128);p.add_argument('--rank',type=int,default=128)
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--task-weight',type=float,default=1.);p.add_argument('--seed',type=int,default=20261028)
    p.add_argument('--no-gradient-checkpointing',dest='gradient_checkpointing',action='store_false',default=True)
    p.add_argument('--spatial',action='store_true',help='SF adaptive Composer and fixed directional answer KL; select validation checkpoint on core-valid questions')
    p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    if set(os.sched_getaffinity(0)) & {4,5}:p.error('Use taskset -c 0-3,6-31')
    try:train(a)
    except torch.cuda.OutOfMemoryError:
        write(Path(a.output)/'memory_probe.json',dict(status='OOM',
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gradient_checkpointing=a.gradient_checkpointing))
        raise


if __name__=='__main__':main()
