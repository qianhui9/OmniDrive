"""Same-carrier real frozen-LLM representation diagnostics, outside timings."""
import argparse
import json
import os
from pathlib import Path
import torch
from tools.worldstream_m123.benchmark.common import write,typed_digest,sha256
from tools.worldstream_m123.latency_controls import load_capture
from tools.worldstream_m123.benchmark.gpu_lock import gpu_job_lock
from tools.worldstream_m2.global_residual_benchmark import wait_gpu


def structure(root):
    from worldstream.m3.world_prefix_composer import WorldPrefixComposer,SpatiallyFaithfulComposer
    from worldstream.m3.stable_world import StableAnchorManager
    prior=Path('outputs/m3_semantic_core_wrcps_run01')
    sequence=json.loads((prior/'adaptation_sequence.json').read_text())
    torch.manual_seed(20261028)
    old=WorldPrefixComposer.load_adapter(prior/'adapter_training/adapter.pt',4096,'cpu',torch.float32)
    new=SpatiallyFaithfulComposer(4096,64,128).eval();records=[];managers={}
    # One prespecified first core-valid frame per scene, plus each first cold
    # start. These are existing development captures, not acceptance samples.
    selected=set();cold=set()
    with torch.no_grad():
        for token in sequence['tokens']:
            path=prior/'adaptation_capture/m3_inputs'/(token+'.pt')
            capture=load_capture(path);s=capture['stable_snapshot']
            a,_=managers.setdefault(s.scene_id,StableAnchorManager()).update(s,capture['stable_context'])
            key=(s.scene_id,a.stable_state_valid)
            if key in selected:continue
            selected.add(key);current=capture['carriers'].float()
            legacy=old(current,a);spatial=new(current,a)
            def retained(output):
                return int(sum(bool(torch.eq(output,slot[None,None]).all(-1).any()) for slot in current[0]))
            groups={}
            for kind,values,compressor in [('object',current[:,:256],new.objects),('map',current[:,257:],new.maps)]:
                src,dst,keep=compressor.plan(values)
                cos=torch.nn.functional.cosine_similarity(values[0,src],values[0,dst],dim=-1)
                groups[kind]=dict(removed_slots=len(src),retained_slots=len(keep),merged_destination_slots=len(dst.unique()),
                    exact_unmixed_slots=len(keep)-len(dst.unique()),mean_selected_cosine=float(cos.mean()))
            records.append(dict(scene_id=s.scene_id,frame_id=token,core_valid=a.stable_state_valid,
                legacy_exact_native_slots=retained(legacy),sf_exact_native_slots=retained(spatial),
                sf_no_core_storage_identity=spatial is current,groups=groups,
                scope='Representation structure only: old adapter trained, new adapter untrained; not LLM quality or K optimum evidence'))
    write(root/'structural_diagnostic.json',dict(frames=records,trained_spatial_adapter=False,
        conclusion='Native slot retention is structurally improved; preservation of spatial answers still requires real frozen-LLM validation on core-valid inputs.'))


def no_core(root):
    import projects.mmdet3d_plugin
    from tools.worldstream_m123.benchmark.worker import effective_config,load_model
    from worldstream.m3 import M3Runtime,M3Config
    from worldstream.m3.visual_state import base_model,expand_input,CarrierLayout
    from worldstream.m3.world_prefix_composer import SpatiallyFaithfulComposer
    dest=root/'no_core_equivalence'
    if dest.exists():raise FileExistsError('Preserve diagnostic')
    dest.mkdir(parents=True)
    torch.manual_seed(20261028)
    cfg=effective_config('B0','language',dest)
    model=load_model(cfg,'ckpts/iter_10548.pth',dest).eval().requires_grad_(False)
    base=base_model(model.lm_head)
    for name,module in model.named_children():
        if name!='lm_head':module.cpu()
    torch.cuda.empty_cache()
    runtime=M3Runtime(M3Config(prefix_layout='sf_adaptive',reuse_strategy='exact_prefix',memory_budget_mb=0))
    layout=CarrierLayout(256,1,256)
    prior=Path('outputs/m3_semantic_core_wrcps_run01')
    paths=list((prior/'failure_diagnostic/m3_inputs').glob('*.pt'))
    paths.sort(key=lambda p:load_capture(p)['stable_snapshot'].timestamp)
    records=[]
    with torch.no_grad():
        for path in paths:
            capture=load_capture(path);snapshot=capture['stable_snapshot']
            if snapshot.facts:raise ValueError('No-core regression must use genuinely empty real snapshots')
            native=capture['carriers'].cuda();ctx=dict(capture['context'],stable_state_version=snapshot.state_version,profile=True)
            for qi,(ids,question) in enumerate(zip(capture['ids'],capture['questions'])):
                ids=ids.cuda();cpu_rng,cuda_rng=capture['rng'][qi]
                original=expand_input(model.lm_head,ids,native,carrier_layout=layout)
                expected_logits=base(inputs_embeds=original.embeddings,attention_mask=original.attention_mask,
                    position_ids=original.position_ids,use_cache=False).logits[:,-1].clone()
                torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
                teacher=model.lm_head.generate(inputs=ids.reshape(1,-1),images=native,**capture['generation'])
                torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
                actual,diagnostics=runtime.generate(model.lm_head,[ids],native,context=ctx,questions=[question],
                    layout=layout,stable_world_state=snapshot,**capture['generation'])
                identity=expand_input(model.lm_head,ids,runtime.last_composed_carriers,carrier_layout=layout)
                logits=base(inputs_embeds=identity.embeddings,attention_mask=identity.attention_mask,
                    position_ids=identity.position_ids,use_cache=False).logits[:,-1]
                checks={name:torch.equal(getattr(original,name),getattr(identity,name))
                    for name in ('embeddings','position_ids','attention_mask','token_ids','visual_mask')}
                checks.update(storage_identity=runtime.last_composed_carriers is native,
                    prefill_logits=torch.equal(expected_logits,logits),generated_tokens=torch.equal(teacher,actual[0]))
                record=dict(scene_id=snapshot.scene_id,frame_id=snapshot.frame_id,question_index=qi,question=question,
                    original_capture=str(path),capture_sha256=sha256(path),checks=checks,
                    native_answer=model.tokenizer.batch_decode(teacher,skip_special_tokens=True)[0],
                    sf_answer=model.tokenizer.batch_decode(actual[0],skip_special_tokens=True)[0],
                    native_tokens=teacher.cpu().tolist(),sf_tokens=actual[0].cpu().tolist(),
                    visual=typed_digest(identity.embeddings[:,identity.visual_mask[0]]),
                    prefill_logits=typed_digest(logits),history_kv_tokens=diagnostics['requests'][0]['reused_tokens'],
                    scope='Old failure scene: regression diagnostic only, not independent quality/performance')
                with (dest/'per_request.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                records.append(record)
                print('NO_CORE',snapshot.frame_id,qi,checks,flush=True)
                if not all(checks.values()):raise ValueError('No-Core Native equivalence failed')
    write(dest/'summary.json',dict(passed=all(all(r['checks'].values()) for r in records),requests=len(records),frames=len(paths),
        tests=['actual expanded visual embeddings','positions/mask/IDs','prefill last-token logits','full free-generation tokens'],
        original_generation='Unchanged sample temperature .1/top_p .75/max_new_tokens 320; captured RNG per question',
        independent_acceptance=False,base_checkpoint_sha256=sha256('ckpts/iter_10548.pth')))


def core_quality(root):
    """Entire frozen development validation sequence, paired at identical RNG.

    This is AFTER checkpoint selection, before opening final acceptance logs.
    Parser flags are review aids; they never certify physical truth.
    """
    import projects.mmdet3d_plugin
    from tools.worldstream_m123.benchmark.worker import effective_config,load_model
    from tools.worldstream_m123.benchmark.quality_calibration import paired_quality
    from worldstream.m3 import M3Runtime,M3Config
    from worldstream.m3.visual_state import base_model,CarrierLayout
    dest=root/'core_valid_quality'
    if dest.exists():raise FileExistsError('Preserve development diagnostic')
    manifest=json.loads((root/'training_manifest.json').read_text())
    training=json.loads((root/'adapter_training/training_result.json').read_text())
    if training['trained_steps']<1:raise ValueError('Trained SF Composer required')
    dest.mkdir(parents=True)
    cfg=effective_config('B0','language',dest)
    model=load_model(cfg,'ckpts/iter_10548.pth',dest).eval().requires_grad_(False)
    for name,module in model.named_children():
        if name!='lm_head':module.cpu()
    torch.cuda.empty_cache()
    runtime=M3Runtime(M3Config(prefix_layout='sf_adaptive',reuse_strategy='exact_prefix',
        prefix_adapter=str(root/'adapter_training/adapter.pt'),memory_budget_mb=0,
        semantic_qualification=str(root/'semantic_qualification.json')))
    layout=CarrierLayout(256,1,256);records=[]
    with torch.no_grad():
        for sample in manifest['samples']:
            if manifest['scene_splits'][sample['scene_id']]!='validation':continue
            capture=load_capture(sample['capture']);snapshot=capture['stable_snapshot']
            native=capture['carriers'].cuda()
            context=dict(capture['context'],stable_state_version=snapshot.state_version,profile=True)
            for pair in sample['pairs']:
                qi=pair['question_index'];ids=capture['ids'][qi].cuda();cpu_rng,cuda_rng=capture['rng'][qi]
                torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
                teacher=model.lm_head.generate(inputs=ids.reshape(1,-1),images=native,**capture['generation'])
                torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng)
                actual,diagnostics=runtime.generate(model.lm_head,[ids],native,context=context,
                    questions=[pair['question']],layout=layout,stable_world_state=snapshot,**capture['generation'])
                diag=diagnostics['requests'][0]
                if diag['reused_tokens']:raise ValueError('Representation diagnostic admitted history')
                native_answer=model.tokenizer.batch_decode(teacher,skip_special_tokens=True)[0]
                sf_answer=model.tokenizer.batch_decode(actual[0],skip_special_tokens=True)[0]
                quality=paired_quality(sf_answer,native_answer,pair['answer'])
                record=dict(scene_id=snapshot.scene_id,frame_id=snapshot.frame_id,question_index=qi,
                    question=pair['question'],reference=pair['answer'],native_answer=native_answer,sf_answer=sf_answer,
                    core_valid=diag['stable_anchor_valid'],anchor_id=diag['stable_anchor_id'],
                    visual_layout_mode=diag['visual_layout_mode'],native_tokens=teacher.cpu().tolist(),
                    sf_tokens=actual[0].cpu().tolist(),history_kv_tokens=0,quality=quality,
                    capture_sha256=sample['capture_sha256'],diagnostics=diag,
                    scope='Entire development validation sequence; same real current carriers/RNG; not final acceptance')
                if not record['core_valid'] and not torch.equal(teacher,actual[0]):
                    raise ValueError('No-core development input lost native equivalence')
                with (dest/'per_request.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                records.append(record)
                print('CORE_QUALITY',snapshot.frame_id,qi,record['core_valid'],quality['critical_regressions'],flush=True)
    def aggregate(rr):
        return dict(requests=len(rr),frames=len({r['frame_id'] for r in rr}),
            mean_native_token_f1=sum(r['quality']['baseline_f1'] for r in rr)/len(rr) if rr else None,
            mean_sf_token_f1=sum(r['quality']['f1'] for r in rr)/len(rr) if rr else None,
            identical_generated_tokens=sum(r['native_tokens']==r['sf_tokens'] for r in rr),
            critical_proxy_regressions=sum(r['quality']['critical_regressions'] for r in rr))
    write(dest/'summary.json',dict(core_valid=aggregate([r for r in records if r['core_valid']]),
        no_core=aggregate([r for r in records if not r['core_valid']]),
        trained_adapter_sha256=sha256(root/'adapter_training/adapter.pt'),
        semantic_qualification_sha256=sha256(root/'semantic_qualification.json'),
        human_critical_review_required=True,stage_two_pass=False,
        limitation='Changed spatial/motion/count/traffic claims need separate reference/image review; Teacher is not physical truth.'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--stage',choices=['no-core','structure','core-quality'],required=True);a=p.parse_args()
    if set(os.sched_getaffinity(0))&{4,5}:p.error('Use taskset -c 0-3,6-31')
    if a.stage=='structure':structure(Path(a.root))
    else:
        with gpu_job_lock():wait_gpu();(no_core if a.stage=='no-core' else core_quality)(Path(a.root))


if __name__=='__main__':main()
