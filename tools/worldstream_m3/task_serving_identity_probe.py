"""Fresh actual expanded-input identity mutations, including false->true masks."""
from dataclasses import replace
import hashlib
from time import perf_counter
from pathlib import Path
import torch
from worldstream.m3 import expand_input
from worldstream.m3.visual_state import base_model,execution_signature
from worldstream.m3.prefix_matcher import exact_lcp
from worldstream.m3.task_evidence import native_prompt_matches
from tools.worldstream_m1.io_utils import write_json


@torch.no_grad()
def verify(model,snapshots,path):
    lm=model.lm_head;device=base_model(lm).get_input_embeddings().weight.device;snap=snapshots[0]
    x=expand_input(lm,snap['ids'][0].to(device),snap['carriers'].to(device),scope=(snap['scene'],snap['frame']),signature=execution_signature(lm),source=(snap['scene'],snap['frame']))
    rows=[]
    for name in ('position_ids','attention_mask','token_ids','visual_mask'):
        y=x.prefix(x.length,clone=True);t=getattr(y,name);old=t[0,3].item()
        t[0,3]=not bool(old) if name=='visual_mask' else 0 if name=='attention_mask' else int(old)+1
        n=exact_lcp(x,y);rows.append(dict(field=name,index=3,old_value=old,new_value=t[0,3].item(),actual_value_changed=t[0,3].item()!=old,matching_prefix=n,passed=n==3))
    changed=snap['carriers'].to(device).clone();changed[0,0,0]+=1
    y=expand_input(lm,snap['ids'][0].to(device),changed,scope=x.scope,signature=x.signature)
    first_visual=int(x.visual_mask[0].nonzero()[0]);visual_prefix=exact_lcp(x,y)
    from projects.mmdet3d_plugin.datasets.utils.data_utils import preprocess
    question='What is the distance to observed object fixture-1 in meters?'
    raw=[[{'from':'human','value':'<image>\nYou are driving in boston. '+question},{'from':'gpt','value':''}]]
    native_ids=preprocess(raw,model.tokenizer,True,False)['input_ids'][0]
    bound=native_prompt_matches(model.tokenizer,native_ids,question)
    prefix_path=Path(path).parents[1]/'m2_oracle/prefix_only_profile.json'
    if not prefix_path.exists():
        from projects.mmdet3d_plugin.models.backbones.eva_vit import get_abs_pos
        image_input=snap.get('prefix_images')
        if image_input is None:
            from mmcv import Config
            from mmcv.parallel import collate,scatter
            from mmdet3d.datasets import build_dataset
            cfg=Config.fromfile('projects/configs/OmniDrive/worldstream_m1_coverage_observe.py');cfg.data.test.test_mode=True;dataset=build_dataset(cfg.data.test)
            i=next(i for i,v in enumerate(dataset.data_infos) if v['token']==snap['frame'])
            batch=scatter(collate([dataset[i]],samples_per_gpu=1),[torch.cuda.current_device()])[0];image_input=batch['img'][0]
        images=image_input.to(device).reshape(-1,3,640,640);backbone=model.img_backbone
        embeddings=backbone.patch_embed(images)
        if backbone.pos_embed is not None:embeddings=embeddings+get_abs_pos(backbone.pos_embed,backbone.pretrain_use_cls_token,embeddings.shape[1:3])
        for _ in range(3):backbone.blocks[1](backbone.blocks[0](embeddings))
        samples=[]
        for _ in range(20):
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);torch.cuda.synchronize();tick=perf_counter();a.record()
            backbone.blocks[1](backbone.blocks[0](embeddings));b.record();b.synchronize();samples.append(dict(wall_ms=(perf_counter()-tick)*1000,GPU_ms=a.elapsed_time(b)))
        write_json(prefix_path,dict(samples=samples,scope='Only actual original blocks 0/1 on normalized six-camera input of final-source full model; excludes PE/ABS. Impossible all-prefix-deletion upper bound; no partial-token speed estimate.',producer=str(Path(__file__)),scene=snap['scene'],frame=snap['frame'],online_qualified_saving_ms=0))
    import inspect,marshal
    caller=inspect.currentframe().f_back.f_code
    write_json(path,dict(caller_loaded_bytecode_sha256=hashlib.sha256(marshal.dumps(caller)).hexdigest(),actual_original_tokenizer_supported_question_binding=bound,scene=snap['scene'],frame=snap['frame'],input_length=x.length,rows=rows,
        changed_vision_prefix=visual_prefix,first_visual_index=first_visual,changed_vision_rejects_full_visual_prefix=visual_prefix==first_visual,
        scene_change=exact_lcp(x,replace(x,scope=('new_scene',))),model_signature_change=exact_lcp(x,replace(x,signature=('new_model',))),
        all_mutations_verified=all(r['passed'] and r['actual_value_changed'] for r in rows),
        first_supplement_probe_visual_mask_mutation_note='Initial auxiliary probe assigned 0 to a text token already False, so it was a no-op. Its raw results are retained; this fresh final-source witness toggles the actual value.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
